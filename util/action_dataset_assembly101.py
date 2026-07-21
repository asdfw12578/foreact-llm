import os
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import torch.utils.data as Data


class Assembly101ActionDataset(Data.Dataset):
    """Assembly101 coarse-action dataset for the SDR ActionLLM pipeline.

    The loader follows the GTDA/MANTA Assembly101 protocol: features are stored
    as ``TSM_features/{video_id}/{view}/features.npy`` with shape [D, T], and
    coarse labels are stored as segment files under ``coarse-annotations``.
    """

    def __init__(self, args, mode, actions_dict, data_path, n_class, pad_idx):
        super().__init__()
        assert mode in {"train", "val"}, f"unsupported mode: {mode}"

        self.args = args
        self.mode = mode
        self.actions_dict = actions_dict
        self.sample_rate = int(args.sample_rate)
        self.pred_ratio = float(args.pred_ratio)
        self.n_class = n_class
        self.pad_idx = pad_idx
        self.NONE = self.n_class - 1
        self.n_query = int(args.n_query)

        self.data_path = Path(data_path)
        self.features_root = self.data_path / "TSM_features"
        self.annotation_root = self.data_path / "annotations" / "coarse-annotations"
        self.coarse_label_root = self.annotation_root / "coarse_labels"
        self.metadata_path = self.data_path / "splits" / f"{mode}.csv"

        if not self.metadata_path.exists():
            raise FileNotFoundError(
                f"Assembly101 metadata not found: {self.metadata_path}. "
                "Run scripts/prepare_assembly101.py first."
            )
        if not self.coarse_label_root.exists():
            raise FileNotFoundError(f"coarse_labels not found: {self.coarse_label_root}")

        self.metadata = pd.read_csv(self.metadata_path)
        required = {"video_id", "view", "action_type", "video_end_frame"}
        missing = required - set(self.metadata.columns)
        if missing:
            raise ValueError(f"{self.metadata_path} is missing columns: {sorted(missing)}")

        self.base_samples = []
        skipped = 0
        for row in self.metadata.to_dict("records"):
            sample = {
                "video_id": str(row["video_id"]),
                "view": str(row["view"]),
                "action_type": str(row["action_type"]),
                "video_end_frame": int(row["video_end_frame"]),
            }
            if self._feature_path(sample).is_file() and self._label_path(sample).is_file():
                self.base_samples.append(sample)
            else:
                skipped += 1

        if not self.base_samples:
            raise RuntimeError(
                f"No usable Assembly101 samples found from {self.metadata_path}. "
                "Check TSM_features and coarse_labels paths."
            )

        self.samples = []
        if self.mode == "train":
            # Match the GTDA/MANTA training protocol: random observation during
            # training is approximated here by multiple fixed observation ratios.
            obs_list = [0.2, 0.3, 0.5]
        else:
            obs_list = [0.2, 0.3]

        for sample in self.base_samples:
            for obs in obs_list:
                self.samples.append((sample, obs))

        print(
            f"Assembly101 {mode}: base_samples={len(self.base_samples)}, "
            f"expanded_samples={len(self.samples)}, skipped={skipped}, "
            f"sample_rate={self.sample_rate}"
        )

        # Fail fast on the first item, as the original ActionDataset does.
        self._make_input(*self.samples[0])

    def _feature_path(self, sample):
        return self.features_root / sample["video_id"] / sample["view"] / "features.npy"

    def _label_path(self, sample):
        name = f"{sample['action_type']}_{sample['video_id']}.txt"
        return self.coarse_label_root / name

    def _load_segmentation(self, sample):
        labels = []
        start_indices = []
        end_indices = []
        label_path = self._label_path(sample)

        with open(label_path, "r", encoding="utf-8") as f:
            for raw in f:
                raw = raw.strip()
                if not raw:
                    continue
                parts = raw.split("\t")
                if len(parts) < 3:
                    parts = raw.split()
                if len(parts) < 3:
                    continue

                start = int(float(parts[0]))
                end = int(float(parts[1]))
                action = parts[2]
                if action not in self.actions_dict:
                    raise KeyError(f"Unknown action '{action}' in {label_path}")
                if end <= start:
                    continue

                start_indices.append(start)
                end_indices.append(end)
                labels.extend([action] * (end - start))

        if not labels:
            raise RuntimeError(f"Empty segmentation: {label_path}")

        start_frame = min(start_indices)
        end_frame = min(max(end_indices), int(sample["video_end_frame"]))
        expected_len = max(0, end_frame - start_frame)
        labels = labels[:expected_len]
        return labels, start_frame, end_frame

    def _load_features(self, sample, start_frame, end_frame):
        feature_file = self._feature_path(sample)
        features = np.load(feature_file, mmap_mode="r")  # [D, T]
        if features.ndim != 2:
            raise ValueError(f"Expected [D,T] features, got {features.shape}: {feature_file}")

        max_t = features.shape[1]
        start_frame = max(0, min(start_frame, max_t))
        end_frame = max(start_frame, min(end_frame, max_t))
        features = features[:, start_frame:end_frame].T  # [T, D]
        return np.asarray(features, dtype=np.float32)

    def _make_input(self, sample, obs_perc):
        if self.mode == "train" and random.random() < 0.4:
            # Match GTDA/MANTA Assembly101 training: a subset of batches uses
            # a random observation ratio in [0.15, 0.40].
            obs_perc = 0.15 + 0.25 * random.random()

        content, start_frame, end_frame = self._load_segmentation(sample)
        features = self._load_features(sample, start_frame, end_frame)

        min_len = min(len(content), features.shape[0])
        content = content[:min_len]
        features = features[:min_len]

        vid_len = len(content)
        observed_len = min(max(1, int(float(obs_perc) * vid_len)), vid_len)
        pred_len = max(1, int(self.pred_ratio * vid_len))
        future_start = observed_len
        future_end = min(vid_len, observed_len + pred_len)

        past_features = features[:observed_len][:: self.sample_rate]
        # Assembly101 has no text feature branch in this setup. The SDR model
        # expects text tokens to already live in the LLaMA hidden space, while
        # visual TSM features are still projected from 2048 to that space by
        # adapter_proj. For LLaMA-7B this hidden dimension is 4096.
        text_dim = int(getattr(self.args, "text_feature_dim", 4096))
        text_features = np.zeros((past_features.shape[0], text_dim), dtype=np.float32)

        past_content = content[:observed_len][:: self.sample_rate]
        if len(past_content) == 0 or past_features.shape[0] == 0:
            raise RuntimeError(f"Empty sampled sequence for {sample}")
        past_label = torch.tensor(self.seq2idx(past_content), dtype=torch.long)

        future_content = content[future_start:future_end][:: self.sample_rate]
        if len(future_content) == 0:
            future_content = [content[-1]]

        trans_future, trans_future_dur = self.seq2transcript(future_content)
        trans_future = np.append(trans_future, self.NONE)

        trans_seq_len = len(trans_future)
        diff = self.n_query - trans_seq_len
        if diff > 0:
            trans_future = np.concatenate((trans_future, np.ones(diff) * self.pad_idx))
            trans_future_dur = np.concatenate((trans_future_dur, np.ones(diff + 1) * self.pad_idx))
        elif diff < 0:
            trans_future = trans_future[: self.n_query]
            trans_future_dur = trans_future_dur[: self.n_query]
        else:
            trans_future_dur = np.concatenate((trans_future_dur, np.ones(1) * self.pad_idx))

        return {
            "inputs_embeds": torch.tensor(past_features, dtype=torch.float32),
            "text_inputs_embeds": torch.tensor(text_features, dtype=torch.float32),
            "labels_action": torch.tensor(trans_future, dtype=torch.long),
            "past_labels": past_label,
            "labels_duration": torch.tensor(trans_future_dur, dtype=torch.float32),
        }

    def __getitem__(self, idx):
        sample, obs_perc = self.samples[idx]
        return self._make_input(sample, obs_perc)

    def __len__(self):
        return len(self.samples)

    def my_collate(self, batch):
        b_features = [item["inputs_embeds"] for item in batch]
        b_text_features = [item["text_inputs_embeds"] for item in batch]
        b_labels_action = [item["labels_action"] for item in batch]
        b_past_labels = [item["past_labels"] for item in batch]
        b_labels_duration = [item["labels_duration"] for item in batch]

        sizes = [t.shape[0] for t in b_past_labels]
        max_size = max(sizes)
        sequence_lengths = [len(seq) for seq in b_features]
        padding_lengths = [max_size - length for length in sequence_lengths]

        inputs_embeds = torch.nn.utils.rnn.pad_sequence(
            [F.pad(seq, (0, 0, padding, 0), value=0) for seq, padding in zip(b_features, padding_lengths)],
            batch_first=True,
            padding_value=0,
        )
        text_inputs_embeds = torch.nn.utils.rnn.pad_sequence(
            [F.pad(seq, (0, 0, padding, 0), value=0) for seq, padding in zip(b_text_features, padding_lengths)],
            batch_first=True,
            padding_value=0,
        )
        past_labels = torch.nn.utils.rnn.pad_sequence(
            [F.pad(seq, (padding, 0), value=-100) for seq, padding in zip(b_past_labels, padding_lengths)],
            batch_first=True,
            padding_value=-100,
        )
        labels_action = torch.nn.utils.rnn.pad_sequence(
            b_labels_action, batch_first=True, padding_value=self.pad_idx
        )
        labels_duration = torch.nn.utils.rnn.pad_sequence(
            b_labels_duration, batch_first=True, padding_value=self.pad_idx
        )
        return [inputs_embeds, text_inputs_embeds, past_labels, labels_action, labels_duration]

    def shuffle_list(self, values):
        random.shuffle(values)

    def seq2idx(self, seq):
        return np.asarray([self.actions_dict[x] for x in seq], dtype=np.int64)

    def seq2transcript(self, seq):
        transcript_action = []
        transcript_dur = []
        action = seq[0]
        transcript_action.append(self.actions_dict[action])
        last_i = 0
        for i, label in enumerate(seq):
            if action != label:
                action = label
                transcript_action.append(self.actions_dict[action])
                transcript_dur.append((i - last_i) / max(1, len(seq)))
                last_i = i
        transcript_dur.append((len(seq) - last_i) / max(1, len(seq)))
        return np.asarray(transcript_action), np.asarray(transcript_dur)

