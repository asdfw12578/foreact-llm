import argparse
import csv
import gc
import os
import re
from pathlib import Path

import numpy as np
import pandas as pd
import torch

import util.misc as misc
from foreactllm.builder import foreactllm
from util.action_tool import normalize_duration, read_mapping_dict
from util.opts import get_args_parser


def checkpoint_number(path):
    match = re.search(r"checkpoint-(\d+)\.pth$", str(path))
    if match is None:
        raise ValueError(f"Cannot parse checkpoint number from: {path}")
    return int(match.group(1))


def label_path(row, data_path):
    return (
        Path(data_path)
        / "annotations"
        / "coarse-annotations"
        / "coarse_labels"
        / f"{row['action_type']}_{row['video_id']}.txt"
    )


def feature_path(row, data_path):
    return Path(data_path) / "TSM_features" / str(row["video_id"]) / str(row["view"]) / "features.npy"


def load_segmentation(row, data_path, actions_dict):
    labels, starts, ends = [], [], []
    path = label_path(row, data_path)
    with open(path, "r", encoding="utf-8") as f:
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
            if action not in actions_dict or end <= start:
                continue
            starts.append(start)
            ends.append(end)
            labels.extend([action] * (end - start))
    if not labels:
        raise RuntimeError(f"empty labels: {path}")
    start_frame = min(starts)
    end_frame = min(max(ends), int(row["video_end_frame"]))
    labels = labels[: max(0, end_frame - start_frame)]
    return labels, start_frame, end_frame


def load_features(row, data_path, start_frame, end_frame):
    path = feature_path(row, data_path)
    features = np.load(path, mmap_mode="r")  # [D, T]
    start_frame = max(0, min(start_frame, features.shape[1]))
    end_frame = max(start_frame, min(end_frame, features.shape[1]))
    return np.asarray(features[:, start_frame:end_frame].T, dtype=np.float32)


def eval_file(gt_content, recog_content, obs_percentage, classes):
    last_frame = min(len(recog_content), len(gt_content))
    recognized = recog_content[int(obs_percentage * len(gt_content)) : last_frame]
    ground_truth = gt_content[int(obs_percentage * len(gt_content)) : last_frame]
    n_t = np.zeros(len(classes))
    n_f = np.zeros(len(classes))
    for gt, pred in zip(ground_truth, recognized):
        if gt == pred:
            n_t[classes[gt]] += 1
        else:
            n_f[classes[gt]] += 1
    return n_t, n_f


def eval_all_none(gt_content, obs_percentage, classes, pred_percentage):
    ground_truth = gt_content[
        int(obs_percentage * len(gt_content)) : int((obs_percentage + pred_percentage) * len(gt_content))
    ]
    n_t = np.zeros(len(classes))
    n_f = np.zeros(len(classes))
    for gt in ground_truth:
        n_f[classes[gt]] += 1
    return n_t, n_f


def predict_one(model, features, gt_seq, obs_p, args, n_class, actions_dict, device):
    pred_p = 0.5
    none_id = n_class - 1
    id_to_action = {v: k for k, v in actions_dict.items()}
    id_to_action[none_id] = "NONE"

    vid_len = len(gt_seq)
    past_len = max(1, int(obs_p * vid_len))
    future_len = max(1, int(pred_p * vid_len))

    past_seq = np.asarray(gt_seq[:past_len])
    inputs = torch.tensor(features[:past_len], dtype=torch.float32, device=device).unsqueeze(0)
    # Text branch is disabled for Assembly101, but SDR expects text tokens in
    # the LLaMA hidden space before adapter_multi_down.
    text_inputs = torch.zeros((inputs.shape[0], inputs.shape[1], 4096), dtype=inputs.dtype, device=device)

    with torch.no_grad():
        outputs = model(
            inputs_embeds=inputs,
            text_inputs_embeds=text_inputs,
            labels_action=None,
            past_labels=None,
            labels_duration=None,
            return_preds=True,
        )

    output_action = outputs["action"]
    output_dur = outputs["duration"]
    output_label = output_action.max(-1)[1]

    none_idx = None
    for i in range(output_label.size(1)):
        if int(output_label[0, i].item()) == none_id:
            none_idx = i
            break
    if none_idx == 0:
        return None

    none_mask = torch.ones_like(output_label, dtype=torch.bool, device=device)
    if none_idx is not None:
        none_mask[:, none_idx:] = False
    output_dur = normalize_duration(output_dur, none_mask)

    pred_len = (0.5 + future_len * output_dur).squeeze(0).long()
    pred_len = torch.cat([torch.zeros(1, device=device, dtype=torch.long), pred_len], dim=0)

    predicted = torch.ones(future_len, device=device, dtype=torch.long) * none_id
    actions = output_label.squeeze(0)
    for i in range(len(actions)):
        action_id = int(actions[i].item())
        if action_id not in id_to_action:
            action_id = none_id
        start = int(pred_len[i].item())
        end = int((pred_len[i] + pred_len[i + 1]).item()) if i + 1 < len(pred_len) else future_len
        start = max(0, min(start, future_len))
        end = max(start, min(end, future_len))
        predicted[start:end] = action_id
        if i + 1 < len(pred_len):
            pred_len[i + 1] = pred_len[i] + pred_len[i + 1]
        if i == len(actions) - 1:
            predicted[start:] = action_id

    pred_names = [id_to_action.get(int(x.item()), "NONE") for x in predicted]
    return np.concatenate([past_seq, np.asarray(pred_names)])


def evaluate_checkpoint(model, eval_rows, args, n_class, actions_dict, device, data_path):
    obs_ps = [0.2, 0.3]
    eval_ps = [0.1, 0.2, 0.3, 0.5]
    results = {}

    model.eval()
    for obs_p in obs_ps:
        t_actions = np.zeros((len(eval_ps), len(actions_dict)))
        f_actions = np.zeros((len(eval_ps), len(actions_dict)))

        for idx, row in enumerate(eval_rows):
            if idx > 0 and idx % 100 == 0:
                print(f"  [progress] obs={obs_p:.1f} {idx}/{len(eval_rows)}", flush=True)
            try:
                labels, start_frame, end_frame = load_segmentation(row, data_path, actions_dict)
                features = load_features(row, data_path, start_frame, end_frame)
            except Exception as exc:
                print(f"  [skip] {idx}: {exc}", flush=True)
                continue

            length = min(len(labels), features.shape[0])
            labels = labels[:length][:: args.sample_rate]
            features = features[:length][:: args.sample_rate]
            if len(labels) < 5 or features.shape[0] < 5:
                continue

            prediction = predict_one(model, features, labels, obs_p, args, n_class, actions_dict, device)
            for i, pred_p in enumerate(eval_ps):
                if prediction is None:
                    t, f = eval_all_none(labels, obs_p, actions_dict, pred_p)
                else:
                    eval_len = int((obs_p + pred_p) * len(labels))
                    t, f = eval_file(labels, prediction[:eval_len], obs_p, actions_dict)
                t_actions[i] += t
                f_actions[i] += f

        for i, pred_p in enumerate(eval_ps):
            acc = 0.0
            n = 0
            total = t_actions[i] + f_actions[i]
            for cls_idx in range(len(actions_dict)):
                if total[cls_idx] != 0:
                    acc += float(t_actions[i, cls_idx] / total[cls_idx])
                    n += 1
            moc = float(acc / max(1, n))
            key = f"obs{obs_p:.1f}_pred{pred_p:.1f}"
            results[key] = moc
            print(f"  obs {obs_p:.1f}, pred {pred_p:.1f} --> MoC: {moc:.4f}", flush=True)

    results["avg"] = sum(results.values()) / 8.0
    print(f"  Avg MoC: {results['avg']:.4f}", flush=True)
    return results


def read_completed(csv_path):
    if not csv_path.exists():
        return set()
    completed = set()
    with open(csv_path, "r", newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if row.get("checkpoint"):
                completed.add(int(row["checkpoint"]))
    return completed


def append_csv(csv_path, ck_num, results):
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "checkpoint",
        "obs0.2_pred0.1",
        "obs0.2_pred0.2",
        "obs0.2_pred0.3",
        "obs0.2_pred0.5",
        "obs0.3_pred0.1",
        "obs0.3_pred0.2",
        "obs0.3_pred0.3",
        "obs0.3_pred0.5",
        "avg",
    ]
    write_header = not csv_path.exists()
    with open(csv_path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        if write_header:
            writer.writeheader()
        row = {"checkpoint": ck_num}
        row.update({key: f"{results[key]:.6f}" for key in fields if key != "checkpoint"})
        writer.writerow(row)


def main(args):
    device = torch.device(args.device)
    seed = args.seed + misc.get_rank()
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    data_path = os.path.join(args.data_root, args.dataset)
    actions_dict = read_mapping_dict(os.path.join(data_path, "mapping.txt"))
    n_class = len(actions_dict) + 1
    args.action_class = len(actions_dict)

    val_csv = Path(data_path) / "splits" / "val.csv"
    eval_rows = pd.read_csv(val_csv).to_dict("records")

    checkpoints = sorted(Path(args.checkpoint_dir).glob(args.checkpoint_pattern), key=checkpoint_number)
    checkpoints = [p for p in checkpoints if checkpoint_number(p) >= args.eval_start_epoch]
    if args.eval_end_epoch >= 0:
        checkpoints = [p for p in checkpoints if checkpoint_number(p) <= args.eval_end_epoch]
    if not checkpoints:
        raise FileNotFoundError(f"No checkpoints to evaluate under {args.checkpoint_dir}")

    csv_path = Path(args.eval_csv)
    completed = read_completed(csv_path)

    print(f"[multi-eval] building SDR model once for {len(checkpoints)} checkpoints", flush=True)
    print(f"[multi-eval] dataset={data_path}, val rows={len(eval_rows)}, classes={len(actions_dict)}", flush=True)
    model = ActionLLM(args)
    model.to(device)
    model_dict = model.state_dict()

    critical_keys = {
        "output_action.weight",
        "output_seg_vis.weight",
        "output_seg_text.weight",
        "adapter_output_duration.weight",
    }

    for checkpoint_path in checkpoints:
        ck_num = checkpoint_number(checkpoint_path)
        if ck_num in completed:
            print(f"[skip] checkpoint-{ck_num} already evaluated", flush=True)
            continue

        print(f"===== checkpoint-{ck_num} =====", flush=True)
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)["model"]
        missing_critical = sorted(critical_keys - set(checkpoint.keys()))
        if missing_critical:
            raise RuntimeError(f"{checkpoint_path} is missing critical keys: {missing_critical}")

        pretrained = {key: value for key, value in checkpoint.items() if key in model_dict}
        missing, unexpected = model.load_state_dict(pretrained, strict=False)
        critical_missing_after_load = sorted(k for k in critical_keys if k in missing)
        if critical_missing_after_load:
            raise RuntimeError(f"Critical keys still missing after load: {critical_missing_after_load}")
        print(f"[load] keys={len(pretrained)}, missing={len(missing)}, unexpected={len(unexpected)}", flush=True)
        del checkpoint, pretrained
        gc.collect()

        results = evaluate_checkpoint(model, eval_rows, args, n_class, actions_dict, device, data_path)
        append_csv(csv_path, ck_num, results)
        completed.add(ck_num)
        print(f"[saved] {csv_path}", flush=True)

    print("[multi-eval] done", flush=True)


if __name__ == "__main__":
    parser = get_args_parser()
    parser.add_argument("--checkpoint_dir", required=True)
    parser.add_argument("--checkpoint_pattern", default="checkpoint-*.pth")
    parser.add_argument("--eval_start_epoch", type=int, default=10)
    parser.add_argument("--eval_end_epoch", type=int, default=-1)
    parser.add_argument("--eval_csv", required=True)
    main(parser.parse_args())

