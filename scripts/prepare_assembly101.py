#!/usr/bin/env python3
"""Prepare Assembly101 coarse annotations for the ActionLLM SDR pipeline.

Expected source layout:
  data/assembly101/
    annotations/coarse-annotations/
      actions.csv
      coarse_labels/
      coarse_splits/          # optional but preferred
    TSM_features/{video_id}/{view}/features.npy

This script writes:
  data/assembly101/mapping.txt
  data/assembly101/splits/train.csv
  data/assembly101/splits/val.csv
"""

import argparse
import os
import re
from pathlib import Path

import pandas as pd


VIDEO_RE = re.compile(r"(nusar-[^/\s,]+)")


def normalize_video_id(text: str) -> str:
    text = str(text).strip().replace("\\", "/")
    text = text.split("/")[-1]
    for suffix in (".txt", ".mp4", ".avi", ".npy"):
        if text.endswith(suffix):
            text = text[: -len(suffix)]
    m = VIDEO_RE.search(text)
    return m.group(1) if m else text


def read_actions(actions_csv: Path):
    df = pd.read_csv(actions_csv)
    if {"action_id", "action_cls"}.issubset(df.columns):
        rows = df[["action_id", "action_cls"]].copy()
    elif {"id", "action"}.issubset(df.columns):
        rows = df[["id", "action"]].rename(columns={"id": "action_id", "action": "action_cls"})
    else:
        # Fall back to the first two columns.
        rows = df.iloc[:, :2].copy()
        rows.columns = ["action_id", "action_cls"]
    rows["action_id"] = rows["action_id"].astype(int)
    rows["action_cls"] = rows["action_cls"].astype(str)
    return rows.sort_values("action_id")


def write_mapping(actions: pd.DataFrame, out_path: Path):
    with open(out_path, "w", encoding="utf-8") as f:
        for _, row in actions.iterrows():
            f.write(f"{int(row['action_id'])} {row['action_cls']}\n")
    print(f"[write] {out_path}")


def scan_coarse_labels(label_dir: Path):
    info = {}
    action_labels = set()
    for path in sorted(label_dir.glob("*.txt")):
        stem = path.stem
        m = VIDEO_RE.search(stem)
        if not m:
            continue
        video_id = m.group(1)
        action_type = stem[: m.start()].rstrip("_") or "action_both"
        max_end = 0
        with open(path, "r", encoding="utf-8") as f:
            for raw in f:
                raw = raw.strip()
                if not raw:
                    continue
                parts = raw.split("\t")
                if len(parts) < 3:
                    parts = raw.split()
                if len(parts) >= 2:
                    max_end = max(max_end, int(float(parts[1])))
                if len(parts) >= 3:
                    action_labels.add(parts[2])
        info[video_id] = {
            "action_type": action_type,
            "video_end_frame": max_end,
            "label_file": str(path),
        }
    if not info:
        raise RuntimeError(f"No coarse label files found in {label_dir}")
    print(f"[scan] coarse label videos={len(info)}, action labels={len(action_labels)}")
    return info, action_labels


def actions_from_labels(action_labels):
    rows = [{"action_id": i, "action_cls": name} for i, name in enumerate(sorted(action_labels))]
    return pd.DataFrame(rows)


def read_split_ids(split_dir: Path):
    splits = {"train": set(), "val": set()}
    if not split_dir.exists():
        print(f"[warn] no coarse_splits dir found: {split_dir}")
        return splits

    for path in sorted(split_dir.rglob("*")):
        if not path.is_file():
            continue
        name = path.name.lower()
        if "train" in name and "val" not in name:
            bucket = "train"
        elif "val" in name or "valid" in name or "test" in name:
            # Assembly101 public test labels are not used; validation is used
            # as the evaluation split in prior LTAA work.
            bucket = "val"
        else:
            continue
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            for raw in f:
                raw = raw.strip()
                if raw:
                    splits[bucket].add(normalize_video_id(raw.split(",")[0]))

    print(f"[split] train ids={len(splits['train'])}, val ids={len(splits['val'])}")
    return splits


def scan_features(features_root: Path):
    rows = []
    for path in sorted(features_root.glob("*/*/features.npy")):
        rows.append({"video_id": path.parents[1].name, "view": path.parent.name})
    if not rows:
        raise RuntimeError(f"No features.npy found under {features_root}")
    df = pd.DataFrame(rows).drop_duplicates()
    print(f"[scan] feature video-view rows={len(df)}, videos={df['video_id'].nunique()}")
    return df


def build_metadata(feature_df: pd.DataFrame, label_info: dict, split_ids: dict):
    rows = []
    missing_labels = 0
    for row in feature_df.to_dict("records"):
        video_id = row["video_id"]
        if video_id not in label_info:
            missing_labels += 1
            continue
        item = {
            "video_id": video_id,
            "view": row["view"],
            "action_type": label_info[video_id]["action_type"],
            "video_end_frame": int(label_info[video_id]["video_end_frame"]),
        }
        if video_id in split_ids["train"]:
            split = "train"
        elif video_id in split_ids["val"]:
            split = "val"
        else:
            split = None
        rows.append((split, item))

    if missing_labels:
        print(f"[warn] feature rows without coarse labels: {missing_labels}")

    train = [item for split, item in rows if split == "train"]
    val = [item for split, item in rows if split == "val"]

    if not train or not val:
        print("[warn] split files did not produce both train and val; using deterministic 80/20 video split")
        videos = sorted({item["video_id"] for _, item in rows})
        cut = max(1, int(len(videos) * 0.8))
        train_videos = set(videos[:cut])
        train = [item for _, item in rows if item["video_id"] in train_videos]
        val = [item for _, item in rows if item["video_id"] not in train_videos]

    return pd.DataFrame(train), pd.DataFrame(val)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", default="/root/autodl-tmp/ActionLLM/data")
    parser.add_argument("--dataset", default="assembly101")
    args = parser.parse_args()

    data_path = Path(args.data_root) / args.dataset
    annotation_root = data_path / "annotations" / "coarse-annotations"
    features_root = data_path / "TSM_features"
    split_dir = annotation_root / "coarse_splits"
    label_dir = annotation_root / "coarse_labels"
    out_split_dir = data_path / "splits"
    out_split_dir.mkdir(parents=True, exist_ok=True)

    actions = read_actions(annotation_root / "actions.csv")
    label_info, label_actions = scan_coarse_labels(label_dir)
    action_set = set(actions["action_cls"].astype(str).tolist())
    if len(actions) < 50 or not label_actions.issubset(action_set):
        missing = sorted(label_actions - action_set)
        print(
            "[warn] actions.csv does not cover coarse label actions; "
            f"using labels scanned from coarse_labels instead. missing={len(missing)}"
        )
        if missing[:10]:
            print("[warn] missing examples:", ", ".join(missing[:10]))
        actions = actions_from_labels(label_actions)
    write_mapping(actions, data_path / "mapping.txt")

    split_ids = read_split_ids(split_dir)
    feature_df = scan_features(features_root)
    train_df, val_df = build_metadata(feature_df, label_info, split_ids)

    train_path = out_split_dir / "train.csv"
    val_path = out_split_dir / "val.csv"
    train_df.to_csv(train_path, index=False)
    val_df.to_csv(val_path, index=False)
    print(f"[write] {train_path} rows={len(train_df)}, videos={train_df['video_id'].nunique() if len(train_df) else 0}")
    print(f"[write] {val_path} rows={len(val_df)}, videos={val_df['video_id'].nunique() if len(val_df) else 0}")
    if len(train_df):
        print("[train views]")
        print(train_df["view"].value_counts().sort_index().to_string())
    if len(val_df):
        print("[val views]")
        print(val_df["view"].value_counts().sort_index().to_string())
    all_views = sorted(set(train_df.get("view", [])) | set(val_df.get("view", [])))
    print(f"[views used] {len(all_views)} views: {', '.join(all_views)}")


if __name__ == "__main__":
    main()

