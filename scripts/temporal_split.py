"""Assign entire capture sessions (dates) to train/val/test splits.

Images from the same session stay together to prevent data leakage.
Sessions are assigned greedily (largest-first) to whichever split is
furthest below its target ratio.

Usage:
    python scripts/temporal_split.py --dataset noisy
    python scripts/temporal_split.py --dataset baseline
    python scripts/temporal_split.py --dataset both
"""

import argparse
import re
from pathlib import Path

import pandas as pd

ARTIFACT_DIR = Path(__file__).resolve().parent / "../notebooks/data_cleaning/artifacts"

DATASETS = {
    "noisy": {
        "input_csv": ARTIFACT_DIR / "noisy/noisy_images.csv",
        "output_csv": ARTIFACT_DIR / "noisy/temporal_split_70_15_15.csv",
        "filename_col": "image_id",
    },
    "baseline": {
        "input_csv": ARTIFACT_DIR / "baseline/baseline_images_no_duplicates.csv",
        "output_csv": ARTIFACT_DIR / "baseline/temporal_split_70_15_15.csv",
        "filename_col": "file_path",
    },
}

TARGET_RATIOS = {"train": 0.70, "val": 0.15, "test": 0.15}
DATE_PATTERN = re.compile(r"(\d{8}-\d{6})")


def extract_date(s: str) -> pd.Timestamp | None:
    m = DATE_PATTERN.search(s)
    if m:
        return pd.to_datetime(m.group(1), format="%Y%m%d-%H%M%S")
    return None


def assign_sessions(session_sizes: dict[str, int], target_ratios: dict[str, float]) -> dict[str, str]:
    """Greedy largest-first assignment of sessions to splits."""
    total = sum(session_sizes.values())
    targets = {k: v * total for k, v in target_ratios.items()}
    current = {k: 0 for k in target_ratios}

    # Sort sessions largest first for better packing
    sorted_sessions = sorted(session_sizes.items(), key=lambda x: -x[1])

    assignment = {}
    for session, size in sorted_sessions:
        # Pick the split that is furthest below its target
        best_split = min(targets, key=lambda s: current[s] / targets[s])
        assignment[session] = best_split
        current[best_split] += size

    return assignment


def temporal_split(dataset_key: str):
    cfg = DATASETS[dataset_key]
    df = pd.read_csv(cfg["input_csv"])

    # For baseline, filter out duplicates
    if "is_duplicate" in df.columns:
        before = len(df)
        df = df[df["is_duplicate"] == False].copy()
        print(f"  Filtered duplicates: {before} -> {len(df)}")

    # Extract date (session = calendar date)
    df["datetime"] = df[cfg["filename_col"]].apply(extract_date)
    no_date_mask = df["datetime"].isna()
    n_no_date = no_date_mask.sum()
    if n_no_date:
        print(f"  {n_no_date} images without parseable date -> assigned to train")

    df["session"] = df["datetime"].dt.date.astype(str)
    df.loc[no_date_mask, "session"] = "__no_date__"

    session_sizes = df[~no_date_mask].groupby("session").size().to_dict()
    print(f"  {len(df)} images across {len(session_sizes)} sessions (+{n_no_date} undated)")

    assignment = assign_sessions(session_sizes, TARGET_RATIOS)
    assignment["__no_date__"] = "train"
    df["split"] = df["session"].map(assignment)

    # Report
    all_session_sizes = df.groupby("session").size().to_dict()
    print(f"\n  Session assignment:")
    for session in sorted(assignment):
        split = assignment[session]
        n = all_session_sizes[session]
        print(f"    {session}  ({n:4d} imgs) -> {split}")

    print(f"\n  Split sizes:")
    for split in ["train", "val", "test"]:
        subset = df[df["split"] == split]
        pct = len(subset) / len(df) * 100
        print(f"    {split:5s}: {len(subset):5d} ({pct:.1f}%)")

    print(f"\n  Class distribution per split:")
    for split in ["train", "val", "test"]:
        subset = df[df["split"] == split]
        dist = subset["label"].value_counts().sort_index()
        print(f"    {split}:")
        for cls, count in dist.items():
            total_cls = len(df[df["label"] == cls])
            print(f"      {cls:20s}: {count:4d} / {total_cls:4d} ({count/total_cls:.0%})")

    # Check for missing classes
    for split in ["train", "val", "test"]:
        missing = set(df["label"].unique()) - set(df[df["split"] == split]["label"].unique())
        if missing:
            print(f"\n  WARNING: {split} is missing classes: {missing}")

    # Save
    out_cols = [c for c in df.columns if c not in ("datetime", "session")]
    df[out_cols].to_csv(cfg["output_csv"], index=False)
    print(f"\n  Saved to {cfg['output_csv']}")


def main():
    parser = argparse.ArgumentParser(description="Temporal session-based dataset splitting")
    parser.add_argument("--dataset", choices=["noisy", "baseline", "both"], default="both")
    args = parser.parse_args()

    datasets = ["noisy", "baseline"] if args.dataset == "both" else [args.dataset]
    for ds in datasets:
        print(f"\n{'='*60}")
        print(f"Dataset: {ds}")
        print(f"{'='*60}")
        temporal_split(ds)


if __name__ == "__main__":
    main()
