"""Apply the cluster-aware split CSV to create train/val/test directories.

Usage:
    python scripts/apply_split.py --dry-run
    python scripts/apply_split.py

Reads:  notebooks/data_cleaning/artifacts/noisy/cluster_aware_split_70_15_15.csv
Source: <source_path>/noisy  (flat class folders)
Output: <output_path>/noisy_clustered/{train,val,test}/<class>/<file>
"""

import argparse
import csv
import shutil
from pathlib import Path

DEFAULT_SOURCE = Path("/Volumes/SamsungT5/data/surfaceClassification/noisy")
DEFAULT_OUTPUT = Path("/Volumes/SamsungT5/data/surfaceClassification/noisy_temporal")
SPLIT_CSV = Path(__file__).parent / "../notebooks/data_cleaning/artifacts/noisy/temporal_split_70_15_15.csv"


def main():
    parser = argparse.ArgumentParser(description="Apply cluster-aware split to noisy dataset")
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE, help="Source noisy dataset path")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT, help="Output split dataset path")
    parser.add_argument("--split-csv", type=Path, default=SPLIT_CSV, help="Path to split CSV")
    parser.add_argument("--dry-run", action="store_true", help="Print actions without copying")
    args = parser.parse_args()

    with open(args.split_csv) as f:
        rows = list(csv.DictReader(f))

    print(f"Loaded {len(rows)} entries from {args.split_csv}")

    # Create directories
    splits = {r["split"] for r in rows}
    classes = {r["label"] for r in rows}
    if not args.dry_run:
        for split in splits:
            for cls in classes:
                (args.output / split / cls).mkdir(parents=True, exist_ok=True)

    copied, skipped = 0, 0
    for row in rows:
        # file_path is like /corrosion/20250513-110537_14.jpg
        src = args.source / row["file_path"].lstrip("/")
        dst = args.output / row["split"] / row["label"] / src.name

        if not src.exists():
            print(f"WARNING: missing {src}")
            skipped += 1
            continue

        if args.dry_run:
            print(f"COPY {src} -> {dst}")
        else:
            shutil.copy2(src, dst)
        copied += 1

    print(f"\nDone: {copied} copied, {skipped} missing")
    if args.dry_run:
        print("(dry run — no files were actually copied)")


if __name__ == "__main__":
    main()
