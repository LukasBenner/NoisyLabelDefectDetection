"""Rebuild the noisy dataset from the baseline dataset using the mapping CSV.

Usage:
    python scripts/build_noisy_from_baseline.py [--dry-run]

Reads:  scripts/noisy_mapping.csv
Source: /Volumes/SamsungT5/data/surfaceClassification/baseline
Output: /Volumes/SamsungT5/data/surfaceClassification/noisy
"""

import argparse
import csv
import shutil
from pathlib import Path

BASELINE = Path("/Volumes/SamsungT5/data/surfaceClassification/baseline")
OUTPUT = Path("/Volumes/SamsungT5/data/surfaceClassification/noisy")
MAPPING = Path(__file__).parent / "noisy_mapping.csv"


def main():
    parser = argparse.ArgumentParser(description="Build noisy dataset from baseline + mapping CSV")
    parser.add_argument("--dry-run", action="store_true", help="Print actions without copying")
    args = parser.parse_args()

    with open(MAPPING) as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    # Collect target classes to create directories
    target_classes = {r["noisy_class"] for r in rows if r["action"] in ("keep", "move")}

    if not args.dry_run:
        OUTPUT.mkdir(parents=True, exist_ok=True)
        for cls in sorted(target_classes):
            (OUTPUT / cls).mkdir(parents=True, exist_ok=True)

    copied, skipped, removed = 0, 0, 0
    for row in rows:
        filename = row["filename"]
        action = row["action"]
        baseline_class = row["baseline_class"]
        noisy_class = row["noisy_class"]

        if action == "remove":
            removed += 1
            continue

        if action in ("keep", "move"):
            src = BASELINE / baseline_class / filename
            dst = OUTPUT / noisy_class / filename

            if not src.exists():
                print(f"WARNING: source not found: {src}")
                skipped += 1
                continue

            if args.dry_run:
                print(f"COPY {src} -> {dst}")
            else:
                shutil.copy2(src, dst)
            copied += 1

    print(f"\nDone: {copied} copied, {removed} removed (skipped), {skipped} missing sources")
    if args.dry_run:
        print("(dry run — no files were actually copied)")


if __name__ == "__main__":
    main()
