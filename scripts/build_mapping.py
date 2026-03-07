"""Build a mapping table (CSV) showing how each file moves from baseline to noisy.

Usage:
    python scripts/build_mapping.py

Outputs: scripts/noisy_mapping.csv
"""

import os
import csv
from pathlib import Path

BASELINE = Path("/Volumes/SamsungT5/data/surfaceClassification/baseline")
NOISY = Path("/Volumes/SamsungT5/data/surfaceClassification/noisy")
OUTPUT = Path(__file__).parent / "noisy_mapping.csv"
EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff")


def scan_dataset(root: Path) -> dict[str, str]:
    file_map = {}
    for dirpath, _, files in os.walk(root):
        cls = os.path.basename(dirpath)
        for f in files:
            if f.lower().endswith(EXTENSIONS):
                file_map[f] = cls
    return file_map


def main():
    baseline_map = scan_dataset(BASELINE)
    noisy_map = scan_dataset(NOISY)

    all_files = sorted(set(baseline_map) | set(noisy_map))

    with open(OUTPUT, "w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["filename", "baseline_class", "noisy_class", "action"])

        for f in all_files:
            src = baseline_map.get(f)
            dst = noisy_map.get(f)

            if src and dst:
                action = "keep" if src == dst else "move"
            elif src and not dst:
                action = "remove"
            else:
                action = "add"

            writer.writerow([f, src or "", dst or "", action])

    # Print summary
    actions = {"keep": 0, "move": 0, "remove": 0, "add": 0}
    transitions = {}
    for f in all_files:
        src = baseline_map.get(f)
        dst = noisy_map.get(f)
        if src and dst:
            action = "keep" if src == dst else "move"
            key = (src, dst)
            transitions[key] = transitions.get(key, 0) + 1
        elif src:
            action = "remove"
        else:
            action = "add"
        actions[action] += 1

    print(f"Wrote {len(all_files)} entries to {OUTPUT}")
    print(f"  keep:   {actions['keep']}")
    print(f"  move:   {actions['move']}")
    print(f"  remove: {actions['remove']}")
    print(f"  add:    {actions['add']}")

    print("\nClass transitions (baseline -> noisy): count")
    for (src, dst), cnt in sorted(transitions.items()):
        print(f"  {src:20s} -> {dst:20s}: {cnt}")


if __name__ == "__main__":
    main()
