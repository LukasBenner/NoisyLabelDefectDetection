#!/usr/bin/env python3
"""
Per-class FID and KID evaluation for synthetic vs real images.

Computes Frechet Inception Distance (FID) and Kernel Inception Distance (KID)
for each defect class, comparing real and synthetic image distributions using
InceptionV3 features.

Usage:
    python evaluate.py \
        --real_dir /path/to/real/train \
        --syn_dir /path/to/synthetic \
        --output results.csv
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path
from typing import Optional

import torch
from PIL import Image
from torchmetrics.image.fid import FrechetInceptionDistance
from torchmetrics.image.kid import KernelInceptionDistance
from torchvision import transforms


MIN_IMAGES_FID = 50
MIN_IMAGES_KID = 10
INCEPTION_SIZE = 299
BATCH_SIZE = 32

# Aliases: maps alternative folder names to canonical class names
CLASS_ALIASES = {
    "silicate_discolor": "silicate_stain",
    "no_deficiencies": "ok",
}


def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def list_images(directory: Path) -> list[Path]:
    exts = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tiff"}
    return sorted(p for p in directory.iterdir() if p.is_file() and p.suffix.lower() in exts)


def discover_classes(
    real_dir: Path, syn_dir: Path
) -> list[dict]:
    """Find matching class folders between real and synthetic directories."""
    real_subs = {d.name: d for d in sorted(real_dir.iterdir()) if d.is_dir()}
    syn_subs = {d.name: d for d in sorted(syn_dir.iterdir()) if d.is_dir()}

    # Build canonical name mapping for synthetic dirs
    syn_canonical: dict[str, Path] = {}
    for name, path in syn_subs.items():
        canonical = CLASS_ALIASES.get(name, name)
        syn_canonical[canonical] = path

    classes = []
    for real_name, real_path in sorted(real_subs.items()):
        canonical = CLASS_ALIASES.get(real_name, real_name)
        syn_path = syn_canonical.get(canonical)
        if syn_path is None:
            print(f"  [skip] {canonical}: no synthetic folder found")
            continue
        classes.append({
            "name": canonical,
            "real_path": real_path,
            "syn_path": syn_path,
        })

    # Report synthetic-only classes
    real_canonical = {CLASS_ALIASES.get(n, n) for n in real_subs}
    for name in sorted(syn_canonical):
        if name not in real_canonical:
            print(f"  [skip] {name}: synthetic only (no real data)")

    return classes


def load_images_as_tensor(
    image_paths: list[Path],
    transform: transforms.Compose,
    batch_size: int = BATCH_SIZE,
) -> list[torch.Tensor]:
    """Load images and return list of batched tensors (uint8, 0-255)."""
    batches = []
    batch = []
    for p in image_paths:
        try:
            img = Image.open(p).convert("RGB")
            tensor = transform(img)  # [3, 299, 299] uint8
            batch.append(tensor)
        except Exception as e:
            print(f"    [warn] Failed to load {p.name}: {e}")
            continue

        if len(batch) >= batch_size:
            batches.append(torch.stack(batch))
            batch = []

    if batch:
        batches.append(torch.stack(batch))

    return batches


def compute_metrics(
    real_batches: list[torch.Tensor],
    syn_batches: list[torch.Tensor],
    device: torch.device,
) -> dict:
    """Compute FID and KID for given real and synthetic image batches.

    Note: Metrics are kept on CPU because the internal covariance matrices
    require float64, which MPS does not support. The Inception forward pass
    inside torchmetrics still benefits from batched processing.
    """
    # Use CPU for metrics (float64 covariance); CUDA works fine, MPS does not.
    metric_device = device if device.type == "cuda" else torch.device("cpu")

    fid = FrechetInceptionDistance(feature=2048, normalize=True).to(metric_device)
    kid = KernelInceptionDistance(feature=2048, subset_size=min(50, min(
        sum(b.shape[0] for b in real_batches),
        sum(b.shape[0] for b in syn_batches),
    )), normalize=True).to(metric_device)

    for batch in real_batches:
        batch = batch.to(metric_device).float() / 255.0
        fid.update(batch, real=True)
        kid.update(batch, real=True)

    for batch in syn_batches:
        batch = batch.to(metric_device).float() / 255.0
        fid.update(batch, real=False)
        kid.update(batch, real=False)

    fid_val = fid.compute().item()
    kid_mean, kid_std = kid.compute()

    return {
        "fid": fid_val,
        "kid_mean": kid_mean.item(),
        "kid_std": kid_std.item(),
    }


def evaluate(
    real_dir: Path,
    syn_dir: Path,
    output_csv: Optional[Path] = None,
    device: Optional[torch.device] = None,
) -> list[dict]:
    """Run per-class FID/KID evaluation. Returns list of result dicts."""
    if device is None:
        device = get_device()

    print(f"Device: {device}")
    print(f"Real:   {real_dir}")
    print(f"Syn:    {syn_dir}")
    print()

    classes = discover_classes(real_dir, syn_dir)
    if not classes:
        print("No matching classes found.")
        return []

    transform = transforms.Compose([
        transforms.Resize((INCEPTION_SIZE, INCEPTION_SIZE)),
        transforms.ToTensor(),
        transforms.Lambda(lambda x: (x * 255).to(torch.uint8)),
    ])

    results = []
    print()

    for cls in classes:
        name = cls["name"]
        real_images = list_images(cls["real_path"])
        syn_images = list_images(cls["syn_path"])

        n_real = len(real_images)
        n_syn = len(syn_images)

        print(f"  {name}: {n_real} real, {n_syn} synthetic", end="")

        if n_real < MIN_IMAGES_KID or n_syn < MIN_IMAGES_KID:
            print(f" -> SKIPPED (need >= {MIN_IMAGES_KID} images each)")
            results.append({
                "class": name, "n_real": n_real, "n_syn": n_syn,
                "fid": None, "kid_mean": None, "kid_std": None,
                "note": "too_few_images",
            })
            continue

        real_batches = load_images_as_tensor(real_images, transform)
        syn_batches = load_images_as_tensor(syn_images, transform)

        can_fid = n_real >= MIN_IMAGES_FID and n_syn >= MIN_IMAGES_FID

        if can_fid:
            metrics = compute_metrics(real_batches, syn_batches, device)
            print(f" -> FID={metrics['fid']:.2f}  KID={metrics['kid_mean']:.4f}±{metrics['kid_std']:.4f}")
        else:
            # Only compute KID (works with fewer samples)
            metric_device = device if device.type == "cuda" else torch.device("cpu")
            kid = KernelInceptionDistance(
                feature=2048,
                subset_size=min(50, min(n_real, n_syn)),
                normalize=True,
            ).to(metric_device)
            for batch in real_batches:
                kid.update((batch.to(metric_device).float() / 255.0), real=True)
            for batch in syn_batches:
                kid.update((batch.to(metric_device).float() / 255.0), real=False)
            kid_mean, kid_std = kid.compute()
            metrics = {"fid": None, "kid_mean": kid_mean.item(), "kid_std": kid_std.item()}
            print(f" -> FID=N/A (<{MIN_IMAGES_FID} imgs)  KID={metrics['kid_mean']:.4f}±{metrics['kid_std']:.4f}")

        results.append({
            "class": name, "n_real": n_real, "n_syn": n_syn,
            "fid": metrics["fid"], "kid_mean": metrics["kid_mean"],
            "kid_std": metrics["kid_std"], "note": "",
        })

    # Summary table
    print("\n" + "=" * 80)
    print(f"{'Class':<20} {'N_real':>7} {'N_syn':>7} {'FID':>10} {'KID_mean':>10} {'KID_std':>10}")
    print("-" * 80)
    for r in results:
        fid_str = f"{r['fid']:.2f}" if r["fid"] is not None else "N/A"
        kid_str = f"{r['kid_mean']:.4f}" if r["kid_mean"] is not None else "N/A"
        kstd_str = f"{r['kid_std']:.4f}" if r["kid_std"] is not None else "N/A"
        print(f"{r['class']:<20} {r['n_real']:>7} {r['n_syn']:>7} {fid_str:>10} {kid_str:>10} {kstd_str:>10}")
    print("=" * 80)

    # CSV output
    if output_csv:
        output_csv.parent.mkdir(parents=True, exist_ok=True)
        with open(output_csv, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["class", "n_real", "n_syn", "fid", "kid_mean", "kid_std", "note"])
            writer.writeheader()
            writer.writerows(results)
        print(f"\nResults saved to {output_csv}")

    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="Per-class FID/KID evaluation: real vs synthetic images")
    parser.add_argument("--real_dir", required=True, type=Path, help="Root of real dataset (ImageFolder with class subfolders)")
    parser.add_argument("--syn_dir", required=True, type=Path, help="Root of synthetic dataset (ImageFolder with class subfolders)")
    parser.add_argument("--output", type=Path, default=None, help="Output CSV path (optional)")
    parser.add_argument("--device", default=None, help="Device (cuda/mps/cpu, auto-detected if omitted)")
    args = parser.parse_args()

    device = torch.device(args.device) if args.device else get_device()
    evaluate(args.real_dir, args.syn_dir, output_csv=args.output, device=device)


if __name__ == "__main__":
    main()
