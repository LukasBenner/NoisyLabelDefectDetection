#!/usr/bin/env python3
"""
Synthetic image quality evaluation: real vs synthetic.

Computes standard generative image quality metrics comparing a folder of real
images against a folder of synthetic images (flat directories, no subfolders).

Metrics:
    - FID  (Fréchet Inception Distance)   — lower is better
    - KID  (Kernel Inception Distance)    — lower is better
    - IS   (Inception Score)              — higher is better
    - Improved Precision & Recall         — both higher is better

Usage:
    python evaluate.py \
        --real_dir /path/to/real \
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
from torchmetrics.image.inception import InceptionScore
from torchvision import transforms


INCEPTION_SIZE = 299
BATCH_SIZE = 32
MIN_IMAGES_KID = 10
MIN_IMAGES_FID = 2


def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def list_images(directory: Path) -> list[Path]:
    exts = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tiff"}
    return sorted(p for p in directory.iterdir() if p.is_file() and p.suffix.lower() in exts)


def load_images_as_tensor(
    image_paths: list[Path],
    transform: transforms.Compose,
    batch_size: int = BATCH_SIZE,
) -> list[torch.Tensor]:
    """Load images and return list of batched uint8 tensors [B, 3, 299, 299]."""
    batches = []
    batch = []
    for p in image_paths:
        try:
            img = Image.open(p).convert("RGB")
            tensor = transform(img)
            batch.append(tensor)
        except Exception as e:
            print(f"  [warn] Failed to load {p.name}: {e}")
            continue

        if len(batch) >= batch_size:
            batches.append(torch.stack(batch))
            batch = []

    if batch:
        batches.append(torch.stack(batch))

    return batches


def compute_precision_recall(
    real_batches: list[torch.Tensor],
    syn_batches: list[torch.Tensor],
    device: torch.device,
    k: int = 3,
) -> dict[str, float]:
    """Compute improved precision and recall using k-nearest neighbors in
    InceptionV3 feature space.

    Precision = fraction of synthetic samples whose nearest real neighbor is
    close (synthetic images look realistic).
    Recall = fraction of real samples whose nearest synthetic neighbor is
    close (real distribution is covered).
    """
    from torchvision.models import inception_v3, Inception_V3_Weights

    model = inception_v3(weights=Inception_V3_Weights.DEFAULT)
    # Use the penultimate layer (avgpool output = 2048-d features)
    model.fc = torch.nn.Identity()  # type: ignore[assignment]
    model.eval()
    model.to(device)

    def extract_features(batches: list[torch.Tensor]) -> torch.Tensor:
        feats = []
        with torch.no_grad():
            for batch in batches:
                batch = batch.to(device).float() / 255.0
                out = model(batch)
                # inception_v3 may return InceptionOutputs namedtuple
                f = out.logits if hasattr(out, "logits") else out
                feats.append(f.cpu())
        return torch.cat(feats, dim=0)

    real_feats = extract_features(real_batches)
    syn_feats = extract_features(syn_batches)

    def kth_nearest_distance(features: torch.Tensor, k: int) -> torch.Tensor:
        """For each sample, find distance to its k-th nearest neighbor."""
        # Compute pairwise distances in chunks to save memory
        n = features.shape[0]
        kth_dists = torch.zeros(n)
        chunk = 512
        for i in range(0, n, chunk):
            end = min(i + chunk, n)
            dists = torch.cdist(features[i:end], features)
            # k+1 because the closest is the point itself
            topk = dists.topk(k + 1, largest=False, dim=1).values
            kth_dists[i:end] = topk[:, k]
        return kth_dists

    # Manifold radii for real and synthetic distributions
    real_radii = kth_nearest_distance(real_feats, k)
    syn_radii = kth_nearest_distance(syn_feats, k)

    # Precision: fraction of synthetic samples falling within real manifold
    n_syn = syn_feats.shape[0]
    precision_count = 0
    chunk = 512
    for i in range(0, n_syn, chunk):
        end = min(i + chunk, n_syn)
        dists_to_real = torch.cdist(syn_feats[i:end], real_feats)
        # A synthetic sample is "covered" if it's within the radius of any real sample
        min_ratio = (dists_to_real / real_radii.unsqueeze(0)).min(dim=1).values
        precision_count += (min_ratio <= 1.0).sum().item()
    precision = precision_count / n_syn

    # Recall: fraction of real samples falling within synthetic manifold
    n_real = real_feats.shape[0]
    recall_count = 0
    for i in range(0, n_real, chunk):
        end = min(i + chunk, n_real)
        dists_to_syn = torch.cdist(real_feats[i:end], syn_feats)
        min_ratio = (dists_to_syn / syn_radii.unsqueeze(0)).min(dim=1).values
        recall_count += (min_ratio <= 1.0).sum().item()
    recall = recall_count / n_real

    return {"precision": precision, "recall": recall}


def compute_metrics(
    real_batches: list[torch.Tensor],
    syn_batches: list[torch.Tensor],
    device: torch.device,
) -> dict:
    """Compute FID, KID, IS, and Precision/Recall."""
    n_real = sum(b.shape[0] for b in real_batches)
    n_syn = sum(b.shape[0] for b in syn_batches)

    # Use CPU for metrics (float64 covariance); CUDA works fine, MPS does not.
    metric_device = device if device.type == "cuda" else torch.device("cpu")

    results = {}

    # --- FID ---
    if n_real >= MIN_IMAGES_FID and n_syn >= MIN_IMAGES_FID:
        fid = FrechetInceptionDistance(feature=2048, normalize=True).to(metric_device)
        for batch in real_batches:
            fid.update((batch.to(metric_device).float() / 255.0), real=True)
        for batch in syn_batches:
            fid.update((batch.to(metric_device).float() / 255.0), real=False)
        results["fid"] = fid.compute().item()
        del fid
    else:
        results["fid"] = None

    # --- KID ---
    if n_real >= MIN_IMAGES_KID and n_syn >= MIN_IMAGES_KID:
        subset = min(50, min(n_real, n_syn))
        kid = KernelInceptionDistance(feature=2048, subset_size=subset, normalize=True).to(metric_device)
        for batch in real_batches:
            kid.update((batch.to(metric_device).float() / 255.0), real=True)
        for batch in syn_batches:
            kid.update((batch.to(metric_device).float() / 255.0), real=False)
        kid_mean, kid_std = kid.compute()
        results["kid_mean"] = kid_mean.item()
        results["kid_std"] = kid_std.item()
        del kid
    else:
        results["kid_mean"] = None
        results["kid_std"] = None

    # --- IS (on synthetic images only) ---
    inception_score = InceptionScore(normalize=True).to(metric_device)
    for batch in syn_batches:
        inception_score.update(batch.to(metric_device).float() / 255.0)
    is_mean, is_std = inception_score.compute()
    results["is_mean"] = is_mean.item()
    results["is_std"] = is_std.item()
    del inception_score

    # --- Improved Precision & Recall ---
    print("  Computing precision & recall (InceptionV3 features)...")
    pr = compute_precision_recall(real_batches, syn_batches, device)
    results["precision"] = pr["precision"]
    results["recall"] = pr["recall"]

    return results


def evaluate(
    real_dir: Path,
    syn_dir: Path,
    output_csv: Optional[Path] = None,
    device: Optional[torch.device] = None,
) -> dict:
    """Run evaluation on flat image folders. Returns result dict."""
    if device is None:
        device = get_device()

    print(f"Device: {device}")
    print(f"Real:   {real_dir}")
    print(f"Syn:    {syn_dir}")
    print()

    real_images = list_images(real_dir)
    syn_images = list_images(syn_dir)
    n_real = len(real_images)
    n_syn = len(syn_images)

    print(f"Found {n_real} real images, {n_syn} synthetic images")

    if n_real == 0 or n_syn == 0:
        print("Error: need images in both directories.")
        return {}

    transform = transforms.Compose([
        transforms.Resize((INCEPTION_SIZE, INCEPTION_SIZE)),
        transforms.ToTensor(),
        transforms.Lambda(lambda x: (x * 255).to(torch.uint8)),
    ])

    print("Loading real images...")
    real_batches = load_images_as_tensor(real_images, transform)
    print("Loading synthetic images...")
    syn_batches = load_images_as_tensor(syn_images, transform)

    print("Computing metrics...")
    metrics = compute_metrics(real_batches, syn_batches, device)
    metrics["n_real"] = n_real
    metrics["n_syn"] = n_syn

    # Print results
    print()
    print("=" * 60)
    print("  Synthetic Image Quality Metrics")
    print("=" * 60)
    print(f"  Images:     {n_real} real, {n_syn} synthetic")
    print(f"  FID:        {metrics['fid']:.2f}" if metrics["fid"] is not None else "  FID:        N/A")
    if metrics["kid_mean"] is not None:
        print(f"  KID:        {metrics['kid_mean']:.4f} ± {metrics['kid_std']:.4f}")
    else:
        print("  KID:        N/A")
    print(f"  IS:         {metrics['is_mean']:.2f} ± {metrics['is_std']:.2f}")
    print(f"  Precision:  {metrics['precision']:.4f}")
    print(f"  Recall:     {metrics['recall']:.4f}")
    print("=" * 60)

    # CSV output
    if output_csv:
        output_csv.parent.mkdir(parents=True, exist_ok=True)
        fieldnames = ["n_real", "n_syn", "fid", "kid_mean", "kid_std",
                      "is_mean", "is_std", "precision", "recall"]
        with open(output_csv, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerow({k: metrics.get(k) for k in fieldnames})
        print(f"\nResults saved to {output_csv}")

    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate synthetic image quality (FID, KID, IS, Precision/Recall)")
    parser.add_argument("--real_dir", required=True, type=Path,
                        help="Folder containing real images")
    parser.add_argument("--syn_dir", required=True, type=Path,
                        help="Folder containing synthetic images")
    parser.add_argument("--output", type=Path, default=None,
                        help="Output CSV path (optional)")
    parser.add_argument("--device", default=None,
                        help="Device (cuda/mps/cpu, auto-detected if omitted)")
    args = parser.parse_args()

    device = torch.device(args.device) if args.device else get_device()
    evaluate(args.real_dir, args.syn_dir, output_csv=args.output, device=device)


if __name__ == "__main__":
    main()
