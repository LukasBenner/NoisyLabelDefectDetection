#!/usr/bin/env python3
"""
Automated quality feedback for synthetic images.

Runs FID/KID/IS/Precision/Recall evaluation on flat real vs synthetic image
folders and generates a structured quality report with actionable suggestions.

Usage:
    python feedback.py \
        --real_dir /path/to/real \
        --syn_dir /path/to/synthetic
"""

from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path
from typing import Optional

import torch

from evaluate import evaluate, get_device


# ---------------------------------------------------------------------------
# Quality classification thresholds
# ---------------------------------------------------------------------------

def classify_fid(fid: float) -> str:
    if fid < 80:
        return "good"
    if fid < 150:
        return "moderate"
    if fid < 250:
        return "poor"
    return "very_poor"


def classify_kid(kid_mean: float) -> str:
    if kid_mean < 0.02:
        return "good"
    if kid_mean < 0.05:
        return "moderate"
    if kid_mean < 0.10:
        return "poor"
    return "very_poor"


def classify_is(is_mean: float) -> str:
    """Higher IS is better — more diverse and recognizable outputs."""
    if is_mean >= 3.0:
        return "good"
    if is_mean >= 2.0:
        return "moderate"
    if is_mean >= 1.5:
        return "poor"
    return "very_poor"


def classify_precision(p: float) -> str:
    if p >= 0.7:
        return "good"
    if p >= 0.5:
        return "moderate"
    if p >= 0.3:
        return "poor"
    return "very_poor"


def classify_recall(r: float) -> str:
    if r >= 0.5:
        return "good"
    if r >= 0.3:
        return "moderate"
    if r >= 0.15:
        return "poor"
    return "very_poor"


def overall_quality(metrics: dict) -> str:
    """Aggregate quality label from individual metric ratings."""
    ratings = []
    if metrics.get("fid") is not None:
        ratings.append(classify_fid(metrics["fid"]))
    if metrics.get("kid_mean") is not None:
        ratings.append(classify_kid(metrics["kid_mean"]))
    if metrics.get("is_mean") is not None:
        ratings.append(classify_is(metrics["is_mean"]))
    if metrics.get("precision") is not None:
        ratings.append(classify_precision(metrics["precision"]))
    if metrics.get("recall") is not None:
        ratings.append(classify_recall(metrics["recall"]))

    if not ratings:
        return "unknown"

    order = {"very_poor": 0, "poor": 1, "moderate": 2, "good": 3}
    avg = sum(order[r] for r in ratings) / len(ratings)
    if avg >= 2.5:
        return "good"
    if avg >= 1.5:
        return "moderate"
    if avg >= 0.5:
        return "poor"
    return "very_poor"


# ---------------------------------------------------------------------------
# Suggestion generation
# ---------------------------------------------------------------------------

def generate_suggestions(metrics: dict, prompt: Optional[str]) -> list[str]:
    suggestions = []
    fid = metrics.get("fid")
    kid_mean = metrics.get("kid_mean")
    is_mean = metrics.get("is_mean")
    precision = metrics.get("precision")
    recall = metrics.get("recall")
    n_syn = metrics.get("n_syn", 0)

    # Data quantity
    if n_syn < 200:
        suggestions.append(
            f"Only {n_syn} synthetic images. Generate more to improve distribution "
            "coverage and reduce metric variance."
        )

    # FID
    if fid is not None:
        q = classify_fid(fid)
        if q == "very_poor":
            suggestions.append(
                "FID is very high — synthetic images may look unrealistic. "
                "Review prompts for overly generic descriptions. Add specific "
                "material, lighting, and texture details."
            )
        elif q == "poor":
            suggestions.append(
                "FID indicates a noticeable distribution gap. Refine texture "
                "descriptions, surface finish details, or defect characteristics."
            )
        elif q == "moderate":
            suggestions.append(
                "FID is acceptable but could improve. Fine-tune defect severity "
                "and ensure lighting/background variety matches real data."
            )

    # KID
    if kid_mean is not None and classify_kid(kid_mean) in ("poor", "very_poor"):
        suggestions.append(
            "High KID suggests feature-level mismatch. Synthetic images may have "
            "systematic differences in texture, color, or structure."
        )

    # IS
    if is_mean is not None:
        q = classify_is(is_mean)
        if q in ("poor", "very_poor"):
            suggestions.append(
                "Low Inception Score — synthetic images may lack diversity or "
                "contain ambiguous content. Increase prompt variation and ensure "
                "distinct visual features."
            )

    # Precision & Recall
    if precision is not None and classify_precision(precision) in ("poor", "very_poor"):
        suggestions.append(
            f"Low precision ({precision:.2f}) — many synthetic images fall outside "
            "the real data manifold. They may contain artifacts or unrealistic "
            "features. Tighten prompts to stay closer to real image characteristics."
        )

    if recall is not None and classify_recall(recall) in ("poor", "very_poor"):
        suggestions.append(
            f"Low recall ({recall:.2f}) — synthetic images don't cover enough of "
            "the real distribution. Increase prompt diversity, vary backgrounds, "
            "angles, lighting, and defect severity."
        )

    # Prompt-specific
    if prompt is not None:
        if len(prompt) < 50:
            suggestions.append(
                "Prompt is very short. Add more descriptive detail about defect "
                "appearance, distribution, and material interaction."
            )

    if not suggestions:
        suggestions.append("Quality looks good. No immediate changes needed.")

    return suggestions


# ---------------------------------------------------------------------------
# Report generation
# ---------------------------------------------------------------------------

def generate_report(
    metrics: dict,
    prompt: Optional[str],
    output_dir: Path,
) -> Path:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir.mkdir(parents=True, exist_ok=True)
    report_path = output_dir / f"{timestamp}_quality_report.md"

    quality = overall_quality(metrics)
    suggestions = generate_suggestions(metrics, prompt)

    fid_str = f"{metrics['fid']:.2f}" if metrics.get("fid") is not None else "N/A"
    kid_str = (f"{metrics['kid_mean']:.4f} ± {metrics['kid_std']:.4f}"
               if metrics.get("kid_mean") is not None else "N/A")
    is_str = (f"{metrics['is_mean']:.2f} ± {metrics['is_std']:.2f}"
              if metrics.get("is_mean") is not None else "N/A")
    prec_str = f"{metrics['precision']:.4f}" if metrics.get("precision") is not None else "N/A"
    rec_str = f"{metrics['recall']:.4f}" if metrics.get("recall") is not None else "N/A"

    lines = [
        "# Synthetic Image Quality Report",
        f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "",
        "## Dataset",
        "",
        f"- **Real images:** {metrics.get('n_real', '?')}",
        f"- **Synthetic images:** {metrics.get('n_syn', '?')}",
        "",
        "## Metrics",
        "",
        "| Metric | Value | Rating |",
        "|--------|-------|--------|",
        f"| FID ↓ | {fid_str} | {classify_fid(metrics['fid']) if metrics.get('fid') is not None else 'N/A'} |",
        f"| KID ↓ | {kid_str} | {classify_kid(metrics['kid_mean']) if metrics.get('kid_mean') is not None else 'N/A'} |",
        f"| IS ↑ | {is_str} | {classify_is(metrics['is_mean']) if metrics.get('is_mean') is not None else 'N/A'} |",
        f"| Precision ↑ | {prec_str} | {classify_precision(metrics['precision']) if metrics.get('precision') is not None else 'N/A'} |",
        f"| Recall ↑ | {rec_str} | {classify_recall(metrics['recall']) if metrics.get('recall') is not None else 'N/A'} |",
        "",
        f"**Overall quality: {quality}**",
        "",
        "## Suggestions",
        "",
    ]

    for s in suggestions:
        lines.append(f"- {s}")

    if prompt is not None:
        lines.extend([
            "",
            "## Current Prompt",
            "",
            "```",
            prompt,
            "```",
        ])

    # Metric interpretation guide
    lines.extend([
        "",
        "## Metric Guide",
        "",
        "| Metric | What it measures | Good range |",
        "|--------|-----------------|------------|",
        "| **FID** | Overall distributional similarity (lower = more similar) | < 80 |",
        "| **KID** | Unbiased distributional similarity, works with fewer samples | < 0.02 |",
        "| **IS** | Diversity and recognizability of synthetic images (higher = better) | > 3.0 |",
        "| **Precision** | Fidelity — fraction of synthetic images that look realistic | > 0.7 |",
        "| **Recall** | Diversity — fraction of real distribution covered by synthetic | > 0.5 |",
        "",
    ])

    report_text = "\n".join(lines)
    report_path.write_text(report_text, encoding="utf-8")
    return report_path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Automated synthetic image quality feedback")
    parser.add_argument("--real_dir", required=True, type=Path,
                        help="Folder containing real images")
    parser.add_argument("--syn_dir", required=True, type=Path,
                        help="Folder containing synthetic images")
    parser.add_argument("--prompt_file", type=Path, default=None,
                        help="Optional .txt file with the generation prompt")
    parser.add_argument("--report_dir", type=Path,
                        default=Path(__file__).parent / "reports",
                        help="Output directory for reports")
    parser.add_argument("--device", default=None)
    args = parser.parse_args()

    device = torch.device(args.device) if args.device else get_device()

    prompt = None
    if args.prompt_file and args.prompt_file.exists():
        prompt = args.prompt_file.read_text(encoding="utf-8").strip()

    # Run evaluation
    csv_path = args.report_dir / "latest_metrics.csv"
    metrics = evaluate(args.real_dir, args.syn_dir, output_csv=csv_path, device=device)

    if not metrics:
        print("No results to report.")
        return

    # Generate report
    report_path = generate_report(metrics, prompt, args.report_dir)
    print(f"\nReport saved to: {report_path}")


if __name__ == "__main__":
    main()
