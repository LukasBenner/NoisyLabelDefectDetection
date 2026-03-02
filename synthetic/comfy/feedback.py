#!/usr/bin/env python3
"""
Automated quality feedback for synthetic defect images.

Runs FID/KID evaluation per class, ranks classes by quality,
and generates a structured report with actionable suggestions.

Usage:
    python feedback.py \
        --real_dir /path/to/real/train \
        --syn_dir /path/to/synthetic \
        --prompts_dir ./defects
"""

from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path
from typing import Optional

import torch

from evaluate import evaluate, get_device


def load_prompt(prompts_dir: Path, class_name: str) -> Optional[str]:
    """Load the prompt file for a given class."""
    prompt_file = prompts_dir / f"{class_name}.txt"
    if prompt_file.exists():
        return prompt_file.read_text(encoding="utf-8").strip()
    return None


def classify_fid(fid: float) -> str:
    """Rough quality bucket for FID scores (domain-specific imagery)."""
    if fid < 80:
        return "good"
    if fid < 150:
        return "moderate"
    if fid < 250:
        return "poor"
    return "very_poor"


def classify_kid(kid_mean: float) -> str:
    """Rough quality bucket for KID scores."""
    if kid_mean < 0.02:
        return "good"
    if kid_mean < 0.05:
        return "moderate"
    if kid_mean < 0.10:
        return "poor"
    return "very_poor"


def generate_suggestions(result: dict, prompt: Optional[str]) -> list[str]:
    """Generate actionable suggestions for a class based on its metrics."""
    suggestions = []
    n_real = result["n_real"]
    n_syn = result["n_syn"]
    fid = result.get("fid")
    kid_mean = result.get("kid_mean")

    # Data quantity issues
    if n_real < 100:
        suggestions.append(
            f"Only {n_real} real reference images. Consider collecting more real samples "
            "for this class to improve metric reliability and training."
        )
    if n_syn < 200:
        suggestions.append(
            f"Only {n_syn} synthetic images. Generate more to improve distribution coverage "
            "and reduce KID variance."
        )

    # Quality issues based on FID
    if fid is not None:
        quality = classify_fid(fid)
        if quality == "very_poor":
            suggestions.append(
                "FID is very high — synthetic images may look unrealistic. "
                "Review prompt for overly generic descriptions. Add specific material/lighting details."
            )
        elif quality == "poor":
            suggestions.append(
                "FID indicates noticeable distribution gap. Consider refining texture descriptions, "
                "adding surface finish details, or adjusting defect coverage percentages."
            )
        elif quality == "moderate":
            suggestions.append(
                "FID is acceptable but could improve. Fine-tune defect severity descriptions "
                "and ensure lighting/background variety matches real data."
            )

    # Quality issues based on KID
    if kid_mean is not None:
        kid_quality = classify_kid(kid_mean)
        if kid_quality in ("poor", "very_poor"):
            suggestions.append(
                "High KID suggests feature-level mismatch. The synthetic images may have "
                "systematic differences in texture, color, or structure."
            )

    # Prompt-specific suggestions
    if prompt is not None:
        if len(prompt) < 50:
            suggestions.append(
                "Prompt is very short. Add more descriptive detail about defect appearance, "
                "distribution, and material interaction."
            )
        if "distribution" not in prompt.lower() and "cover" not in prompt.lower():
            suggestions.append(
                "Prompt lacks spatial distribution info. Consider adding coverage percentages "
                "and location patterns (e.g., 'near edges', 'across flat areas')."
            )

    if not suggestions:
        suggestions.append("Quality looks good. No immediate changes needed.")

    return suggestions


def generate_report(
    results: list[dict],
    prompts_dir: Optional[Path],
    output_dir: Path,
) -> Path:
    """Generate a markdown quality report."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir.mkdir(parents=True, exist_ok=True)
    report_path = output_dir / f"{timestamp}_quality_report.md"

    # Filter to classes with computed metrics and sort by FID (worst first)
    scored = [r for r in results if r.get("kid_mean") is not None]
    # Sort: use FID if available, otherwise KID
    scored.sort(key=lambda r: r.get("fid") or (r["kid_mean"] * 5000), reverse=True)

    # Compute averages
    fid_vals = [r["fid"] for r in scored if r["fid"] is not None]
    kid_vals = [r["kid_mean"] for r in scored if r["kid_mean"] is not None]
    avg_fid = sum(fid_vals) / len(fid_vals) if fid_vals else None
    avg_kid = sum(kid_vals) / len(kid_vals) if kid_vals else None

    lines = [
        f"# Synthetic Image Quality Report",
        f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "",
        "## Summary",
        "",
        f"- **Classes evaluated:** {len(scored)}",
        f"- **Average FID:** {avg_fid:.2f}" if avg_fid else "- **Average FID:** N/A",
        f"- **Average KID:** {avg_kid:.4f}" if avg_kid else "- **Average KID:** N/A",
        "",
        "## Per-Class Results (worst first)",
        "",
        "| Rank | Class | N_real | N_syn | FID | KID | Quality |",
        "|------|-------|--------|-------|-----|-----|---------|",
    ]

    for i, r in enumerate(scored, 1):
        fid_str = f"{r['fid']:.1f}" if r["fid"] is not None else "N/A"
        kid_str = f"{r['kid_mean']:.4f}" if r["kid_mean"] is not None else "N/A"
        if r["fid"] is not None:
            quality = classify_fid(r["fid"])
        elif r["kid_mean"] is not None:
            quality = classify_kid(r["kid_mean"])
        else:
            quality = "unknown"
        lines.append(
            f"| {i} | {r['class']} | {r['n_real']} | {r['n_syn']} | {fid_str} | {kid_str} | {quality} |"
        )

    lines.extend(["", "## Detailed Analysis & Suggestions", ""])

    for i, r in enumerate(scored, 1):
        name = r["class"]
        prompt = load_prompt(prompts_dir, name) if prompts_dir else None
        suggestions = generate_suggestions(r, prompt)

        fid_str = f"{r['fid']:.2f}" if r["fid"] is not None else "N/A"
        kid_str = f"{r['kid_mean']:.4f}±{r['kid_std']:.4f}" if r["kid_mean"] is not None else "N/A"

        lines.append(f"### {i}. {name}")
        lines.append(f"- **FID:** {fid_str}  |  **KID:** {kid_str}")
        lines.append(f"- **Images:** {r['n_real']} real, {r['n_syn']} synthetic")

        if r["fid"] is not None and avg_fid is not None:
            delta = r["fid"] - avg_fid
            if delta > 20:
                lines.append(f"- **{delta:.0f} above average FID** — priority for improvement")
            elif delta < -20:
                lines.append(f"- **{abs(delta):.0f} below average FID** — performing well")

        lines.append("")
        lines.append("**Suggestions:**")
        for s in suggestions:
            lines.append(f"- {s}")

        if prompt is not None:
            lines.append("")
            lines.append(f"<details><summary>Current prompt ({name}.txt)</summary>")
            lines.append("")
            lines.append("```")
            lines.append(prompt)
            lines.append("```")
            lines.append("</details>")

        lines.append("")

    # Priority action items
    poor_classes = [r for r in scored if (r.get("fid") and classify_fid(r["fid"]) in ("poor", "very_poor"))
                    or (r.get("kid_mean") and classify_kid(r["kid_mean"]) in ("poor", "very_poor"))]

    if poor_classes:
        lines.extend([
            "## Priority Action Items",
            "",
            "Classes needing immediate attention (poor/very_poor quality):",
            "",
        ])
        for r in poor_classes:
            lines.append(f"1. **{r['class']}** — Revise prompt, increase variety, check reference images")
        lines.append("")

    report_text = "\n".join(lines)
    report_path.write_text(report_text, encoding="utf-8")
    return report_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Automated synthetic image quality feedback")
    parser.add_argument("--real_dir", required=True, type=Path)
    parser.add_argument("--syn_dir", required=True, type=Path)
    parser.add_argument("--prompts_dir", type=Path, default=Path(__file__).parent / "defects",
                        help="Directory with per-class .txt prompt files")
    parser.add_argument("--report_dir", type=Path, default=Path(__file__).parent / "reports",
                        help="Output directory for reports")
    parser.add_argument("--device", default=None)
    args = parser.parse_args()

    device = torch.device(args.device) if args.device else get_device()

    # Run evaluation
    csv_path = args.report_dir / "latest_metrics.csv"
    results = evaluate(args.real_dir, args.syn_dir, output_csv=csv_path, device=device)

    if not results:
        print("No results to report.")
        return

    # Generate report
    report_path = generate_report(results, args.prompts_dir, args.report_dir)
    print(f"\nReport saved to: {report_path}")


if __name__ == "__main__":
    main()
