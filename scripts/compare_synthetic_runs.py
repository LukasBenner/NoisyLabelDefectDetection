"""
Compare all Synthetic training runs against the baseline.

Loads test_summary/all_runs_metrics.csv from each experiment under
logs/train/Synthetic/{baseline,lora,pretrained}/, prints summary statistics,
and runs paired statistical tests (paired t-test, Wilcoxon, sign-flip)
comparing each method to the baseline.
"""

import itertools
import math
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats as scipy_stats

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
LOG_ROOT = Path(
    "/Users/lukasb/Projekte/NoisyLabelDefectDetection/logs/train/Synthetic"
)

# Discover all experiments: each subfolder (baseline/lora/pretrained) contains
# dated experiment directories.  We pair (category, experiment_dir).
EXPERIMENTS: list[tuple[str, Path]] = []
for category_dir in sorted(LOG_ROOT.iterdir()):
    if not category_dir.is_dir():
        continue
    for exp_dir in sorted(category_dir.iterdir()):
        if not exp_dir.is_dir():
            continue
        EXPERIMENTS.append((category_dir.name, exp_dir))


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
ALL_CLASSES = [
    "background", "black_stain", "corrosion", "crack", "deformation",
    "missing_part", "ok", "other", "silicate_stain", "water_stain",
]

# Map experiment name keywords to the class that received synthetic data.
TARGET_CLASS_MAP = {
    "water_stain": "water_stain",
    "waterstain": "water_stain",
    "missing-part": "missing_part",
    "missing_part": "missing_part",
}


def friendly_name(category: str, exp_dir: Path) -> str:
    """Derive a human-readable name from the experiment path."""
    # Directory names look like "2026-03-13_11-24-55_" or
    # "2026-03-13_17-31-34_waterstain_plain".
    # Strip the leading timestamp.
    parts = exp_dir.name.split("_", 2)
    suffix = parts[2] if len(parts) > 2 else ""
    suffix = suffix.strip("_") or "default"
    return f"{category}/{suffix}"


def target_class_for(name: str) -> str | None:
    """Return the class that received synthetic data, or None for baseline."""
    lower = name.lower()
    for keyword, cls in TARGET_CLASS_MAP.items():
        if keyword in lower:
            return cls
    return None


def paired_comparison(
    base_vals: np.ndarray, method_vals: np.ndarray
) -> dict:
    """Run paired statistical tests and return results dict."""
    d = method_vals - base_vals
    n = len(d)
    mean_d = d.mean()
    sd_d = d.std(ddof=1)
    se_d = sd_d / math.sqrt(n) if n > 0 else float("nan")
    t_crit = scipy_stats.t.ppf(0.975, df=n - 1) if n > 1 else float("nan")
    ci_low = mean_d - t_crit * se_d
    ci_high = mean_d + t_crit * se_d

    t_stat, p_ttest = scipy_stats.ttest_rel(method_vals, base_vals)

    try:
        w_stat, p_wilcoxon = scipy_stats.wilcoxon(d)
    except ValueError:
        w_stat, p_wilcoxon = float("nan"), float("nan")

    signs = np.array(list(itertools.product([1.0, -1.0], repeat=n)))
    perm_means = (signs * d).mean(axis=1)
    p_signflip = float(np.mean(np.abs(perm_means) >= abs(mean_d)))

    return {
        "baseline_mean": float(base_vals.mean()),
        "method_mean": float(method_vals.mean()),
        "mean_d": mean_d,
        "sd_d": sd_d,
        "ci_low": ci_low,
        "ci_high": ci_high,
        "p_ttest": p_ttest,
        "p_wilcoxon": p_wilcoxon,
        "p_signflip": p_signflip,
        "n": n,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    # ------------------------------------------------------------------
    # 1. Load & print summary statistics for every experiment
    # ------------------------------------------------------------------
    loaded: dict[str, pd.DataFrame] = {}  # friendly_name -> all_runs_metrics

    print(f"Found {len(EXPERIMENTS)} experiments:\n")

    for category, exp_dir in EXPERIMENTS:
        name = friendly_name(category, exp_dir)
        metrics_path = exp_dir / "test_summary" / "all_runs_metrics.csv"
        summary_path = exp_dir / "test_summary" / "summary_statistics.csv"

        if not metrics_path.exists():
            print(f"  SKIP {name}: no test_summary found")
            continue

        df = pd.read_csv(metrics_path)
        loaded[name] = df

        summary_df = pd.read_csv(summary_path)
        # Old CSVs have test/recall_weighted; remap to acc_micro for display
        summary_df["metric"] = summary_df["metric"].replace(
            {"test/recall_weighted": "test/acc_micro"}
        )
        print(f"  {name}  (n={len(df)} runs)")
        for _, row in summary_df.iterrows():
            if row["metric"] in (
                "test/f1_macro",
                "test/acc_micro",
                "test/precision_macro",
                "test/recall_macro",
            ):
                print(
                    f"    {row['metric']}: {row['mean']:.4f} ± {row['std']:.4f} "
                    f"(CI: [{row['ci_lower']:.4f}, {row['ci_upper']:.4f}])"
                )
        print()

    # ------------------------------------------------------------------
    # 2. Statistical comparison (all methods vs baseline) — macro metrics
    # ------------------------------------------------------------------
    print("\n" + "=" * 80)
    print("STATISTICAL COMPARISONS (paired, vs baseline)")
    print("=" * 80 + "\n")

    baseline_name = None
    for name in loaded:
        if name.startswith("baseline/"):
            baseline_name = name
            break

    if baseline_name is None:
        print("Baseline not found, skipping comparisons.")
        return

    baseline_df = loaded[baseline_name]
    print(f"Baseline: {baseline_name}  "
          f"(mean test/f1_macro = {baseline_df['test/f1_macro'].mean():.4f})\n")

    comparison_results = []

    for name, method_df in loaded.items():
        if name == baseline_name:
            continue

        merged = baseline_df.merge(
            method_df, on="run_idx", suffixes=("_base", "_method")
        )

        # --- Macro F1 comparison ---
        res = paired_comparison(
            merged["test/f1_macro_base"].values,
            merged["test/f1_macro_method"].values,
        )

        print(f"  {name} vs {baseline_name}:")
        print(
            f"    Δ(test/f1_macro) = {res['mean_d']:.6f}  "
            f"95% CI [{res['ci_low']:.6f}, {res['ci_high']:.6f}]"
        )
        print(
            f"    paired t-test p={res['p_ttest']:.6f}  "
            f"wilcoxon p={res['p_wilcoxon']:.6f}  "
            f"sign-flip p={res['p_signflip']:.6f}"
        )
        print(
            f"    baseline mean={res['baseline_mean']:.4f}  "
            f"{name} mean={res['method_mean']:.4f}"
        )

        comparison_results.append({"method": name, "metric": "test/f1_macro", **res})

        # --- Per-class F1 comparison ---
        target_cls = target_class_for(name)
        if target_cls:
            # Compare target class first, then all other classes
            classes_ordered = [target_cls] + [
                c for c in ALL_CLASSES if c != target_cls
            ]
        else:
            classes_ordered = ALL_CLASSES

        print(f"    {'—' * 60}")
        print(f"    Per-class F1 breakdown"
              + (f"  (target class: {target_cls})" if target_cls else "")
              + ":")

        for cls in classes_ordered:
            col = f"test/f1_{cls}"
            if col not in merged.columns:
                # Column may exist only with _base / _method suffix after merge
                col_base = f"{col}_base"
                col_method = f"{col}_method"
            else:
                col_base = f"{col}_base"
                col_method = f"{col}_method"

            if col_base not in merged.columns or col_method not in merged.columns:
                continue

            cls_res = paired_comparison(
                merged[col_base].values,
                merged[col_method].values,
            )

            marker = " <<<" if cls == target_cls else ""
            sig = " *" if cls_res["p_ttest"] < 0.05 else ""
            print(
                f"      {cls:>16s}: "
                f"base={cls_res['baseline_mean']:.4f}  "
                f"method={cls_res['method_mean']:.4f}  "
                f"Δ={cls_res['mean_d']:+.4f}  "
                f"p={cls_res['p_ttest']:.4f}{sig}{marker}"
            )

            comparison_results.append(
                {"method": name, "metric": col, **cls_res}
            )

        print()

    if comparison_results:
        comp_df = pd.DataFrame(comparison_results)

        # Holm-Bonferroni correction per metric group (macro F1 and each per-class F1 separately)
        for metric_val, group_df in comp_df.groupby("metric"):
            m = len(group_df)
            if m < 2:
                for p_col in ("p_ttest", "p_wilcoxon", "p_signflip"):
                    comp_df.loc[group_df.index, f"{p_col}_holm"] = group_df[p_col]
                continue
            for p_col in ("p_ttest", "p_wilcoxon", "p_signflip"):
                sorted_idx = group_df[p_col].sort_values().index
                adjusted = []
                for rank, idx in enumerate(sorted_idx, start=1):
                    adjusted.append((idx, group_df.loc[idx, p_col] * (m - rank + 1)))
                # Enforce monotonicity and cap at 1.0
                running_max = 0.0
                for idx, val in adjusted:
                    running_max = max(running_max, val)
                    comp_df.loc[idx, f"{p_col}_holm"] = min(running_max, 1.0)

        comp_path = LOG_ROOT / "comparison_results.csv"
        comp_df.to_csv(comp_path, index=False)
        print(f"\nComparison results saved to: {comp_path}")
        print("\nMacro-level summary (with Holm-corrected p-values):")
        macro_df = comp_df[comp_df["metric"] == "test/f1_macro"]
        print(macro_df.to_string(index=False))


if __name__ == "__main__":
    main()
