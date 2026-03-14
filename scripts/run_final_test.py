"""
Run final test-set evaluation for all *_no_crop methods.

For each method, loads the best checkpoint from each of the 10 runs,
evaluates on the held-out test set, and saves per-run + summary metrics.
Results are stored under each experiment's `test_summary/` directory.
"""

import itertools
import math
import os
import sys
from pathlib import Path
from typing import Any, Dict, List

import hydra
import numpy as np
import rootutils
import torch
import pandas as pd
from hydra.core.global_hydra import GlobalHydra
from lightning import Trainer, seed_everything
from omegaconf import OmegaConf

# Setup root directory
project_root = rootutils.setup_root(
    __file__,
    indicator=".project-root",
    pythonpath=True,
)

src_root = project_root / "src"
if str(src_root) not in sys.path:
    sys.path.insert(0, str(src_root))

from utils.utils import (
    calculate_summary_statistics,
    get_class_names,
    to_float,
)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
LOG_ROOT = Path(
    "/Users/lukasb/Projekte/NoisyLabelDefectDetection/logs/train/RobustLearning/noisy_new"
)
N_RUNS = 10
BASE_SEED = 42

# Local data path — overrides whatever the saved configs used during training
DATA_ROOT = Path("/Users/lukasb/Documents/data/surfaceClassification/noisy_clustered_new")

# All *_no_crop experiment directories
EXPERIMENTS = sorted(
    [d for d in LOG_ROOT.iterdir() if d.is_dir() and d.name.endswith("_no_crop")]
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def select_best_checkpoint(run_dir: Path) -> Path:
    """Select checkpoint with highest val F1, falling back to last.ckpt."""
    ckpt_dir = run_dir / "checkpoints"
    ckpts = list(ckpt_dir.glob("epoch_*-val_f1_*.ckpt"))
    if ckpts:

        def parse_f1(path: Path) -> float:
            try:
                return float(path.stem.split("val_f1_")[-1])
            except ValueError:
                return -1.0

        return max(ckpts, key=parse_f1)
    last_ckpt = ckpt_dir / "last.ckpt"
    if last_ckpt.exists():
        return last_ckpt
    raise FileNotFoundError(f"No checkpoints found in {ckpt_dir}")


def collect_per_class_metrics(
    cm: Dict, prefix: str, datamodule: Any
) -> Dict[str, Any]:
    """Collect per-class precision/recall/f1 from callback_metrics."""
    metrics: Dict[str, Any] = {}
    class_names = get_class_names(datamodule)
    n_classes = None
    try:
        n_classes = int(getattr(datamodule, "num_classes"))
    except Exception:
        pass
    if n_classes is None and class_names:
        n_classes = len(class_names)

    class_names_for_metrics = None
    if class_names and n_classes is not None and len(class_names) >= n_classes:
        class_names_for_metrics = class_names

    if n_classes is not None and n_classes > 0:
        for i in range(n_classes):
            for metric_name in ("precision", "recall", "f1"):
                key_idx = f"{prefix}/{metric_name}_c{i}"
                if class_names_for_metrics:
                    class_name = class_names_for_metrics[i]
                    key_named = f"{prefix}/{metric_name}_{class_name}"
                    if key_named in cm:
                        metrics[key_named] = to_float(cm.get(key_named))
                    elif key_idx in cm:
                        metrics[key_named] = to_float(cm.get(key_idx))
                else:
                    if key_idx in cm:
                        metrics[key_idx] = to_float(cm.get(key_idx))
    return metrics


def run_test(cfg, run_idx: int, seed: int, checkpoint_path: Path) -> Dict[str, Any]:
    """Run test-set evaluation for a single run."""
    seed_everything(seed, workers=True)

    datamodule = hydra.utils.instantiate(
        cfg.data,
        seed=seed,
        train_path=str(DATA_ROOT / "train"),
        val_path=str(DATA_ROOT / "val"),
        test_path=str(DATA_ROOT / "test"),
    )
    datamodule.prepare_data()
    datamodule.setup(stage="fit")
    datamodule.setup(stage="test")

    model = hydra.utils.instantiate(cfg.model, datamodule=datamodule)

    # NAL checkpoints save the wrapped net (InstanceNoiseAdaptionNet or NoiseAdaptionNet).
    # Detect this from the checkpoint keys and initialise the wrapper so state_dict loads.
    ckpt_sd = torch.load(str(checkpoint_path), map_location="cpu", weights_only=False)["state_dict"]
    has_instance_noise = any(k.startswith("net.base_transition") for k in ckpt_sd)
    has_noise_layer = any(k.startswith("net.noise_layer.") for k in ckpt_sd)

    if has_instance_noise:
        from src.models.components.noise_adaption_layer import InstanceNoiseAdaptionNet
        model.net = InstanceNoiseAdaptionNet(
            base_net=model.net,
            num_classes=int(model.num_classes),
        )
        model._noise_initialized = True
        model._instance_noise_initialized = True
    elif has_noise_layer:
        from src.models.components.noise_adaption_layer import NoiseAdaptionNet
        model.net = NoiseAdaptionNet(
            base_net=model.net,
            num_classes=int(model.num_classes),
        )
        model._noise_initialized = True

    del ckpt_sd

    test_trainer: Trainer = hydra.utils.instantiate(
        cfg.trainer,
        callbacks=[],
        logger=False,
        devices=1,
        strategy="auto",
        accelerator="mps",
        precision="32-true",
    )

    test_trainer.test(
        model=model,
        datamodule=datamodule,
        ckpt_path=str(checkpoint_path),
        weights_only=False
    )

    cm = test_trainer.callback_metrics

    test_metrics: Dict[str, Any] = {
        "run_idx": run_idx,
        "test/f1_macro": to_float(cm.get("test/f1_macro")),
        "test/precision_macro": to_float(cm.get("test/precision_macro")),
        "test/recall_macro": to_float(cm.get("test/recall_macro")),
        # recall_weighted == micro accuracy for single-label multiclass
        "test/acc_micro": to_float(cm.get("test/recall_weighted")),
        "test/f1_weighted": to_float(cm.get("test/f1_weighted")),
        "test/precision_weighted": to_float(cm.get("test/precision_weighted")),
        "test/recall_weighted": to_float(cm.get("test/recall_weighted")),
        "test/loss": to_float(cm.get("test/loss")),
    }
    test_metrics.update(collect_per_class_metrics(cm, "test", datamodule))

    return test_metrics


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    torch.set_float32_matmul_precision("medium")

    print(f"Found {len(EXPERIMENTS)} no_crop experiments:")
    for exp in EXPERIMENTS:
        print(f"  - {exp.name}")
    print()

    for exp_dir in EXPERIMENTS:
        print("=" * 80)
        print(f"Evaluating: {exp_dir.name}")
        print("=" * 80)

        # If test_summary already exists, just print and skip
        test_summary_dir = exp_dir / "test_summary"
        if (test_summary_dir / "summary_statistics.csv").exists():
            print(f"  SKIP {exp_dir.name}: test_summary already exists")
            summary_df = pd.read_csv(test_summary_dir / "summary_statistics.csv")
            # Old CSVs have test/acc (macro, wrong) and test/recall_weighted;
            # remap recall_weighted → acc_micro for display.
            summary_df["metric"] = summary_df["metric"].replace(
                {"test/recall_weighted": "test/acc_micro"}
            )
            # Still print summary
            print(f"\n  Summary for {exp_dir.name}:")
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
            continue

        # Load config from val_summary/hyperparameters.yaml
        hparams_path = exp_dir / "val_summary" / "hyperparameters.yaml"
        if not hparams_path.exists():
            print(f"  SKIP: no hyperparameters.yaml found at {hparams_path}")
            continue

        cfg = OmegaConf.load(hparams_path)

        # Initialize Hydra for instantiation
        if GlobalHydra.instance().is_initialized():
            GlobalHydra.instance().clear()
        with hydra.initialize(version_base="1.3", config_path="../configs"):
            _ = hydra.compose(config_name="train.yaml")

        all_metrics: List[Dict[str, Any]] = []

        for run_idx in range(1, N_RUNS + 1):
            seed = BASE_SEED + (run_idx - 1)
            run_dir = exp_dir / f"run_{run_idx}_seed_{seed}"

            if not run_dir.exists():
                print(f"  SKIP run {run_idx}: {run_dir} not found")
                continue

            checkpoint_path = select_best_checkpoint(run_dir)
            print(f"  Run {run_idx} (seed={seed}): {checkpoint_path.name}")

            test_metrics = run_test(cfg, run_idx, seed, checkpoint_path)
            all_metrics.append(test_metrics)

            print(
                f"    test/f1_macro={test_metrics.get('test/f1_macro', 'N/A'):.4f}  "
                f"test/acc_micro={test_metrics.get('test/acc_micro', 'N/A'):.4f}"
            )

        if not all_metrics:
            print(f"  No runs evaluated for {exp_dir.name}")
            continue

        # Calculate and save summary statistics
        all_metrics_df, summary_df = calculate_summary_statistics(all_metrics)

        test_summary_dir.mkdir(parents=True, exist_ok=True)
        all_metrics_df.to_csv(test_summary_dir / "all_runs_metrics.csv", index=False)
        summary_df.to_csv(test_summary_dir / "summary_statistics.csv", index=False)

        print(f"\n  Summary for {exp_dir.name}:")
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
        print(f"  Saved to: {test_summary_dir}")
        print()

    # ---------------------------------------------------------------------------
    # Statistical comparison (all methods vs CE baseline)
    # ---------------------------------------------------------------------------
    print("\n" + "=" * 80)
    print("STATISTICAL COMPARISONS (paired, vs CE baseline)")
    print("=" * 80 + "\n")

    ce_dir = None
    for exp_dir in EXPERIMENTS:
        if "_ce_no_crop" in exp_dir.name:
            ce_dir = exp_dir
            break

    if ce_dir is None:
        print("CE baseline not found, skipping comparisons.")
        return

    ce_metrics_path = ce_dir / "test_summary" / "all_runs_metrics.csv"
    if not ce_metrics_path.exists():
        print(f"CE test metrics not found at {ce_metrics_path}")
        return

    ce_df = pd.read_csv(ce_metrics_path)

    from scipy import stats as scipy_stats

    comparison_results = []

    for exp_dir in EXPERIMENTS:
        if exp_dir == ce_dir:
            continue

        # e.g. "2026-03-05_23-30-37_sce_no_crop" -> "sce"
        method_name = exp_dir.name.rsplit("_no_crop", 1)[0].split("_", 2)[-1]
        metrics_path = exp_dir / "test_summary" / "all_runs_metrics.csv"
        if not metrics_path.exists():
            print(f"  SKIP {method_name}: no test metrics found")
            continue

        method_df = pd.read_csv(metrics_path)

        # Align by run_idx
        merged = ce_df.merge(
            method_df, on="run_idx", suffixes=("_ce", f"_{method_name}")
        )

        f1_ce = merged["test/f1_macro_ce"].values
        f1_method = merged[f"test/f1_macro_{method_name}"].values
        d = f1_method - f1_ce
        n = len(d)
        mean_d = d.mean()
        sd_d = d.std(ddof=1)
        se_d = sd_d / math.sqrt(n)
        t_crit = scipy_stats.t.ppf(0.975, df=n - 1)
        ci_low = mean_d - t_crit * se_d
        ci_high = mean_d + t_crit * se_d

        # Paired t-test (two-sided)
        t_stat, p_ttest = scipy_stats.ttest_rel(f1_method, f1_ce)

        # Wilcoxon signed-rank test
        try:
            w_stat, p_wilcoxon = scipy_stats.wilcoxon(d)
        except ValueError:
            w_stat, p_wilcoxon = float("nan"), float("nan")

        # Sign-flip test (exact for n<=20)
        signs = np.array(list(itertools.product([1.0, -1.0], repeat=n)))
        perm_means = (signs * d).mean(axis=1)
        p_signflip = float(np.mean(np.abs(perm_means) >= abs(mean_d)))

        print(f"  {method_name} vs CE:")
        print(
            f"    Δ(test/f1_macro) = {mean_d:.6f}  "
            f"95% CI [{ci_low:.6f}, {ci_high:.6f}]"
        )
        print(
            f"    paired t-test p={p_ttest:.6f}  "
            f"wilcoxon p={p_wilcoxon:.6f}  "
            f"sign-flip p={p_signflip:.6f}"
        )
        print(f"    CE mean={f1_ce.mean():.4f}  {method_name} mean={f1_method.mean():.4f}")
        print()

        comparison_results.append(
            {
                "method": method_name,
                "ce_mean": f1_ce.mean(),
                "method_mean": f1_method.mean(),
                "mean_d": mean_d,
                "sd_d": sd_d,
                "ci_low": ci_low,
                "ci_high": ci_high,
                "p_ttest": p_ttest,
                "p_wilcoxon": p_wilcoxon,
                "p_signflip": p_signflip,
                "n": n,
            }
        )

    if comparison_results:
        comp_df = pd.DataFrame(comparison_results)

        # Holm-Bonferroni correction for multiple comparisons
        m = len(comp_df)
        for p_col in ("p_ttest", "p_wilcoxon", "p_signflip"):
            sorted_idx = comp_df[p_col].sort_values().index
            adjusted = []
            for rank, idx in enumerate(sorted_idx, start=1):
                adjusted.append((idx, comp_df.loc[idx, p_col] * (m - rank + 1)))
            # Enforce monotonicity and cap at 1.0
            holm_vals = pd.Series(dtype=float, index=comp_df.index)
            running_max = 0.0
            for idx, val in adjusted:
                running_max = max(running_max, val)
                holm_vals[idx] = min(running_max, 1.0)
            comp_df[f"{p_col}_holm"] = holm_vals

        comp_path = LOG_ROOT / "test_comparison_results.csv"
        comp_df.to_csv(comp_path, index=False)
        print(f"\nComparison results saved to: {comp_path}")
        print("\nComparison summary (with Holm-corrected p-values):")
        print(comp_df.to_string(index=False))


if __name__ == "__main__":
    main()
