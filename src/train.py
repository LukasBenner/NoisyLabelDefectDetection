import os
from datetime import date
from pathlib import Path
from typing import Dict, List, Any

import hydra
import numpy as np
import pandas as pd
import rootutils
from lightning import Callback, Trainer
from lightning.pytorch.loggers import CSVLogger
from omegaconf import DictConfig, OmegaConf
from scipy import stats
from lightning.pytorch.callbacks import ModelCheckpoint
import torch

from utils.instantiators import instantiate_callbacks
from utils.utils import calculate_summary_statistics, to_float

# Setup root
rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

from src.utils.pylogger import RankedLogger
from lightning import seed_everything

log = RankedLogger(__name__, rank_zero_only=True)


def train_single_run(cfg: DictConfig, run_idx: int, base_save_dir: Path) -> Dict[str, Any]:
    """Train a single run with a specific seed."""

    # Set seed for this run
    seed = cfg.get("seed", 42) + (run_idx - 1)
    seed_everything(seed, workers=True)
    log.info(f"Run {run_idx}/{cfg.n_runs} with seed {seed}")

    # Create run-specific directory (for checkpoints and metrics only)
    run_save_dir = base_save_dir / f"run_{run_idx}"
    run_save_dir.mkdir(parents=True, exist_ok=True)

    # Instantiate datamodule with run-specific seed
    log.info(f"Instantiating datamodule for run {run_idx}")
    datamodule = hydra.utils.instantiate(cfg.data, seed=seed)
    datamodule.prepare_data()
    datamodule.setup(stage="fit")

    # Instantiate model
    log.info(f"Instantiating model for run {run_idx}")
    model = hydra.utils.instantiate(cfg.model, datamodule=datamodule)    

    callbacks: List[Callback] = instantiate_callbacks(cfg.get("callbacks"))

    # Update ModelCheckpoint dirpath to use run-specific directory
    for callback in callbacks:
        if isinstance(callback, ModelCheckpoint) and callback.dirpath is None:
            callback.dirpath = str(run_save_dir / "checkpoints")
            log.info(f"Set checkpoint directory to: {callback.dirpath}")

    # Setup logger
    logger = CSVLogger(save_dir=run_save_dir, name="", version="")

    # Instantiate trainer
    log.info(f"Instantiating trainer for run {run_idx}")
    trainer: Trainer = hydra.utils.instantiate(cfg.trainer, callbacks=callbacks, logger=logger)

    # Train the model
    log.info(f"Starting training for run {run_idx}")
    trainer.fit(model=model, datamodule=datamodule)

    val_f1_macro_best = trainer.callback_metrics.get("val/f1_macro_best")

    # Fallback: read directly from the model if not present.
    if val_f1_macro_best is None and hasattr(model, "val_f1_macro_best"):
        try:
            val_f1_macro_best = model.val_f1_macro_best.compute()
        except Exception:
            val_f1_macro_best = None

    # Test the model on best checkpoint
    log.info(f"Starting testing for run {run_idx}")
    best_model_path = trainer.checkpoint_callback.best_model_path

    test_dir = run_save_dir / "final_test"
    test_dir.mkdir(parents=True, exist_ok=True)
    test_logger = CSVLogger(save_dir=test_dir, name="", version="")
    test_trainer: Trainer = hydra.utils.instantiate(
        cfg.trainer,
        callbacks=[],
        logger=test_logger,
        devices=1,
        strategy="auto",
    )

    if best_model_path:
        log.info(f"Loading best checkpoint: {best_model_path}")
        test_trainer.test(model=model, datamodule=datamodule, ckpt_path=best_model_path)
    else:
        log.warning("No best checkpoint found, testing with final model")
        test_trainer.test(model=model, datamodule=datamodule)

    # Extract test metrics
    cm = test_trainer.callback_metrics

    test_metrics: Dict[str, Any] = {
        "run_idx": run_idx,

        # Primary reporting (imbalance + equal class importance)
        "test/f1_macro": to_float(cm.get("test/f1_macro")),
        "test/precision_macro": to_float(cm.get("test/precision_macro")),
        "test/recall_macro": to_float(cm.get("test/recall_macro")),

        # Context / secondary metrics
        "test/acc": to_float(cm.get("test/acc")),
        "test/f1_weighted": to_float(cm.get("test/f1_weighted")),
        "test/precision_weighted": to_float(cm.get("test/precision_weighted")),
        "test/recall_weighted": to_float(cm.get("test/recall_weighted")),
        "test/loss": to_float(cm.get("test/loss")),

        # Best validation metric(s)
        "val/f1_macro_best": to_float(val_f1_macro_best),
    }

    # Optionally capture per-class test metrics if your model logs them (e.g., test/f1_c0..c9)
    # This keeps the script compatible whether per-class logging is enabled or not.
    n_classes = None
    try:
        n_classes = int(getattr(datamodule, "num_classes"))
    except Exception:
        n_classes = None

    if n_classes is not None and n_classes > 0:
        for i in range(n_classes):
            for metric_name in ("precision", "recall", "f1"):
                key = f"test/{metric_name}_c{i}"
                if key in cm:
                    test_metrics[key] = to_float(cm.get(key))

    log.info(
        f"Run {run_idx}/{cfg.n_runs} completed | "
        f"test/acc: {(test_metrics.get('test/acc') or 0.0):.4f} | "
        f"test/f1_macro: {(test_metrics.get('test/f1_macro') or 0.0):.4f}"
    )

    return test_metrics


@hydra.main(version_base="1.3", config_path="../configs", config_name="train.yaml")
def main(cfg: DictConfig) -> None:
    """
    Main training function using PyTorch Lightning.

    Supports multiple runs with different seeds for statistical robustness.
    """
    torch.set_float32_matmul_precision("medium")

    # Print configuration
    if cfg.get("print_config"):
        log.info(f"Config:\n{OmegaConf.to_yaml(cfg)}")

    log_path = cfg.get("log_path", "logs")
    research_name = cfg.get("research_name", "default")
    experiment_name = cfg.get("experiment_name", "default")
    run_name = cfg.get("run_name", date.today().strftime("%Y-%m-%d"))
    base_save_dir = Path(log_path) / "train" / research_name / experiment_name / run_name

    log.info(f"Saving results to: {base_save_dir}")

    # Get number of runs
    n_runs = cfg.get("n_runs", 1)
    log.info(f"Running {n_runs} training run(s)")

    # Save hyperparameters
    summary_dir = base_save_dir / "summary"
    summary_dir.mkdir(parents=True, exist_ok=True)
    hparams_file = summary_dir / "hyperparameters.yaml"
    with open(hparams_file, "w") as f:
        f.write(OmegaConf.to_yaml(cfg, resolve=True))
    log.info(f"Saved hyperparameters to: {hparams_file}")

    # Run multiple training runs
    all_metrics: List[Dict[str, Any]] = []
    for run_idx in range(1, n_runs + 1):
        run_metrics = train_single_run(cfg=cfg, run_idx=run_idx, base_save_dir=base_save_dir)
        all_metrics.append(run_metrics)

    # Calculate and save summary statistics
    if n_runs > 1:
        log.info("Calculating summary statistics across runs")
        all_metrics_df, summary_df = calculate_summary_statistics(all_metrics)

        all_metrics_df.to_csv(summary_dir / "all_runs_metrics.csv", index=False)
        summary_df.to_csv(summary_dir / "summary_statistics.csv", index=False)

        log.info(f"\nSummary Statistics:\n{summary_df.to_string()}")
        log.info(f"Saved summary to: {summary_dir}")
    else:
        # Save single run metrics
        metrics_df = pd.DataFrame([all_metrics[0]])
        metrics_df.to_csv(summary_dir / "metrics.csv", index=False)
        log.info(f"Saved metrics to: {summary_dir / 'metrics.csv'}")

    log.info("All training runs complete!")


if __name__ == "__main__":
    main()
