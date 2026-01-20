"""
Train script using PyTorch Lightning.

This replaces the old manual training loop with Lightning's Trainer API.
"""

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
import torch

from utils.instantiators import instantiate_callbacks

# Setup root
rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

from src.utils.pylogger import RankedLogger

from lightning import seed_everything

log = RankedLogger(__name__, rank_zero_only=True)


def calculate_summary_statistics(
    all_metrics: List[Dict[str, Any]],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Calculate summary statistics across multiple runs."""
    all_metrics_df = pd.DataFrame(all_metrics)
    summary_statistics = []
    metrics_to_analyze = [
        m for m in all_metrics_df.columns if m not in ["run_idx"]
    ]

    for metric in metrics_to_analyze:
        values = all_metrics_df[metric].dropna()
        n = len(values)
        mean = values.mean()
        std = values.std(ddof=1)
        se = std / np.sqrt(n) if n > 0 else 0.0

        if n > 1:
            t = stats.t.ppf(0.975, df=n - 1)
            margin = t * se
            ci_lower = mean - margin
            ci_upper = mean + margin
        else:
            ci_lower = ci_upper = mean

        summary_statistics.append(
            {
                "metric": metric,
                "mean": mean,
                "std": std,
                "ci_lower": ci_lower,
                "ci_upper": ci_upper,
                "min": values.min(),
                "max": values.max(),
            }
        )

    return all_metrics_df, pd.DataFrame(summary_statistics)


def train_single_run(
    cfg: DictConfig,
    run_idx: int,
    base_save_dir: Path
) -> Dict[str, Any]:
    """Train a single run with a specific seed."""
    
    # Set seed for this run
    seed = cfg.get("seed", 42) + run_idx
    seed_everything(seed, workers=True)
    log.info(f"Run {run_idx + 1}/{cfg.n_runs} with seed {seed}")

    # Create run-specific directory (for checkpoints and metrics only)
    run_save_dir = base_save_dir / f"run_{run_idx}"
    run_save_dir.mkdir(parents=True, exist_ok=True)

    # Instantiate datamodule with run-specific seed
    log.info(f"Instantiating datamodule for run {run_idx}")
    datamodule = hydra.utils.instantiate(cfg.data, seed=seed)
    datamodule.prepare_data()
    datamodule.setup()
    
    # Instantiate model
    log.info(f"Instantiating model for run {run_idx}")
    model = hydra.utils.instantiate(
        cfg.model, 
        datamodule=datamodule
    )

    # Setup callbacks with checkpoint directory override
    from lightning.pytorch.callbacks import ModelCheckpoint
    
    callbacks: List[Callback] = instantiate_callbacks(cfg.get("callbacks"))
    
    # Update ModelCheckpoint dirpath to use run-specific directory
    for callback in callbacks:
        if isinstance(callback, ModelCheckpoint) and callback.dirpath is None:
            callback.dirpath = str(run_save_dir / "checkpoints")
            log.info(f"Set checkpoint directory to: {callback.dirpath}")

    # Setup logger
    logger = CSVLogger(
        save_dir=run_save_dir,
        name="",
        version="",
    )

    # Instantiate trainer
    log.info(f"Instantiating trainer for run {run_idx}")
    trainer: Trainer = hydra.utils.instantiate(
        cfg.trainer, 
        callbacks=callbacks, 
        logger=logger
    )

    # Train the model
    log.info(f"Starting training for run {run_idx}")
    trainer.fit(model=model, datamodule=datamodule)

    # Extract validation-best metrics right after fit.
    # NOTE: Lightning's `trainer.callback_metrics` will be updated/overwritten by the test loop,
    # so reading val metrics after `trainer.test()` often yields missing values.
    def to_float(value):
        """Convert tensor or numeric value to float."""
        if value is None:
            return None
        if hasattr(value, "item"):
            return float(value.item())
        return float(value)

    val_acc_best = trainer.callback_metrics.get("val/acc_best")
    val_f1_best = trainer.callback_metrics.get("val/f1_best")

    # Fallback: read directly from the model metrics if not present.
    if val_acc_best is None and hasattr(model, "val_acc_best"):
        try:
            val_acc_best = model.val_acc_best.compute()
        except Exception:
            val_acc_best = None
    if val_f1_best is None and hasattr(model, "val_f1_best"):
        try:
            val_f1_best = model.val_f1_best.compute()
        except Exception:
            val_f1_best = None

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

    # Extract metrics (convert tensors to floats)
    test_metrics = {
        "run_idx": run_idx,
        "test/loss": to_float(test_trainer.callback_metrics.get("test/loss", 0.0)),
        "test/acc": to_float(test_trainer.callback_metrics.get("test/acc", 0.0)),
        "test/precision": to_float(test_trainer.callback_metrics.get("test/precision", 0.0)),
        "test/recall": to_float(test_trainer.callback_metrics.get("test/recall", 0.0)),
        "test/f1": to_float(test_trainer.callback_metrics.get("test/f1", 0.0)),
        "val/acc_best": to_float(val_acc_best) if val_acc_best is not None else 0.0,
        "val/f1_best": to_float(val_f1_best) if val_f1_best is not None else 0.0,
    }

    log.info(
        f"Run {run_idx + 1}/{cfg.n_runs} completed | "
        f"test/acc: {test_metrics['test/acc']:.4f} | "
        f"test/f1: {test_metrics['test/f1']:.4f}"
    )

    return test_metrics


@hydra.main(version_base="1.3", config_path="../configs", config_name="train.yaml")
def main(cfg: DictConfig) -> None:
    """
    Main training function using PyTorch Lightning.
    
    Supports multiple runs with different seeds for statistical robustness.
    """
    torch.set_float32_matmul_precision('medium')
    
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
    all_metrics = []
    for run_idx in range(n_runs):
        run_metrics = train_single_run(
            cfg=cfg,
            run_idx=run_idx,
            base_save_dir=base_save_dir
        )
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
