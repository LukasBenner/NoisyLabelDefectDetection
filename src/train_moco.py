from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional
import os

import hydra
import numpy as np
import pandas as pd
import rootutils
import torch
import torch.distributed as dist
from lightning import Callback, Trainer, seed_everything
from lightning.pytorch.callbacks import ModelCheckpoint, LearningRateMonitor
from lightning.pytorch.loggers import CSVLogger
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf
from scipy import stats

from utils.instantiators import instantiate_callbacks

rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

from src.utils.pylogger import RankedLogger

log = RankedLogger(__name__, rank_zero_only=True)

def to_float(value):
    """Convert tensor or numeric value to float."""
    if value is None:
        return None
    if hasattr(value, "item"):
        return float(value.item())
    return float(value)


def calculate_summary_statistics(all_metrics: List[Dict[str, Any]]) -> tuple[pd.DataFrame, pd.DataFrame]:
    all_metrics_df = pd.DataFrame(all_metrics)
    summary_statistics = []
    metrics_to_analyze = [m for m in all_metrics_df.columns if m not in ["run_idx"]]

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


def _to_float(value: Any) -> Optional[float]:
    if value is None:
        return None
    if hasattr(value, "item"):
        return float(value.item())
    return float(value)


def _find_checkpoint_callback(callbacks: List[Callback]) -> Optional[ModelCheckpoint]:
    for cb in callbacks:
        if isinstance(cb, ModelCheckpoint):
            return cb
    return None


def train_single_run(cfg: DictConfig, run_idx: int, base_save_dir: Path):
    seed = cfg.get("seed", 42) + (run_idx - 1)
    seed_everything(seed, workers=True)
    log.info(f"Run {run_idx}/{cfg.n_runs} with seed {seed}")

    run_save_dir = base_save_dir / f"run_{run_idx}"
    run_save_dir.mkdir(parents=True, exist_ok=True)
    
    
    log.info("Instantiating ssl datamodule")
    ssl_datamodule = hydra.utils.instantiate(cfg.ssl_data)
    ssl_datamodule.setup()
    
    log.info("Instantiating ssl model")
    ssl_model = hydra.utils.instantiate(cfg.ssl_model)
    
    ssl_ckpt = ModelCheckpoint(
        monitor="ssl_loss_epoch",
        mode="min",
        save_top_k=1,
        dirpath = str(run_save_dir / "moco_checkpoints"),
        filename="moco-v2-{epoch:03d}-{ssl_loss_epoch:.4f}",
    )
    
    ssl_logger = CSVLogger(
        save_dir=run_save_dir,
        name="ssl_log",
        version="",
    )
    
    ssl_trainer: Trainer = hydra.utils.instantiate(
        cfg.trainer,
        callbacks=[ssl_ckpt, LearningRateMonitor(logging_interval="epoch")],
        max_epochs=cfg.ssl_model.max_epochs,
        logger=ssl_logger
    )
    
    ssl_trainer.fit(ssl_model, train_dataloaders=ssl_datamodule.train_dataloader())

    best_moco_path = ssl_ckpt.best_model_path
    log.info(f"Best MoCo checkpoint: {best_moco_path}")

    # Instantiate datamodule
    log.info("Instantiating datamodule")
    datamodule = hydra.utils.instantiate(cfg.data, seed=seed)
    datamodule.setup()

    model = hydra.utils.instantiate(
        cfg.model,
    )
    model.load_from_moco(best_moco_path, strict=False)

    callbacks: List[Callback] = instantiate_callbacks(cfg.get("callbacks"))

    # Ensure checkpointing writes into iteration folder
    ckpt_cb = _find_checkpoint_callback(callbacks)
    if ckpt_cb is not None and ckpt_cb.dirpath is None:
        ckpt_cb.dirpath = str(run_save_dir / "checkpoints")

    logger = CSVLogger(save_dir=run_save_dir, name="", version="")

    trainer: Trainer = hydra.utils.instantiate(
        cfg.trainer,
        callbacks=callbacks,
        logger=logger
    )

    trainer.fit(model=model, datamodule=datamodule)

    # If Lightning launched via DDP script/subprocess, this function continues on every rank.
    # For a single-GPU final test (to avoid duplicated evaluation), first tear down the
    # distributed process group on all ranks, then only run the test on local rank 0.
    if dist.is_available() and dist.is_initialized():
        try:
            dist.barrier()
        finally:
            dist.destroy_process_group()

    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    if local_rank != 0:
        return {"run_idx": run_idx, "test/loss": None, "test/acc": None}

    # Track best checkpoint across iterations (by callback score)
    if ckpt_cb is not None and ckpt_cb.best_model_path:
        best_ckpt_path = ckpt_cb.best_model_path
    else:
        best_ckpt_path = None

    # --- Final test ---
    log.info("Instantiating final model for testing")
    final_model = hydra.utils.instantiate(
        cfg.model,
    )

    # We need a trainer for test (can reuse last trainer settings)
    test_dir = run_save_dir / "final_test"
    test_dir.mkdir(parents=True, exist_ok=True)
    test_logger = CSVLogger(save_dir=test_dir, name="", version="")
    test_trainer: Trainer = hydra.utils.instantiate(
        cfg.trainer,
        logger=test_logger,
        callbacks=[],
        devices=1,
        strategy="auto",
    )

    log.info("Running test")
    if best_ckpt_path:
        test_trainer.test(model=final_model, datamodule=datamodule, ckpt_path=best_ckpt_path, weights_only=False)
    else:
        test_trainer.test(model=final_model, datamodule=datamodule)
        
    test_metrics = {
        "run_idx": run_idx,
        "test/loss": to_float(test_trainer.callback_metrics.get("test/loss", 0.0)),
        "test/acc": to_float(test_trainer.callback_metrics.get("test/acc", 0.0)),
    }

    log.info(
        f"Run {run_idx}/{cfg.n_runs} completed | "
        f"test/acc: {test_metrics['test/acc']:.4f} | "
        f"test/loss: {test_metrics['test/loss']:.4f}"
    )

    return test_metrics

@hydra.main(version_base="1.3", config_path="../configs", config_name="train.yaml")
def main(cfg: DictConfig) -> None:
    torch.set_float32_matmul_precision("medium")

    local_rank = int(os.environ.get("LOCAL_RANK", "0"))

    if cfg.get("print_config"):
        log.info(f"Config:\n{OmegaConf.to_yaml(cfg)}")

    base_save_dir = Path(HydraConfig.get().runtime.output_dir)

    log.info(f"Saving results to: {base_save_dir}")

    n_runs = int(cfg.get("n_runs", 1))
    log.info(f"Running {n_runs} training run(s)")

    # Save hyperparameters (rank 0 only)
    summary_dir = base_save_dir / "summary"
    summary_dir.mkdir(parents=True, exist_ok=True)
    hparams_file = summary_dir / "hyperparameters.yaml"
    if local_rank == 0:
        with open(hparams_file, "w") as f:
            f.write(OmegaConf.to_yaml(cfg, resolve=True))

    all_metrics = []
    for run_idx in range(1, n_runs + 1):
        run_metrics = train_single_run(cfg=cfg, run_idx=run_idx, base_save_dir=base_save_dir)
        all_metrics.append(run_metrics)

    # Avoid duplicate summary writes from non-zero ranks in DDP script launch.
    if local_rank != 0:
        return

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
