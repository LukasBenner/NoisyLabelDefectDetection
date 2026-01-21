from __future__ import annotations

from datetime import date
from pathlib import Path
from typing import Any, Dict, List, Optional

import hydra
import numpy as np
import pandas as pd
import rootutils
import torch
from lightning import Callback, Trainer, seed_everything
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import CSVLogger
from omegaconf import DictConfig, OmegaConf
from scipy import stats

from utils.instantiators import instantiate_callbacks

rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

from src.utils.pylogger import RankedLogger

log = RankedLogger(__name__, rank_zero_only=True)


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


def train_single_run(cfg: DictConfig, run_idx: int, base_save_dir: Path) -> Dict[str, Any]:
    seed = cfg.get("seed", 42) + (run_idx - 1)
    seed_everything(seed, workers=True)
    log.info(f"Run {run_idx}/{cfg.n_runs} with seed {seed}")

    run_save_dir = base_save_dir / f"run_{run_idx}"
    run_save_dir.mkdir(parents=True, exist_ok=True)

    # Instantiate datamodule
    log.info("Instantiating datamodule")
    datamodule = hydra.utils.instantiate(cfg.data, seed=seed)
    datamodule.setup()

    # SEAL outer loop params
    num_iterations = int(cfg.model.get("num_iterations"))
    epochs_per_iteration = int(cfg.get("seal_epochs_per_iteration", cfg.model.get("epochs_per_iteration", cfg.trainer.max_epochs)))

    log.info(f"SEAL iterations: {num_iterations} | epochs per iteration (T): {epochs_per_iteration}")

    soft_labels = None  # carried across iterations
    best_ckpt_path = None
    best_ckpt_score = None

    model = hydra.utils.instantiate(
        cfg.model,
        datamodule=datamodule,
        initial_soft_labels=soft_labels,
    )

    callbacks: List[Callback] = instantiate_callbacks(cfg.get("callbacks"))

    # Ensure checkpointing writes into iteration folder
    ckpt_cb = _find_checkpoint_callback(callbacks)
    if ckpt_cb is not None and ckpt_cb.dirpath is None:
        ckpt_cb.dirpath = str(run_save_dir / "checkpoints")

    logger = CSVLogger(save_dir=run_save_dir, name="", version="")

    trainer: Trainer = hydra.utils.instantiate(
        cfg.trainer,
        max_epochs=epochs_per_iteration*num_iterations,
        callbacks=callbacks,
        logger=logger,
    )

    trainer.fit(model=model, datamodule=datamodule)

    # Carry soft labels to next iteration
    soft_labels = model.soft_labels.detach().cpu()

    # Track best checkpoint across iterations (by callback score)
    if ckpt_cb is not None and ckpt_cb.best_model_path:
        score = _to_float(ckpt_cb.best_model_score)
        if score is not None and (best_ckpt_score is None or score > best_ckpt_score):
            best_ckpt_score = score
            best_ckpt_path = ckpt_cb.best_model_path

    # --- Final test ---
    log.info("Instantiating final model for testing")
    final_model = hydra.utils.instantiate(
        cfg.model,
        datamodule=datamodule,
        initial_soft_labels=soft_labels,
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

    # Extract metrics
    val_acc_best = test_trainer.callback_metrics.get("val/acc_best")
    val_f1_best = test_trainer.callback_metrics.get("val/f1_best")

    test_metrics = {
        "run_idx": run_idx,
        "test/loss": _to_float(test_trainer.callback_metrics.get("test/loss", 0.0)) or 0.0,
        "test/acc": _to_float(test_trainer.callback_metrics.get("test/acc", 0.0)) or 0.0,
        "test/f1": _to_float(test_trainer.callback_metrics.get("test/f1", 0.0)) or 0.0,
        "val/acc_best": _to_float(val_acc_best) or 0.0,
        "val/f1_best": _to_float(val_f1_best) or 0.0,
        "best_ckpt_score": best_ckpt_score or 0.0,
    }

    log.info(
        f"Run {run_idx}/{cfg.n_runs} completed | "
        f"test/acc: {test_metrics['test/acc']:.4f} | test/f1: {test_metrics['test/f1']:.4f}"
    )
    return test_metrics


@hydra.main(version_base="1.3", config_path="../configs", config_name="train.yaml")
def main(cfg: DictConfig) -> None:
    torch.set_float32_matmul_precision("medium")

    if cfg.get("print_config"):
        log.info(f"Config:\n{OmegaConf.to_yaml(cfg)}")

    log_path = cfg.get("log_path", "logs")
    research_name = cfg.get("research_name", "default")
    experiment_name = cfg.get("experiment_name", "default")
    run_name = cfg.get("run_name", date.today().strftime("%Y-%m-%d"))
    base_save_dir = Path(log_path) / "train" / research_name / experiment_name / run_name
    base_save_dir.mkdir(parents=True, exist_ok=True)

    log.info(f"Saving results to: {base_save_dir}")

    n_runs = int(cfg.get("n_runs", 1))
    log.info(f"Running {n_runs} training run(s)")

    # Save hyperparameters
    summary_dir = base_save_dir / "summary"
    summary_dir.mkdir(parents=True, exist_ok=True)
    hparams_file = summary_dir / "hyperparameters.yaml"
    with open(hparams_file, "w") as f:
        f.write(OmegaConf.to_yaml(cfg, resolve=True))

    all_metrics: List[Dict[str, Any]] = []
    for run_idx in range(1, n_runs + 1):
        run_metrics = train_single_run(cfg=cfg, run_idx=run_idx, base_save_dir=base_save_dir)
        all_metrics.append(run_metrics)

    # Summaries
    if n_runs > 1:
        all_metrics_df, summary_df = calculate_summary_statistics(all_metrics)
        all_metrics_df.to_csv(summary_dir / "all_runs_metrics.csv", index=False)
        summary_df.to_csv(summary_dir / "summary_statistics.csv", index=False)
        log.info(f"\nSummary Statistics:\n{summary_df.to_string(index=False)}")
    else:
        pd.DataFrame([all_metrics[0]]).to_csv(summary_dir / "metrics.csv", index=False)

    log.info("All training runs complete.")


if __name__ == "__main__":
    main()
