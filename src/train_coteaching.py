"""Train script for Co-Teaching using PyTorch Lightning."""

from datetime import date
from pathlib import Path
from typing import Any, Dict, List

import hydra
import numpy as np
import pandas as pd
import rootutils
import torch
from lightning import Callback, Trainer, seed_everything
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


def _to_float(value):
    if value is None:
        return None
    if hasattr(value, "item"):
        return float(value.item())
    return float(value)


def train_single_run(cfg: DictConfig, run_idx: int, base_save_dir: Path) -> Dict[str, Any]:
    seed = cfg.get("seed", 42) + (run_idx - 1)
    seed_everything(seed, workers=True)
    log.info(f"Run {run_idx}/{cfg.n_runs} with seed {seed}")

    run_save_dir = base_save_dir / f"run_{run_idx}"
    run_save_dir.mkdir(parents=True, exist_ok=True)

    log.info("Instantiating datamodule")
    datamodule = hydra.utils.instantiate(cfg.data, seed=seed)
    datamodule.prepare_data()
    datamodule.setup()

    log.info("Instantiating model")
    model = hydra.utils.instantiate(cfg.model, datamodule=datamodule)

    callbacks: List[Callback] = instantiate_callbacks(cfg.get("callbacks"))

    logger = CSVLogger(save_dir=run_save_dir, name="", version="")

    trainer_kwargs = {}
    if "reload_dataloaders_every_n_epochs" not in cfg.trainer:
        trainer_kwargs["reload_dataloaders_every_n_epochs"] = 1

    trainer: Trainer = hydra.utils.instantiate(
        cfg.trainer,
        callbacks=callbacks,
        logger=logger,
        **trainer_kwargs,
    )

    log.info("Starting training")
    trainer.fit(model=model, datamodule=datamodule)

    val_acc_best = trainer.callback_metrics.get("val/acc_best")
    val_f1_best = trainer.callback_metrics.get("val/f1_best")

    log.info("Starting testing")
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

    ckpt_cb = getattr(trainer, "checkpoint_callback", None)
    best_model_path = getattr(ckpt_cb, "best_model_path", None) if ckpt_cb is not None else None

    if best_model_path:
        test_trainer.test(model=model, datamodule=datamodule, ckpt_path=best_model_path)
    else:
        test_trainer.test(model=model, datamodule=datamodule)

    test_metrics = {
        "run_idx": run_idx,
        "test/loss": _to_float(test_trainer.callback_metrics.get("test/loss", 0.0)) or 0.0,
        "test/acc": _to_float(test_trainer.callback_metrics.get("test/acc", 0.0)) or 0.0,
        "test/f1": _to_float(test_trainer.callback_metrics.get("test/f1", 0.0)) or 0.0,
        "val/acc_best": _to_float(val_acc_best) if val_acc_best is not None else 0.0,
        "val/f1_best": _to_float(val_f1_best) if val_f1_best is not None else 0.0,
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

    summary_dir = base_save_dir / "summary"
    summary_dir.mkdir(parents=True, exist_ok=True)
    hparams_file = summary_dir / "hyperparameters.yaml"
    with open(hparams_file, "w") as f:
        f.write(OmegaConf.to_yaml(cfg, resolve=True))

    all_metrics = []
    for run_idx in range(1, n_runs + 1):
        run_metrics = train_single_run(cfg=cfg, run_idx=run_idx, base_save_dir=base_save_dir)
        all_metrics.append(run_metrics)

    if n_runs > 1:
        all_metrics_df, summary_df = calculate_summary_statistics(all_metrics)
        all_metrics_df.to_csv(summary_dir / "all_runs_metrics.csv", index=False)
        summary_df.to_csv(summary_dir / "summary_statistics.csv", index=False)
        log.info(f"\nSummary Statistics:\n{summary_df.to_string(index=False)}")
    else:
        pd.DataFrame([all_metrics[0]]).to_csv(summary_dir / "metrics.csv", index=False)

    log.info("All training runs complete!")


if __name__ == "__main__":
    main()
