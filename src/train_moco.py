from __future__ import annotations

from datetime import date
from pathlib import Path
from typing import Any, Dict, List

import hydra
import pandas as pd
import rootutils
import torch
from lightning import Callback, Trainer, seed_everything
from lightning.pytorch.callbacks import LearningRateMonitor, ModelCheckpoint
from lightning.pytorch.loggers import CSVLogger
from omegaconf import DictConfig, OmegaConf

from utils.instantiators import instantiate_callbacks
from utils.utils import (
    calculate_summary_statistics,
    collect_preds_targets,
    create_confusion_matrix,
    get_class_names,
    to_float,
)

rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

from src.utils.pylogger import RankedLogger

log = RankedLogger(__name__, rank_zero_only=True)


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
    datamodule.prepare_data()
    datamodule.setup(stage="fit")

    model = hydra.utils.instantiate(cfg.model)
    model.load_from_moco(best_moco_path, strict=False)

    callbacks: List[Callback] = instantiate_callbacks(cfg.get("callbacks"))

    # Ensure checkpointing writes into run-specific folder
    for callback in callbacks:
        if isinstance(callback, ModelCheckpoint) and callback.dirpath is None:
            callback.dirpath = str(run_save_dir / "checkpoints")
            log.info(f"Set checkpoint directory to: {callback.dirpath}")

    logger = CSVLogger(save_dir=run_save_dir, name="", version="")

    trainer: Trainer = hydra.utils.instantiate(
        cfg.trainer,
        callbacks=callbacks,
        logger=logger
    )

    trainer.fit(model=model, datamodule=datamodule)

    val_f1_macro_best = trainer.callback_metrics.get("val/f1_macro_best")
    if val_f1_macro_best is None and hasattr(model, "val_f1_macro_best"):
        try:
            val_f1_macro_best = model.val_f1_macro_best.compute()
        except Exception:
            val_f1_macro_best = None

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
        test_trainer.test(
            model=model,
            datamodule=datamodule,
            ckpt_path=best_model_path,
            weights_only=False,
        )
    else:
        log.warning("No best checkpoint found, testing with final model")
        test_trainer.test(
            model=model,
            datamodule=datamodule,
            weights_only=False,
        )

    log.info("Generating confusion matrix")
    datamodule.setup(stage="test")
    class_names = get_class_names(datamodule)
    if class_names is None:
        log.warning("No class names found; skipping confusion matrix")
    else:
        preds, targets = collect_preds_targets(model, datamodule.test_dataloader())
        create_confusion_matrix(
            preds=preds.numpy(),
            targets=targets.numpy(),
            class_names=class_names,
            logging_directory=str(test_dir),
        )

    cm = test_trainer.callback_metrics

    test_metrics: Dict[str, Any] = {
        "run_idx": run_idx,
        "test/f1_macro": to_float(cm.get("test/f1_macro")),
        "test/precision_macro": to_float(cm.get("test/precision_macro")),
        "test/recall_macro": to_float(cm.get("test/recall_macro")),
        "test/acc": to_float(cm.get("test/acc")),
        "test/f1_weighted": to_float(cm.get("test/f1_weighted")),
        "test/precision_weighted": to_float(cm.get("test/precision_weighted")),
        "test/recall_weighted": to_float(cm.get("test/recall_weighted")),
        "test/loss": to_float(cm.get("test/loss")),
        "val/f1_macro_best": to_float(val_f1_macro_best),
    }

    n_classes = None
    try:
        n_classes = int(getattr(datamodule, "num_classes"))
    except Exception:
        n_classes = None

    if n_classes is None and class_names:
        n_classes = len(class_names)

    class_names_for_metrics = None
    if class_names and n_classes is not None and len(class_names) >= n_classes:
        class_names_for_metrics = class_names

    if n_classes is not None and n_classes > 0:
        for i in range(n_classes):
            for metric_name in ("precision", "recall", "f1"):
                key_idx = f"test/{metric_name}_c{i}"
                if class_names_for_metrics:
                    class_name = class_names_for_metrics[i]
                    key_named = f"test/{metric_name}_{class_name}"
                    if key_named in cm:
                        test_metrics[key_named] = to_float(cm.get(key_named))
                    elif key_idx in cm:
                        test_metrics[key_named] = to_float(cm.get(key_idx))
                else:
                    if key_idx in cm:
                        test_metrics[key_idx] = to_float(cm.get(key_idx))

    log.info(
        f"Run {run_idx}/{cfg.n_runs} completed | "
        f"test/acc: {(test_metrics.get('test/acc') or 0.0):.4f} | "
        f"test/f1_macro: {(test_metrics.get('test/f1_macro') or 0.0):.4f}"
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
    base_save_dir = (
        Path(log_path) / "train" / research_name / experiment_name / run_name
    )

    log.info(f"Saving results to: {base_save_dir}")

    n_runs = int(cfg.get("n_runs", 1))
    log.info(f"Running {n_runs} training run(s)")

    summary_dir = base_save_dir / "summary"
    summary_dir.mkdir(parents=True, exist_ok=True)
    hparams_file = summary_dir / "hyperparameters.yaml"
    with open(hparams_file, "w") as f:
        f.write(OmegaConf.to_yaml(cfg, resolve=True))
    log.info(f"Saved hyperparameters to: {hparams_file}")

    all_metrics = []
    for run_idx in range(1, n_runs + 1):
        run_metrics = train_single_run(cfg=cfg, run_idx=run_idx, base_save_dir=base_save_dir)
        all_metrics.append(run_metrics)

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
