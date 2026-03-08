"""Two-phase training: pretrain on synthetic data, then finetune on real data.

Usage:
    python src/pretrain_finetune.py experiment=multilabel/pretrain_finetune_ce
"""

import torch
from datetime import date
from pathlib import Path
from typing import Any, Dict, List, Optional

import hydra
import pandas as pd
import rootutils
from hydra.core.hydra_config import HydraConfig
from lightning import Callback, Trainer, seed_everything
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import CSVLogger
from omegaconf import DictConfig, OmegaConf

rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

from src.models.pretrain_finetune_module import PretrainFinetuneModule
from utils.instantiators import instantiate_callbacks
from utils.pylogger import RankedLogger
from utils.utils import (
    calculate_summary_statistics,
    get_class_names,
    to_float,
)

log = RankedLogger(__name__, rank_zero_only=True)


def _collect_per_class_metrics(
    cm: Dict, prefix: str, datamodule: Any
) -> Dict[str, Any]:
    """Collect per-class precision/recall/f1 from callback_metrics."""
    metrics: Dict[str, Any] = {}
    n_classes: Optional[int] = None
    try:
        n_classes = int(getattr(datamodule, "num_classes"))
    except Exception:
        pass

    class_names = get_class_names(datamodule)
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


# ─────────────────────────── Phase 1: Pretrain ───────────────────────────


def pretrain_phase(cfg: DictConfig, seed: int, save_dir: Path) -> str:
    """Pretrain on synthetic data. Returns path to best checkpoint."""
    pretrain_cfg = cfg.pretrain
    log.info("=" * 60)
    log.info("PHASE 1: PRETRAINING ON SYNTHETIC DATA")
    log.info("=" * 60)

    pretrain_dir = save_dir / "pretrain"
    pretrain_dir.mkdir(parents=True, exist_ok=True)

    # Instantiate datamodule
    datamodule = hydra.utils.instantiate(pretrain_cfg.data, seed=seed)
    datamodule.prepare_data()
    datamodule.setup(stage="fit")

    has_val = datamodule.val_dataloader() is not None

    # Instantiate model (uses BaseRobustModule or PretrainFinetuneModule)
    model = hydra.utils.instantiate(pretrain_cfg.model, datamodule=datamodule)

    # Callbacks
    callbacks: List[Callback] = instantiate_callbacks(pretrain_cfg.get("callbacks"))
    if not has_val:
        # Remove callbacks that monitor validation metrics
        callbacks = [
            cb for cb in callbacks
            if not isinstance(cb, (ModelCheckpoint,))
            and not hasattr(cb, "monitor")
        ]
        # Add epoch-based checkpoint (save last)
        ckpt_cb = ModelCheckpoint(
            dirpath=str(pretrain_dir / "checkpoints"),
            filename="pretrain-epoch_{epoch:03d}",
            save_last=True,
            save_top_k=0,  # only save last
        )
        callbacks.append(ckpt_cb)
        log.info("No validation data — using epoch-based checkpointing (save last).")
    else:
        for callback in callbacks:
            if isinstance(callback, ModelCheckpoint) and callback.dirpath is None:
                callback.dirpath = str(pretrain_dir / "checkpoints")

    logger = CSVLogger(save_dir=pretrain_dir, name="", version="")

    # Trainer — disable validation if no val data
    trainer_overrides = {}
    if not has_val:
        trainer_overrides["limit_val_batches"] = 0

    trainer: Trainer = hydra.utils.instantiate(
        pretrain_cfg.trainer, callbacks=callbacks, logger=logger, **trainer_overrides
    )

    # Train
    log.info("Starting pretraining...")
    trainer.fit(model=model, datamodule=datamodule)

    # Resolve checkpoint path: prefer best (val-monitored), fall back to last
    best_path = None
    if trainer.checkpoint_callback:
        best_path = trainer.checkpoint_callback.best_model_path or None
        if not best_path and hasattr(trainer.checkpoint_callback, "last_model_path"):
            best_path = trainer.checkpoint_callback.last_model_path or None
    if not best_path:
        fallback_path = str(pretrain_dir / "checkpoints" / "last_pretrain.ckpt")
        trainer.save_checkpoint(fallback_path)
        best_path = fallback_path

    if has_val:
        val_f1 = trainer.callback_metrics.get("val/f1_macro", 0.0)
        log.info(f"Pretraining complete. Best checkpoint: {best_path}")
        log.info(f"Pretrain val/f1_macro: {val_f1:.4f}")
    else:
        log.info(f"Pretraining complete (no validation). Checkpoint: {best_path}")

    return best_path


# ─────────────────────────── Phase 2: Finetune ───────────────────────────


def finetune_phase(
    cfg: DictConfig,
    seed: int,
    save_dir: Path,
    pretrained_ckpt: str,
    run_idx: int,
) -> Dict[str, Any]:
    """Finetune on real data using pretrained backbone weights."""
    finetune_cfg = cfg.finetune
    log.info("=" * 60)
    log.info("PHASE 2: FINETUNING ON REAL DATA")
    log.info("=" * 60)

    finetune_dir = save_dir / "finetune"
    finetune_dir.mkdir(parents=True, exist_ok=True)

    # Instantiate datamodule
    datamodule = hydra.utils.instantiate(finetune_cfg.data, seed=seed)
    datamodule.prepare_data()
    datamodule.setup(stage="fit")

    # Instantiate model
    model: PretrainFinetuneModule = hydra.utils.instantiate(
        finetune_cfg.model, datamodule=datamodule
    )

    # Load pretrained backbone weights
    log.info(f"Loading backbone weights from: {pretrained_ckpt}")
    PretrainFinetuneModule.load_backbone_weights(model, pretrained_ckpt)

    # Callbacks
    callbacks: List[Callback] = instantiate_callbacks(finetune_cfg.get("callbacks"))
    for callback in callbacks:
        if isinstance(callback, ModelCheckpoint) and callback.dirpath is None:
            callback.dirpath = str(finetune_dir / "checkpoints")

    logger = CSVLogger(save_dir=finetune_dir, name="", version="")

    # Trainer
    trainer: Trainer = hydra.utils.instantiate(
        finetune_cfg.trainer, callbacks=callbacks, logger=logger
    )

    # Train
    log.info("Starting finetuning...")
    trainer.fit(model=model, datamodule=datamodule)

    val_f1_macro_best = trainer.callback_metrics.get("val/f1_macro_best")
    if val_f1_macro_best is None and hasattr(model, "val_f1_macro_best"):
        try:
            val_f1_macro_best = model.val_f1_macro_best.compute()
        except Exception:
            val_f1_macro_best = None

    best_model_path = None
    if trainer.checkpoint_callback:
        best_model_path = trainer.checkpoint_callback.best_model_path or None

    # ---- Evaluate ----
    eval_prefix = "test" if cfg.get("run_test", False) else "val"

    eval_dir = finetune_dir / f"final_{eval_prefix}"
    eval_dir.mkdir(parents=True, exist_ok=True)
    eval_logger = CSVLogger(save_dir=eval_dir, name="", version="")
    eval_trainer: Trainer = hydra.utils.instantiate(
        finetune_cfg.trainer,
        callbacks=[],
        logger=eval_logger,
        devices=1,
        strategy="auto",
    )

    if cfg.get("run_test", False):
        if best_model_path:
            eval_trainer.test(model=model, datamodule=datamodule, ckpt_path=best_model_path, weights_only=False)
        else:
            eval_trainer.test(model=model, datamodule=datamodule)
    else:
        if best_model_path:
            eval_trainer.validate(model=model, datamodule=datamodule, ckpt_path=best_model_path, weights_only=False)
        else:
            eval_trainer.validate(model=model, datamodule=datamodule)

    cm = eval_trainer.callback_metrics
    result_metrics: Dict[str, Any] = {
        "run_idx": run_idx,
        f"{eval_prefix}/f1_macro": to_float(cm.get(f"{eval_prefix}/f1_macro")),
        f"{eval_prefix}/precision_macro": to_float(cm.get(f"{eval_prefix}/precision_macro")),
        f"{eval_prefix}/recall_macro": to_float(cm.get(f"{eval_prefix}/recall_macro")),
        f"{eval_prefix}/acc": to_float(cm.get(f"{eval_prefix}/acc")),
        f"{eval_prefix}/f1_weighted": to_float(cm.get(f"{eval_prefix}/f1_weighted")),
        f"{eval_prefix}/precision_weighted": to_float(cm.get(f"{eval_prefix}/precision_weighted")),
        f"{eval_prefix}/recall_weighted": to_float(cm.get(f"{eval_prefix}/recall_weighted")),
        f"{eval_prefix}/loss": to_float(cm.get(f"{eval_prefix}/loss")),
        "val/f1_macro_best": to_float(val_f1_macro_best),
    }
    result_metrics.update(_collect_per_class_metrics(cm, eval_prefix, datamodule))

    log.info(
        f"Run {run_idx} completed | "
        f"{eval_prefix}/f1_macro: {(result_metrics.get(f'{eval_prefix}/f1_macro') or 0.0):.4f}"
    )
    return result_metrics


# ─────────────────────────── Main ───────────────────────────


def train_single_run(
    cfg: DictConfig, run_idx: int, base_save_dir: Path
) -> Dict[str, Any]:
    """Run both phases for a single seed."""
    seed = cfg.get("seed", 42) + (run_idx - 1)
    seed_everything(seed, workers=True)
    log.info(f"Run {run_idx}/{cfg.n_runs} with seed {seed}")

    run_dir = base_save_dir / f"run_{run_idx}_seed_{seed}"
    run_dir.mkdir(parents=True, exist_ok=True)

    # Phase 1: Pretrain (only once per run)
    pretrained_ckpt = pretrain_phase(cfg, seed, run_dir)

    # Phase 2: Finetune
    result_metrics = finetune_phase(cfg, seed, run_dir, pretrained_ckpt, run_idx)

    return result_metrics


@hydra.main(
    version_base="1.3",
    config_path="../configs",
    config_name="pretrain_finetune",
)
def main(cfg: DictConfig) -> None:
    """Two-phase training: pretrain on synthetic, finetune on real data."""
    torch.set_float32_matmul_precision("medium")

    if cfg.get("print_config"):
        log.info(f"Config:\n{OmegaConf.to_yaml(cfg)}")

    base_save_dir = Path(HydraConfig.get().runtime.output_dir)
    log.info(f"Saving results to: {base_save_dir}")

    n_runs = cfg.get("n_runs", 1)
    log.info(f"Running {n_runs} pretrain+finetune run(s)")

    # Save hyperparameters
    eval_prefix = "test" if cfg.get("run_test", False) else "val"
    summary_dir = base_save_dir / f"{eval_prefix}_summary"
    summary_dir.mkdir(parents=True, exist_ok=True)
    with open(summary_dir / "hyperparameters.yaml", "w") as f:
        f.write(OmegaConf.to_yaml(cfg, resolve=True))

    # Run training
    all_metrics: List[Dict[str, Any]] = []
    for run_idx in range(1, n_runs + 1):
        run_metrics = train_single_run(cfg, run_idx, base_save_dir)
        all_metrics.append(run_metrics)

    # Summary statistics
    if n_runs > 1:
        all_metrics_df, summary_df = calculate_summary_statistics(all_metrics)
        all_metrics_df.to_csv(summary_dir / "all_runs_metrics.csv", index=False)
        summary_df.to_csv(summary_dir / "summary_statistics.csv", index=False)
        log.info(f"\nSummary Statistics:\n{summary_df.to_string()}")
    else:
        metrics_df = pd.DataFrame([all_metrics[0]])
        metrics_df.to_csv(summary_dir / "metrics.csv", index=False)

    log.info("All pretrain+finetune runs complete!")


if __name__ == "__main__":
    main()
