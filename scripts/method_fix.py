import os
from datetime import date
from pathlib import Path
import sys
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
from hydra.core.global_hydra import GlobalHydra
from lightning import Callback, Trainer

project_root = rootutils.setup_root(
    __file__ if "__file__" in dir() else os.getcwd(),
    indicator=".project-root",
    pythonpath=True,
)

src_root = project_root / "src"

if src_root and str(src_root) not in sys.path:
    sys.path.insert(0, str(src_root))

from utils.instantiators import instantiate_callbacks
from utils.utils import (
    calculate_summary_statistics,
    collect_preds_targets,
    create_confusion_matrix,
    get_class_names,
    to_float,
)

# Setup root


from lightning import seed_everything

def validate_one_run(cfg: DictConfig, run_idx: int, seed: int, checkpoint_path: Path) -> Dict[str, Any]:
    # Set seed for reproducibility
    seed_everything(seed)
    # Run validation
    print(f"Run {run_idx} with seed {seed}")
    
    datamodule = hydra.utils.instantiate(cfg.data, seed=seed)
    datamodule.prepare_data()
    datamodule.setup(stage="fit")
    
    model = hydra.utils.instantiate(cfg.model, datamodule=datamodule)
    
    # nal specific stuff
    if type(model).__name__ == "NoiseAdaptionModule":
        trainer: Trainer = hydra.utils.instantiate(
            cfg.trainer,
            max_epochs=31,
        )
        trainer.fit(model=model, datamodule=datamodule)

    val_trainer: Trainer = hydra.utils.instantiate(
        cfg.trainer,
        callbacks=[],
        devices=1,
        strategy="auto",
    )
    
    val_trainer.validate(
        model=model,
        datamodule=datamodule,
        ckpt_path=checkpoint_path,
        weights_only=False,
    )  
    class_names = get_class_names(datamodule)

    cm = val_trainer.callback_metrics
    val_metrics: Dict[str, Any] = {
        "run_idx": run_idx,
        # Primary reporting (imbalance + equal class importance)
        "val/f1_macro": to_float(cm.get("val/f1_macro")),
        "val/precision_macro": to_float(cm.get("val/precision_macro")),
        "val/recall_macro": to_float(cm.get("val/recall_macro")),
        # Context / secondary metrics
        "val/acc": to_float(cm.get("val/acc")),
        "val/f1_weighted": to_float(cm.get("val/f1_weighted")),
        "val/precision_weighted": to_float(cm.get("val/precision_weighted")),
        "val/recall_weighted": to_float(cm.get("val/recall_weighted")),
        "val/loss": to_float(cm.get("val/loss")),
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
                key_idx = f"val/{metric_name}_c{i}"
                if class_names_for_metrics:
                    class_name = class_names_for_metrics[i]
                    key_named = f"val/{metric_name}_{class_name}"
                    if key_named in cm:
                        val_metrics[key_named] = to_float(cm.get(key_named))
                    elif key_idx in cm:
                        val_metrics[key_named] = to_float(cm.get(key_idx))
                else:
                    if key_idx in cm:
                        val_metrics[key_idx] = to_float(cm.get(key_idx))

    return val_metrics
    
    
def select_best_checkpoint(run_dir: Path) -> Path:
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
  

def run_validation(experiment_directory: Path):
    if GlobalHydra.instance().is_initialized():
        GlobalHydra.instance().clear()
    
    hyperparams_path = experiment_directory / "summary" / "hyperparameters.yaml"
    cfg: DictConfig = OmegaConf.load(hyperparams_path)
    
    all_metrics = []
    for run_idx in range(1, 11):
        seed = 41 + run_idx
        run_dir = experiment_directory / f"run_{run_idx}_seed_{seed}"
        checkpoint_path = select_best_checkpoint(run_dir)
        
        val_metrics = validate_one_run(cfg, run_idx, seed, checkpoint_path)
        all_metrics.append(val_metrics)
        
    all_metrics_df, summary_df = calculate_summary_statistics(all_metrics)

    summary_dir = experiment_directory / "val_summary"
    summary_dir.mkdir(parents=True, exist_ok=True)
    all_metrics_df.to_csv(summary_dir / "all_runs_metrics.csv", index=False)
    summary_df.to_csv(summary_dir / "summary_statistics.csv", index=False)
    
    with open(summary_dir / "summary_statistics.txt", "w") as f:
        f.write(summary_df.to_string(index=False))



if __name__ == "__main__":
    experiments = [
        "/home/lukasb/Documents/NoisyLabelDefectDetection/logs/train/SurfaceDefectDetection/noisy_new_nal/2026-02-25_08-21-57_not_precomputed"
    ]
    
    for exp in experiments:
        print(f"Validating experiment: {exp}")
        run_validation(Path(exp))