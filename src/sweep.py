"""Hyperparameter sweep using Optuna with PyTorch Lightning."""

from pathlib import Path
from typing import Any, Dict, List, Optional

import hydra
import pandas as pd
import rootutils
from lightning import Callback, Trainer, seed_everything
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import CSVLogger
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf
import optuna

from utils.instantiators import instantiate_callbacks
from utils.pylogger import RankedLogger
from utils.utils import (
    collect_preds_targets,
    create_confusion_matrix,
    get_class_names,
    get_metric_value,
    to_float,
)

rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

log = RankedLogger(__name__, rank_zero_only=True)


def suggest_hyperparameters(
    trial: optuna.trial.BaseTrial, search_space: DictConfig
) -> Dict[str, Any]:

    def _suggest_recursive(space: DictConfig, prefix: str = ""):
        """Recursively suggest hyperparameters."""
        result = {}

        for key, value in space.items():
            full_key = f"{prefix}.{key}" if prefix else str(key)

            if isinstance(value, DictConfig):
                if "type" in value:
                    param_type = value.type

                    if param_type == "float":
                        result[key] = trial.suggest_float(
                            full_key, value.low, value.high, log=value.get("log", False)
                        )
                    elif param_type == "int":
                        result[key] = trial.suggest_int(
                            full_key, value.low, value.high, log=value.get("log", False)
                        )
                    elif param_type == "categorical":
                        result[key] = trial.suggest_categorical(
                            full_key, value.choices
                        )  # type: ignore
                    elif param_type == "uniform":
                        result[key] = trial.suggest_uniform(
                            full_key, value.low, value.high
                        )
                    else:
                        log.warning(f"Unknown parameter type: {param_type}")
                else:
                    # Nested configuration
                    result[key] = _suggest_recursive(value, full_key)

        return result

    return _suggest_recursive(search_space)


def update_config_with_suggestions(
    cfg: DictConfig, suggestions: Dict[str, Any]
) -> DictConfig:
    """Update configuration with suggested hyperparameters.

    Args:
        cfg: Original configuration
        suggestions: Nested dictionary of suggested values

    Returns:
        Updated configuration
    """

    def _update_recursive(config, updates):
        for key, value in updates.items():
            if isinstance(value, dict):
                if key not in config:
                    config[key] = {}
                _update_recursive(config[key], value)
            else:
                config[key] = value

    cfg_copy = OmegaConf.create(OmegaConf.to_container(cfg, resolve=True))
    assert isinstance(cfg_copy, DictConfig), "Config must be a DictConfig"
    _update_recursive(cfg_copy, suggestions)
    return cfg_copy


@hydra.main(version_base="1.3", config_path="../configs", config_name="sweep.yaml")
def main(cfg: DictConfig) -> Optional[float]:
    """Main sweep function using PyTorch Lightning and Optuna."""
    n_trials = cfg.get("n_trials", 10)
    optimized_metric = cfg.get("optimized_metric", "val/f1_macro_best")
    direction = cfg.get("direction", "maximize")
    seed = cfg.get("seed", 42)
    
    # Set global seed
    seed_everything(seed, workers=True)

    # Setup logging paths
    base_log_path = Path(HydraConfig.get().runtime.output_dir)
    experiment_name = cfg.get("experiment_name", "default")

    log.info(f"Saving sweep results to: {base_log_path}")

    # Setup datamodule once to get num_classes
    log.info("Setting up initial datamodule to get dataset info")
    temp_datamodule = hydra.utils.instantiate(cfg.data, seed=seed)
    temp_datamodule.prepare_data()
    temp_datamodule.setup(stage="fit")

    # Get search space from config
    if "search_space" not in cfg:
        raise ValueError(
            "No search_space defined in config. Please add search_space to your hparams_search config."
        )

    search_space = cfg.search_space

    # Define objective function for Optuna
    def objective(trial: optuna.Trial) -> float:
        """Objective function for a single trial using PyTorch Lightning."""
        try:
            # Set seed for this trial
            trial_seed = seed + trial.number
            seed_everything(trial_seed, workers=True)
            
            # Create a copy of config for this trial
            run_cfg = OmegaConf.create(OmegaConf.to_container(cfg, resolve=True))
            assert isinstance(run_cfg, DictConfig)

            # Suggest hyperparameters from search space
            suggestions = suggest_hyperparameters(trial, search_space)
            run_cfg = update_config_with_suggestions(run_cfg, suggestions)

            # Flatten suggestions for logging
            flat_suggestions = {}

            def _flatten(d, prefix=""):
                for k, v in d.items():
                    key = f"{prefix}.{k}" if prefix else k
                    if isinstance(v, dict):
                        _flatten(v, key)
                    else:
                        flat_suggestions[key] = v

            _flatten(suggestions)

            # Log trial info
            log.info(f"\n{'='*60}")
            log.info(f"Trial {trial.number}:")
            for key, value in flat_suggestions.items():
                log.info(f"  {key}: {value}")
            log.info(f"{'='*60}\n")

            # Create trial-specific directory
            trial_dir = base_log_path / f"trial_{trial.number}"
            trial_dir.mkdir(parents=True, exist_ok=True)

            # Save trial hyperparameters
            hparams_file = trial_dir / "hyperparameters.yaml"
            with open(hparams_file, "w") as f:
                f.write(OmegaConf.to_yaml(run_cfg, resolve=True))
                f.write(f"trial_number: {trial.number}\n")
                f.write(f"trial_seed: {trial_seed}\n")

            # Instantiate datamodule with trial-specific seed
            log.info(f"Instantiating datamodule for trial {trial.number}")
            datamodule = hydra.utils.instantiate(run_cfg.data, seed=trial_seed)
            datamodule.prepare_data()
            datamodule.setup(stage="fit")

            # Instantiate model
            log.info(f"Instantiating model for trial {trial.number}")
            model = hydra.utils.instantiate(
                run_cfg.model,
                datamodule=datamodule
            )

            # Setup callbacks and finalize any factories (e.g., Optuna pruning)
            callbacks: List[Callback] = instantiate_callbacks(run_cfg.get("callbacks"))

            finalized_callbacks: List[Callback] = []
            for cb in callbacks:
                # Update ModelCheckpoint dirpath to use trial-specific directory
                if isinstance(cb, ModelCheckpoint) and cb.dirpath is None:
                    cb.dirpath = str(trial_dir / "checkpoints")
                    log.info(f"Set checkpoint directory to: {cb.dirpath}")
                # Finalize partial/factory callbacks that require the Optuna trial
                if callable(cb):
                    try:
                        cb = cb(trial)  # works for _partial_ callbacks expecting trial
                        log.info(f"Finalized instantiation of callback {cb}")
                    except TypeError:
                        pass
                finalized_callbacks.append(cb)
            callbacks = finalized_callbacks

            # Setup logger
            logger = CSVLogger(
                save_dir=str(trial_dir),
                name="",
                version="",
            )

            # Instantiate trainer
            log.info(f"Instantiating trainer for trial {trial.number}")
            trainer: Trainer = hydra.utils.instantiate(
                run_cfg.trainer,
                callbacks=callbacks,
                logger=logger
            )

            # Train the model
            log.info(f"Starting training for trial {trial.number}")
            trainer.fit(model=model, datamodule=datamodule)

            # Get metric value for optimization
            metric_value = get_metric_value(trainer.callback_metrics, optimized_metric)
            if metric_value is None and hasattr(model, "val_f1_macro_best"):
                try:
                    metric_value = model.val_f1_macro_best.compute()
                except Exception:
                    metric_value = None

            if metric_value is None:
                log.warning(
                    f"Metric '{optimized_metric}' not found in callback_metrics. "
                    f"Available metrics: {list(trainer.callback_metrics.keys())}"
                )
                metric_value = (
                    float("-inf") if direction == "maximize" else float("inf")
                )

            log.info(
                f"\nTrial {trial.number} completed: {optimized_metric} = {metric_value:.4f}\n"
            )

            return float(metric_value)

        except Exception as e:
            log.error(f"Trial {trial.number} failed with error: {e}")
            import traceback
            traceback.print_exc()
            # Return worst possible value based on direction
            return float("-inf") if direction == "maximize" else float("inf")

    # Create Optuna study
    study_name = cfg.get(
        "study_name", f"optuna_study_{experiment_name}"
    )

    # Create study with direction
    study = optuna.create_study(
        study_name=study_name,
        direction=direction,
        sampler=optuna.samplers.TPESampler(seed=seed),
    )

    log.info(f"Created Optuna study: {study_name}")

    # Run optimization
    study.optimize(objective, n_trials=n_trials)

    # Log best results
    log.info(f"{'='*60}")
    log.info("OPTIMIZATION COMPLETED")
    log.info(f"{'='*60}")
    log.info(f"Best trial: {study.best_trial.number}")
    log.info(f"Best {optimized_metric}: {study.best_value:.4f}")
    log.info(f"Best hyperparameters:")
    for key, value in study.best_params.items():
        log.info(f"  {key}: {value}")
    log.info(f"{'='*60}\n")

    # Save best hyperparameters
    best_hparams_file = base_log_path / "best_hyperparameters.yaml"
    with open(best_hparams_file, "w") as f:
        f.write(f"# Best trial: {study.best_trial.number}\n")
        f.write(f"# Best {optimized_metric}: {study.best_value:.4f}\n\n")
        for key, value in study.best_params.items():
            f.write(f"{key}: {value}\n")
    log.info(f"Saved best hyperparameters to: {best_hparams_file}")

    # Perform a full training run with the best hyperparameters
    log.info("Running final training + test with best hyperparameters")

    # Rebuild best config
    best_suggestions = suggest_hyperparameters(
        optuna.trial.FixedTrial(study.best_params), search_space
    )
    best_cfg = update_config_with_suggestions(cfg, best_suggestions)
    best_run_dir = base_log_path / "best_run"
    best_run_dir.mkdir(parents=True, exist_ok=True)

    summary_dir = best_run_dir / "summary"
    summary_dir.mkdir(parents=True, exist_ok=True)
    hparams_file = summary_dir / "hyperparameters.yaml"
    with open(hparams_file, "w") as f:
        f.write(OmegaConf.to_yaml(best_cfg, resolve=True))
        f.write(f"best_trial: {study.best_trial.number}\n")
    log.info(f"Saved best-run hyperparameters to: {hparams_file}")

    best_datamodule = hydra.utils.instantiate(best_cfg.data, seed=seed)
    best_datamodule.prepare_data()
    best_datamodule.setup(stage="fit")

    best_model = hydra.utils.instantiate(best_cfg.model, datamodule=best_datamodule)

    callbacks: List[Callback] = instantiate_callbacks(best_cfg.get("callbacks"))
    for cb in callbacks:
        if isinstance(cb, ModelCheckpoint) and cb.dirpath is None:
            cb.dirpath = str(best_run_dir / "checkpoints")
            log.info(f"Set checkpoint directory to: {cb.dirpath}")

    logger = CSVLogger(save_dir=str(best_run_dir), name="", version="")
    trainer: Trainer = hydra.utils.instantiate(
        best_cfg.trainer, callbacks=callbacks, logger=logger
    )

    log.info("Starting final training with best hyperparameters")
    trainer.fit(model=best_model, datamodule=best_datamodule)

    val_f1_macro_best = trainer.callback_metrics.get("val/f1_macro_best")
    if val_f1_macro_best is None and hasattr(best_model, "val_f1_macro_best"):
        try:
            val_f1_macro_best = best_model.val_f1_macro_best.compute()
        except Exception:
            val_f1_macro_best = None

    best_model_path = trainer.checkpoint_callback.best_model_path

    test_dir = best_run_dir / "final_test"
    test_dir.mkdir(parents=True, exist_ok=True)
    test_logger = CSVLogger(save_dir=str(test_dir), name="", version="")
    test_trainer: Trainer = hydra.utils.instantiate(
        best_cfg.trainer,
        callbacks=[],
        logger=test_logger,
        devices=1,
        strategy="auto",
    )

    if best_model_path:
        log.info(f"Loading best checkpoint: {best_model_path}")
        test_trainer.test(
            model=best_model,
            datamodule=best_datamodule,
            ckpt_path=best_model_path,
            weights_only=False,
        )
    else:
        log.warning("No best checkpoint found, testing with final model")
        test_trainer.test(
            model=best_model,
            datamodule=best_datamodule,
            weights_only=False,
        )

    class_names = get_class_names(best_datamodule)

    cm = test_trainer.callback_metrics
    
    test_metrics: Dict[str, Any] = {
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
        n_classes = int(getattr(best_datamodule, "num_classes"))
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

    metrics_df = pd.DataFrame([test_metrics])
    metrics_file = summary_dir / "metrics.csv"
    metrics_df.to_csv(metrics_file, index=False)
    log.info(f"Saved best-run metrics to: {metrics_file}")

    log.info("Final test results with best hyperparameters:")
    for key, value in test_metrics.items():
        if key.startswith("test/"):
            log.info(f"  {key}: {(value or 0.0):.4f}")

    return study.best_value


if __name__ == "__main__":
    main()
