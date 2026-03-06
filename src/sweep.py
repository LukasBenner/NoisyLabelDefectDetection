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
    import torch
    torch.set_float32_matmul_precision("medium")

    n_trials = cfg.get("n_trials", 10)
    optimized_metric = cfg.get("optimized_metric", "val/f1_macro_best")
    direction = cfg.get("direction", "maximize")
    seed = cfg.get("seed", 42)
    timeout_hours = cfg.get("timeout_hours", None)
    timeout_seconds = timeout_hours * 3600 if timeout_hours is not None else None

    # Pruner config: n_startup_trials before pruning begins, n_warmup_steps per trial
    pruner_n_startup_trials = cfg.get("pruner_n_startup_trials", 5)
    pruner_n_warmup_steps = cfg.get("pruner_n_warmup_steps", 10)

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
            trainer_kwargs = dict(callbacks=callbacks, logger=logger)

            # SEAL requires max_epochs = epochs_per_iteration * num_iterations
            if "epochs_per_iteration" in run_cfg.model and "num_iterations" in run_cfg.model:
                seal_max_epochs = int(run_cfg.model.epochs_per_iteration) * int(run_cfg.model.num_iterations)
                trainer_kwargs["max_epochs"] = seal_max_epochs
                log.info(f"SEAL mode: max_epochs = {run_cfg.model.epochs_per_iteration} * {run_cfg.model.num_iterations} = {seal_max_epochs}")

            trainer: Trainer = hydra.utils.instantiate(
                run_cfg.trainer,
                **trainer_kwargs
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

    study = optuna.create_study(
        study_name=study_name,
        direction=direction,
        sampler=optuna.samplers.TPESampler(
            seed=seed,
            multivariate=True,
            n_startup_trials=max(10, n_trials // 5),
        ),
        pruner=optuna.pruners.MedianPruner(
            n_startup_trials=pruner_n_startup_trials,
            n_warmup_steps=pruner_n_warmup_steps,
        )
    )

    log.info(f"Created/loaded Optuna study: {study_name}")

    # Warm-start: enqueue any explicitly specified trials from config
    if "enqueue_trials" in cfg:
        for i, params in enumerate(cfg.enqueue_trials):
            study.enqueue_trial(OmegaConf.to_container(params, resolve=True))
            log.info(f"Enqueued warm-start trial {i}: {dict(params)}")

    # Run optimization
    study.optimize(objective, n_trials=n_trials, timeout=timeout_seconds)

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

    # Save all trial results as CSV
    trials_df = study.trials_dataframe()
    trials_csv = base_log_path / "all_trials.csv"
    trials_df.to_csv(trials_csv, index=False)
    log.info(f"Saved all trial results to: {trials_csv}")

    # Save Optuna visualizations (requires plotly)
    try:
        import optuna.visualization as vis
        plots = {
            "optimization_history.html": vis.plot_optimization_history(study),
            "param_importances.html": vis.plot_param_importances(study),
            "parallel_coordinate.html": vis.plot_parallel_coordinate(study),
            "contour.html": vis.plot_contour(study),
        }
        for filename, fig in plots.items():
            fig.write_html(str(base_log_path / filename))
            log.info(f"Saved plot: {filename}")
    except Exception as e:
        log.warning(f"Could not save Optuna visualizations (install plotly): {e}")

    return study.best_value


if __name__ == "__main__":
    main()
