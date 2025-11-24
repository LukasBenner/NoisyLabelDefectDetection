"""Hyperparameter sweep using Optuna with MLflow parent-child run hierarchy."""

from datetime import date
from typing import Any, Dict, Optional

import hydra
import rootutils
from lightning import Fabric
from omegaconf import DictConfig, OmegaConf, ListConfig
import optuna
from sklearn.model_selection import train_test_split
import torch
import torchvision

from train import (
    calculate_class_weights,
    save_hyperparameters,
    setup_data_loaders,
    setup_metrics,
    setup_model_and_optimizer,
    train_one_epoch,
    validate_one_epoch,
)
from utils.pylogger import RankedLogger
from utils.utils import get_metric_value
from lightning.fabric.loggers.csv_logs import CSVLogger

rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

log = RankedLogger(__name__, rank_zero_only=True)


def suggest_hyperparameters(
    trial: optuna.Trial, search_space: DictConfig
) -> Dict[str, Any]:
    suggested = {}

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
    cfg: DictConfig | ListConfig, suggestions: Dict[str, Any]
) -> DictConfig | ListConfig:
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
    _update_recursive(cfg_copy, suggestions)
    return cfg_copy


@hydra.main(version_base="1.3", config_path="../configs", config_name="sweep.yaml")
def main(cfg: DictConfig) -> Optional[float]:
    n_trials = cfg.get("n_trials", 10)
    optimized_metric = cfg.get("optimized_metric", "best_val/acc")
    direction = cfg.get("direction", "maximize")
    seed = cfg.get("seed", 42)

    device_id = cfg.trainer.device_id
    fabric = Fabric(accelerator="gpu", precision="16-mixed", devices=[device_id])
    device = fabric.device
    fabric.seed_everything(seed)
    torch.set_float32_matmul_precision("high")

    # Setup logging paths
    log_path = cfg.get("log_path", "logs")
    experiment_name = cfg.get("experiment_name")
    run_name = cfg.get("run_name", date.today().strftime("%Y-%m-%d"))
    log_path = f"{log_path}/sweep/{experiment_name}/{run_name}"

    # Load initial data to get class info
    train_data_path = cfg.data.train_data_path
    test_data_path = cfg.data.test_data_path
    train_image_folder_set = torchvision.datasets.ImageFolder(root=train_data_path)

    num_classes = len(train_image_folder_set.classes)
    print(f"Number of classes: {num_classes}")

    val_split = cfg.data.get("val_split", 0.2)

    # Get search space from config
    if "search_space" not in cfg:
        raise ValueError(
            "No search_space defined in config. Please add search_space to your hparams_search config."
        )

    search_space = cfg.search_space

    # Define objective function for Optuna
    def objective(trial: optuna.Trial) -> float:
        """Objective function for a single trial."""
        try:
            # Create a copy of config for this trial
            run_cfg = OmegaConf.create(OmegaConf.to_container(cfg, resolve=True))
            assert type(run_cfg) is DictConfig

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
                        flat_suggestions[k] = v

            _flatten(suggestions)

            # Log trial info
            log.info(f"\n{'='*60}")
            log.info(f"Trial {trial.number}:")
            for key, value in flat_suggestions.items():
                log.info(f"  {key}: {value}")
            log.info(f"{'='*60}\n")

            logger = CSVLogger(log_path, name=f"trial_{trial.number + 1}", version="")
            trial_log_path = log_path + f"/trial_{trial.number + 1}"

            # Split data
            all_indices = list(range(len(train_image_folder_set)))
            all_targets = [train_image_folder_set.samples[i][1] for i in all_indices]
            train_indices, val_indices = train_test_split(
                all_indices,
                test_size=val_split,
                random_state=seed + trial.number,
                stratify=all_targets,
            )

            # Setup data loaders
            train_loader, val_loader, test_loader, classes = setup_data_loaders(
                train_data_path=train_data_path,
                test_data_path=test_data_path,
                train_indices=train_indices,
                val_indices=val_indices,
                batch_size=run_cfg.data.get("batch_size"),
                num_workers=run_cfg.data.get("num_workers", 4),
                fabric=fabric,
            )

            train_targets = [all_targets[i] for i in train_indices]
            class_weights, samples_per_class = calculate_class_weights(
                train_targets, num_classes, device
            )

            # Setup model, optimizer, and criterion
            model, optimizer, criterion, scheduler = setup_model_and_optimizer(
                cfg, num_classes, class_weights, fabric
            )

            train_metrics = setup_metrics(num_classes, device)
            val_metrics = setup_metrics(num_classes, device)

            num_epochs = run_cfg.get("num_epochs")
            patience = run_cfg.get("early_stopping_patience", 999)
            epochs_no_improvement = 0
            best_val_acc = 0.0
            best_val_f1 = 0.0
            best_epoch = -1

            # Save hyperparameters once

            save_hyperparameters(
                trial_log_path, run_cfg, num_classes, samples_per_class, class_weights
            )

            for epoch in range(num_epochs):
                train_result = train_one_epoch(
                    run_cfg,
                    model=model,
                    train_loader=train_loader,
                    optimizer=optimizer,
                    criterion=criterion,
                    metrics=train_metrics,
                    fabric=fabric,
                    epoch=epoch,
                    num_epochs=num_epochs,
                    num_classes=num_classes,
                    run_idx=trial.number,
                )
                train_result["epoch"] = epoch + 1
                train_result["lr"] = optimizer.param_groups[0]["lr"]
                logger.log_metrics(train_result)

                # Validate one epoch
                val_result = validate_one_epoch(
                    model=model,
                    val_loader=val_loader,
                    criterion=criterion,
                    metrics=val_metrics,
                    epoch=epoch,
                    run_idx=trial.number,
                )

                if scheduler:
                    scheduler.step()

                # Early stopping check
                current_val_acc = val_result["val/acc"]
                current_val_f1 = val_result["val/f1"]

                best_val_f1 = max(best_val_f1, current_val_f1)
                if current_val_acc > best_val_acc:
                    best_val_acc = current_val_acc
                    epochs_no_improvement = 0
                else:
                    epochs_no_improvement += 1

                val_result["epoch"] = epoch + 1
                val_result["best_val/acc"] = best_val_acc
                val_result["best_val/f1"] = best_val_f1
                logger.log_metrics(val_result)
                logger.save()

                if epochs_no_improvement >= patience:
                    break

            metric_value = get_metric_value(val_result, optimized_metric)

            if metric_value is None:
                log.warning(
                    f"Metric '{optimized_metric}' not found in results. Available metrics: {list(val_result.keys())}"
                )
                metric_value = (
                    float("-inf") if direction == "maximize" else float("inf")
                )

            log.info(
                f"\nTrial {trial.number} completed: {optimized_metric} = {metric_value:.4f}\n"
            )

            return metric_value

        except Exception as e:
            log.error(f"Trial {trial.number} failed with error: {e}")
            import traceback

            traceback.print_exc()
            # Return worst possible value based on direction
            return float("-inf") if direction == "maximize" else float("inf")

    # Create Optuna study

    study_name = cfg.get(
        "study_name", f"optuna_study_{cfg.get('experiment_name', 'default')}"
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
    log.info(f"\n{'='*80}")
    log.info("OPTIMIZATION COMPLETED")
    log.info(f"{'='*80}")
    log.info(f"Best trial: {study.best_trial.number}")
    log.info(f"Best {optimized_metric}: {study.best_value:.4f}")
    log.info(f"Best hyperparameters:")
    for key, value in study.best_params.items():
        log.info(f"  {key}: {value}")
    log.info(f"{'='*80}\n")

    return study.best_value


if __name__ == "__main__":
    main()
