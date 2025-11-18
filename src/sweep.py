"""Hyperparameter sweep using Optuna with MLflow parent-child run hierarchy."""

from typing import Any, Dict, List, Optional

import hydra
import rootutils
from lightning import Callback, LightningDataModule, LightningModule, Trainer, seed_everything
from lightning.pytorch.loggers import Logger
from omegaconf import DictConfig, OmegaConf
import optuna
import torch

from utils.instantiators import instantiate_callbacks, instantiate_loggers
from utils.logging_utils import log_hyperparameters, log_training_results
from utils.pylogger import RankedLogger
from utils.utils import get_metric_value

rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

log = RankedLogger(__name__, rank_zero_only=True)

def suggest_hyperparameters(trial: optuna.Trial, search_space: DictConfig) -> Dict[str, Any]:
    """Suggest hyperparameters based on search space configuration.
    
    Args:
        trial: Optuna trial object
        search_space: DictConfig with search space definitions
        
    Returns:
        Dictionary of suggested hyperparameters with nested keys
    """
    suggested = {}
    
    def _suggest_recursive(space: DictConfig, prefix: str = ""):
        """Recursively suggest hyperparameters."""
        result = {}
        
        for key, value in space.items():
            full_key = f"{prefix}.{key}" if prefix else key
            
            if isinstance(value, DictConfig):
                if "type" in value:
                    # This is a hyperparameter definition
                    param_type = value.type
                    
                    if param_type == "float":
                        result[key] = trial.suggest_float(
                            full_key,
                            value.low,
                            value.high,
                            log=value.get("log", False)
                        )
                    elif param_type == "int":
                        result[key] = trial.suggest_int(
                            full_key,
                            value.low,
                            value.high,
                            log=value.get("log", False)
                        )
                    elif param_type == "categorical":
                        result[key] = trial.suggest_categorical(
                            full_key,
                            value.choices
                        ) # type: ignore
                    elif param_type == "uniform":
                        result[key] = trial.suggest_uniform(
                            full_key,
                            value.low,
                            value.high
                        )
                    else:
                        log.warning(f"Unknown parameter type: {param_type}")
                else:
                    # Nested configuration
                    result[key] = _suggest_recursive(value, full_key)
        
        return result
    
    return _suggest_recursive(search_space)


def update_config_with_suggestions(cfg: DictConfig, suggestions: Dict[str, Any]) -> DictConfig:
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
    """Run Optuna hyperparameter sweep with MLflow tracking."""

    n_trials = cfg.get("n_trials", 20)
    optimized_metric = cfg.get("optimized_metric", "val/f1_best")
    direction = cfg.get("direction", "maximize")
    seed = cfg.get("seed", 42)

    torch.set_float32_matmul_precision('high')
    seed_everything(seed)
    
    # Save the original output dir
    original_output_dir = cfg.paths.output_dir
    
     # Get search space from config
    if "search_space" not in cfg:
        raise ValueError("No search_space defined in config. Please add search_space to your hparams_search config.")
    
    search_space = cfg.search_space
    
    log.info(f"\n{'='*80}")
    log.info(f"Starting Optuna Sweep")
    log.info(f"{'='*80}")
    log.info(f"  Number of trials: {n_trials}")
    log.info(f"  Optimized metric: {optimized_metric}")
    log.info(f"  Direction: {direction}")
    log.info(f"  Seed: {seed}")
    log.info(f"  Output directory: {original_output_dir}")
    log.info(f"{'='*80}\n")
    
    # Setup MLflow parent run if configured
    parent_run = None
    parent_run_id = None
    
    if "logger" in cfg and "mlflow" in cfg.logger:
        try:
            import mlflow
            if "tracking_uri" in cfg.logger.mlflow:
                mlflow.set_tracking_uri(cfg.logger.mlflow.tracking_uri)
            experiment_name = cfg.get("experiment_name", "optuna_sweep")
            mlflow.set_experiment(experiment_name)
            
            parent_run = mlflow.start_run(run_name=f"optuna_sweep_{cfg.get('run_name', 'default')}")
            
            if "tags" in cfg and cfg.tags:
                mlflow.set_tags(OmegaConf.to_container(cfg.tags, resolve=True))
            
            parent_run_id = parent_run.info.run_id
            
            # Log sweep configuration
            mlflow.log_params({
                "n_trials": n_trials,
                "optimized_metric": optimized_metric,
                "direction": direction,
                "base_seed": seed,
            })
            
            log.info(f"Created MLflow parent run: {parent_run_id}")
        except Exception as e:
            log.warning(f"Could not create MLflow parent run: {e}")
            parent_run = None
            parent_run_id = None
    
    # Define objective function for Optuna
    def objective(trial: optuna.Trial) -> float:
        """Objective function for a single trial."""
        try:
            # Create a copy of config for this trial
            run_cfg = OmegaConf.create(OmegaConf.to_container(cfg, resolve=True))
            
            # Set unique output directory for this trial
            run_cfg.paths.output_dir = f"{original_output_dir}/trial_{trial.number}"
            
            # Suggest hyperparameters from search space
            suggestions = suggest_hyperparameters(trial, search_space)
            
            # Update config with suggestions
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
            
            # Update run name
            if "run_name" in run_cfg:
                run_cfg.run_name = f"{run_cfg.run_name}_trial_{trial.number}"

            if "trainer" in run_cfg:
                run_cfg.trainer.default_root_dir = run_cfg.paths.output_dir

            if "csv" in run_cfg.logger:
                run_cfg.logger.csv.save_dir = f"{run_cfg.paths.output_dir}"
            
            # If using MLflow, tag this run as child of parent
            if parent_run_id and "logger" in run_cfg and "mlflow" in run_cfg.logger:
                if "tags" not in run_cfg.logger.mlflow:
                    run_cfg.logger.mlflow.tags = {}
                run_cfg.logger.mlflow.tags["mlflow.parentRunId"] = str(parent_run_id)
                run_cfg.logger.mlflow.tags["trial_number"] = str(trial.number)
            
            # Instantiate components
            log.info(f"Instantiating datamodule <{run_cfg.data._target_}>")
            datamodule: LightningDataModule = hydra.utils.instantiate(run_cfg.data)
            
            log.info(f"Instantiating model <{run_cfg.model._target_}>")
            model: LightningModule = hydra.utils.instantiate(run_cfg.model)
            
            log.info("Instantiating callbacks...")
            callbacks: List[Callback] = instantiate_callbacks(run_cfg.get("callbacks"))
            
            log.info("Instantiating loggers...")
            loggers: List[Logger] = instantiate_loggers(run_cfg.get("logger"))
            
            log.info(f"Instantiating trainer <{run_cfg.trainer._target_}>")
            trainer: Trainer = hydra.utils.instantiate(run_cfg.trainer, callbacks=callbacks, logger=loggers)
            
            object_dict = {
                "cfg": run_cfg,
                "datamodule": datamodule,
                "model": model,
                "callbacks": callbacks,
                "logger": loggers,
                "trainer": trainer,
            }
            
            if loggers:
                log.info("Logging hyperparameters!")
                log_hyperparameters(object_dict)
            
            log.info("Starting training!")
            trainer.fit(model=model, datamodule=datamodule)
            
            # Collect final metrics
            metric_dict = {}
            if loggers:
                log.info("Logging training results!")
                metric_dict = log_training_results(trainer, model)
            
            # Get the optimized metric value
            metric_value = get_metric_value(metric_dict, optimized_metric)
            
            if metric_value is None:
                log.warning(f"Metric '{optimized_metric}' not found in results. Available metrics: {list(metric_dict.keys())}")
                # Return a default value based on direction
                metric_value = float('-inf') if direction == "maximize" else float('inf')
            
            log.info(f"\nTrial {trial.number} completed: {optimized_metric} = {metric_value:.4f}\n")
            
            return metric_value
            
        except Exception as e:
            log.error(f"Trial {trial.number} failed with error: {e}")
            import traceback
            traceback.print_exc()
            # Return worst possible value based on direction
            return float('-inf') if direction == "maximize" else float('inf')
    
    # Create Optuna study
    try:
        study_name = cfg.get("study_name", f"optuna_study_{cfg.get('experiment_name', 'default')}")
        
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
        
        # Log to MLflow parent run
        if parent_run:
            try:
                import mlflow
                mlflow.log_params({f"best_{k}": v for k, v in study.best_params.items()})
                mlflow.log_metric("best_value", study.best_value)
                mlflow.log_metric("best_trial", study.best_trial.number)
                
                # Log optimization history
                for trial in study.trials:
                    if trial.state == optuna.trial.TrialState.COMPLETE:
                        mlflow.log_metric(f"trial_{trial.number}_value", trial.value, step=trial.number)
                
                log.info(f"Logged best results to MLflow parent run: {parent_run_id}")
            except Exception as e:
                log.warning(f"Could not log to MLflow: {e}")
        
        return study.best_value
        
    finally:
        # Close MLflow parent run
        if parent_run:
            try:
                import mlflow
                mlflow.end_run()
                log.info("Closed MLflow parent run")
            except Exception as e:
                log.warning(f"Could not close MLflow parent run: {e}")


if __name__ == "__main__":
    main()