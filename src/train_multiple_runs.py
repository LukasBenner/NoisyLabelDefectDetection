from pathlib import Path

import hydra
from omegaconf import DictConfig, OmegaConf
from lightning import seed_everything

from train import train
from utils.pylogger import RankedLogger

log = RankedLogger(__name__, rank_zero_only=True)


@hydra.main(version_base="1.2", config_path="../configs", config_name="train.yaml")
def main(cfg: DictConfig) -> None:
    """Run multiple training rounds with different seeds.
    
    Args:
        cfg: Hydra configuration.
    """
    # Get number of runs and base seed from config
    
    if not cfg.get("num_runs"):
        raise Exception("num_runs not specified")
    
    num_runs = cfg.get("num_runs")
    base_seed = cfg.get("seed")
    
    log.info(f"Starting {num_runs} training runs with base seed {base_seed}")
    
    if "logger" in cfg and "mlflow" in cfg.logger:
        try:
            import mlflow
            if "tracking_uri" in cfg.logger.mlflow:
                mlflow.set_tracking_uri(cfg.logger.mlflow.tracking_uri)
            experiment_name = cfg.get("experiment_name")
            mlflow.set_experiment(experiment_name)
            
            parent_run = mlflow.start_run(run_name=f"multi_run_{cfg.run_name}_{num_runs}_runs")
            mlflow.set_tags(cfg.tags)

            parent_run_id = parent_run.info.run_id
            
            # Log multi-run config
            mlflow.log_params({
                "num_runs": num_runs,
                "base_seed": base_seed
            })
            
            log.info(f"Created MLflow parent run: {parent_run_id}")
        except Exception as e:
            log.warning(f"Could not create MLflow parent run: {e}")
            parent_run = None
            parent_run_id = None
    else:
        parent_run = None
        parent_run_id = None
    
    # Store results for each run
    all_results = []
    
    # Save the original output dir to restore it for statistics
    original_output_dir = cfg.paths.output_dir
    
    log.info(f"\n{'='*80}")
    log.info(f"Multi-run directory structure:")
    log.info(f"  Base directory: {original_output_dir}")
    log.info(f"  Individual runs will be saved to: {original_output_dir}/run_<idx>_seed<seed>/")
    log.info(f"  Summary results will be saved to: {original_output_dir}/")
    log.info(f"{'='*80}\n")
    
    try:
        for run_idx in range(num_runs):
            current_seed = base_seed + run_idx
            
            log.info(f"\n{'='*80}")
            log.info(f"Starting run {run_idx + 1}/{num_runs} with seed {current_seed}")
            log.info(f"{'='*80}\n")
            
            # Create a copy of config and update seed
            run_cfg = OmegaConf.create(OmegaConf.to_container(cfg, resolve=True))
            run_cfg.seed = current_seed
            
            # Create unique output directory for this run
            # This prevents overwriting logs, checkpoints, and hydra outputs
            run_cfg.paths.output_dir = f"{original_output_dir}/run_{run_idx}_seed{current_seed}"
            
            # Update run name to include seed
            if "run_name" in run_cfg:
                run_cfg.run_name = f"{run_cfg.run_name}_seed{current_seed}"
            
            # Update checkpoint dirpath to use new output dir
            if "callbacks" in run_cfg and "model_checkpoint" in run_cfg.callbacks:
                run_cfg.callbacks.model_checkpoint.dirpath = f"{run_cfg.paths.output_dir}/checkpoints"
            
            # Update trainer's default_root_dir
            if "trainer" in run_cfg:
                run_cfg.trainer.default_root_dir = run_cfg.paths.output_dir

            if "csv" in run_cfg.logger:
                run_cfg.logger.csv.save_dir = f"{run_cfg.paths.output_dir}"
            
            # If using MLflow, set the parent run
            if parent_run_id and "logger" in run_cfg and "mlflow" in run_cfg.logger:
                run_cfg.logger.mlflow.tags = run_cfg.logger.mlflow.get("tags", {})
                run_cfg.logger.mlflow.tags["mlflow.parentRunId"] = str(parent_run_id)
                run_cfg.logger.mlflow.tags["run_index"] = str(run_idx)
                run_cfg.logger.mlflow.tags["seed"] = str(current_seed)
            
            # Set seed for reproducibility
            seed_everything(current_seed, workers=True)
            
            # Run training
            try:
                metric_dict, _ = train(run_cfg)
                
                # Store results
                result = {
                    "run_idx": run_idx,
                    "seed": current_seed,
                    **metric_dict
                }
                all_results.append(result)
                
                log.info(f"\nRun {run_idx + 1} completed successfully")
                if "val/f1_best" in metric_dict:
                    log.info(f"Best val/f1: {metric_dict['val/f1_best']:.4f}")
                if "val/acc_best" in metric_dict:
                    log.info(f"Best val/acc: {metric_dict['val/acc_best']:.4f}")
                
            except Exception as e:
                log.error(f"Run {run_idx + 1} failed with error: {e}")
                continue
        
        # Compute and log statistics
        log.info(f"\n{'='*80}")
        log.info("All runs completed!")
        log.info(f"{'='*80}\n")
        
        if all_results:
            import pandas as pd
            import numpy as np
            from scipy import stats
            
            df = pd.DataFrame(all_results)
            
            # Save detailed results in the original output directory
            # (not in a subdirectory to avoid nesting)
            output_dir = Path(original_output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)
            
            experiment_name = cfg.get("experiment_name", "experiment")
            results_file = output_dir / f"results_{experiment_name}_{num_runs}runs.csv"
            df.to_csv(results_file, index=False)
            log.info(f"Saved detailed results to: {results_file}")
            
            # Compute statistics for key metrics
            metrics_to_analyze = [col for col in df.columns if col not in ["run_idx", "seed"]]
            
            stats_results = []
            mlflow_summary = {}
            
            log.info("\n" + "="*80)
            log.info("STATISTICAL SUMMARY (95% Confidence Intervals)")
            log.info("="*80 + "\n")
            
            for metric in metrics_to_analyze:
                if metric in df.columns:
                    values = df[metric].dropna()
                    
                    if len(values) > 0:
                        mean = values.mean()
                        std = values.std()
                        
                        # Compute 95% confidence interval using t-distribution
                        confidence = 0.95
                        dof = len(values) - 1
                        t_critical = stats.t.ppf((1 + confidence) / 2, dof) if dof > 0 else 0
                        margin_error = t_critical * (std / np.sqrt(len(values))) if len(values) > 1 else 0
                        ci_lower = mean - margin_error
                        ci_upper = mean + margin_error
                        
                        stats_results.append({
                            "metric": metric,
                            "mean": mean,
                            "std": std,
                            "ci_lower": ci_lower,
                            "ci_upper": ci_upper,
                            "min": values.min(),
                            "max": values.max(),
                            "n_runs": len(values)
                        })
                        
                        # Prepare for MLflow logging
                        metric_name = metric.replace("/", "_")
                        mlflow_summary[f"summary/{metric_name}_mean"] = mean
                        mlflow_summary[f"summary/{metric_name}_std"] = std
                        mlflow_summary[f"summary/{metric_name}_ci_lower"] = ci_lower
                        mlflow_summary[f"summary/{metric_name}_ci_upper"] = ci_upper
                        mlflow_summary[f"summary/{metric_name}_min"] = values.min()
                        mlflow_summary[f"summary/{metric_name}_max"] = values.max()
                        
                        log.info(f"{metric}:")
                        log.info(f"  Mean ± Std:  {mean:.4f} ± {std:.4f}")
                        log.info(f"  95% CI:      [{ci_lower:.4f}, {ci_upper:.4f}]")
                        log.info(f"  Range:       [{values.min():.4f}, {values.max():.4f}]")
                        log.info(f"  N runs:      {len(values)}")
                        log.info("")
            
            # Save statistics summary
            stats_df = pd.DataFrame(stats_results)
            stats_file = output_dir / f"statistics_{experiment_name}_{num_runs}runs.csv"
            stats_df.to_csv(stats_file, index=False)
            log.info(f"Saved statistics summary to: {stats_file}")
            
            # Log to MLflow parent run
            if parent_run:
                try:
                    # Log summary metrics
                    mlflow.log_metrics(mlflow_summary)
                    
                    # Log result files as artifacts
                    mlflow.log_artifact(str(results_file))
                    mlflow.log_artifact(str(stats_file))
                    
                    log.info(f"Logged summary statistics to MLflow parent run: {parent_run_id}")
                except Exception as e:
                    log.warning(f"Could not log to MLflow: {e}")
            
            log.info("\n" + "="*80)
            log.info("Multi-run experiment completed!")
            log.info("="*80)
        else:
            log.error("No successful runs completed!")
    
    finally:
        # End MLflow parent run
        if parent_run:
            try:
                mlflow.end_run()
                log.info("Closed MLflow parent run")
            except Exception as e:
                log.warning(f"Could not close MLflow parent run: {e}")


if __name__ == "__main__":
    main()