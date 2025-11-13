from typing import Any, Dict

from lightning_utilities.core.rank_zero import rank_zero_only
from omegaconf import OmegaConf
from rich.console import Console
from rich.table import Table

from utils import pylogger

log = pylogger.RankedLogger(__name__, rank_zero_only=True)


@rank_zero_only
def log_hyperparameters(object_dict: Dict[str, Any]) -> None:
    hparams = {}

    cfg = OmegaConf.to_container(object_dict["cfg"], resolve=True)
    model = object_dict["model"]
    trainer = object_dict["trainer"]

    if not trainer.logger:
        log.warning("Logger not found! Skipping hyperparameter logging...")
        return

    hparams["model"] = cfg["model"]

    # save number of model parameters
    hparams["model/params/total"] = sum(p.numel() for p in model.parameters())
    hparams["model/params/trainable"] = sum(
        p.numel() for p in model.parameters() if p.requires_grad
    )
    hparams["model/params/non_trainable"] = sum(
        p.numel() for p in model.parameters() if not p.requires_grad
    )

    hparams["data"] = cfg["data"]
    hparams["trainer"] = cfg["trainer"]

    hparams["callbacks"] = cfg.get("callbacks")
    hparams["extras"] = cfg.get("extras")

    hparams["task_name"] = cfg.get("task_name")
    hparams["tags"] = cfg.get("tags")
    hparams["ckpt_path"] = cfg.get("ckpt_path")
    hparams["seed"] = cfg.get("seed")

    # send hparams to all loggers
    for logger in trainer.loggers:
        logger.log_hyperparams(hparams)


@rank_zero_only
def log_training_results(trainer, model) -> Dict[str, float]:
    """Log training results in a nicely formatted table and return metrics dict.
    
    Args:
        trainer: Lightning Trainer instance
        model: Lightning Module instance
    
    Returns:
        Dictionary containing final metrics
    """
    if not trainer.logger:
        log.warning("Logger not found! Skipping results logging...")
        return {}
    
    # Collect metrics from the model
    metrics = {}
    
    # Best validation metrics
    if hasattr(model, 'val_acc_best'):
        metrics['val/acc_best'] = model.val_acc_best.compute().item()
    if hasattr(model, 'val_f1_best'):
        metrics['val/f1_best'] = model.val_f1_best.compute().item()

    # Test metrics
    if hasattr(model, 'test_loss'):
        metrics['test/loss'] = model.test_loss.compute().item()
    if hasattr(model, 'test_acc'):
        metrics['test/acc'] = model.test_acc.compute().item()
    if hasattr(model, 'test_f1'):
        metrics['test/f1'] = model.test_f1.compute().item()
    if hasattr(model, 'test_precision'):
        metrics['test/precision'] = model.test_precision.compute().item()
    if hasattr(model, 'test_recall'):
        metrics['test/recall'] = model.test_recall.compute().item()
    
    # Create a Rich table for pretty console output
    table = Table(title="Training Results", show_header=True, header_style="bold magenta")
    table.add_column("Metric", style="cyan", no_wrap=True)
    table.add_column("Value", style="green", justify="right")
    
    for metric_name, metric_value in sorted(metrics.items()):
        table.add_row(metric_name, f"{metric_value:.4f}")
    
    # Print table to console
    console = Console()
    console.print("\n")
    console.print(table)
    console.print("\n")
    
    # Log metrics to all loggers
    for logger in trainer.loggers:
        logger.log_metrics({"final/" + k: v for k, v in metrics.items()})
    
    log.info(f"Final results logged! Best Val F1: {metrics.get('val/f1_best', 0):.4f}, Best Val Acc: {metrics.get('val/acc_best', 0):.4f}")
    
    return metrics
