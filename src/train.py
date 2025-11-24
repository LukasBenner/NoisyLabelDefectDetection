import os

from datetime import date
import hydra
import rootutils

from omegaconf import DictConfig
from sklearn.model_selection import train_test_split
import torch
import torchvision
from torchvision.transforms import v2
from data.components.transform_subset import TransformSubset
import pandas as pd
import numpy as np
from scipy import stats
from torchmetrics.classification import (
    MulticlassAccuracy,
    MulticlassF1Score,
    MulticlassRecall,
    MulticlassPrecision,
)
from torchmetrics import MeanMetric
from lightning.fabric.loggers.csv_logs import CSVLogger
from lightning.fabric import Fabric

from models.components.gce_loss import GceLoss
from utils.pylogger import RankedLogger

import tqdm

rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

log = RankedLogger(__name__, rank_zero_only=True)

train_transforms = v2.Compose(
    [
        v2.Resize((640, 480), antialias=True),
        v2.RandomHorizontalFlip(),
        v2.RandomVerticalFlip(),
        v2.ToImage(),
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)
test_transforms = v2.Compose(
    [
        v2.Resize((640, 480), antialias=True),
        v2.ToImage(),
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)


def setup_data_loaders(
    train_data_path: str,
    test_data_path: str,
    train_indices: list,
    val_indices: list,
    batch_size: int,
    num_workers: int = 4,
    fabric: Fabric | None = None,
):
    """Setup train, validation, and test data loaders."""
    train_image_folder_set = torchvision.datasets.ImageFolder(root=train_data_path)
    test_image_folder_set = torchvision.datasets.ImageFolder(root=test_data_path)

    train_dataset = TransformSubset(
        train_image_folder_set,
        indices=train_indices,
        transform=train_transforms,
    )
    val_dataset = TransformSubset(
        train_image_folder_set,
        indices=val_indices,
        transform=test_transforms,
    )
    test_dataset = TransformSubset(
        test_image_folder_set,
        indices=list(range(len(test_image_folder_set))),
        transform=test_transforms,
    )

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=False,
    )
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=False,
    )
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=False,
    )

    if fabric:
        train_loader, val_loader, test_loader = fabric.setup_dataloaders(
            train_loader, val_loader, test_loader
        )

    return train_loader, val_loader, test_loader, train_image_folder_set.classes


def calculate_class_weights(
    train_targets: list, num_classes: int, device: torch.device
):
    """Calculate class weights for handling class imbalance."""
    samples_per_class = np.bincount(train_targets, minlength=num_classes)
    inv_freq = 1.0 / np.maximum(samples_per_class, 1)
    class_weights = inv_freq / inv_freq.sum() * num_classes
    return (
        torch.tensor(class_weights, dtype=torch.float32).to(device),
        samples_per_class,
    )


def setup_model_and_optimizer(
    cfg: DictConfig,
    num_classes: int,
    class_weights: torch.Tensor,
    fabric: Fabric,
):
    """Setup model, optimizer, criterion, and optional scheduler."""
    model = hydra.utils.instantiate(cfg.model, num_classes=num_classes)
    
    if cfg.get("mae_loss"):
        criterion = torch.nn.L1Loss()
    else:
        criterion = torch.nn.CrossEntropyLoss(weight=class_weights)

    optimizer = hydra.utils.instantiate(cfg.get("optimizer"), model.parameters())

    scheduler_cfg = cfg.get("scheduler", None)
    if scheduler_cfg:
        scheduler = hydra.utils.instantiate(scheduler_cfg, optimizer)
    else:
        scheduler = None

    model, optimizer = fabric.setup(model, optimizer, move_to_device=True)
    return model, optimizer, criterion, scheduler


def setup_metrics(num_classes: int, device: torch.device):
    """Setup metrics for training, validation, or testing."""
    return {
        "loss": MeanMetric().to(device),
        "acc": MulticlassAccuracy(num_classes=num_classes, average="weighted").to(
            device
        ),
        "precision": MulticlassPrecision(
            num_classes=num_classes, average="weighted"
        ).to(device),
        "recall": MulticlassRecall(num_classes=num_classes, average="weighted").to(
            device
        ),
        "f1": MulticlassF1Score(num_classes=num_classes, average="weighted").to(device),
    }


def reset_metrics(metrics: dict):
    """Reset all metrics in the metrics dictionary."""
    for metric in metrics.values():
        metric.reset()


def compute_metrics(metrics: dict, prefix: str = "train"):
    """Compute all metrics and return as dictionary with prefix."""
    return {
        f"{prefix}/loss": metrics["loss"].compute().item(),
        f"{prefix}/acc": metrics["acc"].compute().item(),
        f"{prefix}/precision": metrics["precision"].compute().item(),
        f"{prefix}/recall": metrics["recall"].compute().item(),
        f"{prefix}/f1": metrics["f1"].compute().item(),
    }


def train_one_epoch(
    cfg: DictConfig,
    model,
    train_loader,
    optimizer,
    criterion,
    metrics: dict,
    fabric: Fabric,
    epoch: int,
    num_epochs: int,
    num_classes: int,
    run_idx: int,
):
    """Train model for one epoch."""
    model.train()
    reset_metrics(metrics)

    train_bar = tqdm.tqdm(
        train_loader,
        desc=f"Run {run_idx + 1}, Epoch {epoch + 1}/{num_epochs}",
        leave=False,
        ncols=100,
        mininterval=0.2,
    )
    
    use_mixup = cfg.mixup.get("enabled", False)
    mixup = None
    if use_mixup:
        alpha = cfg.mixup.get("alpha", 1.0)
        mixup = v2.MixUp(num_classes=num_classes, alpha=alpha)
    
    # Check if using L1Loss (needs one-hot targets)
    use_one_hot = not isinstance(criterion, torch.nn.CrossEntropyLoss)

    for batch in train_bar:
        inputs, targets = batch
        if mixup is not None:
            inputs, targets = mixup(inputs, targets)
            targets_for_metrics = targets.argmax(dim=1)
        else:
            targets_for_metrics = targets
            # Convert to one-hot for L1Loss
            if use_one_hot:
                targets = torch.nn.functional.one_hot(targets, num_classes=num_classes).float()

        optimizer.zero_grad(set_to_none=True)
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        fabric.backward(loss)
        optimizer.step()        
        with torch.no_grad():
            preds = torch.argmax(outputs, dim=1)
            metrics["loss"].update(loss)
            metrics["acc"].update(preds, targets_for_metrics)
            metrics["precision"].update(preds, targets_for_metrics)
            metrics["recall"].update(preds, targets_for_metrics)
            metrics["f1"].update(preds, targets_for_metrics)

        train_bar.set_postfix(
            {
                "train/loss": f"{metrics['loss'].compute().item():.4f}",
                "train/acc": f"{metrics['acc'].compute().item():.4f}",
            },
            refresh=False,
        )
    train_bar.close()

    return compute_metrics(metrics, "train")


def validate_one_epoch(
    model,
    val_loader,
    criterion,
    metrics: dict,
    epoch: int,
    run_idx: int,
    num_classes: int = None,
):
    """Validate model for one epoch."""
    model.eval()
    reset_metrics(metrics)
    
    # Check if using L1Loss (needs one-hot targets)
    use_one_hot = not isinstance(criterion, torch.nn.CrossEntropyLoss)

    with torch.no_grad():
        val_bar = tqdm.tqdm(
            val_loader,
            desc=f"Run {run_idx + 1}, Val Epoch {epoch + 1}",
            leave=False,
            ncols=100,
            mininterval=0.2,
        )
        for batch in val_bar:
            inputs, targets = batch
            targets_for_metrics = targets
            
            # Convert to one-hot for L1Loss
            if use_one_hot and num_classes is not None:
                targets = torch.nn.functional.one_hot(targets, num_classes=num_classes).float()
            
            outputs = model(inputs)
            loss = criterion(outputs, targets)

            preds = torch.argmax(outputs, dim=1)
            metrics["loss"].update(loss)
            metrics["acc"].update(preds, targets_for_metrics)
            metrics["precision"].update(preds, targets_for_metrics)
            metrics["recall"].update(preds, targets_for_metrics)
            metrics["f1"].update(preds, targets_for_metrics)

            val_bar.set_postfix(
                {
                    "val/loss": f"{metrics['loss'].compute().item():.4f}",
                    "val/acc": f"{metrics['acc'].compute().item():.4f}",
                },
                refresh=False,
            )
        val_bar.close()

    return compute_metrics(metrics, "val")


def test_model(
    model,
    test_loader,
    criterion,
    metrics: dict,
    device: torch.device,
    num_classes: int = None,
):
    """Test the model and return metrics."""
    model.eval()
    reset_metrics(metrics)
    
    # Check if using L1Loss (needs one-hot targets)
    use_one_hot = not isinstance(criterion, torch.nn.CrossEntropyLoss)

    with torch.no_grad():
        for batch in test_loader:
            inputs, targets = batch
            inputs = inputs.to(device)
            targets = targets.to(device)
            targets_for_metrics = targets
            
            # Convert to one-hot for L1Loss
            if use_one_hot and num_classes is not None:
                targets = torch.nn.functional.one_hot(targets, num_classes=num_classes).float()

            outputs = model(inputs)
            loss = criterion(outputs, targets)

            preds = torch.argmax(outputs, dim=1)
            metrics["loss"].update(loss)
            metrics["acc"].update(preds, targets_for_metrics)
            metrics["precision"].update(preds, targets_for_metrics)
            metrics["recall"].update(preds, targets_for_metrics)
            metrics["f1"].update(preds, targets_for_metrics)

    return compute_metrics(metrics, "test")


def calculate_summary_statistics(
    all_metrics: list,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Calculate summary statistics across multiple runs."""
    all_metrics_df = pd.DataFrame(all_metrics)
    summary_statistics = []
    metrics_to_analyze = [
        m for m in all_metrics_df.columns if m not in ["run_idx", "best_val/epoch"]
    ]

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


def save_hyperparameters(
    log_path: str,
    cfg: DictConfig,
    num_classes: int,
):
    """Save hyperparameters to a YAML file."""
    os.makedirs(f"{log_path}", exist_ok=True)
    with open(f"{log_path}/hyperparameters.yaml", "w") as f:
        from omegaconf import OmegaConf

        f.write(OmegaConf.to_yaml(cfg, resolve=True))
        f.write(f"\nnum_classes: {num_classes}\n")


@hydra.main(version_base="1.3", config_path="../configs", config_name="train.yaml")
def main(cfg: DictConfig) -> None:
    num_runs = cfg.get("num_runs", 1)
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
    log_path = f"{log_path}/train/{experiment_name}/{run_name}"

    # Load initial data to get class info
    train_data_path = cfg.data.train_data_path
    test_data_path = cfg.data.test_data_path
    train_image_folder_set = torchvision.datasets.ImageFolder(root=train_data_path)

    num_classes = len(train_image_folder_set.classes)
    print(f"Number of classes: {num_classes}")

    val_split = cfg.data.get("val_split", 0.2)

    save_hyperparameters(
        log_path + "/summary",
        cfg,
        num_classes,
    )

    all_metrics = []

    # Training loop over multiple runs
    for run_idx in range(num_runs):
        logger = CSVLogger(log_path, name=f"run_{run_idx + 1}", version="")

        # Split data
        all_indices = list(range(len(train_image_folder_set)))
        all_targets = [train_image_folder_set.samples[i][1] for i in all_indices]
        train_indices, val_indices = train_test_split(
            all_indices,
            test_size=val_split,
            random_state=seed + run_idx,
            stratify=all_targets,
        )

        # Setup data loaders
        train_loader, val_loader, test_loader, classes = setup_data_loaders(
            train_data_path=train_data_path,
            test_data_path=test_data_path,
            train_indices=train_indices,
            val_indices=val_indices,
            batch_size=cfg.data.get("batch_size"),
            num_workers=cfg.data.get("num_workers", 4),
            fabric=fabric,
        )

        # Calculate class weights
        train_targets = [all_targets[i] for i in train_indices]
        class_weights, samples_per_class = calculate_class_weights(
            train_targets, num_classes, device
        )

        # Setup model, optimizer, and criterion
        model, optimizer, criterion, scheduler = setup_model_and_optimizer(
            cfg, num_classes, class_weights, fabric
        )

        # Setup metrics
        train_metrics = setup_metrics(num_classes, device)
        val_metrics = setup_metrics(num_classes, device)
        test_metrics = setup_metrics(num_classes, device)

        # Training configuration
        num_epochs = cfg.get("num_epochs")
        patience = cfg.get("early_stopping_patience", 999)
        epochs_no_improvement = 0
        best_val_acc = 0.0
        best_val_f1 = 0.0
        best_epoch = -1

        # Training epochs
        for epoch in range(num_epochs):
            # Train one epoch
            train_result = train_one_epoch(
                cfg,
                model=model,
                train_loader=train_loader,
                optimizer=optimizer,
                criterion=criterion,
                metrics=train_metrics,
                fabric=fabric,
                epoch=epoch,
                num_epochs=num_epochs,
                num_classes=num_classes,
                run_idx=run_idx,
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
                run_idx=run_idx,
                num_classes=num_classes,
            )

            if scheduler:
                scheduler.step()

            # Early stopping check
            current_val_acc = val_result["val/acc"]
            current_val_f1 = val_result["val/f1"]
            improved = current_val_acc > best_val_acc

            if improved:
                best_val_acc = current_val_acc
                best_val_f1 = max(best_val_f1, current_val_f1)
                best_epoch = epoch + 1
                epochs_no_improvement = 0

                best_model_path = f"{log_path}/run_{run_idx + 1}/best_model.pth"
                torch.save(model.state_dict(), best_model_path)
            else:
                epochs_no_improvement += 1

            val_result["epoch"] = epoch + 1
            val_result["best_val/acc"] = best_val_acc
            val_result["best_val/f1"] = best_val_f1
            logger.log_metrics(val_result)
            logger.save()

            if epochs_no_improvement >= patience:
                break

        # Test loop
        ckpt_path = f"{log_path}/run_{run_idx + 1}/best_model.pth"
        model.load_state_dict(torch.load(ckpt_path, map_location=device))

        test_result = test_model(
            model=model,
            test_loader=test_loader,
            criterion=criterion,
            metrics=test_metrics,
            device=device,
            num_classes=num_classes,
        )

        final_test_metrics = {
            "run_idx": run_idx + 1,
            **test_result,
            "best_val/acc": best_val_acc,
            "best_val/f1": best_val_f1,
            "best_val/epoch": best_epoch,
        }
        logger.log_metrics(final_test_metrics)
        all_metrics.append(final_test_metrics)

        logger.finalize("success")
        logger.save()
        print(
            f"Run {run_idx + 1}/{num_runs} | test/acc: {final_test_metrics['test/acc']:.4f} test/f1: {final_test_metrics['test/f1']:.4f}"
        )

    # Calculate and save summary statistics
    all_metrics_df, summary_df = calculate_summary_statistics(all_metrics)
    os.makedirs(f"{log_path}/summary", exist_ok=True)
    all_metrics_df.to_csv(f"{log_path}/summary/summary_metrics.csv", index=False)
    summary_df.to_csv(f"{log_path}/summary/summary_statistics.csv", index=False)


if __name__ == "__main__":
    main()
