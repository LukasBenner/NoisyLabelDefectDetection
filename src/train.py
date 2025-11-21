import os
from typing import Any, Dict

from datetime import date
import hydra
import rootutils

from omegaconf import DictConfig
from sklearn.model_selection import train_test_split
import torch
import torchvision
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
from torchmetrics import MeanMetric, MaxMetric
from lightning.fabric.loggers import CSVLogger

import tqdm


from torchvision.transforms import v2

rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)


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


def train(cfg: DictConfig) -> Dict[str, Any]:
    # Logger
    log_path = cfg.get("log_path", "logs")
    experiment_name = cfg.get("experiment_name")
    run_name = cfg.get("run_name", date.today().strftime("%Y-%m-%d"))

    log_path = f"{log_path}/train/{experiment_name}/{run_name}"

    hyperparams_logged = False

    use_cuda = torch.cuda.is_available()
    if use_cuda:
        device_id = cfg.trainer.device_id
        device = torch.device("cuda", index=device_id)
        torch.set_float32_matmul_precision("high")
    else:
        device = torch.device("cpu")

    # Seeding
    seed = cfg.get("seed", 42)
    torch.manual_seed(seed)
    np.random.seed(seed)
    if use_cuda:
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = not torch.backends.cudnn.deterministic

    # Setup data
    train_data_path = cfg.data.train_data_path
    test_data_path = cfg.data.test_data_path

    train_image_folder_set = torchvision.datasets.ImageFolder(root=train_data_path)
    test_image_folder_set = torchvision.datasets.ImageFolder(root=test_data_path)

    num_classes = len(train_image_folder_set.classes)
    print(f"Number of classes: {num_classes}")

    num_runs = cfg.get("num_runs", 1)
    val_split = cfg.data.get("val_split", 0.2)

    all_metrics = []

    # Training loop over multiple runs
    for run_idx in range(num_runs):
        logger = CSVLogger(log_path, name=f"run_{run_idx + 1}", version="")

        all_indices = list(range(len(train_image_folder_set)))
        all_targets = [train_image_folder_set.samples[i][1] for i in all_indices]

        train_indices, val_indices = train_test_split(
            all_indices,
            test_size=val_split,
            random_state=seed + run_idx,
            stratify=all_targets,
        )
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

        # Calculate class weights for handling class imbalance
        train_targets = [all_targets[i] for i in train_indices]
        samples_per_class = np.bincount(train_targets, minlength=num_classes)
        inv_freq = 1.0 / np.maximum(samples_per_class, 1)
        class_weights = inv_freq / inv_freq.sum() * num_classes
        class_weights = torch.tensor(class_weights, dtype=torch.float32)

        train_loader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=cfg.data.get("batch_size"),
            shuffle=True,
            num_workers=cfg.data.get("num_workers", 4),
            pin_memory=True,
        )
        val_loader = torch.utils.data.DataLoader(
            val_dataset,
            batch_size=cfg.data.get("batch_size"),
            shuffle=False,
            num_workers=cfg.data.get("num_workers", 4),
            pin_memory=True,
        )
        test_loader = torch.utils.data.DataLoader(
            test_dataset,
            batch_size=cfg.data.get("batch_size"),
            shuffle=False,
            num_workers=cfg.data.get("num_workers", 4),
            pin_memory=True,
        )

        # Create model
        model = hydra.utils.instantiate(cfg.model, num_classes=num_classes)
        model.to(device)

        scaler = torch.amp.grad_scaler.GradScaler("cuda") if use_cuda else None

        criterion = torch.nn.CrossEntropyLoss(weight=class_weights.to(device))
        criterion = criterion.to(device)

        optimizer: torch.optim.Optimizer = hydra.utils.instantiate(
            cfg.get("optimizer"),
            model.parameters(),
        )

        num_epochs = cfg.get("num_epochs")

        scheduler_cfg = cfg.get("scheduler", None)
        if scheduler_cfg:
            scheduler = hydra.utils.instantiate(scheduler_cfg, optimizer)
        else:
            scheduler = None

        train_loss = MeanMetric().to(device)
        train_acc = MulticlassAccuracy(num_classes=num_classes, average="weighted").to(device)
        train_precision = MulticlassPrecision(num_classes=num_classes, average="weighted").to(device)
        train_recall = MulticlassRecall(num_classes=num_classes, average="weighted").to(device)
        train_f1 = MulticlassF1Score(num_classes=num_classes, average="weighted").to(device)

        val_loss = MeanMetric().to(device)
        val_acc = MulticlassAccuracy(num_classes=num_classes, average="weighted").to(device)
        val_precision = MulticlassPrecision(num_classes=num_classes, average="weighted").to(device)
        val_recall = MulticlassRecall(num_classes=num_classes, average="weighted").to(device)
        val_f1 = MulticlassF1Score(num_classes=num_classes, average="weighted").to(device)

        test_loss = MeanMetric().to(device)
        test_acc = MulticlassAccuracy(num_classes=num_classes, average="weighted").to(device)
        test_precision = MulticlassPrecision(num_classes=num_classes, average="weighted").to(device)
        test_recall = MulticlassRecall(num_classes=num_classes, average="weighted").to(device)
        test_f1 = MulticlassF1Score(num_classes=num_classes, average="weighted").to(device)
        
        best_val_acc = 0.0
        best_val_f1 = 0.0
        best_epoch = -1

        if not hyperparams_logged:
            os.makedirs(f"{log_path}/summary", exist_ok=True)
            with open(f"{log_path}/summary/hyperparameters.yaml", "w") as f:
                from omegaconf import OmegaConf
                f.write(OmegaConf.to_yaml(cfg, resolve=True))
                f.write(f"\nnum_classes: {num_classes}\n")
                f.write(f"class_counts: {samples_per_class.tolist()}\n")
                f.write(f"class_weights: {class_weights.tolist()}\n")
            hyperparams_logged = True

        patience = cfg.get("early_stopping_patience", 999)
        epochs_no_improvement = 0

        for epoch in range(num_epochs):
            model.train()
            train_loss.reset()
            train_acc.reset()
            train_precision.reset()
            train_recall.reset()
            train_f1.reset()

            # Training loop
            train_bar = tqdm.tqdm(
                train_loader,
                desc=f"Run {run_idx + 1}, Epoch {epoch + 1}/{num_epochs}",
                leave=False,
                ncols=100,
                mininterval=0.2,
            )
            for batch in train_bar:
                inputs, targets = batch
                inputs = inputs.to(device)
                targets = targets.to(device)

                optimizer.zero_grad(set_to_none=True)

                if use_cuda:
                    with torch.amp.autocast("cuda"):
                        outputs = model(inputs)
                        loss = criterion(outputs, targets)
                else:
                    outputs = model(inputs)
                    loss = criterion(outputs, targets)

                if scaler:
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    optimizer.step()

                with torch.no_grad():
                    preds = torch.argmax(outputs, dim=1)
                    train_loss.update(loss)
                    train_acc.update(preds, targets)
                    train_precision.update(preds, targets)
                    train_recall.update(preds, targets)
                    train_f1.update(preds, targets)

                train_bar.set_postfix(
                    {
                        "train/loss": f"{train_loss.compute().item():.4f}",
                        "train/acc": f"{train_acc.compute().item():.4f}",
                    },
                    refresh=False,
                )
            train_bar.close()

            logger.log_metrics(
                {
                    "epoch": epoch + 1,
                    "train/loss": train_loss.compute().item(),
                    "train/acc": train_acc.compute().item(),
                    "train/precision": train_precision.compute().item(),
                    "train/recall": train_recall.compute().item(),
                    "train/f1": train_f1.compute().item(),
                    "lr": optimizer.param_groups[0]["lr"],
                }
            )

            # Validation loop
            model.eval()
            val_loss.reset()
            val_acc.reset()
            val_precision.reset()
            val_recall.reset()
            val_f1.reset()

            with torch.no_grad():
                val_bar = tqdm.tqdm(
                    val_loader,
                    desc=f"Run {run_idx + 1}, Val Epoch {epoch + 1}",
                    leave=False,
                    ncols=100,
                    mininterval=.2,
                )
                for batch in val_bar:
                    inputs, targets = batch
                    inputs = inputs.to(device)
                    targets = targets.to(device)

                    if use_cuda:
                        with torch.amp.autocast("cuda"):
                            outputs = model(inputs)
                            loss = criterion(outputs, targets)
                    else:
                        outputs = model(inputs)
                        loss = criterion(outputs, targets)

                    preds = torch.argmax(outputs, dim=1)
                    val_loss.update(loss)
                    val_acc.update(preds, targets)
                    val_precision.update(preds, targets)
                    val_recall.update(preds, targets)
                    val_f1.update(preds, targets)

                    val_bar.set_postfix(
                        {
                            "val/loss": f"{val_loss.compute().item():.4f}",
                            "val/acc": f"{val_acc.compute().item():.4f}",
                        },
                        refresh=False,
                    )
                val_bar.close()
            
            if scheduler:
                scheduler.step()

            # Early stopping check

            current_val_acc = val_acc.compute().item()
            current_val_f1 = val_f1.compute().item()

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

            logger.log_metrics(
                {
                    "epoch": epoch + 1,
                    "val/loss": val_loss.compute().item(),
                    "val/acc": val_acc.compute().item(),
                    "val/precision": val_precision.compute().item(),
                    "val/recall": val_recall.compute().item(),
                    "val/f1": val_f1.compute().item(),
                    "best_val/acc": best_val_acc,
                    "best_val/f1": best_val_f1,
                }
            )
            logger.save()

            if epochs_no_improvement >= patience:
                break

        # Test loop
        test_loss.reset(); test_acc.reset(); test_precision.reset(); test_recall.reset(); test_f1.reset()
        ckpt_path = f"{log_path}/run_{run_idx + 1}/best_model.pth"
        model.load_state_dict(torch.load(ckpt_path, map_location=device))

        model.eval()
        with torch.no_grad():
            for batch in test_loader:
                inputs, targets = batch
                inputs = inputs.to(device)
                targets = targets.to(device)

                if use_cuda:
                    with torch.amp.autocast("cuda"):
                        outputs = model(inputs)
                        loss = criterion(outputs, targets)
                else:
                    outputs = model(inputs)
                    loss = criterion(outputs, targets)

                preds = torch.argmax(outputs, dim=1)
                test_loss.update(loss)
                test_acc.update(preds, targets)
                test_precision.update(preds, targets)
                test_recall.update(preds, targets)
                test_f1.update(preds, targets)

        final_test_metrics = {
            "run_idx": run_idx + 1,
            "test/loss": test_loss.compute().item(),
            "test/acc": test_acc.compute().item(),
            "test/precision": test_precision.compute().item(),
            "test/recall": test_recall.compute().item(),
            "test/f1": test_f1.compute().item(),
            "best_val/acc": best_val_acc,
            "best_val/f1": best_val_f1,
            "best_val/epoch": best_epoch,
        }
        logger.log_metrics(final_test_metrics)
        all_metrics.append(final_test_metrics)

        logger.finalize("success")
        logger.save()
        print(f"Run {run_idx + 1}/{num_runs} | test/acc: {final_test_metrics['test/acc']:.4f} test/f1: {final_test_metrics['test/f1']:.4f}")

    # Summary stats
    all_metrics_df = pd.DataFrame(all_metrics)
    summary_statistics = []
    metrics_to_analyze = [m for m in all_metrics_df.columns if m not in ["run_idx", "best_val/epoch"]]
    for metric in metrics_to_analyze:
        values = all_metrics_df[metric].dropna()
        n = len(values)
        mean = values.mean()
        std = values.std(ddof=1)
        n = len(values)
        se = std / np.sqrt(n) if n > 0 else 0.0
        if n > 1:
            from scipy import stats as _stats
            t = _stats.t.ppf(0.975, df=n - 1)
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

    summary_df = pd.DataFrame(summary_statistics)
    os.makedirs(f"{log_path}/summary", exist_ok=True)
    all_metrics_df.to_csv(f"{log_path}/summary/summary_metrics.csv", index=False)
    summary_df.to_csv(f"{log_path}/summary/summary_statistics.csv", index=False)


@hydra.main(
    version_base="1.2", config_path="../configs", config_name="train.yaml"
)
def main(cfg: DictConfig) -> None:
    train(cfg)


if __name__ == "__main__":
    main()
