from typing import Any, Dict, Optional, List
import re

import torch
from lightning import LightningModule
from torchmetrics import MaxMetric, MeanMetric, MetricCollection
from torchmetrics.classification import (
    MulticlassAccuracy,
    MulticlassF1Score,
    MulticlassPrecision,
    MulticlassRecall,
)


class BaseRobustModule(LightningModule):
    def __init__(
        self,
        net: torch.nn.Module,
        num_classes: int,
        optimizer: Optional[torch.optim.Optimizer] = None,
        scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
        criterion: Optional[torch.nn.Module] = None,
        compile: bool = False,
        datamodule: Optional[Any] = None,
        log_per_class: bool = True,  # per-class metrics for val/test
    ) -> None:
        super().__init__()
        self.save_hyperparameters(
            logger=False,
            ignore=["net", "optimizer", "scheduler", "criterion", "datamodule"],
        )

        self.net = net
        self.num_classes = num_classes
        self.datamodule = datamodule
        self._optimizer = optimizer
        self._scheduler = scheduler
        self.log_per_class = log_per_class

        # --- class weights: store as buffer (device-safe) ---
        cw = None
        if self.datamodule is not None and hasattr(self.datamodule, "class_weights"):
            cw = self.datamodule.class_weights.detach().to(dtype=torch.float32).cpu()
        if cw is None:
            cw = torch.ones(num_classes, dtype=torch.float32)

        self.register_buffer("class_weights", cw, persistent=True)

        # --- criterion ---
        if criterion is None:
            raise ValueError("criterion has to be set")
        # Instantiate criterion with weights on the right device automatically via buffer
        self.criterion = criterion(num_classes=num_classes, weight=self.class_weights)

        # --- losses ---
        self.train_loss = MeanMetric()
        self.val_loss = MeanMetric()
        self.test_loss = MeanMetric()

        # --- primary metrics: macro (fair to all classes) ---
        macro_metrics = MetricCollection(
            {
                "precision_macro": MulticlassPrecision(num_classes=num_classes, average="macro"),
                "recall_macro": MulticlassRecall(num_classes=num_classes, average="macro"),
                "f1_macro": MulticlassF1Score(num_classes=num_classes, average="macro"),
            }
        )

        # --- secondary context metrics: weighted + accuracy ---
        # Note: For single-label multiclass, weighted recall == accuracy (redundant).
        # We still log accuracy + weighted F1; weighted recall optional, but kept here for completeness.
        context_metrics = MetricCollection(
            {
                "acc": MulticlassAccuracy(num_classes=num_classes),  # default is effectively micro
                "precision_weighted": MulticlassPrecision(num_classes=num_classes, average="weighted"),
                "recall_weighted": MulticlassRecall(num_classes=num_classes, average="weighted"),
                "f1_weighted": MulticlassF1Score(num_classes=num_classes, average="weighted"),
            }
        )

        combined_metrics = MetricCollection(
            {
                **dict(macro_metrics.items()),
                **dict(context_metrics.items()),
            }
        )

        # Split-wise metric collections
        self.train_metrics = combined_metrics.clone(prefix="train/")
        self.val_metrics = combined_metrics.clone(prefix="val/")
        self.test_metrics = combined_metrics.clone(prefix="test/")

        # Per-class metrics (only val/test; high signal for imbalance)
        self.val_per_class = MetricCollection(
            {
                "precision": MulticlassPrecision(num_classes=num_classes, average=None),
                "recall": MulticlassRecall(num_classes=num_classes, average=None),
                "f1": MulticlassF1Score(num_classes=num_classes, average=None),
            }
        ).clone(prefix="val/per_class/")

        self.test_per_class = MetricCollection(
            {
                "precision": MulticlassPrecision(num_classes=num_classes, average=None),
                "recall": MulticlassRecall(num_classes=num_classes, average=None),
                "f1": MulticlassF1Score(num_classes=num_classes, average=None),
            }
        ).clone(prefix="test/per_class/")

        # Best tracking (use macro-F1 as the primary selection metric)
        self.val_f1_macro_best = MaxMetric()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

    def _get_class_names(self) -> Optional[List[str]]:
        if self.datamodule is None or not hasattr(self.datamodule, "class_names"):
            return None
        class_names = list(self.datamodule.class_names)
        if len(class_names) < self.num_classes:
            return None
        return class_names

    @staticmethod
    def _sanitize_class_name(name: str) -> str:
        name = name.strip()
        if not name:
            return "unknown"
        return re.sub(r"[^A-Za-z0-9_]+", "_", name)

    def _class_metric_key(self, prefix: str, metric: str, idx: int) -> str:
        class_names = self._get_class_names()
        if class_names is None:
            return f"{prefix}{metric}_c{idx}"
        safe_name = self._sanitize_class_name(class_names[idx])
        return f"{prefix}{metric}_{safe_name}"

    def compute_loss(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        return self.criterion(logits, targets)

    def model_step(self, batch: Any):
        inputs, targets = batch
        logits = self.forward(inputs)
        loss = self.compute_loss(logits, targets)
        preds = torch.argmax(logits, dim=1)
        return loss, preds, targets

    def on_train_start(self) -> None:
        self.val_f1_macro_best.reset()

    # ---------------- TRAIN ----------------
    def training_step(self, batch: Any, batch_idx: int) -> torch.Tensor:
        loss, preds, targets = self.model_step(batch)

        self.train_loss(loss)
        self.train_metrics(preds, targets)

        self.log("train/loss", self.train_loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log_dict(self.train_metrics, on_step=False, on_epoch=True, prog_bar=False)

        return loss

    # ---------------- VAL ----------------
    def validation_step(self, batch: Any, batch_idx: int) -> None:
        loss, preds, targets = self.model_step(batch)

        self.val_loss(loss)
        self.val_metrics(preds, targets)

        if self.log_per_class:
            self.val_per_class(preds, targets)

        self.log("val/loss", self.val_loss, on_step=False, on_epoch=True, prog_bar=True)
        # Show the primary metric on prog bar
        self.log_dict(
            {k: v for k, v in self.val_metrics.items() if k != "val/f1_macro"},
            on_step=False,
            on_epoch=True,
            prog_bar=False,
        )
        self.log("val/f1_macro", self.val_metrics["val/f1_macro"], on_step=False, on_epoch=True, prog_bar=True)

    def on_validation_epoch_end(self) -> None:
        # Track best macro-F1
        current = self.val_metrics["val/f1_macro"].compute()
        self.val_f1_macro_best(current)
        self.log("val/f1_macro_best", self.val_f1_macro_best.compute(), prog_bar=True, sync_dist=True)

        # Log per-class scalars (val)
        if self.log_per_class:
            pc = self.val_per_class.compute()  # dict: val/per_class/{precision,recall,f1} -> tensor [C]
            for i in range(self.num_classes):
                self.log(
                    self._class_metric_key("val/", "precision", i),
                    pc["val/per_class/precision"][i],
                    sync_dist=True,
                )
                self.log(
                    self._class_metric_key("val/", "recall", i),
                    pc["val/per_class/recall"][i],
                    sync_dist=True,
                )
                self.log(
                    self._class_metric_key("val/", "f1", i),
                    pc["val/per_class/f1"][i],
                    sync_dist=True,
                )
            self.val_per_class.reset()

    # ---------------- TEST ----------------
    def test_step(self, batch: Any, batch_idx: int) -> None:
        loss, preds, targets = self.model_step(batch)

        self.test_loss(loss)
        self.test_metrics(preds, targets)

        if self.log_per_class:
            self.test_per_class(preds, targets)

        self.log("test/loss", self.test_loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log_dict(
            {k: v for k, v in self.test_metrics.items() if k != "test/f1_macro"},
            on_step=False,
            on_epoch=True,
            prog_bar=False,
        )
        self.log("test/f1_macro", self.test_metrics["test/f1_macro"], on_step=False, on_epoch=True, prog_bar=True)

    def on_test_epoch_end(self) -> None:
        if self.log_per_class:
            pc = self.test_per_class.compute()
            for i in range(self.num_classes):
                self.log(
                    self._class_metric_key("test/", "precision", i),
                    pc["test/per_class/precision"][i],
                    sync_dist=True,
                )
                self.log(
                    self._class_metric_key("test/", "recall", i),
                    pc["test/per_class/recall"][i],
                    sync_dist=True,
                )
                self.log(
                    self._class_metric_key("test/", "f1", i),
                    pc["test/per_class/f1"][i],
                    sync_dist=True,
                )
            self.test_per_class.reset()

    # ---------------- OPTIM ----------------
    def configure_optimizers(self) -> Dict[str, Any]:
        if self._optimizer is None:
            raise ValueError("Optimizer not provided. Pass it to __init__ or override configure_optimizers()")

        optimizer = self._optimizer(params=self.trainer.model.parameters())

        if self._scheduler is not None:
            scheduler = self._scheduler(optimizer=optimizer)
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "monitor": "val/f1_macro",   # primary metric for imbalance + equal class importance
                    "interval": "epoch",
                    "frequency": 1,
                },
            }
        return {"optimizer": optimizer}

    def setup(self, stage: str) -> None:
        if self.hparams.compile and stage == "fit":
            self.net = torch.compile(self.net)
