"""LightningModule for simple Co-Teaching with MobileNetV3-Large."""

from __future__ import annotations

from typing import Any, Dict, Optional

from lightning import LightningModule
import torch
import torch.nn.functional as F

from src.models.components.mobile_net import MobileNet
from src.models.base_robust_module import BaseRobustModule
from torchmetrics import MaxMetric, MeanMetric, MetricCollection
from torchmetrics.classification import (
    MulticlassAccuracy,
    MulticlassF1Score,
    MulticlassPrecision,
    MulticlassRecall,
)


class CoTeachingModule(LightningModule):
    """Simple co-teaching implementation using two MobileNetV3-Large models."""

    def __init__(
        self,
        num_classes: int,
        lr: float = 0.001,
        sgd_momentum: float = 0.9,
        weight_decay: float = 1e-5,
        forget_rate: float = 0.2,
        num_gradual: int = 10,
        exponent: float = 1.0,
        datamodule: Optional[Any] = None,
        log_per_class: bool = True,  # per-class metrics for val/test
        compile: bool = False,
    ) -> None:
        super().__init__()
        self.save_hyperparameters(logger=False, ignore=["datamodule"])

        self.num_classes = num_classes
        self.datamodule = datamodule
        self.log_per_class = log_per_class
        self._compile = compile

        self.model1 = MobileNet(
            num_classes=self.num_classes,
            pretrained=True,
            variant="large",
        )
        self.model2 = MobileNet(
            num_classes=self.num_classes,
            pretrained=True,
            variant="large",
        )

        self.automatic_optimization = False
        
        # --- class weights: store as buffer (device-safe) ---
        cw = None
        if self.datamodule is not None and hasattr(self.datamodule, "class_weights"):
            cw = self.datamodule.class_weights.detach().to(dtype=torch.float32).cpu()
        if cw is None:
            cw = torch.ones(num_classes, dtype=torch.float32)

        self.register_buffer("class_weights", cw, persistent=True)

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


    def setup(self, stage: str) -> None:
        if self._compile and stage == "fit":
            self.model1 = torch.compile(self.model1)
            self.model2 = torch.compile(self.model2)

    def _forget_rate(self) -> float:
        if self.hparams.num_gradual <= 0:
            return float(self.hparams.forget_rate)
        progress = min(
            1.0,
            float(self.current_epoch + 1) / float(self.hparams.num_gradual),
        )
        rate = float(self.hparams.forget_rate) * (progress ** float(self.hparams.exponent))
        return float(min(max(rate, 0.0), float(self.hparams.forget_rate)))

    def _co_teaching_loss(
        self, logits1: torch.Tensor, logits2: torch.Tensor, targets: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        loss1_all = F.cross_entropy(logits1, targets, reduction="none")
        loss2_all = F.cross_entropy(logits2, targets, reduction="none")

        forget_rate = self._forget_rate()
        remember_rate = 1.0 - forget_rate
        batch_size = targets.size(0)
        num_keep = max(1, int(remember_rate * batch_size))

        idx1 = torch.argsort(loss1_all)[:num_keep]
        idx2 = torch.argsort(loss2_all)[:num_keep]

        loss1 = F.cross_entropy(logits1[idx2], targets[idx2])
        loss2 = F.cross_entropy(logits2[idx1], targets[idx1])
        loss = 0.5 * (loss1 + loss2)
        return loss, loss1, loss2
    
    def on_train_start(self) -> None:
        self.val_f1_macro_best.reset()

    def on_train_epoch_end(self) -> None:
        # Manual optimization requires manual scheduler stepping.
        schedulers = self.lr_schedulers()
        if not schedulers:
            return

        if not isinstance(schedulers, (list, tuple)):
            schedulers = [schedulers]

        for scheduler in schedulers:
            if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                metric = self.val_metrics["val/f1_macro"].compute()
                scheduler.step(metric)
            else:
                scheduler.step()

    def training_step(self, batch: Any, batch_idx: int) -> torch.Tensor:
        inputs, targets = batch

        logits1 = self.model1(inputs)
        logits2 = self.model2(inputs)

        loss, loss1, loss2 = self._co_teaching_loss(logits1, logits2, targets)

        opt1, opt2 = self.optimizers()
        opt1.zero_grad()
        opt2.zero_grad()
        self.manual_backward(loss)
        opt1.step()
        opt2.step()

        avg_logits = 0.5 * (logits1 + logits2)
        preds = torch.argmax(avg_logits, dim=1)

        self.train_loss(loss)
        self.train_metrics(preds, targets)

        self.log("train/loss", self.train_loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("train/loss_model1", loss1, on_step=False, on_epoch=True)
        self.log("train/loss_model2", loss2, on_step=False, on_epoch=True)
        self.log_dict(self.train_metrics, on_step=False, on_epoch=True, prog_bar=False)

        return loss

    def validation_step(self, batch: Any, batch_idx: int) -> None:
        inputs, targets = batch
        logits1 = self.model1(inputs)
        logits2 = self.model2(inputs)
        avg_logits = 0.5 * (logits1 + logits2)
        loss = F.cross_entropy(avg_logits, targets)
        preds = torch.argmax(avg_logits, dim=1)

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
                self.log(f"val/precision_c{i}", pc["val/per_class/precision"][i], sync_dist=True)
                self.log(f"val/recall_c{i}", pc["val/per_class/recall"][i], sync_dist=True)
                self.log(f"val/f1_c{i}", pc["val/per_class/f1"][i], sync_dist=True)

    def test_step(self, batch: Any, batch_idx: int) -> None:
        inputs, targets = batch
        logits1 = self.model1(inputs)
        logits2 = self.model2(inputs)
        avg_logits = 0.5 * (logits1 + logits2)
        loss = F.cross_entropy(avg_logits, targets)
        preds = torch.argmax(avg_logits, dim=1)

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
                self.log(f"test/precision_c{i}", pc["test/per_class/precision"][i], sync_dist=True)
                self.log(f"test/recall_c{i}", pc["test/per_class/recall"][i], sync_dist=True)
                self.log(f"test/f1_c{i}", pc["test/per_class/f1"][i], sync_dist=True)

    def configure_optimizers(self) -> Any:
        opt1 = torch.optim.SGD(
            self.model1.parameters(),
            lr=float(self.hparams.lr),
            momentum=float(self.hparams.sgd_momentum),
            weight_decay=float(self.hparams.weight_decay),
        )
        opt2 = torch.optim.SGD(
            self.model2.parameters(),
            lr=float(self.hparams.lr),
            momentum=float(self.hparams.sgd_momentum),
            weight_decay=float(self.hparams.weight_decay),
        )

        t_max = int(getattr(self.trainer, "max_epochs", 1) or 1)
        scheduler1 = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer=opt1, T_max=t_max)
        scheduler2 = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer=opt2, T_max=t_max)

        return [
            {"optimizer": opt1, "lr_scheduler": {"scheduler": scheduler1, "interval": "epoch"}},
            {"optimizer": opt2, "lr_scheduler": {"scheduler": scheduler2, "interval": "epoch"}},
        ]
