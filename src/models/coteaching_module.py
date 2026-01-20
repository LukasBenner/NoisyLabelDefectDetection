"""LightningModule for simple Co-Teaching with MobileNetV3-Large."""

from __future__ import annotations

from typing import Any, Dict, Optional

import torch
import torch.nn.functional as F
from lightning import LightningModule
from torchmetrics import MaxMetric, MeanMetric
from torchmetrics.classification import (
    MulticlassAccuracy,
    MulticlassF1Score,
    MulticlassPrecision,
    MulticlassRecall,
)

from src.models.components.mobile_net import MobileNet


class CoTeachingModule(LightningModule):
    """Simple co-teaching implementation using two MobileNetV3-Large models."""

    def __init__(
        self,
        num_classes: int = 10,
        pretrained: bool = True,
        lr: float = 1e-3,
        weight_decay: float = 1e-4,
        forget_rate: float = 0.2,
        num_gradual: int = 10,
        exponent: float = 1.0,
        compile: bool = False,
        datamodule: Optional[Any] = None,
    ) -> None:
        super().__init__()

        self.save_hyperparameters(logger=False, ignore=["datamodule"])

        self.model1 = MobileNet(
            num_classes=num_classes,
            pretrained=pretrained,
            variant="large",
        )
        self.model2 = MobileNet(
            num_classes=num_classes,
            pretrained=pretrained,
            variant="large",
        )

        self.automatic_optimization = False
        self._compile = compile
        self.datamodule = datamodule

        self.train_acc = MulticlassAccuracy(num_classes=num_classes, average="weighted")
        self.val_acc = MulticlassAccuracy(num_classes=num_classes, average="weighted")
        self.test_acc = MulticlassAccuracy(num_classes=num_classes, average="weighted")

        self.train_precision = MulticlassPrecision(num_classes=num_classes, average="weighted")
        self.val_precision = MulticlassPrecision(num_classes=num_classes, average="weighted")
        self.test_precision = MulticlassPrecision(num_classes=num_classes, average="weighted")

        self.train_recall = MulticlassRecall(num_classes=num_classes, average="weighted")
        self.val_recall = MulticlassRecall(num_classes=num_classes, average="weighted")
        self.test_recall = MulticlassRecall(num_classes=num_classes, average="weighted")

        self.train_f1 = MulticlassF1Score(num_classes=num_classes, average="weighted")
        self.val_f1 = MulticlassF1Score(num_classes=num_classes, average="weighted")
        self.test_f1 = MulticlassF1Score(num_classes=num_classes, average="weighted")

        self.train_loss = MeanMetric()
        self.val_loss = MeanMetric()
        self.test_loss = MeanMetric()

        self.val_acc_best = MaxMetric()
        self.val_f1_best = MaxMetric()

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
        self.val_loss.reset()
        self.val_acc_best.reset()
        self.val_f1_best.reset()

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
        self.train_acc(preds, targets)
        self.train_precision(preds, targets)
        self.train_recall(preds, targets)
        self.train_f1(preds, targets)

        self.log("train/loss", self.train_loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("train/acc", self.train_acc, on_step=False, on_epoch=True, prog_bar=True)
        self.log("train/precision", self.train_precision, on_step=False, on_epoch=True)
        self.log("train/recall", self.train_recall, on_step=False, on_epoch=True)
        self.log("train/f1", self.train_f1, on_step=False, on_epoch=True)
        self.log("train/forget_rate", self._forget_rate(), on_step=False, on_epoch=True)
        self.log("train/loss_model1", loss1, on_step=False, on_epoch=True)
        self.log("train/loss_model2", loss2, on_step=False, on_epoch=True)

        return loss

    def validation_step(self, batch: Any, batch_idx: int) -> None:
        inputs, targets = batch
        logits1 = self.model1(inputs)
        logits2 = self.model2(inputs)
        avg_logits = 0.5 * (logits1 + logits2)
        loss = F.cross_entropy(avg_logits, targets)
        preds = torch.argmax(avg_logits, dim=1)

        self.val_loss(loss)
        self.val_acc(preds, targets)
        self.val_precision(preds, targets)
        self.val_recall(preds, targets)
        self.val_f1(preds, targets)

        self.log("val/loss", self.val_loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("val/acc", self.val_acc, on_step=False, on_epoch=True, prog_bar=True)
        self.log("val/precision", self.val_precision, on_step=False, on_epoch=True)
        self.log("val/recall", self.val_recall, on_step=False, on_epoch=True)
        self.log("val/f1", self.val_f1, on_step=False, on_epoch=True)

    def on_validation_epoch_end(self) -> None:
        acc = self.val_acc.compute()
        f1 = self.val_f1.compute()

        self.val_acc_best(acc)
        self.val_f1_best(f1)

        self.log("val/acc_best", self.val_acc_best.compute(), sync_dist=True, prog_bar=True)
        self.log("val/f1_best", self.val_f1_best.compute(), sync_dist=True)

    def test_step(self, batch: Any, batch_idx: int) -> None:
        inputs, targets = batch
        logits1 = self.model1(inputs)
        logits2 = self.model2(inputs)
        avg_logits = 0.5 * (logits1 + logits2)
        loss = F.cross_entropy(avg_logits, targets)
        preds = torch.argmax(avg_logits, dim=1)

        self.test_loss(loss)
        self.test_acc(preds, targets)
        self.test_precision(preds, targets)
        self.test_recall(preds, targets)
        self.test_f1(preds, targets)

        self.log("test/loss", self.test_loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("test/acc", self.test_acc, on_step=False, on_epoch=True, prog_bar=True)
        self.log("test/precision", self.test_precision, on_step=False, on_epoch=True)
        self.log("test/recall", self.test_recall, on_step=False, on_epoch=True)
        self.log("test/f1", self.test_f1, on_step=False, on_epoch=True)

    def configure_optimizers(self) -> Dict[str, Any]:
        opt1 = torch.optim.AdamW(
            self.model1.parameters(),
            lr=float(self.hparams.lr),
            weight_decay=float(self.hparams.weight_decay),
        )
        opt2 = torch.optim.AdamW(
            self.model2.parameters(),
            lr=float(self.hparams.lr),
            weight_decay=float(self.hparams.weight_decay),
        )
        return [opt1, opt2]
