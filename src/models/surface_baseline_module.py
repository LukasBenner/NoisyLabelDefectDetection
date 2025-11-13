from typing import Any, Dict, Tuple

import torch
from lightning import LightningModule
from torchmetrics import MaxMetric, MeanMetric
from torchmetrics.classification import MulticlassAccuracy, MulticlassF1Score, MulticlassPrecision, MulticlassRecall
from torch.optim.lr_scheduler import LRScheduler


class SurfaceBaselineModule(LightningModule):
    def __init__(
        self,
        net: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: LRScheduler,
        compile: bool,
        num_classes: int,
        class_weights: torch.Tensor = None,
    ) -> None:
        super().__init__()

        # this line allows to access init params with 'self.hparams' attribute
        # also ensures init params will be stored in ckpt
        self.save_hyperparameters(logger=False, ignore=["net"])

        self.net = net

        # loss function - will be set in setup() to get class weights from datamodule
        self.criterion = None

        # metric objects for calculating and averaging metrics across batches
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

        # for averaging loss across batches
        self.train_loss = MeanMetric()
        self.val_loss = MeanMetric()
        self.test_loss = MeanMetric()

        # for tracking best so far validation metrics
        self.val_acc_best = MaxMetric()
        self.val_f1_best = MaxMetric()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

    def on_train_start(self) -> None:
        self.val_loss.reset()
        self.val_acc.reset()
        self.val_precision.reset()
        self.val_recall.reset()
        self.val_f1.reset()
        self.val_acc_best.reset()
        self.val_f1_best.reset()

    def setup(self, stage: str) -> None:
        """Lightning hook that is called at the beginning of fit (train + validate), validate,
        test, or predict.

        This is a good hook when you need to build models dynamically or adjust something about
        them. This hook is called on every process when using DDP.

        :param stage: Either `"fit"`, `"validate"`, `"test"`, or `"predict"`.
        """
        # Initialize loss function with class weights from datamodule
        if self.criterion is None:
            class_weights = None
            if hasattr(self.trainer.datamodule, "class_weights") and self.trainer.datamodule.class_weights is not None:
                class_weights = torch.tensor(
                    self.trainer.datamodule.class_weights.values, dtype=torch.float, device=self.device
                )
            self.criterion = torch.nn.CrossEntropyLoss(weight=class_weights)

        if self.hparams.compile and stage == "fit":
            self.net = torch.compile(self.net)

    def model_step(self, batch: Tuple[torch.Tensor, torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        x, y = batch
        logits = self.forward(x)
        loss = self.criterion(logits, y)
        preds = torch.argmax(logits, dim=1)
        return loss, preds, y

    def training_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> torch.Tensor:
        loss, preds, targets = self.model_step(batch)

        # update and log metrics
        self.train_loss.update(loss)
        self.train_acc.update(preds, targets)
        self.train_precision.update(preds, targets)
        self.train_recall.update(preds, targets)
        self.train_f1.update(preds, targets)

        return loss

    def on_train_epoch_end(self) -> None:
        # Log training metrics at the end of each training epoch
        self.log("train/loss", self.train_loss.compute(), on_step=False, on_epoch=True, prog_bar=True)
        self.log("train/acc", self.train_acc.compute(), on_step=False, on_epoch=True, prog_bar=True)
        self.log("train/precision", self.train_precision.compute(), on_step=False, on_epoch=True, prog_bar=False)
        self.log("train/recall", self.train_recall.compute(), on_step=False, on_epoch=True, prog_bar=False)
        self.log("train/f1", self.train_f1.compute(), on_step=False, on_epoch=True, prog_bar=True)
        self.log("train/lr", self.trainer.optimizers[0].param_groups[0]['lr'], on_step=False, on_epoch=True)

    def validation_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> None:
        """Perform a single validation step on a batch of data from the validation set.

        :param batch: A batch of data (a tuple) containing the input tensor of images and target
            labels.
        :param batch_idx: The index of the current batch.
        """
        loss, preds, targets = self.model_step(batch)

        # update and log metrics
        self.val_loss.update(loss)
        self.val_acc.update(preds, targets)
        self.val_precision.update(preds, targets)
        self.val_recall.update(preds, targets)
        self.val_f1.update(preds, targets)

    def on_validation_epoch_end(self) -> None:
        "Lightning hook that is called when a validation epoch ends."
        # Log validation metrics
        self.log("val/loss", self.val_loss.compute(), on_step=False, on_epoch=True, prog_bar=True)
        self.log("val/acc", self.val_acc.compute(), on_step=False, on_epoch=True, prog_bar=True)
        self.log("val/precision", self.val_precision.compute(), on_step=False, on_epoch=True, prog_bar=False)
        self.log("val/recall", self.val_recall.compute(), on_step=False, on_epoch=True, prog_bar=False)
        self.log("val/f1", self.val_f1.compute(), on_step=False, on_epoch=True, prog_bar=True)
        
        # Update and log best metrics
        acc = self.val_acc.compute()  # get current val acc
        f1 = self.val_f1.compute()  # get current val f1
        self.val_acc_best(acc)  # update best so far val acc
        self.val_f1_best(f1)  # update best so far val f1

        # log `val_acc_best` and `val_f1_best` as values through `.compute()` method
        # otherwise metric would be reset by lightning after each epoch
        self.log("val/acc_best", self.val_acc_best.compute(), sync_dist=True, prog_bar=True)
        self.log("val/f1_best", self.val_f1_best.compute(), sync_dist=True, prog_bar=True)

    def test_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> None:
        loss, preds, targets = self.model_step(batch)

        # update metrics
        self.test_loss.update(loss)
        self.test_acc.update(preds, targets)
        self.test_precision.update(preds, targets)
        self.test_recall.update(preds, targets)
        self.test_f1.update(preds, targets)

    def on_test_epoch_end(self) -> None:
        # Log test metrics at the end of testing
        self.log("test/loss", self.test_loss.compute(), on_step=False, on_epoch=True, prog_bar=True)
        self.log("test/acc", self.test_acc.compute(), on_step=False, on_epoch=True, prog_bar=True)
        self.log("test/precision", self.test_precision.compute(), on_step=False, on_epoch=True, prog_bar=False)
        self.log("test/recall", self.test_recall.compute(), on_step=False, on_epoch=True, prog_bar=False)
        self.log("test/f1", self.test_f1.compute(), on_step=False, on_epoch=True, prog_bar=True)

    def configure_optimizers(self) -> Dict[str, Any]:
        optimizer = self.hparams.optimizer(params=self.trainer.model.parameters())
        if self.hparams.scheduler is not None:
            scheduler = self.hparams.scheduler(optimizer=optimizer)
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "monitor": "val/loss",
                    "interval": "epoch",
                    "frequency": 1,
                },
            }
        return {"optimizer": optimizer}


if __name__ == "__main__":
    _ = SurfaceBaselineModule(None, None, None, None, 10)
