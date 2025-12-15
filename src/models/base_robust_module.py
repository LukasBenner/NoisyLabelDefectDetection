"""Base LightningModule for robust learning methods."""

from typing import Any, Dict, Optional

import torch
from lightning import LightningModule
from torchmetrics import MaxMetric, MeanMetric
from torchmetrics.classification import (
    MulticlassAccuracy,
    MulticlassF1Score,
    MulticlassPrecision,
    MulticlassRecall,
)


class BaseRobustModule(LightningModule):
    """
    Base LightningModule for robust learning methods.
    
    All robust learning methods should inherit from this class and implement:
    - compute_loss(): Define custom loss computation
    - configure_optimizers(): Define optimizer and scheduler
    
    This provides:
    - Standard training/validation/test loops
    - Metric tracking (accuracy, precision, recall, F1)
    - Logging to multiple backends
    """

    def __init__(
        self,
        net: torch.nn.Module,
        num_classes: int,
        optimizer: Optional[torch.optim.Optimizer] = None,
        scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
        criterion: Optional[torch.nn.Module] = None,
        compile: bool = False,
        datamodule: Optional[Any] = None,
    ) -> None:
        super().__init__()

        # Save hyperparameters (except net to avoid serialization issues)
        self.save_hyperparameters(logger=False, ignore=["net", "optimizer", "scheduler", "criterion", "datamodule"])

        self.net = net

        # Store optimizer and scheduler for configure_optimizers
        self._optimizer = optimizer
        self._scheduler = scheduler

        # Loss function - can be overridden in subclasses
        self.num_classes = num_classes
        self.datamodule = datamodule
        
        class_weights = None
        if self.datamodule is not None and hasattr(self.datamodule, 'class_weights'):
            class_weights = torch.tensor(
                self.datamodule.class_weights.values, 
                dtype=torch.float32,
                device=self.device
            )
        
        # Initialize criterion with num_classes and class_weights
        self.criterion = criterion(
            num_classes=self.num_classes,
            weight=class_weights
        )

        # Metrics for train/val/test
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

        # Loss tracking
        self.train_loss = MeanMetric()
        self.val_loss = MeanMetric()
        self.test_loss = MeanMetric()

        # Track best validation metrics
        self.val_acc_best = MaxMetric()
        self.val_f1_best = MaxMetric()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the network."""
        return self.net(x)

    def on_train_start(self) -> None:
        """Lightning hook called at the beginning of training."""
        # Reset best metrics
        self.val_loss.reset()
        self.val_acc_best.reset()
        self.val_f1_best.reset()
        self.val_acc_best.reset()

    def model_step(self, batch: Any) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Perform a single model step (forward pass + loss computation).
        
        Args:
            batch: Tuple of (inputs, targets)
            
        Returns:
            Tuple of (loss, predictions, targets)
        """
        inputs, targets = batch
        logits = self.forward(inputs)
        loss = self.compute_loss(logits, targets)
        preds = torch.argmax(logits, dim=1)
        return loss, preds, targets

    def compute_loss(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Compute loss. Override this in subclasses for custom loss functions.
        
        Args:
            logits: Model outputs (before softmax)
            targets: Ground truth labels
            
        Returns:
            Loss value
        """
        return self.criterion(logits, targets)

    def training_step(self, batch: Any, batch_idx: int) -> torch.Tensor:
        """Training step."""
        loss, preds, targets = self.model_step(batch)

        # Update and log metrics
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

        return loss
    
    def on_train_epoch_end(self) -> None:
        pass

    def validation_step(self, batch: Any, batch_idx: int) -> None:
        """Validation step."""
        loss, preds, targets = self.model_step(batch)

        # Update and log metrics
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
        """Lightning hook called at the end of validation epoch."""
        acc = self.val_acc.compute()
        f1 = self.val_f1.compute()
        
        self.val_acc_best(acc)
        self.val_f1_best(f1)
        
        self.log("val/acc_best", self.val_acc_best.compute(), sync_dist=True, prog_bar=True)
        self.log("val/f1_best", self.val_f1_best.compute(), sync_dist=True)

    def test_step(self, batch: Any, batch_idx: int) -> None:
        """Test step."""
        loss, preds, targets = self.model_step(batch)

        # Update and log metrics
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
        """
        Configure optimizers and learning rate schedulers.
        
        Override this if you need custom optimizer configuration.
        """
        if self._optimizer is None:
            raise ValueError("Optimizer not provided. Pass it to __init__ or override configure_optimizers()")
        
        optimizer = self._optimizer(params=self.trainer.model.parameters())
        
        if self._scheduler is not None:
            scheduler = self._scheduler(optimizer=optimizer)
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

    def setup(self, stage: str) -> None:
        """Lightning hook called at the beginning of fit/test/predict."""
        if self.hparams.compile and stage == "fit":
            self.net = torch.compile(self.net)
