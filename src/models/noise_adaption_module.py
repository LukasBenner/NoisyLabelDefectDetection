"""LightningModule for training with Noise Adaption Layer."""

from typing import Any, Dict, Optional

import torch
from torch import nn

from src.models.base_robust_module import BaseRobustModule
from src.models.components.noise_adaption_layer import NoiseAdaptionNet


class NoiseAdaptionModule(BaseRobustModule):
    """
    LightningModule that uses a Noise Adaption Layer for handling noisy labels.
    
    This module wraps a base network with a learnable noise transition matrix
    that adapts the model's predictions to account for label noise during training.
    
    The noise adaption layer learns a confusion matrix T where T[i,j] represents
    the probability that a sample with true label i is mislabeled as j.
    
    Args:
        net: Base neural network (e.g., EfficientNet, ResNet)
        num_classes: Number of classes
        init_value: Initial diagonal value for noise transition matrix (default: 0.9)
        trainable_noise: Whether the noise transition matrix is trainable (default: True)
        optimizer: Optimizer (partial instantiation)
        scheduler: Learning rate scheduler (partial instantiation)
        criterion: Loss function (partial instantiation)
        compile: Whether to compile the model with torch.compile
        datamodule: DataModule for accessing data properties
    """
    
    def __init__(
        self,
        net: nn.Module,
        num_classes: int,
        init_value: float = 0.9,
        optimizer: Optional[torch.optim.Optimizer] = None,
        scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
        criterion: Optional[nn.Module] = None,
        compile: bool = False,
        datamodule: Optional[Any] = None,
    ) -> None:
        # Wrap the base network with noise adaption layer
        noise_adapted_net = NoiseAdaptionNet(
            base_net=net,
            num_classes=num_classes,
            init_value=init_value
        )
        
        # Initialize parent class with wrapped network
        super().__init__(
            net=noise_adapted_net,
            num_classes=num_classes,
            optimizer=optimizer,
            scheduler=scheduler,
            criterion=criterion,
            compile=compile,
            datamodule=datamodule,
        )
        
        # Save additional hyperparameters
        self.save_hyperparameters(
            logger=False,
            ignore=["net", "optimizer", "scheduler", "criterion", "datamodule"]
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through network with noise adaption.
        
        During training, applies noise adaption layer.
        During validation/test, can optionally use clean predictions.
        """
        return self.net(x)
    
    def get_clean_predictions(self, x: torch.Tensor) -> torch.Tensor:
        """Get predictions from base network without noise adaption."""
        return self.net.get_clean_output(x)
    
    def validation_step(self, batch: Any, batch_idx: int) -> None:
        """
        Validation step.
        
        Uses clean predictions (without noise adaption) for validation
        to get true model performance on clean data.
        """
        inputs, targets = batch
        
        # Get clean predictions for validation
        logits = self.get_clean_predictions(inputs)
        loss = self.criterion(logits, targets)
        preds = torch.argmax(logits, dim=1)
        
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
    
    def test_step(self, batch: Any, batch_idx: int) -> None:
        """
        Test step.
        
        Uses clean predictions (without noise adaption) for testing.
        """
        inputs, targets = batch
        
        # Get clean predictions for testing
        logits = self.get_clean_predictions(inputs)
        loss = self.criterion(logits, targets)
        preds = torch.argmax(logits, dim=1)
        
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
    
    def on_train_epoch_end(self) -> None:
        """Log the learned noise transition matrix at the end of each epoch."""
        if hasattr(self.net, 'get_transition_matrix'):
            transition_matrix = self.net.get_transition_matrix()
            
            # Log some statistics about the transition matrix
            diagonal_mean = torch.diagonal(transition_matrix).mean().item()
            off_diagonal_mean = (
                transition_matrix.sum() - torch.diagonal(transition_matrix).sum()
            ).item() / (self.num_classes * (self.num_classes - 1))
            
            self.log("noise/diagonal_mean", diagonal_mean, on_epoch=True)
            self.log("noise/off_diagonal_mean", off_diagonal_mean, on_epoch=True)
