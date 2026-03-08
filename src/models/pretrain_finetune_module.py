from typing import Any, Dict, List, Optional

import torch
from lightning import LightningModule

from src.models.base_robust_module import BaseRobustModule


class PretrainFinetuneModule(BaseRobustModule):
    """Extension of BaseRobustModule that supports two-phase training.

    Phase 1 (pretrain): Uses this module as-is (identical to BaseRobustModule).
    Phase 2 (finetune): Loads pretrained backbone weights, optionally freezes
    the backbone for the first N epochs, and uses discriminative learning rates
    (lower LR for backbone, higher for classifier head).
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
        log_per_class: bool = True,
        freeze_backbone_epochs: int = 0,
        backbone_lr_factor: float = 0.1,
    ) -> None:
        super().__init__(
            net=net,
            num_classes=num_classes,
            optimizer=optimizer,
            scheduler=scheduler,
            criterion=criterion,
            compile=compile,
            datamodule=datamodule,
            log_per_class=log_per_class,
        )
        self.freeze_backbone_epochs = freeze_backbone_epochs
        self.backbone_lr_factor = backbone_lr_factor

    # ---- backbone / head parameter splitting ----

    def _get_backbone_params(self) -> List[torch.nn.Parameter]:
        """Return backbone (feature extractor) parameters."""
        inner = self.net.model if hasattr(self.net, "model") else self.net
        if hasattr(inner, "features"):
            return list(inner.features.parameters())
        raise ValueError(
            "Cannot identify backbone parameters. "
            "Expected net.model.features to exist (MobileNet / EfficientNet)."
        )

    def _get_head_params(self) -> List[torch.nn.Parameter]:
        """Return classifier head parameters."""
        inner = self.net.model if hasattr(self.net, "model") else self.net
        if hasattr(inner, "classifier"):
            return list(inner.classifier.parameters())
        raise ValueError(
            "Cannot identify head parameters. "
            "Expected net.model.classifier to exist (MobileNet / EfficientNet)."
        )

    # ---- freeze / unfreeze ----

    def freeze_backbone(self) -> None:
        for p in self._get_backbone_params():
            p.requires_grad = False

    def unfreeze_backbone(self) -> None:
        for p in self._get_backbone_params():
            p.requires_grad = True

    def on_train_epoch_start(self) -> None:
        if self.freeze_backbone_epochs <= 0:
            return
        if self.current_epoch < self.freeze_backbone_epochs:
            self.freeze_backbone()
        elif self.current_epoch == self.freeze_backbone_epochs:
            self.unfreeze_backbone()

    # ---- discriminative LR ----

    def configure_optimizers(self) -> Dict[str, Any]:
        if self._optimizer is None:
            raise ValueError("Optimizer not provided.")

        if self.backbone_lr_factor >= 1.0:
            # No discriminative LR, use base implementation
            return super().configure_optimizers()

        # Extract base LR from the optimizer partial
        base_lr = self._optimizer.keywords.get("lr", 0.01)

        param_groups = [
            {
                "params": self._get_backbone_params(),
                "lr": base_lr * self.backbone_lr_factor,
                "name": "backbone",
            },
            {
                "params": self._get_head_params(),
                "lr": base_lr,
                "name": "head",
            },
        ]

        optimizer = self._optimizer(params=param_groups)

        if self._scheduler is not None:
            scheduler = self._scheduler(optimizer=optimizer)
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "monitor": "val/f1_macro",
                    "interval": "epoch",
                    "frequency": 1,
                },
            }
        return {"optimizer": optimizer}

    # ---- weight loading utilities ----

    @staticmethod
    def load_backbone_weights(
        model: "PretrainFinetuneModule",
        checkpoint_path: str,
    ) -> None:
        """Load only backbone weights from a pretrained checkpoint.

        Loads feature extractor weights while ignoring the classifier head,
        so pretrain and finetune can have different num_classes.
        """
        ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        state_dict = ckpt.get("state_dict", ckpt)

        # Filter to only backbone keys (net.model.features.*)
        backbone_keys = {
            k: v for k, v in state_dict.items() if "classifier" not in k and k.startswith("net.")
        }

        missing, unexpected = model.load_state_dict(backbone_keys, strict=False)

        # The missing keys should only be classifier + metric keys
        classifier_missing = [k for k in missing if "classifier" in k]
        other_missing = [
            k for k in missing
            if "classifier" not in k
            and not any(s in k for s in ("criterion", "loss", "metrics", "per_class", "f1_macro_best"))
        ]

        if other_missing:
            raise RuntimeError(
                f"Unexpected missing backbone keys when loading pretrained weights: {other_missing}"
            )

        n_loaded = len(backbone_keys)
        n_head_skipped = len(classifier_missing)
        print(
            f"Loaded {n_loaded} backbone parameters from {checkpoint_path} "
            f"(skipped {n_head_skipped} classifier parameters)"
        )
