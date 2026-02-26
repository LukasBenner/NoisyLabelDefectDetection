from __future__ import annotations

from typing import Any

from torchvision.transforms import v2
import torch
from src.models.base_robust_module import BaseRobustModule


class CutMixModule(BaseRobustModule):
    def __init__(
        self,
        *args,
        mixup_alpha: float = 1.0,
        cutmix_alpha: float = 1.0,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.save_hyperparameters(logger=False, ignore=["net", "optimizer", "scheduler", "criterion", "datamodule"])
        self.hparams: Any
        
        
        cutmix = v2.CutMix(alpha=self.hparams.cutmix_alpha, num_classes=self.hparams.num_classes)
        mixup = v2.MixUp(alpha=self.hparams.mixup_alpha, num_classes=self.hparams.num_classes)
        self.cutmix_or_mixup = v2.RandomChoice([cutmix, mixup])


    def _mixup_batch(self,
        inputs: torch.Tensor,
        targets: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        
        mixed_inputs, mixed_targets = self.cutmix_or_mixup(inputs, targets)
        return mixed_inputs, mixed_targets

    def training_step(self, batch: Any, batch_idx: int) -> torch.Tensor:
        inputs, targets = batch

        mixed_inputs, mixed_targets = self._mixup_batch(
            inputs, targets
        )

        logits = self.forward(mixed_inputs)
        loss = self.compute_loss(logits, mixed_targets)

        preds = torch.argmax(logits, dim=1)

        self.train_loss(loss)
        self.train_metrics(preds, mixed_targets)

        self.log("train/loss", self.train_loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log_dict(self.train_metrics, on_step=False, on_epoch=True, prog_bar=False)

        return loss
