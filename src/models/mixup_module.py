from __future__ import annotations

from typing import Any

import torch
from src.models.base_robust_module import BaseRobustModule


class MixupModule(BaseRobustModule):
    def __init__(
        self,
        *args,
        mixup_alpha: float = 0.2,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.save_hyperparameters(logger=False, ignore=["net", "optimizer", "scheduler", "criterion", "datamodule"])
        self.mixup_alpha = float(mixup_alpha)

    @staticmethod
    def _mixup_batch(
        inputs: torch.Tensor,
        targets: torch.Tensor,
        alpha: float,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, float]:
        if alpha <= 0:
            return inputs, targets, targets, 1.0
        lam = float(torch.distributions.Beta(alpha, alpha).sample().item())
        batch_size = inputs.size(0)
        index = torch.randperm(batch_size, device=inputs.device)
        mixed_inputs = lam * inputs + (1.0 - lam) * inputs[index]
        targets_a = targets
        targets_b = targets[index]
        return mixed_inputs, targets_a, targets_b, lam

    def _mixup_criterion(
        self,
        logits: torch.Tensor,
        targets_a: torch.Tensor,
        targets_b: torch.Tensor,
        lam: float,
    ) -> torch.Tensor:
        return lam * self.criterion(logits, targets_a) + (1.0 - lam) * self.criterion(
            logits, targets_b
        )

    def training_step(self, batch: Any, batch_idx: int) -> torch.Tensor:
        inputs, targets = batch

        mixed_inputs, targets_a, targets_b, lam = self._mixup_batch(
            inputs, targets, self.mixup_alpha
        )

        logits = self.forward(mixed_inputs)
        if lam < 1.0:
            loss = self._mixup_criterion(logits, targets_a, targets_b, lam)
        else:
            loss = self.compute_loss(logits, targets)

        preds = torch.argmax(logits, dim=1)

        self.train_loss(loss)
        self.train_metrics(preds, targets)

        self.log("train/loss", self.train_loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log_dict(self.train_metrics, on_step=False, on_epoch=True, prog_bar=False)

        return loss
