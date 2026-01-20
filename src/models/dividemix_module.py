"""LightningModule for DivideMix training."""

from __future__ import annotations

from typing import Any, Callable, Optional

import copy

import numpy as np
import torch
import torch.nn.functional as F
from lightning import LightningModule
from torchmetrics import MaxMetric, MeanMetric
from torchmetrics.classification import MulticlassAccuracy, MulticlassF1Score


def _soft_cross_entropy(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    logp = F.log_softmax(logits, dim=1)
    return -(targets * logp).sum(dim=1).mean()


def _sharpen(probs: torch.Tensor, temperature: float) -> torch.Tensor:
    if temperature <= 0:
        return probs
    p = probs ** (1.0 / temperature)
    return p / p.sum(dim=1, keepdim=True).clamp_min(1e-12)


def _linear_rampup(current: float, rampup_length: int) -> float:
    if rampup_length <= 0:
        return 1.0
    current = np.clip(current, 0, rampup_length)
    return float(current) / float(rampup_length)


def _normal_pdf(x: torch.Tensor, mean: torch.Tensor, var: torch.Tensor) -> torch.Tensor:
    return torch.exp(-0.5 * (x - mean) ** 2 / var.clamp_min(1e-6)) / torch.sqrt(
        2.0 * torch.pi * var.clamp_min(1e-6)
    )


class DivideMixModule(LightningModule):
    """DivideMix with two networks and MixMatch-style semi-supervised learning."""

    def __init__(
        self,
        net1: torch.nn.Module,
        net2: Optional[torch.nn.Module] = None,
        num_classes: int = 10,
        optimizer: Optional[Callable[[Any], torch.optim.Optimizer]] = None,
        scheduler: Optional[Callable[[torch.optim.Optimizer], torch.optim.lr_scheduler._LRScheduler]] = None,
        warmup_epochs: int = 10,
        p_threshold: float = 0.9,
        lambda_u: float = 10.0,
        temperature: float = 0.5,
        mixup_alpha: float = 4.0,
        rampup_length: int = 16,
        gmm_iters: int = 10,
        compile: bool = False,
        datamodule: Optional[Any] = None,
    ) -> None:
        super().__init__()

        self.save_hyperparameters(
            logger=False,
            ignore=["net1", "net2", "optimizer", "scheduler", "datamodule"],
        )

        self.net1 = net1
        self.net2 = net2 if net2 is not None else copy.deepcopy(net1)

        self._optimizer_ctor = optimizer
        self._scheduler_ctor = scheduler
        self._compile = compile

        self.num_classes = int(num_classes)
        self.warmup_epochs = int(warmup_epochs)
        self.p_threshold = float(p_threshold)
        self.lambda_u = float(lambda_u)
        self.temperature = float(temperature)
        self.mixup_alpha = float(mixup_alpha)
        self.rampup_length = int(rampup_length)
        self.gmm_iters = int(gmm_iters)

        if datamodule is None or not hasattr(datamodule, "num_train_samples"):
            raise ValueError("datamodule with num_train_samples is required for DivideMix.")

        self.num_train_samples: int = int(datamodule.num_train_samples)

        self.register_buffer("clean_prob1", torch.ones(self.num_train_samples), persistent=False)
        self.register_buffer("clean_prob2", torch.ones(self.num_train_samples), persistent=False)

        self.automatic_optimization = False

        self.train_loss = MeanMetric()
        self.val_loss = MeanMetric()
        self.test_loss = MeanMetric()

        self.train_acc = MulticlassAccuracy(num_classes=self.num_classes, average="weighted")
        self.val_acc = MulticlassAccuracy(num_classes=self.num_classes, average="weighted")
        self.test_acc = MulticlassAccuracy(num_classes=self.num_classes, average="weighted")

        self.train_f1 = MulticlassF1Score(num_classes=self.num_classes, average="weighted")
        self.val_f1 = MulticlassF1Score(num_classes=self.num_classes, average="weighted")
        self.test_f1 = MulticlassF1Score(num_classes=self.num_classes, average="weighted")

        self.val_acc_best = MaxMetric()
        self.val_f1_best = MaxMetric()

    def setup(self, stage: str) -> None:
        if self._compile and stage == "fit":
            self.net1 = torch.compile(self.net1)
            self.net2 = torch.compile(self.net2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        logits1 = self.net1(x)
        logits2 = self.net2(x)
        return 0.5 * (logits1 + logits2)

    def configure_optimizers(self) -> Any:
        if self._optimizer_ctor is None:
            raise ValueError("optimizer must be provided")

        opt1 = self._optimizer_ctor(self.net1.parameters())
        opt2 = self._optimizer_ctor(self.net2.parameters())

        if self._scheduler_ctor is None:
            return [opt1, opt2]

        sch1 = self._scheduler_ctor(opt1)
        sch2 = self._scheduler_ctor(opt2)
        return [
            {"optimizer": opt1, "lr_scheduler": {"scheduler": sch1, "interval": "epoch"}},
            {"optimizer": opt2, "lr_scheduler": {"scheduler": sch2, "interval": "epoch"}},
        ]

    def on_train_start(self) -> None:
        self.val_acc_best.reset()
        self.val_f1_best.reset()

    def on_train_epoch_start(self) -> None:
        if self.current_epoch < self.warmup_epochs:
            self.clean_prob1 = torch.ones_like(self.clean_prob1)
            self.clean_prob2 = torch.ones_like(self.clean_prob2)
            return

        self._estimate_clean_probabilities()

    def _estimate_clean_probabilities(self) -> None:
        dm = getattr(self.trainer, "datamodule", None)
        if dm is None or not hasattr(dm, "train_eval_dataloader"):
            raise ValueError("Datamodule must provide train_eval_dataloader() for DivideMix.")

        losses1 = torch.zeros(self.num_train_samples, device=self.device)
        losses2 = torch.zeros(self.num_train_samples, device=self.device)

        self.net1.eval()
        self.net2.eval()

        with torch.inference_mode():
            for x, y, idx in dm.train_eval_dataloader():
                x = x.to(self.device, non_blocking=True)
                y = y.to(self.device, non_blocking=True)
                idx = idx.to(self.device, non_blocking=True)

                logits1 = self.net1(x)
                logits2 = self.net2(x)

                loss1 = F.cross_entropy(logits1, y, reduction="none")
                loss2 = F.cross_entropy(logits2, y, reduction="none")

                losses1[idx] = loss1
                losses2[idx] = loss2

        self.net1.train()
        self.net2.train()

        prob1 = self._gmm_posterior(losses1)
        prob2 = self._gmm_posterior(losses2)

        self.clean_prob1 = prob1.detach()
        self.clean_prob2 = prob2.detach()

        self.log("dividemix/clean_ratio_net1", prob1.mean(), on_epoch=True, prog_bar=False)
        self.log("dividemix/clean_ratio_net2", prob2.mean(), on_epoch=True, prog_bar=False)

    def _gmm_posterior(self, losses: torch.Tensor) -> torch.Tensor:
        eps = 1e-6
        x = (losses - losses.min()) / (losses.max() - losses.min() + eps)
        x = x.clamp(0.0, 1.0)

        mu1 = torch.quantile(x, 0.3)
        mu2 = torch.quantile(x, 0.7)
        var1 = x.var() + eps
        var2 = x.var() + eps
        pi1 = torch.tensor(0.5, device=x.device)
        pi2 = torch.tensor(0.5, device=x.device)

        for _ in range(self.gmm_iters):
            p1 = pi1 * _normal_pdf(x, mu1, var1)
            p2 = pi2 * _normal_pdf(x, mu2, var2)
            denom = (p1 + p2).clamp_min(eps)
            gamma1 = p1 / denom
            gamma2 = p2 / denom

            n1 = gamma1.sum().clamp_min(eps)
            n2 = gamma2.sum().clamp_min(eps)

            mu1 = (gamma1 * x).sum() / n1
            mu2 = (gamma2 * x).sum() / n2

            var1 = (gamma1 * (x - mu1) ** 2).sum() / n1 + eps
            var2 = (gamma2 * (x - mu2) ** 2).sum() / n2 + eps

            pi1 = n1 / x.numel()
            pi2 = n2 / x.numel()

        if mu1 <= mu2:
            return gamma1.detach()
        return gamma2.detach()

    def _mixmatch_loss(
        self,
        net: Callable[[torch.Tensor], torch.Tensor],
        other_net: Callable[[torch.Tensor], torch.Tensor],
        x1: torch.Tensor,
        x2: torch.Tensor,
        y: torch.Tensor,
        idx: torch.Tensor,
        clean_prob_other: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, float]:
        device = x1.device
        clean_prob = clean_prob_other[idx].clamp(0.0, 1.0)
        labeled_mask = clean_prob >= self.p_threshold
        unlabeled_mask = ~labeled_mask

        n_labeled = int(labeled_mask.sum().item())
        labeled_ratio = float(n_labeled / max(1, x1.size(0)))

        if n_labeled > 0:
            x_l1 = x1[labeled_mask]
            x_l2 = x2[labeled_mask]
            y_l = y[labeled_mask]
            y_l = F.one_hot(y_l, num_classes=self.num_classes).float()
            w_x = clean_prob[labeled_mask].unsqueeze(1)
        else:
            x_l1 = torch.empty((0,) + x1.shape[1:], device=device)
            x_l2 = torch.empty((0,) + x1.shape[1:], device=device)
            y_l = torch.empty((0, self.num_classes), device=device)
            w_x = torch.empty((0, 1), device=device)

        if unlabeled_mask.any():
            x_u1 = x1[unlabeled_mask]
            x_u2 = x2[unlabeled_mask]
            with torch.inference_mode():
                pu = (
                    F.softmax(net(x_u1), dim=1)
                    + F.softmax(net(x_u2), dim=1)
                    + F.softmax(other_net(x_u1), dim=1)
                    + F.softmax(other_net(x_u2), dim=1)
                ) / 4.0
                targets_u = _sharpen(pu, self.temperature)
        else:
            x_u1 = torch.empty((0,) + x1.shape[1:], device=device)
            x_u2 = torch.empty((0,) + x1.shape[1:], device=device)
            targets_u = torch.empty((0, self.num_classes), device=device)

        if n_labeled > 0:
            with torch.inference_mode():
                px = (F.softmax(net(x_l1), dim=1) + F.softmax(net(x_l2), dim=1)) / 2.0
                px = w_x * y_l + (1.0 - w_x) * px
                targets_x = _sharpen(px, self.temperature)
        else:
            targets_x = torch.empty((0, self.num_classes), device=device)

        inputs = torch.cat([x_l1, x_l2, x_u1, x_u2], dim=0)
        targets = torch.cat([targets_x, targets_x, targets_u, targets_u], dim=0)

        if inputs.size(0) == 0:
            zero = torch.tensor(0.0, device=device)
            return zero, zero, zero, labeled_ratio

        l = np.random.beta(self.mixup_alpha, self.mixup_alpha)
        l = max(l, 1 - l)

        perm = torch.randperm(inputs.size(0), device=device)
        inputs_b = inputs[perm]
        targets_b = targets[perm]

        mixed_inputs = l * inputs + (1 - l) * inputs_b
        mixed_targets = l * targets + (1 - l) * targets_b

        logits = net(mixed_inputs)

        if n_labeled > 0:
            lx = _soft_cross_entropy(logits[: n_labeled * 2], mixed_targets[: n_labeled * 2])
        else:
            lx = torch.tensor(0.0, device=device)

        if logits.size(0) > n_labeled * 2:
            probs_u = F.softmax(logits[n_labeled * 2 :], dim=1)
            lu = torch.mean((probs_u - mixed_targets[n_labeled * 2 :]) ** 2)
        else:
            lu = torch.tensor(0.0, device=device)

        prior = torch.full((self.num_classes,), 1.0 / self.num_classes, device=device)
        pred_mean = F.softmax(logits, dim=1).mean(0)
        penalty = torch.sum(prior * torch.log(prior / pred_mean.clamp_min(1e-12)))

        return lx, lu, penalty, labeled_ratio

    def training_step(self, batch: Any, batch_idx: int) -> torch.Tensor:
        x1, x2, y, idx = batch
        x1 = x1.to(self.device, non_blocking=True)
        x2 = x2.to(self.device, non_blocking=True)
        y = y.to(self.device, non_blocking=True)
        idx = idx.to(self.device, non_blocking=True)

        optimizers = self.optimizers()
        if isinstance(optimizers, (list, tuple)):
            opt1, opt2 = optimizers
        else:
            opt1 = optimizers
            opt2 = optimizers

        if self.current_epoch < self.warmup_epochs:
            logits1 = self.net1(x1)
            logits2 = self.net2(x1)
            loss1 = F.cross_entropy(logits1, y)
            loss2 = F.cross_entropy(logits2, y)

            opt1.zero_grad()
            self.manual_backward(loss1)
            opt1.step()

            opt2.zero_grad()
            self.manual_backward(loss2)
            opt2.step()

            total_loss = loss1 + loss2
            self.train_loss(total_loss)

            preds = torch.argmax(0.5 * (logits1 + logits2), dim=1)
            self.train_acc(preds, y)
            self.train_f1(preds, y)

            self.log("train/loss", self.train_loss, on_step=False, on_epoch=True, prog_bar=True)
            self.log("train/acc", self.train_acc, on_step=False, on_epoch=True, prog_bar=True)
            self.log("train/f1", self.train_f1, on_step=False, on_epoch=True)
            self.log("dividemix/warmup", 1.0, on_step=False, on_epoch=True)

            return total_loss

        num_iter = max(1, int(getattr(self.trainer, "num_training_batches", 1)))
        epoch_progress = self.current_epoch + batch_idx / num_iter
        ramp = _linear_rampup(epoch_progress - self.warmup_epochs, self.rampup_length)
        w_u = self.lambda_u * ramp

        lx1, lu1, pen1, ratio1 = self._mixmatch_loss(
            self.net1, self.net2, x1, x2, y, idx, self.clean_prob2
        )
        lx2, lu2, pen2, ratio2 = self._mixmatch_loss(
            self.net2, self.net1, x1, x2, y, idx, self.clean_prob1
        )

        loss1 = lx1 + w_u * lu1 + pen1
        loss2 = lx2 + w_u * lu2 + pen2

        opt1.zero_grad()
        self.manual_backward(loss1)
        opt1.step()

        opt2.zero_grad()
        self.manual_backward(loss2)
        opt2.step()

        total_loss = loss1 + loss2
        self.train_loss(total_loss)

        with torch.inference_mode():
            logits = 0.5 * (self.net1(x1) + self.net2(x1))
            preds = torch.argmax(logits, dim=1)

        self.train_acc(preds, y)
        self.train_f1(preds, y)

        self.log("train/loss", self.train_loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("train/acc", self.train_acc, on_step=False, on_epoch=True, prog_bar=True)
        self.log("train/f1", self.train_f1, on_step=False, on_epoch=True)
        self.log("dividemix/labeled_ratio_net1", ratio1, on_step=False, on_epoch=True)
        self.log("dividemix/labeled_ratio_net2", ratio2, on_step=False, on_epoch=True)
        self.log("dividemix/unsup_weight", w_u, on_step=False, on_epoch=True)

        return total_loss

    def validation_step(self, batch: Any, batch_idx: int) -> None:
        x, y = batch
        x = x.to(self.device, non_blocking=True)
        y = y.to(self.device, non_blocking=True)

        logits = self.forward(x)
        loss = F.cross_entropy(logits, y)
        preds = torch.argmax(logits, dim=1)

        self.val_loss(loss)
        self.val_acc(preds, y)
        self.val_f1(preds, y)

        self.log("val/loss", self.val_loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("val/acc", self.val_acc, on_step=False, on_epoch=True, prog_bar=True)
        self.log("val/f1", self.val_f1, on_step=False, on_epoch=True)

    def on_validation_epoch_end(self) -> None:
        acc = self.val_acc.compute()
        f1 = self.val_f1.compute()

        self.val_acc_best(acc)
        self.val_f1_best(f1)

        self.log("val/acc_best", self.val_acc_best.compute(), prog_bar=True, sync_dist=True)
        self.log("val/f1_best", self.val_f1_best.compute(), sync_dist=True)

    def test_step(self, batch: Any, batch_idx: int) -> None:
        x, y = batch
        x = x.to(self.device, non_blocking=True)
        y = y.to(self.device, non_blocking=True)

        logits = self.forward(x)
        loss = F.cross_entropy(logits, y)
        preds = torch.argmax(logits, dim=1)

        self.test_loss(loss)
        self.test_acc(preds, y)
        self.test_f1(preds, y)

        self.log("test/loss", self.test_loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("test/acc", self.test_acc, on_step=False, on_epoch=True, prog_bar=True)
        self.log("test/f1", self.test_f1, on_step=False, on_epoch=True)
