from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, Optional

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


@dataclass
class SEALConfig:
    num_classes: int
    epochs_per_iteration: int  # T in Algorithm 2
    num_train_samples: int     # n
    # If you want to stop after one iteration in a single fit call:
    stop_after_iteration: bool = True
    # Numerical stability:
    eps: float = 1e-12


class SEALIterationModule(LightningModule):
    """
    Implements ONE SEAL iteration (Algorithm 2):
    - Train for T epochs using current soft labels Sbar
    - After each epoch t, record Sbar^t = [f_t(x_i)]_{i=1..n}
    - After T epochs, update Sbar = (1/T) * sum_t Sbar^t

    Paper: Algorithm 2. :contentReference[oaicite:4]{index=4}
    """

    def __init__(
        self,
        net: torch.nn.Module,
        num_classes: Optional[int] = None,
        epochs_per_iteration: int = 10,  # T
        num_iterations: int = 3,
        num_train_samples: Optional[int] = None,  # n
        datamodule: Optional[Any] = None,
        optimizer: Optional[Callable[[Any], torch.optim.Optimizer]] = None,
        scheduler: Optional[Callable[[torch.optim.Optimizer], torch.optim.lr_scheduler._LRScheduler]] = None,
        initial_soft_labels: Optional[torch.Tensor] = None,  # [n, c] on CPU OK
        eps: float = 1e-12,
        compile: bool = False,
    ) -> None:
        super().__init__()
        self.save_hyperparameters(
            logger=False,
            ignore=["net", "optimizer", "scheduler", "datamodule", "initial_soft_labels"],
        )

        self.net = net
        self._optimizer_ctor = optimizer
        self._scheduler_ctor = scheduler
        self._compile = compile

        # Infer n and c from datamodule if not provided
        if datamodule is not None:
            if num_train_samples is None and hasattr(datamodule, "num_train_samples"):
                num_train_samples = int(datamodule.num_train_samples)
            if num_classes is None and hasattr(datamodule, "num_classes"):
                num_classes = int(datamodule.num_classes)

        if num_train_samples is None:
            raise ValueError("num_train_samples must be provided or inferable from datamodule.num_train_samples")
        if num_classes is None:
            raise ValueError("num_classes must be provided or inferable from datamodule.num_classes")

        self.num_train_samples = num_train_samples
        self.num_classes = num_classes
        self.epochs_per_iteration = epochs_per_iteration
        self.num_iterations = num_iterations
        self.eps = float(eps)
        self.current_iteration = 0

        n, c = self.num_train_samples, self.num_classes

        # Buffers (kept on module device)
        self.register_buffer("soft_labels", torch.empty(n, c, dtype=torch.float32), persistent=False)
        self.register_buffer("epoch_outputs_sum", torch.zeros(n, c, dtype=torch.float32), persistent=False)
        self.register_buffer("epoch_counts", torch.zeros(n, dtype=torch.int32), persistent=False)

        self.epoch_in_iteration = 0
        self._soft_labels_initialized = False
        
        # Optionally load initial soft labels (from previous iteration)
        if initial_soft_labels is not None:
            if initial_soft_labels.shape != (n, c):
                raise ValueError(f"initial_soft_labels must have shape {(n, c)}, got {tuple(initial_soft_labels.shape)}")
            # Copy onto buffer (buffer will be moved by Lightning to correct device)
            self.soft_labels.copy_(initial_soft_labels.to(dtype=torch.float32))
            self._soft_labels_initialized = True

        # Metrics (lightweight set; add more if needed)
        self.train_loss = MeanMetric()
        self.val_loss = MeanMetric()
        self.test_loss = MeanMetric()

        self.train_acc = MulticlassAccuracy(num_classes=c, average="weighted")
        self.val_acc = MulticlassAccuracy(num_classes=c, average="weighted")
        self.test_acc = MulticlassAccuracy(num_classes=c, average="weighted")

        self.train_f1 = MulticlassF1Score(num_classes=c, average="weighted")
        self.val_f1 = MulticlassF1Score(num_classes=c, average="weighted")
        self.test_f1 = MulticlassF1Score(num_classes=c, average="weighted")

        self.val_acc_best = MaxMetric()
        self.val_f1_best = MaxMetric()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)
    
    def setup(self, stage: str) -> None:
        if self._compile and stage == "fit":
            self.net = torch.compile(self.net)

    # ---------------------------
    # Initialization of soft labels
    # ---------------------------
    
    def on_fit_start(self) -> None:
        self.val_acc_best.reset()
        self.val_f1_best.reset()

        if self._soft_labels_initialized:
            # If we start from existing soft labels, nothing to do.
            return

        dm = self.trainer.datamodule
        if dm is None or not hasattr(dm, "train_eval_dataloader"):
            raise ValueError(
                "Datamodule must provide train_eval_dataloader() returning (x, noisy_y, idx) "
                "for SEAL initialization."
            )

        self.soft_labels.zero_()
        device = self.soft_labels.device

        self.eval()
        with torch.inference_mode():
            for x, noisy_y, idx in dm.train_eval_dataloader():
                idx = idx.to(device, non_blocking=True)
                noisy_y = noisy_y.to(device, non_blocking=True)
                one_hot = F.one_hot(noisy_y, num_classes=self.num_classes).float()
                self.soft_labels[idx] = one_hot

        # If distributed, synchronize the initialized buffer across ranks.
        # Each rank may have only initialized its shard; all-reduce SUM gives the full one-hot matrix.
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            torch.distributed.all_reduce(self.soft_labels, op=torch.distributed.ReduceOp.SUM)

        self.train()
        self._soft_labels_initialized = True
    
    # ---------------------------
    # Training step (uses current Sbar)
    # ---------------------------
    def training_step(self, batch: Any, batch_idx: int) -> torch.Tensor:
        x, noisy_y, idx = batch

        # Ensure idx indexes the buffer device
        idx = idx.to(self.soft_labels.device, non_blocking=True)

        logits = self(x)
        logp = F.log_softmax(logits, dim=1)

        sbar = self.soft_labels[idx]  # [B, C]
        loss = -(sbar * logp).sum(dim=1).mean()

        preds = torch.argmax(logits, dim=1)

        # Metrics vs noisy labels (informative but may be noisy)
        self.train_loss(loss)
        self.train_acc(preds, noisy_y)
        self.train_f1(preds, noisy_y)

        self.log("train/loss", self.train_loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("train/acc", self.train_acc, on_step=False, on_epoch=True, prog_bar=True)
        self.log("train/f1", self.train_f1, on_step=False, on_epoch=True)
        self.log("seal/epoch_in_iteration", self.epoch_in_iteration, on_step=False, on_epoch=True)

        return loss

    def on_train_epoch_end(self) -> None:
        dm = self.trainer.datamodule
        if dm is None or not hasattr(dm, "train_eval_dataloader"):
            raise ValueError(
                "Datamodule must provide train_eval_dataloader() returning (x, noisy_y, idx) "
                "for SEAL epoch-end recording."
            )

        device = self.epoch_outputs_sum.device

        # Record epoch outputs Sbar^t for all samples (or all shards in DDP)
        self.eval()
        with torch.inference_mode():
            for x, _, idx in dm.train_eval_dataloader():
                x = x.to(self.device, non_blocking=True)
                idx = idx.to(device, non_blocking=True)

                logits = self(x)
                probs = F.softmax(logits, dim=1).to(device)

                self.epoch_outputs_sum[idx] += probs
                self.epoch_counts[idx] += 1

        self.train()

        # DDP sync: sums and counts must be global before updating soft labels
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            torch.distributed.all_reduce(self.epoch_outputs_sum, op=torch.distributed.ReduceOp.SUM)
            torch.distributed.all_reduce(self.epoch_counts, op=torch.distributed.ReduceOp.SUM)

        self.epoch_in_iteration += 1

        # Update Sbar after T epochs
        if self.epoch_in_iteration >= self.epochs_per_iteration:
            counts = self.epoch_counts.to(torch.float32).clamp_min(1.0).unsqueeze(1)
            new_soft = (self.epoch_outputs_sum / counts).clamp_min(self.eps)
            new_soft = new_soft / new_soft.sum(dim=1, keepdim=True).clamp_min(self.eps)
            self.soft_labels.copy_(new_soft)

            # Log soft label statistics to verify they're actually soft
            with torch.no_grad():
                # Entropy: higher = more uniform/soft, lower = more peaked/one-hot
                # Max entropy is log(num_classes), min is 0
                entropy = -torch.sum(new_soft * torch.log(new_soft + self.eps), dim=1)
                mean_entropy = entropy.mean().item()
                max_entropy = torch.log(torch.tensor(self.num_classes, dtype=torch.float32)).item()
                normalized_entropy = mean_entropy / max_entropy  # 0 to 1, where 1 = uniform
                
                # Max probability: lower = more soft, higher = more peaked
                max_probs = new_soft.max(dim=1).values
                mean_max_prob = max_probs.mean().item()
                
                # Second highest probability: higher means more uncertainty
                sorted_probs = torch.sort(new_soft, dim=1, descending=True).values
                second_probs = sorted_probs[:, 1]
                mean_second_prob = second_probs.mean().item()
                
                # Check how many are "almost one-hot" (max_prob > 0.95)
                almost_onehot_ratio = (max_probs > 0.95).float().mean().item()
                
                # Check how many are "very soft" (max_prob < 0.6)
                very_soft_ratio = (max_probs < 0.6).float().mean().item()

            self.log("seal/soft_label_entropy", mean_entropy, prog_bar=False)
            self.log("seal/soft_label_entropy_normalized", normalized_entropy, prog_bar=True)
            self.log("seal/soft_label_max_prob", mean_max_prob, prog_bar=True)
            self.log("seal/soft_label_second_prob", mean_second_prob, prog_bar=False)
            self.log("seal/almost_onehot_ratio", almost_onehot_ratio, prog_bar=False)
            self.log("seal/very_soft_ratio", very_soft_ratio, prog_bar=False)

            # Reset accumulators
            self.epoch_outputs_sum.zero_()
            self.epoch_counts.zero_()
            self.epoch_in_iteration = 0
            
            self.current_iteration += 1

            if self.current_iteration >= self.num_iterations:
                self.trainer.should_stop = True

    # ---------------------------
    # Validation/Test (hard labels)
    # ---------------------------
    def validation_step(self, batch: Any, batch_idx: int) -> None:
        x, y = batch
        logits = self(x)
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
        logits = self(x)
        loss = F.cross_entropy(logits, y)
        preds = torch.argmax(logits, dim=1)

        self.test_loss(loss)
        self.test_acc(preds, y)
        self.test_f1(preds, y)

        self.log("test/loss", self.test_loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("test/acc", self.test_acc, on_step=False, on_epoch=True, prog_bar=True)
        self.log("test/f1", self.test_f1, on_step=False, on_epoch=True)

    def configure_optimizers(self) -> Dict[str, Any]:
        if self._optimizer_ctor is None:
            raise ValueError("optimizer_ctor must be provided (callable that returns an optimizer).")

        opt = self._optimizer_ctor(self.parameters())

        if self._scheduler_ctor is None:
            return {"optimizer": opt}

        sch = self._scheduler_ctor(opt)
        return {"optimizer": opt, "lr_scheduler": {"scheduler": sch, "interval": "epoch"}}