from __future__ import annotations

import re
from typing import Any, Callable, Dict, List, Optional, Sequence

import torch
import torch.nn.functional as F
from lightning import LightningModule
from torchmetrics import MaxMetric, MeanMetric, MetricCollection
from torchmetrics.classification import (
    MulticlassAccuracy,
    MulticlassF1Score,
    MulticlassPrecision,
    MulticlassRecall,
)


def _chunk_params(params, sizes):
    """Split a flat parameter list into chunks of given sizes."""
    it = iter(params)
    for s in sizes:
        yield [next(it) for _ in range(s)]


class SEALIterationModule(LightningModule):
    """
    Implements SEAL (Algorithm 2):
    - Train for T epochs using current soft labels Sbar
    - After each epoch t, record Sbar^t = [f_t(x_i)]_{i=1..n}
    - After T epochs, update Sbar = (1/T) * sum_t Sbar^t

    Metrics match BaseRobustModule: macro + weighted + per-class.
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
        log_per_class: bool = True,
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
        self.datamodule = datamodule
        self.log_per_class = log_per_class

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
            self.soft_labels.copy_(initial_soft_labels.to(dtype=torch.float32))
            self._soft_labels_initialized = True

        # --- losses ---
        self.train_loss = MeanMetric()
        self.val_loss = MeanMetric()
        self.test_loss = MeanMetric()

        # --- primary metrics: macro (fair to all classes) ---
        macro_metrics = MetricCollection(
            {
                "precision_macro": MulticlassPrecision(num_classes=c, average="macro"),
                "recall_macro": MulticlassRecall(num_classes=c, average="macro"),
                "f1_macro": MulticlassF1Score(num_classes=c, average="macro"),
            }
        )

        # --- secondary context metrics: weighted + accuracy ---
        context_metrics = MetricCollection(
            {
                "acc": MulticlassAccuracy(num_classes=c),
                "precision_weighted": MulticlassPrecision(num_classes=c, average="weighted"),
                "recall_weighted": MulticlassRecall(num_classes=c, average="weighted"),
                "f1_weighted": MulticlassF1Score(num_classes=c, average="weighted"),
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

        # Per-class metrics (only val/test)
        self.val_per_class = MetricCollection(
            {
                "precision": MulticlassPrecision(num_classes=c, average=None),
                "recall": MulticlassRecall(num_classes=c, average=None),
                "f1": MulticlassF1Score(num_classes=c, average=None),
            }
        ).clone(prefix="val/per_class/")

        self.test_per_class = MetricCollection(
            {
                "precision": MulticlassPrecision(num_classes=c, average=None),
                "recall": MulticlassRecall(num_classes=c, average=None),
                "f1": MulticlassF1Score(num_classes=c, average=None),
            }
        ).clone(prefix="test/per_class/")

        # Best tracking (use macro-F1 as the primary selection metric)
        self.val_f1_macro_best = MaxMetric()

        # Store initial network state for reinitialization between iterations
        # (Algorithm 2: "Initialize a network f" at start of each iteration)
        self._initial_net_state: Optional[Dict[str, Any]] = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

    def setup(self, stage: str) -> None:
        if self._compile and stage == "fit":
            self.net = torch.compile(self.net)

    def _save_initial_state(self) -> None:
        """Snapshot network weights so we can reinitialize each SEAL iteration."""
        import copy
        self._initial_net_state = copy.deepcopy(self.net.state_dict())

    def _reinitialize_network(self) -> None:
        """Reset network to initial weights (Algorithm 2: 'Initialize a network f')."""
        if self._initial_net_state is not None:
            self.net.load_state_dict(self._initial_net_state)

    def _reset_optimizer_and_scheduler(self) -> None:
        """Reset optimizer state and rebuild LR scheduler for a new SEAL iteration.

        After reinitializing the network, the optimizer's momentum buffers and
        the scheduler's step count are stale.  We clear the optimizer state
        and construct a fresh scheduler pointing at the same optimizer so the
        learning-rate schedule restarts from the beginning.
        """
        for opt in self.trainer.optimizers:
            # Clear momentum buffers / adaptive-lr state
            opt.state.clear()
            # Re-assign param groups to pick up the reinitialized parameters
            new_params = list(self.trainer.model.parameters())
            for pg, new_p_chunk in zip(
                opt.param_groups,
                _chunk_params(new_params, [len(pg["params"]) for pg in opt.param_groups]),
            ):
                pg["params"] = list(new_p_chunk)

        # Rebuild schedulers so they restart from step 0
        if self._scheduler_ctor is not None and self.trainer.lr_scheduler_configs:
            for cfg in self.trainer.lr_scheduler_configs:
                cfg.scheduler = self._scheduler_ctor(optimizer=self.trainer.optimizers[0])

    def _get_class_names(self) -> Optional[List[str]]:
        if self.datamodule is None or not hasattr(self.datamodule, "class_names"):
            return None
        class_names = list(self.datamodule.class_names)
        if len(class_names) < self.num_classes:
            return None
        return class_names

    @staticmethod
    def _sanitize_class_name(name: str) -> str:
        name = name.strip()
        if not name:
            return "unknown"
        return re.sub(r"[^A-Za-z0-9_]+", "_", name)

    def _class_metric_key(self, prefix: str, metric: str, idx: int) -> str:
        class_names = self._get_class_names()
        if class_names is None:
            return f"{prefix}{metric}_c{idx}"
        safe_name = self._sanitize_class_name(class_names[idx])
        return f"{prefix}{metric}_{safe_name}"

    # ---------------------------
    # Initialization of soft labels
    # ---------------------------

    def on_fit_start(self) -> None:
        self.val_f1_macro_best.reset()

        # Save initial network state for reinitialization between iterations
        if self._initial_net_state is None:
            self._save_initial_state()

        if self._soft_labels_initialized:
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
            for raw_batch in dm.train_eval_dataloader():
                batch = dm.preprocess_batch(raw_batch, eval_mode=True)
                if len(batch) == 3:
                    x, noisy_y, idx = batch
                else:
                    raise ValueError("train_eval_dataloader must return (x, noisy_y, idx)")
                idx = idx.to(device, non_blocking=True)
                noisy_y = noisy_y.to(device, non_blocking=True)
                one_hot = F.one_hot(noisy_y, num_classes=self.num_classes).float()
                self.soft_labels[idx] = one_hot

        if torch.distributed.is_available() and torch.distributed.is_initialized():
            torch.distributed.all_reduce(self.soft_labels, op=torch.distributed.ReduceOp.SUM)

        self.train()
        self._soft_labels_initialized = True

    # ---------------------------
    # Training step (uses current Sbar)
    # ---------------------------
    def training_step(self, batch: Any, batch_idx: int) -> torch.Tensor:
        if len(batch) == 3:
            x, noisy_y, idx = batch
        else:
            raise ValueError("Training batch must contain (x, noisy_y, idx)")

        idx = idx.to(self.soft_labels.device, non_blocking=True)

        logits = self(x)
        logp = F.log_softmax(logits, dim=1)

        sbar = self.soft_labels[idx]  # [B, C]
        loss = -(sbar * logp).sum(dim=1).mean()

        preds = torch.argmax(logits, dim=1)

        self.train_loss(loss)
        self.train_metrics(preds, noisy_y)

        self.log("train/loss", self.train_loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log_dict(self.train_metrics, on_step=False, on_epoch=True, prog_bar=False)
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

        self.eval()
        with torch.inference_mode():
            for raw_batch in dm.train_eval_dataloader():
                batch = dm.preprocess_batch(raw_batch, eval_mode=True)
                if len(batch) == 3:
                    x, _, idx = batch
                else:
                    raise ValueError("train_eval_dataloader must return (x, noisy_y, idx)")
                x = x.to(self.device, non_blocking=True)
                idx = idx.to(device, non_blocking=True)

                logits = self(x)
                probs = F.softmax(logits, dim=1).to(device)

                self.epoch_outputs_sum[idx] += probs
                self.epoch_counts[idx] += 1

        self.train()

        if torch.distributed.is_available() and torch.distributed.is_initialized():
            torch.distributed.all_reduce(self.epoch_outputs_sum, op=torch.distributed.ReduceOp.SUM)
            torch.distributed.all_reduce(self.epoch_counts, op=torch.distributed.ReduceOp.SUM)

        self.epoch_in_iteration += 1

        # Update Sbar after T epochs
        if self.epoch_in_iteration >= self.epochs_per_iteration:
            # Paper Algorithm 2: S̄ = Σ S̄^t / T
            # Each S̄^t is a softmax output (valid simplex), so the average
            # is also a valid distribution — no clamping/re-normalization needed.
            counts = self.epoch_counts.to(torch.float32).clamp_min(1.0).unsqueeze(1)
            new_soft = self.epoch_outputs_sum / counts
            self.soft_labels.copy_(new_soft)

            with torch.no_grad():
                entropy = -torch.sum(new_soft * torch.log(new_soft + self.eps), dim=1)
                mean_entropy = entropy.mean().item()
                max_entropy = torch.log(torch.tensor(self.num_classes, dtype=torch.float32)).item()
                normalized_entropy = mean_entropy / max_entropy

                max_probs = new_soft.max(dim=1).values
                mean_max_prob = max_probs.mean().item()

                sorted_probs = torch.sort(new_soft, dim=1, descending=True).values
                second_probs = sorted_probs[:, 1]
                mean_second_prob = second_probs.mean().item()

                almost_onehot_ratio = (max_probs > 0.95).float().mean().item()
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
            else:
                # Algorithm 2: "Initialize a network f" at start of each iteration
                self._reinitialize_network()
                self._reset_optimizer_and_scheduler()

    # ---------------------------
    # Validation (clean labels)
    # ---------------------------
    def validation_step(self, batch: Any, batch_idx: int) -> None:
        x, y = batch
        logits = self(x)
        loss = F.cross_entropy(logits, y)
        preds = torch.argmax(logits, dim=1)

        self.val_loss(loss)
        self.val_metrics(preds, y)

        if self.log_per_class:
            self.val_per_class(preds, y)

        self.log("val/loss", self.val_loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log_dict(
            {k: v for k, v in self.val_metrics.items() if k != "val/f1_macro"},
            on_step=False,
            on_epoch=True,
            prog_bar=False,
        )
        self.log("val/f1_macro", self.val_metrics["val/f1_macro"], on_step=False, on_epoch=True, prog_bar=True)

    def on_validation_epoch_end(self) -> None:
        current = self.val_metrics["val/f1_macro"].compute()
        self.val_f1_macro_best(current)
        self.log("val/f1_macro_best", self.val_f1_macro_best.compute(), prog_bar=True, sync_dist=True)

        if self.log_per_class:
            pc = self.val_per_class.compute()
            for i in range(self.num_classes):
                self.log(
                    self._class_metric_key("val/", "precision", i),
                    pc["val/per_class/precision"][i],
                    sync_dist=True,
                )
                self.log(
                    self._class_metric_key("val/", "recall", i),
                    pc["val/per_class/recall"][i],
                    sync_dist=True,
                )
                self.log(
                    self._class_metric_key("val/", "f1", i),
                    pc["val/per_class/f1"][i],
                    sync_dist=True,
                )
            self.val_per_class.reset()

    # ---------------------------
    # Test (clean labels)
    # ---------------------------
    def test_step(self, batch: Any, batch_idx: int) -> None:
        x, y = batch
        logits = self(x)
        loss = F.cross_entropy(logits, y)
        preds = torch.argmax(logits, dim=1)

        self.test_loss(loss)
        self.test_metrics(preds, y)

        if self.log_per_class:
            self.test_per_class(preds, y)

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
                self.log(
                    self._class_metric_key("test/", "precision", i),
                    pc["test/per_class/precision"][i],
                    sync_dist=True,
                )
                self.log(
                    self._class_metric_key("test/", "recall", i),
                    pc["test/per_class/recall"][i],
                    sync_dist=True,
                )
                self.log(
                    self._class_metric_key("test/", "f1", i),
                    pc["test/per_class/f1"][i],
                    sync_dist=True,
                )
            self.test_per_class.reset()

    def configure_optimizers(self) -> Dict[str, Any]:
        if self._optimizer_ctor is None:
            raise ValueError("optimizer must be provided.")

        opt = self._optimizer_ctor(params=self.trainer.model.parameters())

        if self._scheduler_ctor is None:
            return {"optimizer": opt}

        sch = self._scheduler_ctor(optimizer=opt)
        return {
            "optimizer": opt,
            "lr_scheduler": {
                "scheduler": sch,
                "monitor": "val/f1_macro",
                "interval": "epoch",
                "frequency": 1,
            },
        }
