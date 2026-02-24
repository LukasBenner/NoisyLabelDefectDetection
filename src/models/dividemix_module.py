from __future__ import annotations

from typing import Any, Dict, Optional, List
import re

from lightning import LightningModule
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from sklearn.mixture import GaussianMixture
from torchmetrics import MaxMetric, MeanMetric, MetricCollection
from torchmetrics.classification import (
    MulticlassAccuracy,
    MulticlassF1Score,
    MulticlassPrecision,
    MulticlassRecall,
)


class DivideMixModule(LightningModule):
    """Lightning implementation of DivideMix for ImageFolder-style datasets."""

    def __init__(
        self,
        num_classes: int,
        lr: float = 0.01,
        sgd_momentum: float = 0.9,
        weight_decay: float = 5e-4,
        noise_mode: str = "sym",
        alpha: float = 0.5,
        lambda_u: float = 0.1,
        p_threshold: float = 0.5,
        T: float = 0.5,
        warm_up: int = 10,
        arch: str = "resnet50",
        seed: int = 42,
        datamodule: Optional[Any] = None,
        log_per_class: bool = True,
        compile: bool = False,
    ) -> None:
        super().__init__()
        self.save_hyperparameters(logger=False, ignore=["datamodule"])

        self.num_classes = num_classes
        self.datamodule = datamodule
        self.log_per_class = log_per_class
        self._compile = compile

        self.model1 = self._create_model(arch, num_classes)
        self.model2 = self._create_model(arch, num_classes)

        self.automatic_optimization = False

        self.ce_loss = nn.CrossEntropyLoss()
        self.ce_loss_none = nn.CrossEntropyLoss(reduction="none")

        self.train_loss = MeanMetric()
        self.val_loss = MeanMetric()
        self.test_loss = MeanMetric()

        macro_metrics = MetricCollection(
            {
                "precision_macro": MulticlassPrecision(num_classes=num_classes, average="macro"),
                "recall_macro": MulticlassRecall(num_classes=num_classes, average="macro"),
                "f1_macro": MulticlassF1Score(num_classes=num_classes, average="macro"),
            }
        )

        context_metrics = MetricCollection(
            {
                "acc": MulticlassAccuracy(num_classes=num_classes),
                "precision_weighted": MulticlassPrecision(num_classes=num_classes, average="weighted"),
                "recall_weighted": MulticlassRecall(num_classes=num_classes, average="weighted"),
                "f1_weighted": MulticlassF1Score(num_classes=num_classes, average="weighted"),
            }
        )

        combined_metrics = MetricCollection(
            {
                **dict(macro_metrics.items()),
                **dict(context_metrics.items()),
            }
        )

        self.train_metrics = combined_metrics.clone(prefix="train/")
        self.val_metrics = combined_metrics.clone(prefix="val/")
        self.test_metrics = combined_metrics.clone(prefix="test/")

        self.val_per_class = MetricCollection(
            {
                "precision": MulticlassPrecision(num_classes=num_classes, average=None),
                "recall": MulticlassRecall(num_classes=num_classes, average=None),
                "f1": MulticlassF1Score(num_classes=num_classes, average=None),
            }
        ).clone(prefix="val/per_class/")

        self.test_per_class = MetricCollection(
            {
                "precision": MulticlassPrecision(num_classes=num_classes, average=None),
                "recall": MulticlassRecall(num_classes=num_classes, average=None),
                "f1": MulticlassF1Score(num_classes=num_classes, average=None),
            }
        ).clone(prefix="test/per_class/")

        self.val_f1_macro_best = MaxMetric()

        self._rng: Optional[np.random.Generator] = None

    @staticmethod
    def _create_model(arch: str, num_classes: int) -> nn.Module:
        if arch == "resnet18":
            return models.resnet18(weights=None, num_classes=num_classes)
        if arch == "resnet50":
            return models.resnet50(weights=None, num_classes=num_classes)
        if arch == "mobilenet_small":
            from src.models.components.mobile_net import MobileNet

            return MobileNet(num_classes=num_classes, pretrained=True, variant="small")
        raise ValueError(f"Unsupported arch: {arch}")

    def setup(self, stage: str) -> None:
        if self._compile and stage == "fit":
            self.model1 = torch.compile(self.model1)
            self.model2 = torch.compile(self.model2)

    def on_fit_start(self) -> None:
        base_seed = int(self.hparams.get("seed", 42))
        rank = int(getattr(self, "global_rank", 0))
        self._rng = np.random.default_rng(base_seed + rank)
        if self.datamodule is not None:
            self.datamodule.set_train_mode("warmup")

    def on_train_start(self) -> None:
        self.val_f1_macro_best.reset()

    def on_train_epoch_end(self) -> None:
        if self.trainer is None or self.datamodule is None:
            return
        if getattr(self.trainer, "sanity_checking", False):
            return

        next_epoch = self.current_epoch + 1
        if next_epoch < int(self.hparams.warm_up):
            self.datamodule.set_train_mode("warmup")
        else:
            self.datamodule.set_train_mode("train")
            prob1 = self._eval_train(self.model1)
            prob2 = self._eval_train(self.model2)
            pred1 = prob1 > float(self.hparams.p_threshold)
            pred2 = prob2 > float(self.hparams.p_threshold)
            self.datamodule.set_pred_prob(pred1, prob1, pred2, prob2)

        schedulers = self.lr_schedulers()
        if not schedulers:
            return

        if not isinstance(schedulers, (list, tuple)):
            schedulers = [schedulers]

        for scheduler in schedulers:
            scheduler.step()

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

    @staticmethod
    def _linear_rampup(current: float, warm_up: int, rampup_length: int = 16) -> float:
        current = np.clip((current - warm_up) / rampup_length, 0.0, 1.0)
        return float(current)

    def _semi_loss(
        self,
        outputs_x: torch.Tensor,
        targets_x: torch.Tensor,
        outputs_u: torch.Tensor,
        targets_u: torch.Tensor,
        epoch: float,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        probs_u = torch.softmax(outputs_u, dim=1)
        lx = -torch.mean(torch.sum(F.log_softmax(outputs_x, dim=1) * targets_x, dim=1))
        lu = torch.mean((probs_u - targets_u) ** 2)
        lamb = float(self.hparams.lambda_u) * self._linear_rampup(epoch, int(self.hparams.warm_up))
        return lx, lu, torch.tensor(lamb, device=outputs_x.device, dtype=outputs_x.dtype)

    def _conf_penalty(self, outputs: torch.Tensor) -> torch.Tensor:
        probs = torch.softmax(outputs, dim=1)
        return torch.mean(torch.sum(probs.log() * probs, dim=1))

    def _eval_train(self, model: nn.Module) -> np.ndarray:
        if self.datamodule is None:
            raise RuntimeError("DivideMix requires a datamodule with train_eval_dataloader().")

        eval_loader = self.datamodule.train_eval_dataloader()
        losses = torch.zeros(len(eval_loader.dataset), dtype=torch.float32)
        was_training = model.training
        model.eval()

        with torch.no_grad():
            for inputs, targets, index in eval_loader:
                inputs = inputs.to(self.device)
                targets = targets.to(self.device)
                outputs = model(inputs)
                loss = self.ce_loss_none(outputs, targets)
                losses[index] = loss.detach().cpu()

        if was_training:
            model.train()

        min_val = float(losses.min().item())
        max_val = float(losses.max().item())
        denom = max_val - min_val
        if denom < 1e-12:
            losses = torch.full_like(losses, 0.5)
        else:
            losses = (losses - min_val) / denom

        input_loss = losses.view(-1, 1).numpy()
        gmm = GaussianMixture(
            n_components=2,
            max_iter=10,
            tol=1e-2,
            reg_covar=5e-4,
            random_state=int(self.hparams.get("seed", 42)),
        )
        gmm.fit(input_loss)
        prob = gmm.predict_proba(input_loss)
        prob = prob[:, gmm.means_.argmin()]
        return prob

    def _warmup_step(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        batch: Any,
    ) -> torch.Tensor:
        inputs, labels, _ = batch
        inputs = inputs.to(self.device)
        labels = labels.to(self.device)
        outputs = model(inputs)
        loss = self.ce_loss(outputs, labels)
        if self.hparams.noise_mode == "asym":
            loss = loss + self._conf_penalty(outputs)

        optimizer.zero_grad()
        self.manual_backward(loss)
        optimizer.step()
        return loss

    def _train_dividemix(
        self,
        batch,
        model: nn.Module,
        model_other: nn.Module,
        optimizer: torch.optim.Optimizer,
        epoch_progress: float,
    ) -> Dict[str, torch.Tensor]:
        
        print(len(batch))
        
        labeled = batch[0]
        unlabeled = batch[1]
        
        print(len(labeled), len(unlabeled))

        inputs_x, inputs_x2, labels_x, w_x = labeled
        inputs_u, inputs_u2 = unlabeled

        inputs_x = inputs_x.to(self.device)
        inputs_x2 = inputs_x2.to(self.device)
        labels_x = labels_x.to(self.device)
        w_x = w_x.to(self.device).float().view(-1, 1)
        inputs_u = inputs_u.to(self.device)
        inputs_u2 = inputs_u2.to(self.device)

        batch_size = inputs_x.size(0)
        labels_x = F.one_hot(labels_x, num_classes=self.num_classes).float()

        with torch.no_grad():
            outputs_u11 = model(inputs_u)
            outputs_u12 = model(inputs_u2)
            outputs_u21 = model_other(inputs_u)
            outputs_u22 = model_other(inputs_u2)

            pu = (
                torch.softmax(outputs_u11, dim=1)
                + torch.softmax(outputs_u12, dim=1)
                + torch.softmax(outputs_u21, dim=1)
                + torch.softmax(outputs_u22, dim=1)
            ) / 4
            ptu = pu ** (1 / float(self.hparams.T))
            targets_u = ptu / ptu.sum(dim=1, keepdim=True)

            outputs_x = model(inputs_x)
            outputs_x2 = model(inputs_x2)

            px = (torch.softmax(outputs_x, dim=1) + torch.softmax(outputs_x2, dim=1)) / 2
            px = w_x * labels_x + (1 - w_x) * px
            ptx = px ** (1 / float(self.hparams.T))
            targets_x = ptx / ptx.sum(dim=1, keepdim=True)

        if self._rng is None:
            self._rng = np.random.default_rng(int(self.hparams.get("seed", 42)))
        l = float(self._rng.beta(float(self.hparams.alpha), float(self.hparams.alpha)))
        l = max(l, 1 - l)

        all_inputs = torch.cat([inputs_x, inputs_x2, inputs_u, inputs_u2], dim=0)
        all_targets = torch.cat([targets_x, targets_x, targets_u, targets_u], dim=0)

        idx = torch.randperm(all_inputs.size(0), device=all_inputs.device)
        input_a, input_b = all_inputs, all_inputs[idx]
        target_a, target_b = all_targets, all_targets[idx]

        mixed_input = l * input_a + (1 - l) * input_b
        mixed_target = l * target_a + (1 - l) * target_b

        logits = model(mixed_input)
        logits_x = logits[: batch_size * 2]
        logits_u = logits[batch_size * 2 :]

        lx, lu, lamb = self._semi_loss(
            logits_x,
            mixed_target[: batch_size * 2],
            logits_u,
            mixed_target[batch_size * 2 :],
            epoch_progress,
        )

        prior = torch.full((self.num_classes,), 1.0 / self.num_classes, device=logits.device)
        pred_mean = torch.softmax(logits, dim=1).mean(0)
        penalty = torch.sum(prior * (torch.log(prior + 1e-12) - torch.log(pred_mean + 1e-12)))

        loss = lx + lamb * lu + penalty

        optimizer.zero_grad()
        self.manual_backward(loss)
        optimizer.step()

        return {
            "loss": loss,
            "lx": lx,
            "lu": lu,
            "lamb": lamb.detach(),
        }

    def training_step(self, batch: Any, batch_idx: int, dataloader_idx: int = 0) -> torch.Tensor:
        optimizers = self.optimizers()
        if isinstance(optimizers, (list, tuple)):
            opt1, opt2 = optimizers
        else:
            opt1 = opt2 = optimizers

        dual_batch = isinstance(batch, (list, tuple)) and len(batch) == 2
        if self.current_epoch < int(self.hparams.warm_up):
            if dual_batch:
                loss1 = self._warmup_step(self.model1, opt1, batch[0])
                loss2 = self._warmup_step(self.model2, opt2, batch[1])
                self.log("train/warmup_loss_net1", loss1, on_step=False, on_epoch=True, prog_bar=True)
                self.log("train/warmup_loss_net2", loss2, on_step=False, on_epoch=True, prog_bar=True)
                return 0.5 * (loss1 + loss2)
            if dataloader_idx == 0:
                loss = self._warmup_step(self.model1, opt1, batch)
                self.log("train/warmup_loss_net1", loss, on_step=False, on_epoch=True, prog_bar=True)
            else:
                loss = self._warmup_step(self.model2, opt2, batch)
                self.log("train/warmup_loss_net2", loss, on_step=False, on_epoch=True, prog_bar=True)
            return loss

        num_batches = self.trainer.num_training_batches
        if isinstance(num_batches, (list, tuple)):
            num_batches = num_batches[dataloader_idx]
        epoch_progress = self.current_epoch + batch_idx / max(1, int(num_batches))

        if dual_batch:
            print(len(batch))
            stats1 = self._train_dividemix(batch[0], self.model1, self.model2, opt1, epoch_progress)
            stats2 = self._train_dividemix(batch[1], self.model2, self.model1, opt2, epoch_progress)

            mean_loss = 0.5 * (stats1["loss"] + stats2["loss"])
            self.train_loss(mean_loss)
            self.log("train/loss", self.train_loss, on_step=False, on_epoch=True, prog_bar=True)

            for prefix, stats in (("train/net1", stats1), ("train/net2", stats2)):
                self.log(f"{prefix}_loss", stats["loss"], on_step=False, on_epoch=True)
                self.log(f"{prefix}_lx", stats["lx"], on_step=False, on_epoch=True)
                self.log(f"{prefix}_lu", stats["lu"], on_step=False, on_epoch=True)
                self.log(f"{prefix}_lambda_u", stats["lamb"], on_step=False, on_epoch=True)
            return mean_loss

        if dataloader_idx == 0:
            stats = self._train_dividemix(batch, self.model1, self.model2, opt1, epoch_progress)
            prefix = "train/net1"
        else:
            stats = self._train_dividemix(batch, self.model2, self.model1, opt2, epoch_progress)
            prefix = "train/net2"

        self.train_loss(stats["loss"])
        self.log("train/loss", self.train_loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log(f"{prefix}_loss", stats["loss"], on_step=False, on_epoch=True)
        self.log(f"{prefix}_lx", stats["lx"], on_step=False, on_epoch=True)
        self.log(f"{prefix}_lu", stats["lu"], on_step=False, on_epoch=True)
        self.log(f"{prefix}_lambda_u", stats["lamb"], on_step=False, on_epoch=True)
        return stats["loss"]

    def validation_step(self, batch: Any, batch_idx: int) -> None:
        inputs, targets = batch
        inputs = inputs.to(self.device)
        targets = targets.to(self.device)

        logits1 = self.model1(inputs)
        logits2 = self.model2(inputs)
        avg_logits = 0.5 * (logits1 + logits2)
        loss = F.cross_entropy(avg_logits, targets)
        preds = torch.argmax(avg_logits, dim=1)

        self.val_loss(loss)
        self.val_metrics(preds, targets)
        if self.log_per_class:
            self.val_per_class(preds, targets)

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

    def test_step(self, batch: Any, batch_idx: int) -> None:
        inputs, targets = batch
        inputs = inputs.to(self.device)
        targets = targets.to(self.device)

        logits1 = self.model1(inputs)
        logits2 = self.model2(inputs)
        avg_logits = 0.5 * (logits1 + logits2)
        loss = F.cross_entropy(avg_logits, targets)
        preds = torch.argmax(avg_logits, dim=1)

        self.test_loss(loss)
        self.test_metrics(preds, targets)
        if self.log_per_class:
            self.test_per_class(preds, targets)

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

    def configure_optimizers(self) -> Any:
        opt1 = torch.optim.SGD(
            self.model1.parameters(),
            lr=float(self.hparams.lr),
            momentum=float(self.hparams.sgd_momentum),
            weight_decay=float(self.hparams.weight_decay),
        )
        opt2 = torch.optim.SGD(
            self.model2.parameters(),
            lr=float(self.hparams.lr),
            momentum=float(self.hparams.sgd_momentum),
            weight_decay=float(self.hparams.weight_decay),
        )

        max_epochs = int(getattr(self.trainer, "max_epochs", 1) or 1)
        m1 = max(1, int(max_epochs * 0.5))
        m2 = max(1, int(max_epochs * 0.75))
        milestones = sorted(set([m1, m2]))

        scheduler1 = torch.optim.lr_scheduler.MultiStepLR(opt1, milestones=milestones, gamma=0.1)
        scheduler2 = torch.optim.lr_scheduler.MultiStepLR(opt2, milestones=milestones, gamma=0.1)

        return [
            {"optimizer": opt1, "lr_scheduler": {"scheduler": scheduler1, "interval": "epoch"}},
            {"optimizer": opt2, "lr_scheduler": {"scheduler": scheduler2, "interval": "epoch"}},
        ]
