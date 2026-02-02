from __future__ import annotations

from typing import Any, Optional

import torch
from torch.utils.data import DataLoader

from src.models.components.noise_adaption_layer import NoiseAdaptionNet, InstanceNoiseAdaptionNet
from src.models.base_robust_module import BaseRobustModule


class NoiseAdaptionModule(BaseRobustModule):
    def __init__(
        self,
        *args,
        noise_init_epoch: int = 10,
        noise_instance_epoch: int = 30,
        instance_hidden_dim: int = 128,
        noise_init_eps: float = 1e-6,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.save_hyperparameters(
            logger=False,
            ignore=["net", "optimizer", "scheduler", "criterion", "datamodule"],
        )
        self.noise_init_epoch = int(noise_init_epoch)
        self.noise_instance_epoch = int(noise_instance_epoch)
        self.instance_hidden_dim = int(instance_hidden_dim)
        self.noise_init_eps = float(noise_init_eps)
        self._noise_initialized = False
        self._instance_noise_initialized = False

    def _get_val_dataloader(self) -> Optional[DataLoader]:
        trainer = getattr(self, "trainer", None)
        if trainer is None:
            return None

        val_loaders = getattr(trainer, "val_dataloaders", None)
        if val_loaders:
            if isinstance(val_loaders, list):
                return val_loaders[0]
            return val_loaders

        datamodule = getattr(trainer, "datamodule", None)
        if datamodule is None:
            return None
        if hasattr(datamodule, "val_dataloader"):
            return datamodule.val_dataloader()
        return None

    def _compute_confusion_matrix(self, val_loader: DataLoader) -> torch.Tensor:
        num_classes = int(self.num_classes)
        cm = torch.zeros(num_classes, num_classes, device=self.device, dtype=torch.float32)

        was_training = self.training
        self.eval()
        with torch.no_grad():
            for batch in val_loader:
                inputs, targets = batch
                inputs = inputs.to(self.device, non_blocking=True)
                targets = targets.to(self.device, non_blocking=True).view(-1)
                logits = self.forward(inputs)
                preds = torch.argmax(logits, dim=1)

                idx = num_classes * targets + preds
                cm += torch.bincount(idx, minlength=num_classes * num_classes).view(
                    num_classes, num_classes
                )

        if was_training:
            self.train()

        cm = cm + float(self.noise_init_eps)
        cm = cm / cm.sum(dim=1, keepdim=True)
        return cm

    def _initialize_noise_adaption(self) -> None:
        val_loader = self._get_val_dataloader()
        if val_loader is None:
            return

        confusion = self._compute_confusion_matrix(val_loader)

        noise_net = NoiseAdaptionNet(
            base_net=self.net,
            num_classes=int(self.num_classes),
            init_value=0.9,
        )
        noise_net.noise_layer.to(self.device)
        transition_device = noise_net.noise_layer.transition_matrix.device
        noise_net.noise_layer.transition_matrix.data.copy_(
            torch.log(confusion.to(transition_device))
        )
        self.net = noise_net

        optimizers = getattr(self.trainer, "optimizers", [])
        if optimizers:
            optimizer = optimizers[0]
            existing = {id(p) for group in optimizer.param_groups for p in group.get("params", [])}
            new_params = [
                p for p in self.net.noise_layer.parameters() if id(p) not in existing
            ]
            if new_params:
                base_group = optimizer.param_groups[0]
                param_group = {"params": new_params}
                if "lr" in base_group:
                    param_group["lr"] = base_group["lr"]
                if "weight_decay" in base_group:
                    param_group["weight_decay"] = base_group["weight_decay"]
                optimizer.add_param_group(param_group)

        self._noise_initialized = True
        self.log("noise/initialized", 1.0, on_epoch=True, prog_bar=True)

    def _initialize_instance_noise(self) -> None:
        if not hasattr(self.net, "noise_layer"):
            return
        if not hasattr(self.net, "base_net"):
            return

        base_net = self.net.base_net
        base_transition = self.net.noise_layer.transition_matrix.detach()

        instance_net = InstanceNoiseAdaptionNet(
            base_net=base_net,
            num_classes=int(self.num_classes),
            init_value=0.9,
            hidden_dim=self.instance_hidden_dim,
        )
        instance_net.to(self.device)
        instance_net.base_transition.data.copy_(base_transition.to(self.device))
        self.net = instance_net

        optimizers = getattr(self.trainer, "optimizers", [])
        if optimizers:
            optimizer = optimizers[0]
            existing = {id(p) for group in optimizer.param_groups for p in group.get("params", [])}
            new_params = [
                p for p in self.net.parameters() if id(p) not in existing
            ]
            if new_params:
                base_group = optimizer.param_groups[0]
                param_group = {"params": new_params}
                if "lr" in base_group:
                    param_group["lr"] = base_group["lr"]
                if "weight_decay" in base_group:
                    param_group["weight_decay"] = base_group["weight_decay"]
                optimizer.add_param_group(param_group)

        self._instance_noise_initialized = True
        self.log("noise/instance_initialized", 1.0, on_epoch=True, prog_bar=True)

    def on_train_epoch_end(self) -> None:
        if (
            not self._noise_initialized
            and self.noise_init_epoch > 0
            and (self.current_epoch + 1) >= self.noise_init_epoch
        ):
            self._initialize_noise_adaption()
        if (
            self._noise_initialized
            and not self._instance_noise_initialized
            and self.noise_instance_epoch > 0
            and (self.current_epoch + 1) >= self.noise_instance_epoch
        ):
            self._initialize_instance_noise()

    def _eval_logits(self, inputs: torch.Tensor) -> torch.Tensor:
        if self._noise_initialized and hasattr(self.net, "get_clean_output"):
            return self.net.get_clean_output(inputs)
        return self.forward(inputs)

    def validation_step(self, batch: Any, batch_idx: int) -> None:
        inputs, targets = batch
        logits = self._eval_logits(inputs)
        loss = self.criterion(logits, targets)
        preds = torch.argmax(logits, dim=1)

        self.val_loss(loss)
        self.val_metrics(preds, targets)

        if self.log_per_class:
            self.val_per_class(preds, targets)

        self.log("val/loss", self.val_loss, on_step=False, on_epoch=True, prog_bar=True)
        # Show the primary metric on prog bar
        self.log_dict(
            {k: v for k, v in self.val_metrics.items() if k != "val/f1_macro"},
            on_step=False,
            on_epoch=True,
            prog_bar=False,
        )
        self.log("val/f1_macro", self.val_metrics["val/f1_macro"], on_step=False, on_epoch=True, prog_bar=True)


    def test_step(self, batch: Any, batch_idx: int) -> None:
        inputs, targets = batch
        logits = self._eval_logits(inputs)
        loss = self.criterion(logits, targets)
        preds = torch.argmax(logits, dim=1)

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
                self.log(f"test/precision_c{i}", pc["test/per_class/precision"][i], sync_dist=True)
                self.log(f"test/recall_c{i}", pc["test/per_class/recall"][i], sync_dist=True)
                self.log(f"test/f1_c{i}", pc["test/per_class/f1"][i], sync_dist=True)
