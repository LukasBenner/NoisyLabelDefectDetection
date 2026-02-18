from __future__ import annotations

from typing import Optional
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from lightning import LightningModule

from models.base_robust_module import BaseRobustModule
from src.models.components.resnet_backbone import ResNetBackbone
from src.models.components.mobilenetv3_backbone import MobileNetV3Backbone


def _build_backbone(
    *,
    backbone_name: str,
    pretrained: bool,
    backbone_norm: str,
    backbone_gn_groups: int,
) -> nn.Module:
    name = backbone_name.lower().strip().replace("-", "_")
    if name in {"resnet", "resnet50", "resnet_50"}:
        return ResNetBackbone(
            pretrained=pretrained,
            norm=backbone_norm,
            gn_groups=backbone_gn_groups,
        )
    if name in {"mobilenetv3_large", "mobilenet_v3_large", "mnetv3_large"}:
        return MobileNetV3Backbone(
            variant="large",
            pretrained=pretrained,
            norm=backbone_norm,
            gn_groups=backbone_gn_groups,
        )
    if name in {"mobilenetv3_small", "mobilenet_v3_small", "mnetv3_small"}:
        return MobileNetV3Backbone(
            variant="small",
            pretrained=pretrained,
            norm=backbone_norm,
            gn_groups=backbone_gn_groups,
        )

    raise ValueError(
        f"Unknown backbone_name='{backbone_name}'. Expected one of: resnet50, mobilenet_v3_large, mobilenet_v3_small"
    )


class MLPHead(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int, out_dim: int, num_layers: int):
        super().__init__()
        assert num_layers >= 0, "negative layers?!?"

        if num_layers == 0:
            self.net = torch.nn.Identity()
            return

        if num_layers == 1:
            self.net = torch.nn.Linear(in_dim, out_dim)
            return

        layers = []
        prev_dim = in_dim
        for _ in range(num_layers - 1):
            layers.append(torch.nn.Linear(prev_dim, hidden_dim))
            layers.append(torch.nn.ReLU())
            prev_dim = hidden_dim

        layers.append(torch.nn.Linear(hidden_dim, out_dim))

        self.net = torch.nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


class MoCoV2(LightningModule):
    """
    MoCo v2: momentum encoder + queue + InfoNCE
    """

    def __init__(
        self,
        backbone_name: str = "mobilenetv3_large",
        pretrained: bool = False,
        backbone_norm: str = "bn",
        backbone_gn_groups: int = 32,
        proj_dim: int = 256,
        proj_hidden_dim: int = 2048,
        queue_size: int = 65536,
        proj_num_layers: int = 2,
        momentum: float = 0.996,
        momentum_final: Optional[float] = None,
        temperature: float = 0.2,
        lr: float = 0.03,
        sgd_momentum: float = 0.9,
        eta_min: float = 0.0,
        weight_decay: float = 1e-4,
        max_epochs: int = 300,
        gather_keys_for_queue: bool = True,
    ):
        super().__init__()
        self.save_hyperparameters()

        # Encoders
        self.encoder_q = _build_backbone(
            backbone_name=backbone_name,
            pretrained=pretrained,
            backbone_norm=backbone_norm,
            backbone_gn_groups=backbone_gn_groups,
        )
        self.encoder_k = _build_backbone(
            backbone_name=backbone_name,
            pretrained=pretrained,
            backbone_norm=backbone_norm,
            backbone_gn_groups=backbone_gn_groups,
        )

        # Projectors
        encoder_q_out_dim = int(getattr(self.encoder_q, "out_dim"))
        encoder_k_out_dim = int(getattr(self.encoder_k, "out_dim"))

        self.projector_q = MLPHead(
            encoder_q_out_dim, proj_hidden_dim, proj_dim, proj_num_layers
        )
        self.projector_k = MLPHead(
            encoder_k_out_dim, proj_hidden_dim, proj_dim, proj_num_layers
        )

        # Initialize key encoder to query encoder weights; then freeze grads for key
        self._init_key_encoder()

        # Create the queue
        self.queue: torch.Tensor
        self.register_buffer("queue", torch.randn(proj_dim, queue_size))
        self.queue = F.normalize(self.queue, dim=0)
        self.queue_ptr: torch.Tensor
        self.register_buffer("queue_ptr", torch.zeros(1, dtype=torch.long))

    def setup(self, stage: Optional[str] = None) -> None:
        # Cache total training steps for schedules.
        # NOTE: Lightning attaches `trainer` before calling `setup`.
        total = getattr(
            getattr(self, "trainer", None), "estimated_stepping_batches", None
        )
        if total is None:
            self._total_steps: Optional[int] = None
        else:
            self._total_steps = int(total) if int(total) > 0 else None

    @torch.no_grad()
    def _init_key_encoder(self):
        # Copy full state (including BatchNorm running stats buffers).
        self.encoder_k.load_state_dict(self.encoder_q.state_dict(), strict=True)
        self.projector_k.load_state_dict(self.projector_q.state_dict(), strict=True)

        for param in self.encoder_k.parameters():
            param.requires_grad = False
        for param in self.projector_k.parameters():
            param.requires_grad = False

    @staticmethod
    def _dist_is_initialized() -> bool:
        return dist.is_available() and dist.is_initialized()

    @staticmethod
    @torch.no_grad()
    def _concat_all_gather(tensor: torch.Tensor) -> torch.Tensor:
        """Gathers tensors from all ranks and concatenates on dim=0."""
        if not (dist.is_available() and dist.is_initialized()):
            return tensor

        world_size = dist.get_world_size()
        gathered = [torch.ones_like(tensor) for _ in range(world_size)]
        dist.all_gather(gathered, tensor, async_op=False)
        return torch.cat(gathered, dim=0)

    @torch.no_grad()
    def _batch_shuffle_ddp(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Batch shuffle for ShuffleBN.

        Shuffles the batch across all DDP ranks so BatchNorm in the key encoder
        sees samples from different GPUs.

        Returns:
            x_shuffled: The per-rank shuffled batch
            idx_unshuffle: Indices used to restore the original order
        """
        if not self._dist_is_initialized():
            return x, None

        batch_size_this = x.shape[0]
        x_gather = self._concat_all_gather(x)
        batch_size_all = x_gather.shape[0]

        if batch_size_all % batch_size_this != 0:
            raise RuntimeError(
                "DDP batch shuffle requires equal batch size on each rank. "
                "Ensure drop_last=True and DistributedSampler is used."
            )

        world_size = batch_size_all // batch_size_this

        # random shuffle index (same on all ranks)
        if dist.get_rank() == 0:
            idx_shuffle = torch.randperm(
                batch_size_all, device=x.device, dtype=torch.long
            )
        else:
            idx_shuffle = torch.empty(batch_size_all, device=x.device, dtype=torch.long)
        dist.broadcast(idx_shuffle, src=0)

        idx_unshuffle = torch.argsort(idx_shuffle)

        # select this rank's portion
        idx_this = idx_shuffle.view(world_size, -1)[dist.get_rank()]
        return x_gather[idx_this], idx_unshuffle

    @torch.no_grad()
    def _batch_unshuffle_ddp(
        self, x: torch.Tensor, idx_unshuffle: Optional[torch.Tensor]
    ) -> torch.Tensor:
        """Undo batch shuffle for ShuffleBN."""
        if idx_unshuffle is None or not self._dist_is_initialized():
            return x

        batch_size_this = x.shape[0]
        x_gather = self._concat_all_gather(x)
        batch_size_all = x_gather.shape[0]

        if batch_size_all % batch_size_this != 0:
            raise RuntimeError(
                "DDP batch unshuffle requires equal batch size on each rank. "
                "Ensure drop_last=True and DistributedSampler is used."
            )

        world_size = batch_size_all // batch_size_this
        idx_this = idx_unshuffle.view(world_size, -1)[dist.get_rank()]
        return x_gather[idx_this]

    @torch.no_grad()
    def _momentum_update_key_encoder(self):
        m = self._current_momentum()
        for param_q, param_k in zip(
            self.encoder_q.parameters(), self.encoder_k.parameters()
        ):
            param_k.data.mul_(m).add_(param_q.data, alpha=(1.0 - m))
        for param_q, param_k in zip(
            self.projector_q.parameters(), self.projector_k.parameters()
        ):
            param_k.data.mul_(m).add_(param_q.data, alpha=(1.0 - m))

    def _current_momentum(self) -> float:
        """Cosine schedule momentum toward 1.0 (optional).

        If `momentum_final` is not provided, uses constant `momentum`.
        """
        m0 = float(self.hparams["momentum"])
        m1 = self.hparams.get("momentum_final", None)
        if m1 is None:
            return m0

        m1 = float(m1)
        if m1 <= m0:
            return m0

        total_steps = getattr(self, "_total_steps", None)
        if not total_steps:
            # Fallback: schedule across epochs if step count isn't available.
            max_epochs = int(self.hparams.get("max_epochs", 1))
            progress = 0.0
            if getattr(self, "current_epoch", None) is not None and max_epochs > 0:
                progress = float(self.current_epoch) / float(max_epochs)
        else:
            t = float(getattr(self, "global_step", 0))
            progress = min(max(t / float(total_steps), 0.0), 1.0)

        # m = 1 - (1 - m0) * (cos(pi * t/T) + 1) / 2
        return 1.0 - (1.0 - m0) * (math.cos(math.pi * progress) + 1.0) / 2.0

    @torch.no_grad()
    def _dequeue_and_enqueue(self, keys: torch.Tensor):
        # In DDP, gather keys from all ranks so every process maintains the same queue.
        if bool(self.hparams["gather_keys_for_queue"]) and self._dist_is_initialized():
            keys = self._concat_all_gather(keys)

        batch_size = keys.shape[0]
        queue_size = self.queue.shape[1]
        ptr = int(self.queue_ptr.item())

        # Reference MoCo assumes queue size is a multiple of (effective) batch size.
        # This keeps the FIFO replacement aligned and consistent across ranks.
        if batch_size > queue_size:
            raise RuntimeError(
                f"MoCo queue_size ({queue_size}) must be >= effective batch size ({batch_size}). "
                "Reduce batch_size or increase queue_size."
            )
        if (queue_size % batch_size) != 0:
            raise RuntimeError(
                f"MoCo queue_size ({queue_size}) must be divisible by effective batch size ({batch_size}). "
                "Set queue_size to a multiple of (batch_size_per_gpu * world_size)."
            )

        # Replace entries
        end = ptr + batch_size
        if end <= queue_size:
            self.queue[:, ptr:end] = keys.T
        else:
            first = queue_size - ptr
            self.queue[:, ptr:] = keys[:first].T
            self.queue[:, : end - queue_size] = keys[first:].T

        ptr = (ptr + batch_size) % queue_size
        self.queue_ptr[0] = ptr

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Returns normalized projected embedding from query encoder (useful for inference/analysis).
        """
        feat = self.encoder_q(x)
        z = self.projector_q(feat)
        return F.normalize(z, dim=1)

    def training_step(self, batch, batch_idx):
        """
        batch returns: (x_q, x_k) augmented views
        """
        x_q, x_k = batch

        # Query
        q = self.projector_q(self.encoder_q(x_q))
        q = F.normalize(q, dim=1)

        # Key (no grad), momentum update
        with torch.no_grad():
            self._momentum_update_key_encoder()
            # ShuffleBN for the key encoder in DDP
            x_k_shuf, idx_unshuf = self._batch_shuffle_ddp(x_k)
            k = self.projector_k(self.encoder_k(x_k_shuf))
            k = F.normalize(k, dim=1)

            # restore original sample order
            k = self._batch_unshuffle_ddp(k, idx_unshuf)

        # InfoNCE logits: Nx(1+K)
        # Positive: q·k
        # Do similarity + CE in fp32 for AMP stability.
        q32 = q.float()
        k32 = k.float()
        l_pos = torch.einsum("nc,nc->n", [q32, k32]).unsqueeze(-1)
        # Negative: q·queue
        l_neg = torch.einsum("nc,ck->nk", [q32, self.queue.clone().detach().float()])

        logits = torch.cat([l_pos, l_neg], dim=1)
        logits /= float(self.hparams["temperature"])

        labels = torch.zeros(logits.shape[0], dtype=torch.long, device=logits.device)
        loss = F.cross_entropy(logits, labels)

        # Update queue
        with torch.no_grad():
            self._dequeue_and_enqueue(k)

        self.log(
            "ssl_loss",
            loss,
            prog_bar=True,
            on_epoch=True,
            batch_size=logits.size(0),
            sync_dist=self._dist_is_initialized(),
        )
        return loss

    def configure_optimizers(self):
        """
        MoCo v2 typically uses SGD + cosine LR schedule.
        """
        optimizer = torch.optim.SGD(
            list(self.encoder_q.parameters()) + list(self.projector_q.parameters()),
            lr=float(self.hparams["lr"]),
            momentum=float(self.hparams["sgd_momentum"]),
            weight_decay=float(self.hparams["weight_decay"]),
        )

        # Prefer step-based cosine over the full training run (reference implementations
        # schedule over iterations). Fallback to epoch-based if step count is unavailable.
        total_steps = getattr(
            getattr(self, "trainer", None), "estimated_stepping_batches", None
        )
        if total_steps is not None and int(total_steps) > 0:
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                T_max=int(total_steps),
                eta_min=float(self.hparams["eta_min"]),
            )
            return {
                "optimizer": optimizer,
                "lr_scheduler": {"scheduler": scheduler, "interval": "step"},
            }

        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=int(self.hparams["max_epochs"]),
            eta_min=float(self.hparams["eta_min"]),
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": scheduler, "interval": "epoch"},
        }



class MocoClassifier(BaseRobustModule):
    def __init__(
        self,
        num_classes: int = 10,
        backbone_name: str = "mobilenetv3_large",
        pretrained: bool = True,
        backbone_norm: str = "bn",
        backbone_gn_groups: int = 32,
        optimizer: Optional[torch.optim.Optimizer] = None,
        scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
        criterion: Optional[torch.nn.Module] = None,
        lr_backbone: float = 0.01,
        lr_head: float = 0.1,
        momentum: float = 0.9,
        weight_decay: float = 1e-4,
        freeze_backbone: bool = False,
        datamodule: Optional[LightningModule] = None,
        
    ):
        self.save_hyperparameters(ignore=["criterion", "optimizer", "scheduler", "datamodule"])

        self.backbone = _build_backbone(
            backbone_name=backbone_name,
            pretrained=pretrained,
            backbone_norm=backbone_norm,
            backbone_gn_groups=backbone_gn_groups,
        )

        backbone_out_dim = int(getattr(self.backbone, "out_dim"))
        self.head = nn.Linear(backbone_out_dim, num_classes)
        
        if freeze_backbone:
            for p in self.backbone.parameters():
                p.requires_grad = False
                
        super().__init__(self.backbone, num_classes, optimizer, scheduler, criterion, compile=False, datamodule=datamodule)


    @staticmethod
    def _dist_is_initialized() -> bool:
        return dist.is_available() and dist.is_initialized()

    def load_from_moco(self, moco_ckpt_path: str, strict: bool = True):
        """
        Loads encoder_q weights from a MoCo checkpoint into this classifier backbone.
        """
        ckpt = torch.load(moco_ckpt_path, map_location="cpu")
        state = ckpt.get("state_dict", ckpt)

        # Map keys: encoder_q.* -> backbone.*
        mapped = {}
        for k, v in state.items():
            if k.startswith("encoder_q."):
                mapped["backbone." + k[len("encoder_q.") :]] = v

        missing, unexpected = self.load_state_dict(mapped, strict=False)

        if strict:
            # It's expected that classifier head weights are missing.
            allowed_missing_prefixes = ("head.", "ce_weights")
            filtered_missing = [
                k for k in missing if not k.startswith(allowed_missing_prefixes)
            ]
            if filtered_missing or unexpected:
                raise RuntimeError(
                    "MoCo load mismatch. "
                    f"Missing={filtered_missing}, Unexpected={unexpected}"
                )
        return missing, unexpected

    def forward(self, x):
        feat = self.backbone(x)
        return self.head(feat)


    def configure_optimizers(self):
        # Separate LRs (common fine-tuning best practice)
        params = [
            {
                "params": self.backbone.parameters(),
                "lr": float(self.hparams["lr_backbone"]),
            },
            {
                "params": self.head.parameters(), 
                "lr": float(self.hparams["lr_head"])
            },
        ]
        
        if self._optimizer is None:
            raise ValueError("Optimizer not provided. Pass it to __init__ or override configure_optimizers()")

        optimizer = self._optimizer(params=params)

        if self._scheduler is not None:
            scheduler = self._scheduler(optimizer=optimizer)
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "monitor": "val/f1_macro",   # primary metric for imbalance + equal class importance
                    "interval": "epoch",
                    "frequency": 1,
                },
            }
        return {"optimizer": optimizer}
