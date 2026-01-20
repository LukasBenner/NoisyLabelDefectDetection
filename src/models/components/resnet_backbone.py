from __future__ import annotations

from typing import cast

import torch
import torch.nn as nn
from torchvision import models


class ResNetBackbone(nn.Module):
    def __init__(
        self,
        pretrained: bool = False,
        norm: str = "bn",
        gn_groups: int = 32,
    ) -> None:
        """ResNet-50 feature extractor.

        Returns the pooled feature vector (before the final FC classifier).

        :param pretrained: Whether to use ImageNet pretrained weights.
        :param norm: Normalization strategy. One of:
            - "bn": keep BatchNorm layers (default)
            - "freeze_bn": keep BatchNorm but force eval() + freeze affine params
            - "gn": replace BatchNorm2d with GroupNorm
            - "none": replace BatchNorm2d with Identity
        :param gn_groups: Number of groups for GroupNorm when norm="gn".
        """
        super().__init__()

        weights = models.ResNet50_Weights.DEFAULT if pretrained else None
        self.model = models.resnet50(weights=weights)

        # Stubs can be imprecise; cast for static type checkers.
        fc = cast(nn.Linear, self.model.fc)
        self.out_dim: int = fc.in_features

        norm = norm.lower().strip()
        if norm not in {"bn", "freeze_bn", "gn", "none"}:
            raise ValueError(f"Unknown norm='{norm}'. Expected one of: bn, freeze_bn, gn, none")

        if norm == "freeze_bn":
            self._freeze_batchnorm(self.model)
        elif norm in {"gn", "none"}:
            self._replace_batchnorm(self.model, norm=norm, gn_groups=gn_groups)

    @staticmethod
    def _freeze_batchnorm(module: nn.Module) -> None:
        for m in module.modules():
            if isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
                m.eval()
                if m.affine:
                    m.weight.requires_grad = False
                    m.bias.requires_grad = False

    @staticmethod
    def _replace_batchnorm(module: nn.Module, *, norm: str, gn_groups: int) -> None:
        for name, child in module.named_children():
            if isinstance(child, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
                if norm == "none":
                    new_layer: nn.Module = nn.Identity()
                else:
                    num_channels = child.num_features
                    groups = min(max(1, int(gn_groups)), num_channels)
                    while groups > 1 and (num_channels % groups) != 0:
                        groups -= 1
                    new_layer = nn.GroupNorm(
                        num_groups=groups,
                        num_channels=num_channels,
                        eps=child.eps,
                        affine=True,
                    )
                    if child.affine:
                        with torch.no_grad():
                            new_layer.weight.copy_(child.weight)
                            new_layer.bias.copy_(child.bias)

                setattr(module, name, new_layer)
            else:
                ResNetBackbone._replace_batchnorm(child, norm=norm, gn_groups=gn_groups)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Use the standard ResNet forward up to (and including) global average pooling.
        x = self.model.conv1(x)
        x = self.model.bn1(x)
        x = self.model.relu(x)
        x = self.model.maxpool(x)

        x = self.model.layer1(x)
        x = self.model.layer2(x)
        x = self.model.layer3(x)
        x = self.model.layer4(x)

        x = self.model.avgpool(x)
        x = torch.flatten(x, 1)
        return x
