from __future__ import annotations

import torch
import torch.nn as nn
from torchvision import models


class ViTClassifier(nn.Module):
    def __init__(
        self,
        num_classes: int,
        pretrained: bool = True,
        variant: str = "vit_b_16",
    ) -> None:
        super().__init__()

        variant = variant.lower().strip()
        variant_aliases = {
            "vitb16": "vit_b_16",
            "vit_b_16": "vit_b_16",
            "b16": "vit_b_16",
            "vitl16": "vit_l_16",
            "vit_l_16": "vit_l_16",
            "l16": "vit_l_16",
        }
        variant = variant_aliases.get(variant, variant)

        if variant == "vit_l_16":
            weights = None
            if pretrained:
                weights = getattr(models.ViT_L_16_Weights, "DEFAULT", None)
                if weights is None:
                    weights = models.ViT_L_16_Weights.IMAGENET1K_SWAG_E2E_V1
            self.model = models.vit_l_16(weights=weights)
        elif variant == "vit_b_16":
            weights = None
            if pretrained:
                weights = getattr(models.ViT_B_16_Weights, "DEFAULT", None)
                if weights is None:
                    weights = models.ViT_B_16_Weights.IMAGENET1K_SWAG_E2E_V1
            self.model = models.vit_b_16(weights=weights)
        else:
            raise ValueError(
                "Unknown variant='{variant}'. Expected one of: "
                "vit_b_16 or vit_l_16 (aliases: vitb16/vitl16, b16/l16)."
            )

        if hasattr(self.model, "heads") and hasattr(self.model.heads, "head"):
            head = getattr(self.model.heads, "head")
            if isinstance(head, nn.Linear):
                in_features = head.in_features
                self.model.heads.head = nn.Linear(in_features, num_classes)
            else:
                raise RuntimeError("Unexpected ViT head type")
        else:
            raise RuntimeError("ViT model does not expose heads.head")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)
