from __future__ import annotations

from typing import cast

import torch
import torch.nn as nn
from torchvision import models


class ViTBackbone(nn.Module):
    def __init__(
        self,
        variant: str = "vitb16",
        pretrained: bool = False,
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
                # Prefer DEFAULT if present; otherwise fallback to SWAG weights.
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
                f"Unknown variant='{variant}'. Expected one of: vitb16, vitl16 (aliases: vit_b_16/vit_l_16, b16/l16)."
            )

        # Expose the pooled feature dimension (CLS embedding dim).
        # torchvision ViT uses `model.heads.head: nn.Linear(hidden_dim -> num_classes)`.
        heads = cast(nn.Module, self.model.heads)
        if hasattr(heads, "head") and isinstance(getattr(heads, "head"), nn.Linear):
            head = cast(nn.Linear, getattr(heads, "head"))
            self.out_dim = int(head.in_features)
        elif hasattr(self.model, "hidden_dim"):
            self.out_dim = int(getattr(self.model, "hidden_dim"))
        else:
            raise RuntimeError("Could not infer ViT feature dimension (out_dim).")


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Return CLS pooled representation (pre-classifier).
        forward_features = getattr(self.model, "forward_features", None)
        if callable(forward_features):
            feat = forward_features(x)
            if isinstance(feat, torch.Tensor):
                return feat
            if isinstance(feat, dict):
                for key in ("x", "pooler_output", "last_hidden_state"):
                    val = feat.get(key)
                    if isinstance(val, torch.Tensor):
                        return val
            raise TypeError(
                f"Unexpected output type from ViT.forward_features: {type(feat)}"
            )

        # Fallback for older torchvision: replicate feature path.
        n = x.shape[0]
        x = self.model._process_input(x)
        cls_token = self.model.class_token.expand(n, -1, -1)
        x = torch.cat((cls_token, x), dim=1)
        x = self.model.encoder(x)
        x = x[:, 0]
        return x


class ViTB16Backbone(ViTBackbone):
    def __init__(self, pretrained: bool = False) -> None:
        super().__init__(variant="vitb16", pretrained=pretrained)


class ViTL16Backbone(ViTBackbone):
    def __init__(self, pretrained: bool = False) -> None:
        super().__init__(variant="vitl16", pretrained=pretrained)


# Backwards-compatible name (original file used `VitBackbone`).
VitBackbone = ViTBackbone


if __name__ == "__main__":
    model = ViTBackbone(variant="vitb16", pretrained=False)
    x = torch.randn(2, 3, 224, 224)
    out = model(x)
    print(out.shape, model.out_dim)
