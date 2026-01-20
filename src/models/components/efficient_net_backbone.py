import torch
import torch.nn as nn
from torchvision import models
from typing import cast


class EfficientNetBackbone(nn.Module):
    def __init__(
        self,
        pretrained: bool = True,
        norm: str = "bn",
        gn_groups: int = 32,
    ) -> None:
        """Initialize EfficientNetBaseline.

        :param num_classes: Number of output classes.
        :param pretrained: Whether to use pretrained ImageNet weights. Defaults to True.
        :param norm: Normalization strategy. One of:
            - "bn": keep BatchNorm layers (default)
            - "freeze_bn": keep BatchNorm but force eval() + freeze affine params
            - "gn": replace BatchNorm2d with GroupNorm
            - "none": replace BatchNorm2d with Identity
        :param gn_groups: Number of groups to use for GroupNorm when norm="gn".
        """
        super().__init__()

        if pretrained:
            self.model = models.efficientnet_v2_l(
                weights=models.EfficientNet_V2_L_Weights.IMAGENET1K_V1,
            )
        else:
            self.model = models.efficientnet_v2_l(weights=None)
            
        # torchvision stubs are sometimes imprecise; cast for static type checkers.
        classifier = cast(nn.Sequential, self.model.classifier)
        classifier_fc = cast(nn.Linear, classifier[1])
        self.out_dim: int = classifier_fc.in_features

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
                EfficientNetBackbone._replace_batchnorm(child, norm=norm, gn_groups=gn_groups)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.model.features(x)
        x = self.model.avgpool(x)
        x = torch.flatten(x, 1)
        return x
        


if __name__ == "__main__":
    # Test the model
    model = EfficientNetBackbone(pretrained=False, norm="bn")
    x = torch.randn(2, 3, 640, 480)
    output = model(x)
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {output.shape}")
