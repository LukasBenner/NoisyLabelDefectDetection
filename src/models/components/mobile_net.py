import torch
import torch.nn as nn
from torchvision import models


class MobileNet(nn.Module):
    def __init__(self, num_classes: int, pretrained: bool = True, variant: str = "small",) -> None:
        """Initialize MobileNet.

        :param num_classes: Number of output classes.
        :param pretrained: Whether to use pretrained ImageNet weights. Defaults to True.
        """
        super().__init__()
        
        variant = variant.lower().strip()
        if variant not in {"large", "small"}:
            raise ValueError(f"Unknown variant='{variant}'. Expected 'large' or 'small'.")

        if variant == "large":
            weights = models.MobileNet_V3_Large_Weights.IMAGENET1K_V2 if pretrained else None
            self.model = models.mobilenet_v3_large(weights=weights)
        else:
            weights = models.MobileNet_V3_Small_Weights.IMAGENET1K_V1 if pretrained else None
            self.model = models.mobilenet_v3_small(weights=weights)

        # Replace classifier head
        feature_dim = self.model.classifier[0].in_features
        self.feature_dim = feature_dim
        num_ftrs = self.model.classifier[3].in_features
        self.model.classifier[3] = nn.Linear(num_ftrs, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        x = self.model.features(x)
        x = self.model.avgpool(x)
        x = torch.flatten(x, 1)
        return x


if __name__ == "__main__":
    # Test the model
    model = MobileNet(num_classes=10, pretrained=False)
    x = torch.randn(2, 3, 640, 480)
    output = model(x)
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {output.shape}")
