import torch
import torch.nn as nn
from torchvision import models


class MobileNet(nn.Module):
    def __init__(self, num_classes: int, pretrained: bool = True) -> None:
        """Initialize MobileNet.

        :param num_classes: Number of output classes.
        :param pretrained: Whether to use pretrained ImageNet weights. Defaults to True.
        """
        super().__init__()

        if pretrained:
            self.model = models.mobilenet_v3_small(
                weights=models.MobileNet_V3_Small_Weights.IMAGENET1K_V1
            )
        else:
            self.model = models.mobilenet_v3_small(weights=None)

        # Replace classifier head
        num_ftrs = self.model.classifier[3].in_features
        self.model.classifier[3] = nn.Linear(num_ftrs, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)


if __name__ == "__main__":
    # Test the model
    model = MobileNet(num_classes=10, pretrained=False)
    x = torch.randn(2, 3, 640, 480)
    output = model(x)
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {output.shape}")
