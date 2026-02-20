import torch
import torch.nn as nn
from torchvision import models


class ResNet50(nn.Module):
    def __init__(self, num_classes: int, pretrained: bool = True, variant: str = "default") -> None:
        """Initialize EfficientNetBaseline.

        :param num_classes: Number of output classes.
        :param pretrained: Whether to use pretrained ImageNet weights. Defaults to True.
        """
        super().__init__()

        if pretrained:
            self.model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
            num_features = self.model.fc.in_features
            self.model.fc = nn.Linear(num_features, num_classes)
        else:
            self.model = models.resnet50(weights=None, num_classes=num_classes)
            
        

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)


if __name__ == "__main__":
    # Test the model
    model = ResNet50(num_classes=10, pretrained=True)
    x = torch.randn(2, 3, 640, 480)
    output = model(x)
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {output.shape}")
