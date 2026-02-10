import torch
from torchvision.transforms import v2

class BaselineTransforms:
    @staticmethod
    def train_transforms(mean, std):
        transforms = v2.Compose(
            [
                v2.Resize(480, antialias=True),
                v2.CenterCrop(480),
                v2.RandomHorizontalFlip(),
                v2.RandomVerticalFlip(),
                v2.ToImage(),
                v2.ToDtype(torch.float32, scale=True),
                v2.Normalize(mean=mean, std=std),
            ]
        )
        return transforms
    
    @staticmethod
    def eval_transforms(mean, std):
        transforms = v2.Compose(
            [
                v2.Resize(480, antialias=True),
                v2.CenterCrop(480),
                v2.ToImage(),
                v2.ToDtype(torch.float32, scale=True),
                v2.Normalize(mean=mean, std=std),
            ]
        )
        return transforms
    
    
class MediumTransforms:
    @staticmethod
    def train_transforms(mean, std):
        transforms = v2.Compose(
            [
                v2.Resize(480, antialias=True),
                v2.RandomResizedCrop(480, scale=(0.8,1)),
                v2.RandomHorizontalFlip(p=0.5),
                v2.RandomVerticalFlip(p=0.5),
                v2.RandomRotation(degrees=(-10,10)),
                v2.ToImage(),
                v2.ToDtype(torch.float32, scale=True),
                v2.Normalize(mean=mean, std=std),
            ]
        )
        return transforms
    
    @staticmethod
    def eval_transforms(mean, std):
        transforms = v2.Compose(
            [
                v2.Resize(480, antialias=True),
                v2.CenterCrop(480),
                v2.ToImage(),
                v2.ToDtype(torch.float32, scale=True),
                v2.Normalize(mean=mean, std=std),
            ]
        )
        return transforms

class StrongTransforms:
    @staticmethod
    def train_transforms(mean, std):
        transforms = v2.Compose(
            [
                v2.Resize(480, antialias=True),
                v2.RandomResizedCrop(480, scale=(0.8,1)),
                v2.RandomHorizontalFlip(p=0.5),
                v2.RandomVerticalFlip(p=0.5),
                v2.RandomRotation(degrees=(-45,45)),
                v2.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1),
                v2.ToImage(),
                v2.ToDtype(torch.float32, scale=True),
                v2.Normalize(mean=mean, std=std),
            ]
        )
        return transforms
    
    @staticmethod
    def eval_transforms(mean, std):
        transforms = v2.Compose(
            [
                v2.Resize(480, antialias=True),
                v2.CenterCrop(480),
                v2.ToImage(),
                v2.ToDtype(torch.float32, scale=True),
                v2.Normalize(mean=mean, std=std),
            ]
        )
        return transforms