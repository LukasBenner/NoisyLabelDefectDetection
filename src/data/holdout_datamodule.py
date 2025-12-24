from typing import Any, Optional

from torch.utils.data import DataLoader, random_split
from pytorch_lightning.utilities.combined_loader import CombinedLoader
from lightning import LightningDataModule
from torchvision.datasets import ImageFolder
from torchvision.transforms import v2
import torch


class HoldoutDataModule(LightningDataModule):
    def __init__(
        self,
        train_path: str,
        val_path: str,
        test_path: str,
        batch_size: int = 96,
        meta_batch_size: int = 96,
        num_workers: int = 4,
        pin_memory: bool = True,
        seed: int = 42
    ) -> None:
        super().__init__()
        self.train_path = train_path
        self.val_path = val_path
        self.test_path = test_path
        self.batch_size = batch_size
        self.meta_batch_size = meta_batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory

        self.train_transforms = v2.Compose(
            [
                v2.Resize(480, antialias=True),
                v2.RandomCrop(480),
                v2.RandomHorizontalFlip(p=0.5),
                v2.RandomVerticalFlip(p=0.5),
                v2.RandomRotation(degrees=(-5,5)),
                v2.ToImage(),
                v2.ToDtype(torch.float32, scale=True),
                v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )
        self.test_transforms = v2.Compose(
            [
                v2.Resize(480, antialias=True),
                v2.CenterCrop(480),
                v2.ToImage(),
                v2.ToDtype(torch.float32, scale=True),
                v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )

    def setup(self, stage: Optional[str] = None):
        self.train_ds = ImageFolder(self.train_path, transform=self.train_transforms)
        self.val_ds = ImageFolder(self.val_path, transform=self.test_transforms)
        self.tets_ds = ImageFolder(self.test_path, transform=self.test_transforms)
        
    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.train_ds, 
            batch_size=self.batch_size, 
            shuffle=True,
            num_workers=self.num_workers
        )
    
    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self.val_ds, 
            batch_size=self.batch_size, 
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
        )
        
    def test_dataloader(self) -> DataLoader:
        return DataLoader(
            self.tets_ds, 
            batch_size=self.batch_size, 
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
        )
        
        
        
if __name__ == "__main__":
    from matplotlib import pyplot as plt
    train_path = "data/surface/train_new"
    val_path = "data/surface/val"
    test_path = "data/surface/test_new"
    data = HoldoutDataModule(train_path, val_path, test_path, num_workers=0)
    data.setup()
    loaders = data.train_dataloader()