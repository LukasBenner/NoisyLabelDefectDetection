"""DataModule for SEAL that returns sample indices along with data."""

from typing import Optional

from lightning import LightningDataModule
from src.data.components.transform_subset import TransformSubset
import torch
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import v2


class SEALDataModule(LightningDataModule):
    def __init__(
        self,
        train_path: str,
        val_path: str,
        test_path: str,
        image1k_norm: bool = True,
        batch_size: int = 96,
        num_workers: int = 4,
        pin_memory: bool = True,
        seed: int = 42,
    ) -> None:
        super().__init__()

        self.save_hyperparameters(logger=False)
        
        mean_image1k = [0.485, 0.456, 0.406]
        std_image1k = [0.229, 0.224, 0.225]
        
        mean_custom = [0.3299, 0.3896, 0.4599]
        std_custom = [0.2219, 0.2155, 0.2540]
        
        mean = mean_image1k if image1k_norm else mean_custom
        std = std_image1k if image1k_norm else std_custom

        self.train_transforms = v2.Compose(
            [
                v2.Resize(480, antialias=True),
                v2.RandomCrop(480, padding=8),
                v2.RandomHorizontalFlip(p=0.5),
                v2.RandomVerticalFlip(p=0.5),
                v2.RandomRotation(degrees=(-5, 5)),
                v2.ToImage(),
                v2.ToDtype(torch.float32, scale=True),
                v2.Normalize(mean=mean, std=std),
            ]
        )
        self.test_transforms = v2.Compose(
            [
                v2.Resize(480, antialias=True),
                v2.CenterCrop(480),
                v2.ToImage(),
                v2.ToDtype(torch.float32, scale=True),
                v2.Normalize(mean=mean, std=std),
            ]
        )

    @property
    def class_names(self):
        assert self.train_ds is not None, "Dataset not prepared. Call setup() first."
        return tuple(self.train_ds.classes)

    @property
    def num_classes(self) -> int:
        assert self.train_ds is not None, "Dataset not prepared. Call setup() first."
        return len(self.train_ds.classes)
    
    @property
    def num_train_samples(self) -> int:
        """Return the number of training samples (needed for SEAL soft label buffer)."""
        assert self.train_ds is not None, "Dataset not prepared. Call setup() first."
        return len(self.train_ds)

    def setup(self, stage: Optional[str] = None) -> None:
        self.train_ds = datasets.ImageFolder(self.hparams.train_path)
        self.val_ds = datasets.ImageFolder(self.hparams.val_path)
        self.test_ds = datasets.ImageFolder(self.hparams.test_path)
        
        idxs_train = list(range(len(self.train_ds)))
        idxs_val = list(range(len(self.val_ds)))
        idxs_test = list(range(len(self.test_ds)))

        self.train_dataset = TransformSubset(
            self.train_ds,
            idxs_train,
            transform=self.train_transforms,
            return_index=True,  # SEAL needs indices
        )
        
        self.train_val_dataset = TransformSubset(
            self.train_ds,
            idxs_train,
            transform=self.test_transforms,
            return_index=True,
        )

        self.val_dataset = TransformSubset(
            self.val_ds,
            idxs_val,
            transform=self.test_transforms,
            return_index=False,  # Validation doesn't need indices
        )

        self.test_dataset = TransformSubset(
            self.test_ds,
            idxs_test,
            transform=self.test_transforms,
            return_index=False,  # Test doesn't need indices
        )

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.train_dataset,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=True,
            drop_last=False,
            persistent_workers=True if self.hparams.num_workers > 0 else False,
        )
        
    def train_eval_dataloader(self) -> DataLoader:
        return DataLoader(
            self.train_val_dataset,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=False,
            drop_last=False,
            persistent_workers=True if self.hparams.num_workers > 0 else False,
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self.val_dataset,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=False,
            persistent_workers=True if self.hparams.num_workers > 0 else False,
        )

    def test_dataloader(self) -> DataLoader:
        return DataLoader(
            self.test_dataset,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=False,
            persistent_workers=True if self.hparams.num_workers > 0 else False,
        )
