"""DataModule for DivideMix (two augmented views + indices)."""

from __future__ import annotations

from typing import Optional

import torch
from lightning import LightningDataModule
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import v2

from src.data.components.transform_subset import TransformSubset
from src.data.components.two_view_transform_subset import TwoViewTransformSubset


class DivideMixDataModule(LightningDataModule):
    def __init__(
        self,
        train_path: str,
        val_path: str,
        test_path: str,
        image1k_norm: bool = True,
        img_size: int = 480,
        batch_size: int = 64,
        num_workers: int = 4,
        seed: int = 42,
    ) -> None:
        super().__init__()
        self.save_hyperparameters(logger=False)

        self.train_path = train_path
        self.val_path = val_path
        self.test_path = test_path
        self.batch_size = batch_size
        self.num_workers = num_workers

        mean_image1k = [0.485, 0.456, 0.406]
        std_image1k = [0.229, 0.224, 0.225]

        mean_custom = [0.3299, 0.3896, 0.4599]
        std_custom = [0.2219, 0.2155, 0.2540]

        mean = mean_image1k if image1k_norm else mean_custom
        std = std_image1k if image1k_norm else std_custom

        self.train_transforms = v2.Compose(
            [
                v2.Resize(img_size, antialias=True),
                v2.RandomResizedCrop(img_size, scale=(0.5, 1.0)),
                v2.RandomHorizontalFlip(p=0.5),
                v2.RandomVerticalFlip(p=0.5),
                v2.RandomRotation(degrees=(-10, 10)),
                v2.ToImage(),
                v2.ToDtype(torch.float32, scale=True),
                v2.Normalize(mean=mean, std=std),
            ]
        )

        self.train_transforms_alt = v2.Compose(
            [
                v2.Resize(img_size, antialias=True),
                v2.RandomResizedCrop(img_size, scale=(0.5, 1.0)),
                v2.RandomHorizontalFlip(p=0.5),
                v2.RandomVerticalFlip(p=0.5),
                v2.RandomRotation(degrees=(-10, 10)),
                v2.ToImage(),
                v2.ToDtype(torch.float32, scale=True),
                v2.Normalize(mean=mean, std=std),
            ]
        )

        self.test_transforms = v2.Compose(
            [
                v2.Resize(img_size, antialias=True),
                v2.CenterCrop(img_size),
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
        assert self.train_ds is not None, "Dataset not prepared. Call setup() first."
        return len(self.train_ds)

    def setup(self, stage: Optional[str] = None) -> None:
        self.train_ds = datasets.ImageFolder(self.train_path)
        self.val_ds = datasets.ImageFolder(self.val_path)
        self.test_ds = datasets.ImageFolder(self.test_path)

        idxs_train = list(range(len(self.train_ds)))
        idxs_val = list(range(len(self.val_ds)))
        idxs_test = list(range(len(self.test_ds)))

        self.train_dataset = TwoViewTransformSubset(
            self.train_ds,
            idxs_train,
            transform1=self.train_transforms,
            transform2=self.train_transforms_alt,
            return_index=True,
        )

        self.train_eval_dataset = TransformSubset(
            self.train_ds,
            idxs_train,
            transform=self.test_transforms,
            return_index=True,
        )

        self.val_dataset = TransformSubset(
            self.val_ds,
            idxs_val,
            transform=self.test_transforms,
            return_index=False,
        )

        self.test_dataset = TransformSubset(
            self.test_ds,
            idxs_test,
            transform=self.test_transforms,
            return_index=False,
        )

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=True,
            drop_last=True,
            persistent_workers=True if self.num_workers > 0 else False,
        )

    def train_eval_dataloader(self) -> DataLoader:
        return DataLoader(
            self.train_eval_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=False,
            drop_last=False,
            persistent_workers=True if self.num_workers > 0 else False,
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=False,
            persistent_workers=True if self.num_workers > 0 else False,
        )

    def test_dataloader(self) -> DataLoader:
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=False,
            persistent_workers=True if self.num_workers > 0 else False,
        )
