from __future__ import annotations

from typing import Optional

import torch
from torch.utils.data import DataLoader
from lightning import LightningModule
from torchvision.transforms import v2

from data.components.unlabeled_image_folder import UnlabeledImageFolderOrFlat


class TwoCropsTransform:
    """Take two random augmentations of one image."""

    def __init__(self, base_transform):
        self.base_transform = base_transform

    def __call__(self, x):
        q = self.base_transform(x)
        k = self.base_transform(x)
        return q, k


def build_moco_v2_transforms(img_size: int = 480, image1k_norm: bool = True):
    """
    MoCo v2-style augmentations, slightly conservative for defect imagery.
    You can tune color jitter/blur if it harms corrosion vs scratch discrimination.
    """

    mean_image1k = [0.485, 0.456, 0.406]
    std_image1k = [0.229, 0.224, 0.225]

    mean_custom = [0.4496, 0.4970, 0.5210]
    std_cusotm = [0.2539, 0.2344, 0.2480]

    mean = mean_image1k if image1k_norm else mean_custom
    std = std_image1k if image1k_norm else std_cusotm

    normalize = v2.Normalize(mean=mean, std=std)

    augmentation = v2.Compose(
        [
            v2.Resize(img_size, antialias=True),
            v2.RandomResizedCrop(img_size, scale=(0.5, 1.0)),  # avoid tiny crops
            v2.RandomHorizontalFlip(),
            v2.RandomVerticalFlip(),
            v2.RandomRotation(degrees=(-45, 45)),
            v2.RandomApply([v2.GaussianBlur((5,9), (0.1, 2.0))], p=0.5),
            v2.ToImage(),
            v2.ToDtype(torch.float32, scale=True),
            normalize,
        ]
    )
    return TwoCropsTransform(augmentation)


class UnlabeledDataModule(LightningModule):
    def __init__(
        self,
        root_path: str,
        img_size: int = 480,
        batch_size: int = 64,
        num_workers: int = 4,
        image1k_norm: bool = True,
    ):
        super().__init__()

        self.save_hyperparameters(logger=False)

        self.root_path = root_path
        self.img_size = img_size
        self.batch_size_ssl = batch_size
        self.num_workers = num_workers
        self.image1k_norm = image1k_norm

    def setup(self, stage: Optional[str] = None):
        ssl_tf = build_moco_v2_transforms(self.img_size, self.image1k_norm)
        self.ds_unlabeled = UnlabeledImageFolderOrFlat(self.root_path, transform=ssl_tf)

    def train_dataloader(self):
        # This is for SSL pretraining. For supervised training, call supervised_*_dataloader.
        return DataLoader(
            self.ds_unlabeled,
            batch_size=self.batch_size_ssl,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=self.num_workers > 0,
            drop_last=True,
        )
