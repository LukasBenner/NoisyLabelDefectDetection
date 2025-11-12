import os
from typing import Any, Dict, Optional, Tuple

from lightning import LightningDataModule
import pandas as pd
from sklearn.model_selection import train_test_split
from src.data.components.TransformSubset import TransformSubset
import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets
from torchvision.transforms import v2
import dvc.api


class SimpleDataModule(LightningDataModule):
    def __init__(
        self,
        data_path: str,
        data_repo: str,
        val_split: float = 0.2,
        batch_size: int = 96,
        num_workers: int = 4,
        pin_memory: bool = True,
        use_weighted_sampling: bool = True,
        seed: int = 42,
    ) -> None:
        super().__init__()

        self.data_url = os.path.join(data_repo, data_path, "train")

        self.save_hyperparameters()
        self.train_transforms = v2.Compose(
            [
                v2.Resize((640, 480), antialias=True),
                v2.RandomHorizontalFlip(),
                v2.RandomVerticalFlip(),
                v2.ToImage(),
                v2.ToDtype(torch.float32, scale=True),
                v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )
        self.test_transforms = v2.Compose(
            [
                v2.Resize((640, 480), antialias=True),
                v2.ToImage(),
                v2.ToDtype(torch.float32, scale=True),
                v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )

        self.base_dataset: Optional[Dataset] = None

    @property
    def class_names(self) -> Tuple[str, ...]:
        assert (
            self.base_dataset is not None
        ), "Dataset not prepared. Call prepare_data() first."
        return tuple(self.base_dataset.classes)

    @property
    def num_classes(self) -> int:
        assert (
            self.base_dataset is not None
        ), "Dataset not prepared. Call prepare_data() first."
        return len(self.base_dataset.classes)

    def prepare_data(self):
        self.base_dataset = datasets.ImageFolder(root=self.data_url)

    def setup(self, stage: Optional[str] = None) -> None:
        assert (
            self.base_dataset is not None
        ), "Dataset not prepared. Call prepare_data() first."

        all_indices = list(range(len(self.base_dataset)))
        all_targets = [self.base_dataset.samples[i][1] for i in all_indices]

        train_indices, test_inidices = train_test_split(
            all_indices,
            test_size=self.hparams.val_split,
            random_state=self.hparams.seed,
            stratify=all_targets,
        )

        self.train_datset = TransformSubset(
            self.base_dataset,
            train_indices,
            transform=self.train_transforms,
        )

        self.test_dataset = TransformSubset(
            self.base_dataset,
            test_inidices,
            transform=self.test_transforms,
        )

        train_targets = [all_targets[i] for i in train_indices]
        class_counts = pd.Series(train_targets).value_counts().sort_index()
        self.class_weights = 1.0 / class_counts

        if self.hparams.use_weighted_sampling:
            self.sampling_weights = [
                self.class_weights[target] for target in train_targets
            ]

    def train_dataloader(self) -> DataLoader:
        if self.hparams.use_weighted_sampling:
            sampler = torch.utils.data.WeightedRandomSampler(
                weights=self.sampling_weights,
                num_samples=len(self.sampling_weights),
                replacement=True,
            )
            return DataLoader(
                self.train_datset,
                batch_size=self.hparams.batch_size,
                num_workers=self.hparams.num_workers,
                pin_memory=self.hparams.pin_memory,
                sampler=sampler,
                persistent_workers=True if self.hparams.num_workers > 0 else False,
            )
        else:
            return DataLoader(
                self.train_datset,
                batch_size=self.hparams.batch_size,
                num_workers=self.hparams.num_workers,
                pin_memory=self.hparams.pin_memory,
                shuffle=True,
                persistent_workers=True if self.hparams.num_workers > 0 else False,
            )
    
    def test_dataloader(self):
        return DataLoader(
            dataset=self.test_dataset,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=False,
            persistent_workers=True if self.hparams.num_workers > 0 else False,
        )
