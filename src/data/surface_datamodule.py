import os
from typing import Any, Dict, Optional, Tuple

from lightning import LightningDataModule
import pandas as pd
from sklearn.model_selection import train_test_split
from data.components.transform_subset import TransformSubset
import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets
from torchvision.transforms import v2

from utils.pylogger import RankedLogger

log = RankedLogger(__name__, rank_zero_only=True)

class SurfaceDataModule(LightningDataModule):
    def __init__(
        self,
        data_path: str,
        data_root: str,
        val_split: float = 0.2,
        batch_size: int = 96,
        num_workers: int = 4,
        pin_memory: bool = True,
        use_weighted_sampling: bool = True,
        seed: int = 42,
    ) -> None:
        super().__init__()

        self.save_hyperparameters(logger=False)

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

        self.train_dataset: Optional[Dataset] = None

    @property
    def class_names(self) -> Tuple[str, ...]:
        assert (
            self.train_dataset is not None
        ), "Dataset not prepared. Call prepare_data() first."
        return tuple(self.train_dataset.classes)

    @property
    def num_classes(self) -> int:
        assert (
            self.train_dataset is not None
        ), "Dataset not prepared. Call prepare_data() first."
        return len(self.train_dataset.classes)

    def prepare_data(self):
        """Download/prepare data. Only called on rank 0 in DDP."""
        # Nothing to do

    def setup(self, stage: Optional[str] = None) -> None:
        """Setup datasets. Called on ALL ranks in DDP."""
        # Load base dataset here (not in prepare_data) so it's available on all ranks
        if self.train_dataset is None:
            # Ensure data_url is set
            if not hasattr(self, "data_url"):
                output_path = os.path.join(
                    self.hparams.data_root, self.hparams.data_path
                )
                self.train_data_url = os.path.join(output_path, "train")

            self.train_dataset = datasets.ImageFolder(root=self.train_data_url)

        all_indices = list(range(len(self.train_dataset)))
        all_targets = [self.train_dataset.samples[i][1] for i in all_indices]

        train_indices, val_indices = train_test_split(
            all_indices,
            test_size=self.hparams.val_split,
            random_state=self.hparams.seed,
            stratify=all_targets,
        )

        self.train_datset = TransformSubset(
            self.train_dataset,
            train_indices,
            transform=self.train_transforms,
        )

        self.val_dataset = TransformSubset(
            self.train_dataset,
            val_indices,
            transform=self.test_transforms,
        )

        train_targets = [all_targets[i] for i in train_indices]
        class_counts = pd.Series(train_targets).value_counts().sort_index()
        self.class_weights = 1.0 / class_counts

        if self.hparams.use_weighted_sampling:
            log.info("Using weighted random sampler for training dataloader")
            self.sampling_weights = [
                self.class_weights[target] for target in train_targets
            ]
        else:
            log.info("Using standard random sampler for training dataloader")

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

    def val_dataloader(self) -> Any:
        return DataLoader(
            dataset=self.val_dataset,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=False,
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