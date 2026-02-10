from typing import Dict, Optional, Sequence

from torch.utils.data import DataLoader
from lightning import LightningDataModule
from torchvision.datasets import ImageFolder
import torch

from src.data.components.combined_image_folder import CombinedImageFolder
from src.data.components.transform_subset import TransformSubset
from src.data.components.transforms import (
    BaselineTransforms,
    MediumTransforms,
    StrongTransforms,
)
from src.data.components.utils import filter_classes, merge_classes

class CombinedHoldoutDataModule(LightningDataModule):
    def __init__(
        self,
        train_path: str,
        val_path: str,
        test_path: str,
        syn_path: Optional[str] = None,
        classes: Optional[Sequence[str]] = None,
        merge_classes: Optional[Dict[str, Sequence[str]]] = None,
        transforms: str = "medium",
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

        if transforms == "baseline":
            self.train_transforms = BaselineTransforms.train_transforms(mean, std)
            self.test_transforms = BaselineTransforms.eval_transforms(mean, std)

        elif transforms == "medium":
            self.train_transforms = MediumTransforms.train_transforms(mean, std)
            self.test_transforms = MediumTransforms.eval_transforms(mean, std)

        elif transforms == "strong":
            self.train_transforms = StrongTransforms.train_transforms(mean, std)
            self.test_transforms = StrongTransforms.eval_transforms(mean, std)

        else:
            raise ValueError(f"Transforms '{transforms}' not recognized.")

    @property
    def num_classes(self) -> int:
        assert self.train_ds is not None, "Dataset not prepared. Call setup() first."
        return len(self.train_ds.classes)

    @property
    def class_weights(self) -> torch.Tensor:
        assert hasattr(
            self, "_class_weights"
        ), "Class weights not computed. Call setup('fit') first."
        return self._class_weights

    def _add_synthetic_data(self, dataset: ImageFolder) -> ImageFolder:
        if self.hparams.syn_path is None:
            raise ValueError("Synthetic data path not provided.")

        syn_ds = ImageFolder(self.hparams.syn_path)
        syn_ds = filter_classes(syn_ds, self.hparams.classes, allow_missing=True)
        syn_ds = merge_classes(syn_ds, self.hparams.merge_classes, allow_missing=True)
        combined_ds = CombinedImageFolder([dataset, syn_ds])
        return combined_ds

    def setup(self, stage: Optional[str] = None):
        if stage == "fit":
            self.train_ds = ImageFolder(self.hparams.train_path)
            self.train_ds = filter_classes(self.train_ds, self.hparams.classes)
            self.train_ds = merge_classes(self.train_ds, self.hparams.merge_classes)
            if self.hparams.syn_path is not None:
                self.train_ds = self._add_synthetic_data(self.train_ds)
            idxs_train = list(range(len(self.train_ds)))
            self.train_dataset = TransformSubset(
                self.train_ds,
                idxs_train,
                transform=self.train_transforms,
                return_index=False,
            )

            targets = torch.tensor(self.train_ds.targets, dtype=torch.long)
            num_classes = len(self.train_ds.classes)
            counts = torch.bincount(targets, minlength=num_classes).float()
            counts = torch.clamp(counts, min=1.0)
            N = counts.sum()
            self._class_weights = N / (num_classes * counts)

        if stage == "fit" or stage == "validate":
            self.val_ds = ImageFolder(self.hparams.val_path)
            self.val_ds = filter_classes(self.val_ds, self.hparams.classes)
            self.val_ds = merge_classes(self.val_ds, self.hparams.merge_classes)
            idxs_val = list(range(len(self.val_ds)))
            self.val_dataset = TransformSubset(
                self.val_ds,
                idxs_val,
                transform=self.test_transforms,
                return_index=False,
            )

        if stage == "test" or stage == "predict":
            self.test_ds = ImageFolder(self.hparams.test_path)
            self.test_ds = filter_classes(self.test_ds, self.hparams.classes)
            self.test_ds = merge_classes(self.test_ds, self.hparams.merge_classes)
            idxs_test = list(range(len(self.test_ds)))
            self.test_dataset = TransformSubset(
                self.test_ds,
                idxs_test,
                transform=self.test_transforms,
                return_index=False,
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

    def predict_dataloader(self) -> DataLoader:
        return DataLoader(
            self.test_dataset,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=False,
            persistent_workers=True if self.hparams.num_workers > 0 else False,
        )
