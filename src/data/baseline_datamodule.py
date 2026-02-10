from typing import Optional

from torch.utils.data import DataLoader, random_split
from lightning import LightningDataModule
from torchvision.datasets import ImageFolder
import torch

from src.data.components.transform_subset import TransformSubset
from src.data.components.transforms import BaselineTransforms


class BaselineDataModule(LightningDataModule):
    def __init__(
        self,
        data_path: str,
        batch_size: int = 32,
        num_workers: int = 4,
        pin_memory: bool = True,
        seed: int = 42,
    ) -> None:
        super().__init__()
        
        self.save_hyperparameters(logger=False)
        
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]
        
        self.train_transforms = BaselineTransforms.train_transforms(mean, std)
        self.test_transforms = BaselineTransforms.eval_transforms(mean, std)
        
    @property
    def num_classes(self) -> int:
        assert self.train_dataset is not None, "Dataset not prepared. Call setup() first."
        return len(self.train_dataset.parent.classes)
    
    @property
    def class_weights(self) -> torch.Tensor:
        assert hasattr(self, "_class_weights"), "Class weights not computed. Call setup('fit') first."
        return self._class_weights
        

    def setup(self, stage: Optional[str] = None):
        self.ds = ImageFolder(self.hparams.data_path)

        n_train = int(0.8 * len(self.ds))
        n_val = len(self.ds) - n_train
        train_subset, val_subset = random_split(self.ds, [n_train, n_val])

        # Wrap subsets with TransformSubset to apply transforms properly
        self.train_dataset = TransformSubset(self.ds, train_subset.indices, transform=self.train_transforms)
        self.val_dataset = TransformSubset(self.ds, val_subset.indices, transform=self.test_transforms)

        # Calculate class weights from training set
        targets = torch.tensor([self.ds.targets[i] for i in train_subset.indices], dtype=torch.long)
        num_classes = len(self.ds.classes)
        counts = torch.bincount(targets, minlength=num_classes).float()
        counts = torch.clamp(counts, min=1.0)
        N = counts.sum()
        self._class_weights = N / (num_classes * counts)
        
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
            self.val_dataset,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=False,
            persistent_workers=True if self.hparams.num_workers > 0 else False,
        )