from typing import Any, Optional

from torch.utils.data import DataLoader
from torch.utils.data.dataset import Subset
from torch.utils.data import random_split
from lightning import LightningDataModule
from torchvision.datasets import CIFAR100
from torchvision.transforms import v2
import torch
import numpy as np

def other_class(n_classes: int, current_class: int, rng: np.random.Generator) -> int:
    """
    Returns a list of class indices excluding the class indexed by class_ind
    :param nb_classes: number of classes in the task
    :param class_ind: the class index to be omitted
    :return: one random class that != class_ind
    """
    if current_class < 0 or current_class >= n_classes:
        error_str = "class_ind must be within the range (0, nb_classes - 1)"
        raise ValueError(error_str)

    other_class_list = list(range(n_classes))
    other_class_list.remove(current_class)
    return int(rng.choice(other_class_list))


def make_symmetric_noisy_targets(
    targets: list[int],
    noise_rate: float,
    n_classes: int,
    rng: np.random.Generator,
) -> list[int]:
    """Flip labels within each class to a random other class.

    This matches the paper's description: for each class, randomly flip a fraction of
    its samples to an incorrect label drawn uniformly from the other classes.
    """

    if noise_rate <= 0:
        return list(targets)
    if not (0.0 <= noise_rate < 1.0):
        raise ValueError("noise_rate must be in [0, 1)")

    targets_arr = np.asarray(targets, dtype=np.int64)
    noisy_targets = targets_arr.copy()

    for cls in range(n_classes):
        cls_idx = np.where(targets_arr == cls)[0]
        if cls_idx.size == 0:
            continue
        n_flip = int(round(noise_rate * cls_idx.size))
        if n_flip <= 0:
            continue
        flip_idx = rng.choice(cls_idx, size=n_flip, replace=False)
        for i in flip_idx:
            noisy_targets[i] = other_class(n_classes=n_classes, current_class=int(noisy_targets[i]), rng=rng)

    return noisy_targets.tolist()

class CIFAR100DataModule(LightningDataModule):
    def __init__(
        self,
        data_dir: str = "./data_tmp",
        noise_rate: float = 0.0,
        batch_size: int = 128,
        num_workers: int = 4,
        pin_memory: bool = True,
        seed: int = 42,
    ) -> None:
        super().__init__()
        
        self.save_hyperparameters(logger=False)
        
    @property
    def num_classes(self) -> int:
        return 100

    def prepare_data(self):
        """Download CIFAR-100 dataset."""
        CIFAR100(self.hparams.data_dir, train=True, download=True)
        CIFAR100(self.hparams.data_dir, train=False, download=True)

    def setup(self, stage: Optional[str] = None):
        """Setup train, val, and test datasets."""

        cifar_mean = [0.5071, 0.4865, 0.4409]
        cifar_std = [0.2673, 0.2564, 0.2762]

        train_transforms = v2.Compose(
            [
                # "random width/height shift" in the paper is effectively random crop with padding
                v2.RandomCrop(32, padding=4),
                v2.RandomHorizontalFlip(),
                v2.ToImage(),
                v2.ToDtype(torch.float32, scale=True),
                v2.Normalize(mean=cifar_mean, std=cifar_std),
            ]
        )

        eval_transforms = v2.Compose(
            [
                v2.ToImage(),
                v2.ToDtype(torch.float32, scale=True),
                v2.Normalize(mean=cifar_mean, std=cifar_std),
            ]
        )

        base_train = CIFAR100(
            self.hparams.data_dir,
            train=True,
            transform=None,
            download=False,
        )

        rng = np.random.default_rng(int(self.hparams.seed))
        noisy_targets = make_symmetric_noisy_targets(
            targets=list(base_train.targets),
            noise_rate=float(self.hparams.noise_rate),
            n_classes=self.num_classes,
            rng=rng,
        )

        train_full = CIFAR100(
            self.hparams.data_dir,
            train=True,
            transform=train_transforms,
            download=False,
        )
        train_full.targets = noisy_targets
        
        self.train_dataset = train_full
        
        self.test_dataset = CIFAR100(
            self.hparams.data_dir,
            train=False,
            transform=eval_transforms,
            download=False,
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
            self.test_dataset,
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
