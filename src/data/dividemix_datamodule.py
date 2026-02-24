from __future__ import annotations

from typing import Optional, Sequence, List

from lightning import LightningDataModule
from lightning.pytorch.utilities.combined_loader import CombinedLoader
from torch import device
from torch._C import device
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision import datasets
import numpy as np

from src.data.components.dividemix_imagefolder import DivideMixImageFolderDataset


class DivideMixImageFolderDataModule(LightningDataModule):
    def __init__(
        self,
        train_dir: str,
        val_dir: Optional[str] = None,
        test_dir: Optional[str] = None,
        image1k_norm: bool = True,
        batch_size: int = 32,
        num_workers: int = 8,
        pin_memory: bool = True,
        warmup_batch_mult: int = 2,
        seed: int = 42,
    ) -> None:
        super().__init__()

        self.save_hyperparameters(logger=False)

        mean_image1k = [0.485, 0.456, 0.406]
        std_image1k = [0.229, 0.224, 0.225]

        mean_custom = [0.3299, 0.3896, 0.4599]
        std_custom = [0.2219, 0.2155, 0.2540]

        img_size = 480

        mean = mean_image1k if image1k_norm else mean_custom
        std = std_image1k if image1k_norm else std_custom

        self.transform_train = transforms.Compose(
            [
                transforms.RandomResizedCrop(img_size, scale=(0.8, 1.0)),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(mean, std),
            ]
        )
        self.transform_test = transforms.Compose(
            [
                transforms.CenterCrop(img_size),
                transforms.ToTensor(),
                transforms.Normalize(mean, std),
            ]
        )

        self._train_mode = "warmup"
        self._pred1 = None
        self._prob1 = None
        self._pred2 = None
        self._prob2 = None

    @property
    def num_classes(self) -> int:
        assert hasattr(self, "train_base"), "Dataset not prepared. Call setup() first."
        return len(self.train_base.classes)

    @property
    def class_names(self) -> Sequence[str]:
        assert hasattr(self, "train_base"), "Dataset not prepared. Call setup() first."
        return self.train_base.classes

    def setup(self, stage: Optional[str] = None) -> None:
        if stage == "fit" or stage is None:
            self.train_base = datasets.ImageFolder(self.hparams.train_dir)

        if (stage == "fit" or stage == "validate" or stage is None) and self.hparams.val_dir:
            self.val_dataset = DivideMixImageFolderDataset(
                train_dir=self.hparams.train_dir,
                val_dir=self.hparams.val_dir,
                transform=self.transform_test,
                mode="val",
            )

        if (stage == "test" or stage == "predict" or stage is None) and self.hparams.test_dir:
            self.test_dataset = DivideMixImageFolderDataset(
                train_dir=self.hparams.train_dir,
                test_dir=self.hparams.test_dir,
                transform=self.transform_test,
                mode="test",
            )

    def set_train_mode(self, mode: str) -> None:
        if mode not in ("warmup", "train"):
            raise ValueError(f"Unknown train mode: {mode}")
        self._train_mode = mode

    def set_pred_prob(
        self,
        pred1: Sequence[bool],
        prob1: Sequence[float],
        pred2: Sequence[bool],
        prob2: Sequence[float],
    ) -> None:
        self._pred1 = np.asarray(pred1)
        self._prob1 = np.asarray(prob1)
        self._pred2 = np.asarray(pred2)
        self._prob2 = np.asarray(prob2)

    def _build_warmup_loader(self) -> DataLoader:
        warmup_dataset = DivideMixImageFolderDataset(
            train_dir=self.hparams.train_dir,
            transform=self.transform_train,
            mode="all",
        )
        return DataLoader(
            dataset=warmup_dataset,
            batch_size=int(self.hparams.batch_size) * int(self.hparams.warmup_batch_mult),
            shuffle=True,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            drop_last=False,
            persistent_workers=True if self.hparams.num_workers > 0 else False,
        )

    def _build_train_loader(
        self, pred: Sequence[bool], prob: Sequence[float]
    ) -> CombinedLoader:
        labeled_dataset = DivideMixImageFolderDataset(
            train_dir=self.hparams.train_dir,
            transform=self.transform_train,
            mode="labeled",
            pred=pred,
            probability=prob,
        )
        unlabeled_dataset = DivideMixImageFolderDataset(
            train_dir=self.hparams.train_dir,
            transform=self.transform_train,
            mode="unlabeled",
            pred=pred,
        )
        labeled_loader = DataLoader(
            dataset=labeled_dataset,
            batch_size=self.hparams.batch_size,
            shuffle=True,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            drop_last=False,
            persistent_workers=True if self.hparams.num_workers > 0 else False,
        )
        unlabeled_loader = DataLoader(
            dataset=unlabeled_dataset,
            batch_size=self.hparams.batch_size,
            shuffle=True,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            drop_last=False,
            persistent_workers=True if self.hparams.num_workers > 0 else False,
        )

        return CombinedLoader(
            {"labeled": labeled_loader, "unlabeled": unlabeled_loader},
            mode="max_size_cycle",
        )

    def train_dataloader(self):
        if self._train_mode == "warmup":
            warmup_loader = self._build_warmup_loader()
            return [warmup_loader, warmup_loader]

        if self._pred1 is None or self._pred2 is None:
            raise RuntimeError("Predictions not set for training mode.")

        loader1 = self._build_train_loader(self._pred2, self._prob2)
        loader2 = self._build_train_loader(self._pred1, self._prob1)
        return [loader1, loader2]

    def train_eval_dataloader(self) -> DataLoader:
        eval_dataset = DivideMixImageFolderDataset(
            train_dir=self.hparams.train_dir,
            transform=self.transform_test,
            mode="all",
        )
        return DataLoader(
            dataset=eval_dataset,
            batch_size=self.hparams.batch_size,
            shuffle=False,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            drop_last=False,
            persistent_workers=True if self.hparams.num_workers > 0 else False,
        )

    def val_dataloader(self) -> Optional[DataLoader]:
        if not hasattr(self, "val_dataset"):
            return None
        return DataLoader(
            dataset=self.val_dataset,
            batch_size=self.hparams.batch_size,
            shuffle=False,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            drop_last=False,
            persistent_workers=True if self.hparams.num_workers > 0 else False,
        )

    def test_dataloader(self) -> Optional[DataLoader]:
        if not hasattr(self, "test_dataset"):
            return None
        return DataLoader(
            dataset=self.test_dataset,
            batch_size=self.hparams.batch_size,
            shuffle=False,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            drop_last=False,
            persistent_workers=True if self.hparams.num_workers > 0 else False,
        )
