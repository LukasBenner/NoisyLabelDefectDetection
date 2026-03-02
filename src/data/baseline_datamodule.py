from typing import Any, Optional, Sequence

from torch.utils.data import DataLoader, random_split
from lightning import LightningDataModule
from torchvision.datasets import ImageFolder
from torchvision import transforms as T
import torch

from src.data.components.transform_subset import TransformSubset

from src.data.components.dataloader import collate_keep_images_as_list


class BaselineDataModule(LightningDataModule):
    def __init__(
        self,
        data_path: str,
        batch_size: int = 32,
        num_workers: int = 4,
        pin_memory: bool = True,
        transforms: str = "baseline_code",
        seed: int = 42,
    ) -> None:
        super().__init__()
        
        self.save_hyperparameters(logger=False)
        self.hparams: Any
        
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]
        
        if transforms == "baseline_code":
            self.train_transforms = T.Compose(
            [
                T.Resize((640, 480)),
                T.RandomHorizontalFlip(),
                T.RandomVerticalFlip(),
            ])
            self.test_transforms = T.Compose(
            [
                T.Resize((640, 480))
            ])
            
        elif transforms == "baseline_paper":
            self.train_transforms = T.Compose(
            [
                T.Resize((224, 224)),
                T.RandomRotation(degrees=(180, 180)),
                T.RandomHorizontalFlip(p=1.0),
                T.RandomVerticalFlip(p=1.0),
                T.ColorJitter(saturation=3.0, contrast=3.0),
            ])
            self.test_transforms = T.Compose(
            [
                T.Resize((224, 224))
            ])
        else:
            raise ValueError(f"Unknown transform: {transforms}")
        
        self.to_float = T.ConvertImageDtype(torch.float32)
        self.norm = T.Normalize(mean=mean, std=std)
        
    @property
    def num_classes(self) -> int:
        assert self.train_dataset is not None, "Dataset not prepared. Call setup() first."
        return len(self.train_dataset.parent.classes)
    
    @property
    def class_weights(self) -> torch.Tensor:
        assert hasattr(self, "_class_weights"), "Class weights not computed. Call setup('fit') first."
        return self._class_weights
    
    @property
    def class_names(self) -> Sequence[str]:
        if hasattr(self, "ds") and self.ds is not None:
            return self.ds.classes
        else:
            raise ValueError("Dataset not prepared. Call setup() first.")

    def setup(self, stage: Optional[str] = None):
        if hasattr(self, 'val_dataset'):
            return
        
        self.ds = ImageFolder(self.hparams.data_path)

        n_train = int(0.8 * len(self.ds))
        n_val = len(self.ds) - n_train
        train_subset, val_subset = random_split(self.ds, [n_train, n_val])

        # Wrap subsets with TransformSubset to apply transforms properly
        self.train_dataset = TransformSubset(self.ds, train_subset.indices)
        self.val_dataset = TransformSubset(self.ds, val_subset.indices)

        # Calculate class weights from training set
        targets = torch.tensor([self.ds.targets[i] for i in train_subset.indices], dtype=torch.long)
        num_classes = len(self.ds.classes)
        counts = torch.bincount(targets, minlength=num_classes).float()
        counts = torch.clamp(counts, min=1.0)
        N = counts.sum()
        self._class_weights = N / (num_classes * counts)
        
    def on_after_batch_transfer(self, batch: Any, dataloader_idx: int) -> Any:
        if len(batch) == 3:
            imgs, y, idxs = batch
        else:
            imgs, y = batch
            idxs = None

        geom = self.train_transforms if self.trainer.training else self.test_transforms

        # per-image GPU transforms; each output becomes (C,480,480)
        imgs = [geom(img) for img in imgs]

        # now shapes match -> stack
        x = torch.stack(imgs, dim=0)         # (B,C,480,480)
        x = self.to_float(x)                  # float in [0,1]
        x = self.norm(x)

        return (x, y, idxs) if idxs is not None else (x, y)
        
    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.train_dataset,
            collate_fn=collate_keep_images_as_list,
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
            collate_fn=collate_keep_images_as_list,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=False,
            persistent_workers=True if self.hparams.num_workers > 0 else False,
        )

    def test_dataloader(self) -> DataLoader:
        return DataLoader(
            self.val_dataset,
            collate_fn=collate_keep_images_as_list,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=False,
            persistent_workers=True if self.hparams.num_workers > 0 else False,
        )