from typing import Any, Dict, Optional, Sequence

from torch.utils.data import DataLoader
from lightning import LightningDataModule
from torchvision.datasets import ImageFolder
from torchvision.transforms import v2
import torch

from src.data.components.combined_image_folder import CombinedImageFolder
from src.data.components.transform_subset import TransformSubset
from src.data.components.transforms import (
    MediumTransforms,
    StrongTransforms,
    NoCropTransforms
)
from src.data.components.utils import filter_classes, merge_classes
from src.data.components.dataloader import collate_keep_images_as_list

class HoldoutDataModule(LightningDataModule):
    def __init__(
        self,
        train_path: str,
        val_path: str,
        test_path: str,
        syn_path: Optional[str] = None,
        synthetic_classes: Optional[Sequence[str]] = None,
        synthetic_class_map: Optional[Dict[str, str]] = None,
        classes: Optional[Sequence[str]] = None,
        transforms: str = "medium",
        image1k_norm: bool = True,
        batch_size: int = 96,
        use_weighted_sampler: bool = True,
        weight_alpha: float = 1.0,
        num_workers: int = 4,
        prefetch_factor: int = 2,
        pin_memory: bool = True,
        seed: int = 42,
    ) -> None:
        super().__init__()

        self.save_hyperparameters(logger=False)
        self.hparams: Any

        im_1k_mean = [0.485, 0.456, 0.406]
        im_1k_std = [0.229, 0.224, 0.225]
        
        if transforms == "medium":
            self.train_transforms = MediumTransforms.train_transforms()
            self.test_transforms = MediumTransforms.eval_transforms()
        elif transforms == "strong":
            self.train_transforms = StrongTransforms.train_transforms()
            self.test_transforms = StrongTransforms.eval_transforms()
        elif transforms == "no_crop":
            self.train_transforms = NoCropTransforms.train_transforms()
            self.test_transforms = NoCropTransforms.eval_transforms()
        else:
            raise ValueError(f"Transforms '{transforms}' not recognized.")

        self.resize = v2.Resize(480, antialias=True)
        self.to_float = v2.ToDtype(torch.float32, scale=True)
        self.norm = v2.Normalize(mean=im_1k_mean, std=im_1k_std) if image1k_norm else None

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
    
    @property
    def class_names(self) -> Sequence[str]:
        if hasattr(self, "train_ds") and self.train_ds is not None:
            return self.train_ds.classes
        elif hasattr(self, "val_ds") and self.val_ds is not None:
            return self.val_ds.classes
        elif hasattr(self, "test_ds") and self.test_ds is not None:
            return self.test_ds.classes
        else:
            raise ValueError("Datasets not prepared. Call setup() first.")

    def _add_synthetic_data(self, dataset: ImageFolder) -> CombinedImageFolder:
        if self.hparams.syn_path is None:
            raise ValueError("Synthetic data path not provided.")

        syn_ds = ImageFolder(self.hparams.syn_path)

        if self.hparams.synthetic_classes is not None:
            if self.hparams.classes is not None:
                extra = [c for c in self.hparams.synthetic_classes if c not in self.hparams.classes]
                if extra:
                    raise ValueError(
                        "synthetic_classes must be a subset of classes. "
                        f"Unexpected classes: {extra}"
                    )
            syn_ds = filter_classes(syn_ds, self.hparams.synthetic_classes, allow_missing=False)
        else:
            syn_ds = filter_classes(syn_ds, self.hparams.classes, allow_missing=True)

        if self.hparams.synthetic_class_map is not None:
            merge_map: Dict[str, Sequence[str]] = {
                real_name: [s for s, r in self.hparams.synthetic_class_map.items() if r == real_name]
                for real_name in set(self.hparams.synthetic_class_map.values())
            }
            syn_ds = merge_classes(syn_ds, merge_map, allow_missing=False)

        combined_ds = CombinedImageFolder([dataset, syn_ds])
        return combined_ds
    
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
        x = torch.stack(imgs, dim=0)          # (B,C,480,480)
        x = self.resize(x)                    # ensure CPU resize is applied
        x = self.to_float(x)                  # float in [0,1]
        assert self.norm is not None, "Normalization not initialized. Call setup('fit') first."
        x = self.norm(x)

        return (x, y, idxs) if idxs is not None else (x, y)
    

    def _compute_mean_std(self, dataset: ImageFolder) -> None:
        loader = DataLoader(dataset, batch_size=1, num_workers=self.hparams.num_workers, shuffle=False)
        pixel_sum = torch.zeros(3)
        pixel_sq_sum = torch.zeros(3)
        num_pixels = 0
        to_float = v2.Compose([v2.Resize(480, antialias=True), v2.ToDtype(torch.float32, scale=True)])
        for img, _ in loader:
            img = to_float(img)
            b, c, h, w = img.shape
            pixel_sum += img.sum(dim=[0, 2, 3])
            pixel_sq_sum += (img ** 2).sum(dim=[0, 2, 3])
            num_pixels += b * h * w
        mean = pixel_sum / num_pixels
        std = (pixel_sq_sum / num_pixels - mean ** 2).sqrt()
        self.norm = v2.Normalize(mean=mean.tolist(), std=std.tolist())

    def setup(self, stage: Optional[str] = None):
        if stage == "fit":
            self.train_ds = ImageFolder(self.hparams.train_path)
            self.train_ds = filter_classes(self.train_ds, self.hparams.classes)
            if self.norm is None:
                self._compute_mean_std(self.train_ds)
            if self.hparams.syn_path is not None:
                self.train_ds = self._add_synthetic_data(self.train_ds)
            idxs_train = list(range(len(self.train_ds)))
            self.train_dataset = TransformSubset(
                self.train_ds,
                idxs_train,
                return_index=False,
            )

            targets = torch.tensor(self.train_ds.targets, dtype=torch.long)
            num_classes = len(self.train_ds.classes)
            counts = torch.bincount(targets, minlength=num_classes).float()
            counts = torch.clamp(counts, min=1.0)
            N = counts.sum()
            base = N / (num_classes * counts)
            self._class_weights = base.pow(self.hparams.weight_alpha)

            if self.hparams.use_weighted_sampler:
                # sample weight per sample
                sample_weights = self._class_weights[targets]
                self.sampler = torch.utils.data.WeightedRandomSampler(
                    weights=sample_weights,
                    num_samples=len(sample_weights),
                    replacement=True
                )

        if stage == "fit" or stage == "validate":
            self.val_ds = ImageFolder(self.hparams.val_path)
            self.val_ds = filter_classes(self.val_ds, self.hparams.classes)
            idxs_val = list(range(len(self.val_ds)))
            self.val_dataset = TransformSubset(
                self.val_ds,
                idxs_val,
                return_index=False,
            )

        if stage == "test" or stage == "predict":
            self.test_ds = ImageFolder(self.hparams.test_path)
            self.test_ds = filter_classes(self.test_ds, self.hparams.classes)
            idxs_test = list(range(len(self.test_ds)))
            self.test_dataset = TransformSubset(
                self.test_ds,
                idxs_test,
                return_index=False,
            )

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.train_dataset,
            collate_fn=collate_keep_images_as_list,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=False if self.hparams.use_weighted_sampler else True,
            drop_last=False,
            persistent_workers=True if self.hparams.num_workers > 0 else False,
            sampler=self.sampler if self.hparams.use_weighted_sampler else None,
            prefetch_factor=self.hparams.prefetch_factor,
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
            self.test_dataset,
            collate_fn=collate_keep_images_as_list,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=False,
            persistent_workers=True if self.hparams.num_workers > 0 else False,
        )
