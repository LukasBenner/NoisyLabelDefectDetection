from __future__ import annotations

from typing import Optional, Sequence

from PIL import Image
import numpy as np
from torch.utils.data import Dataset
from torchvision import datasets


class DivideMixImageFolderDataset(Dataset):
    def __init__(
        self,
        train_dir: str,
        transform,
        mode: str,
        val_dir: Optional[str] = None,
        test_dir: Optional[str] = None,
        pred: Optional[Sequence[bool]] = None,
        probability: Optional[Sequence[float]] = None,
    ) -> None:
        self.transform = transform
        self.mode = mode
        self.train_dir = train_dir
        self.val_dir = val_dir
        self.test_dir = test_dir

        if self.mode == "test":
            if not self.test_dir:
                raise ValueError('test_dir is required for mode="test"')
            dataset = datasets.ImageFolder(self.test_dir)
            self.test_imgs = [path for path, _ in dataset.samples]
            self.test_labels = [label for _, label in dataset.samples]
            self.classes = dataset.classes
            self.class_to_idx = dataset.class_to_idx
        elif self.mode == "val":
            if not self.val_dir:
                raise ValueError('val_dir is required for mode="val"')
            dataset = datasets.ImageFolder(self.val_dir)
            self.val_imgs = [path for path, _ in dataset.samples]
            self.val_labels = [label for _, label in dataset.samples]
            self.classes = dataset.classes
            self.class_to_idx = dataset.class_to_idx
        else:
            dataset = datasets.ImageFolder(self.train_dir)
            self.train_imgs = [path for path, _ in dataset.samples]
            self.noise_label = [label for _, label in dataset.samples]
            self.classes = dataset.classes
            self.class_to_idx = dataset.class_to_idx

            if self.mode == "all":
                pass
            elif self.mode == "labeled":
                if pred is None or probability is None:
                    raise ValueError("pred and probability are required for mode='labeled'")
                pred_arr = np.asarray(pred)
                prob_arr = np.asarray(probability)
                if len(pred_arr) != len(self.train_imgs):
                    raise ValueError("pred length must match training set length")
                if len(prob_arr) != len(self.train_imgs):
                    raise ValueError("probability length must match training set length")
                pred_idx = np.nonzero(pred_arr)[0]
                self.train_imgs = [self.train_imgs[i] for i in pred_idx]
                self.noise_label = [self.noise_label[i] for i in pred_idx]
                self.probability = [prob_arr[i] for i in pred_idx]
            elif self.mode == "unlabeled":
                if pred is None:
                    raise ValueError("pred is required for mode='unlabeled'")
                pred_arr = np.asarray(pred)
                if len(pred_arr) != len(self.train_imgs):
                    raise ValueError("pred length must match training set length")
                pred_idx = np.nonzero(1 - pred_arr)[0]
                self.train_imgs = [self.train_imgs[i] for i in pred_idx]
                self.noise_label = [self.noise_label[i] for i in pred_idx]
            else:
                raise ValueError(f"Unknown mode: {self.mode}")

    def __getitem__(self, index: int):
        if self.mode == "labeled":
            img_path = self.train_imgs[index]
            target = self.noise_label[index]
            prob = self.probability[index]
            image = Image.open(img_path).convert("RGB")
            img1 = self.transform(image)
            img2 = self.transform(image)
            return img1, img2, target, prob
        if self.mode == "unlabeled":
            img_path = self.train_imgs[index]
            image = Image.open(img_path).convert("RGB")
            img1 = self.transform(image)
            img2 = self.transform(image)
            return img1, img2
        if self.mode == "all":
            img_path = self.train_imgs[index]
            target = self.noise_label[index]
            image = Image.open(img_path).convert("RGB")
            img = self.transform(image)
            return img, target, index
        if self.mode == "test":
            img_path = self.test_imgs[index]
            target = self.test_labels[index]
            image = Image.open(img_path).convert("RGB")
            img = self.transform(image)
            return img, target
        if self.mode == "val":
            img_path = self.val_imgs[index]
            target = self.val_labels[index]
            image = Image.open(img_path).convert("RGB")
            img = self.transform(image)
            return img, target
        raise ValueError(f"Unknown mode: {self.mode}")

    def __len__(self) -> int:
        if self.mode == "test":
            return len(self.test_imgs)
        if self.mode == "val":
            return len(self.val_imgs)
        return len(self.train_imgs)
