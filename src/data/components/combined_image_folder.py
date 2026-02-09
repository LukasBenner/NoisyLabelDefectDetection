import torch
from torch.utils.data import Dataset
from torchvision.datasets import ImageFolder


from typing import List, Sequence, Tuple


class CombinedImageFolder(Dataset):
    """Concat ImageFolder datasets while preserving ImageFolder attributes.

    Supports union of class sets across datasets by remapping targets to a
    unified class index space.
    """

    def __init__(self, datasets: Sequence[ImageFolder]) -> None:
        if len(datasets) == 0:
            raise ValueError("At least one dataset is required.")
        self.datasets: List[ImageFolder] = list(datasets)

        # Build a unified class list (keep order from first dataset, then append new).
        class_names: List[str] = list(self.datasets[0].classes)
        for ds in self.datasets[1:]:
            for name in ds.classes:
                if name not in class_names:
                    class_names.append(name)

        self.classes = class_names
        self.class_to_idx = {name: idx for idx, name in enumerate(self.classes)}

        # Precompute per-dataset target remapping to unified indices.
        self._target_remap: List[List[int]] = []
        for ds in self.datasets:
            remap = [self.class_to_idx[name] for name in ds.classes]
            self._target_remap.append(remap)

        # Concatenate targets and samples for downstream code (with remapped labels).
        self.targets: List[int] = []
        self.samples: List[Tuple[str, int]] = []
        for ds_idx, ds in enumerate(self.datasets):
            remap = self._target_remap[ds_idx]
            if hasattr(ds, "samples"):
                for path, target in ds.samples:
                    self.samples.append((path, remap[target]))
            if hasattr(ds, "targets"):
                self.targets.extend([remap[t] for t in ds.targets])

        # torchvision also exposes imgs as alias of samples
        self.imgs = self.samples

        lengths = [len(ds) for ds in self.datasets]
        self.cumulative_sizes = torch.cumsum(torch.tensor(lengths), dim=0).tolist()

    def __len__(self) -> int:
        return self.cumulative_sizes[-1]

    def __getitem__(self, index: int):
        if index < 0:
            index = len(self) + index
        if index < 0 or index >= len(self):
            raise IndexError("Index out of range")

        dataset_idx = 0
        while index >= self.cumulative_sizes[dataset_idx]:
            dataset_idx += 1
        if dataset_idx == 0:
            sample_idx = index
        else:
            sample_idx = index - self.cumulative_sizes[dataset_idx - 1]
        img, target = self.datasets[dataset_idx][sample_idx]
        target = self._target_remap[dataset_idx][target]
        return img, target