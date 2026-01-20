from __future__ import annotations

from typing import Callable, Optional, Tuple

from PIL import Image
from torch.utils.data import Dataset


class TwoViewTransformSubset(Dataset):
    """Dataset wrapper that returns two augmented views per sample."""

    def __init__(
        self,
        dataset,
        indices,
        transform1: Optional[Callable] = None,
        transform2: Optional[Callable] = None,
        return_index: bool = False,
    ) -> None:
        self.parent = dataset
        self.indices = indices
        self.transform1 = transform1
        self.transform2 = transform2 if transform2 is not None else transform1
        self.return_index = return_index

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, idx: int):
        real_idx = self.indices[idx]
        path, label = self.parent.samples[real_idx]
        img = Image.open(path).convert("RGB")

        view1 = self.transform1(img) if self.transform1 is not None else img
        view2 = self.transform2(img) if self.transform2 is not None else img

        if self.return_index:
            return view1, view2, label, real_idx
        return view1, view2, label
