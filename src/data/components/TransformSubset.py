from torch.utils.data import Dataset

class TransformSubset(Dataset):
    """A subset of a dataset with custom transforms applied."""

    def __init__(self, dataset, indices, transform):
        """Initialize TransformSubset.

        :param parent_dataset: The parent dataset (e.g., ImageFolder).
        :param indices: List of indices to include in this subset.
        :param transform: Transform to apply to the data.
        """
        self.parent = dataset
        self.indices = indices
        self.transform = transform

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        from PIL import Image

        real_idx = self.indices[idx]
        path, label = self.parent.samples[real_idx]
        img = Image.open(path).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img, label