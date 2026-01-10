from torch.utils.data import Dataset
from PIL import Image

class TransformSubset(Dataset):
    def __init__(self, dataset, indices, transform=None, return_index=False):
        self.parent = dataset
        self.indices = indices
        self.transform = transform
        self.return_index = return_index
        
    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        real_idx = self.indices[idx]
        path, label = self.parent.samples[real_idx]
        img = Image.open(path).convert("RGB")
        if self.transform:
            img = self.transform(img)

        if self.return_index:
            return img, label, real_idx  # <-- global index
        return img, label