from torch.utils.data import Dataset
from torchvision.io import read_image, ImageReadMode

class TransformSubset(Dataset):
    def __init__(self, dataset, indices, return_index=False, cpu_transform=None):
        self.parent = dataset
        self.indices = indices
        self.return_index = return_index
        self.cpu_transform = cpu_transform

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        real_idx = self.indices[idx]
        path, label = self.parent.samples[real_idx]

        img = read_image(path, mode=ImageReadMode.RGB)
        if self.cpu_transform is not None:
            img = self.cpu_transform(img)

        if self.return_index:
            return img, label, real_idx
        return img, label