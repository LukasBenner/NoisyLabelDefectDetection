from torchvision.datasets import ImageFolder
from torch.utils.data import Dataset
import os
from PIL import Image

class UnlabeledImageFolderOrFlat(Dataset):
    """
    Supports:
      - ImageFolder-style: root/class_x/img.jpg (classes ignored)
      - Flat folder: root/img.jpg
    Returns PIL image only (transform handles 2-crops).
    """
    def __init__(self, root: str, transform=None):
        self.root = root
        self.transform = transform

        # detect ImageFolder structure
        has_subdirs = any(os.path.isdir(os.path.join(root, d)) for d in os.listdir(root))
        self.samples = []

        exts = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp")
        if has_subdirs:
            # Use ImageFolder to collect paths (but ignore labels)
            ds = ImageFolder(root=root)
            self.samples = [p for (p, _) in ds.samples]
        else:
            for fn in os.listdir(root):
                if fn.lower().endswith(exts):
                    self.samples.append(os.path.join(root, fn))

        if len(self.samples) == 0:
            raise RuntimeError(f"No images found in {root}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path = self.samples[idx]
        img = Image.open(path).convert("RGB")
        if self.transform is not None:
            return self.transform(img)
        return img