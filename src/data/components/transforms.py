from torchvision.transforms import v2

class MediumTransforms:
    @staticmethod
    def train_transforms():
        transforms = v2.Compose(
            [
                v2.RandomResizedCrop(480, scale=(0.8,1)),
                v2.RandomHorizontalFlip(p=0.5),
                v2.RandomVerticalFlip(p=0.5),
                v2.RandomRotation(degrees=(-10,10)),
            ]
        )
        return transforms

    @staticmethod
    def eval_transforms():
        transforms = v2.Compose(
            [
                v2.Resize(480),
                v2.CenterCrop(480),
            ]
        )
        return transforms

class StrongTransforms:
    @staticmethod
    def train_transforms():
        transforms = v2.Compose(
            [
                v2.RandomResizedCrop(480, scale=(0.8,1)),
                v2.RandomHorizontalFlip(p=0.5),
                v2.RandomVerticalFlip(p=0.5),
                v2.RandomRotation(degrees=(-45,45)),
                v2.RandomApply([v2.GaussianBlur(kernel_size=(5,9), sigma=(0.1,5))], p=0.5),
            ]
        )
        return transforms

    @staticmethod
    def eval_transforms():
        transforms = v2.Compose(
            [
                v2.Resize(480),
                v2.CenterCrop(480),
            ]
        )
        return transforms


class NoCropTransforms:
    @staticmethod
    def train_transforms():
        transforms = v2.Compose(
            [
                v2.Resize(480),
                v2.CenterCrop((480, 640)),
                v2.RandomHorizontalFlip(p=0.5),
                v2.RandomVerticalFlip(p=0.5),
                v2.RandomRotation(degrees=(-45,45)),
            ]
        )
        return transforms

    @staticmethod
    def eval_transforms():
        transforms = v2.Compose(
            [
                v2.Resize(480),
                v2.CenterCrop((480, 640)),
            ]
        )
        return transforms