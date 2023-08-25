import torchvision
from torch.utils.data import DataLoader

from .path_config import data_dir
from .transforms import get_standard_transform


class Cifar10DataLoaderFactory:
    train_transform = get_standard_transform(train=True)
    test_transform = get_standard_transform(train=False)

    train_set = torchvision.datasets.CIFAR10(
        root=data_dir,
        train=True,
        download=True,
        transform=train_transform,
    )

    test_set = torchvision.datasets.CIFAR10(
        root=data_dir,
        train=False,
        download=True,
        transform=test_transform,
    )

    @classmethod
    def create_train_loaders(cls, batch_size: int):
        train_loader = DataLoader(
            dataset=cls.train_set,
            batch_size=batch_size,
            shuffle=True,
        )

        test_loader = DataLoader(
            dataset=cls.test_set,
            batch_size=batch_size,
            shuffle=False,
        )

        return train_loader, test_loader
