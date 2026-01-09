from typing import Tuple

from torch.utils.data import DataLoader


def make_cifar10_loaders(
    root: str = ".",
    batch_size_train: int = 128,
    batch_size_test: int = 256,
    num_workers: int = 2,
    pin_memory: bool = False,
) -> Tuple[DataLoader, DataLoader]:
    import torchvision
    import torchvision.transforms as T

    transform = T.Compose(
        [
            T.ToTensor(),
            T.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ]
    )

    train_set = torchvision.datasets.CIFAR10(root=root, train=True, download=True, transform=transform)
    test_set = torchvision.datasets.CIFAR10(root=root, train=False, download=True, transform=transform)

    train_loader = DataLoader(
        train_set,
        batch_size=batch_size_train,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    test_loader = DataLoader(
        test_set,
        batch_size=batch_size_test,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )

    return train_loader, test_loader
