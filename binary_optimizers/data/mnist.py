from typing import Tuple

from torch.utils.data import DataLoader


def make_mnist_loaders(
    root: str = "./data",
    batch_size_train: int = 100,
    batch_size_test: int = 1000,
    num_workers: int = 0,
    pin_memory: bool = False,
) -> Tuple[DataLoader, DataLoader]:
    import torchvision
    import torchvision.transforms as T
    import torchvision.datasets as datasets

    transform = T.Compose(
        [
            T.ToTensor(),
            T.Normalize((0.1307,), (0.3081,)),
        ]
    )

    train_set = datasets.MNIST(root, train=True, download=True, transform=transform)
    test_set = datasets.MNIST(root, train=False, transform=transform)

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
