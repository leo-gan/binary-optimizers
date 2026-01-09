from typing import Any, Tuple

from torch.utils.data import DataLoader


def make_cifar10_loaders(*args: Any, **kwargs: Any) -> Tuple[DataLoader, DataLoader]:
    try:
        from .cifar10 import make_cifar10_loaders as _impl
    except ImportError as e:
        raise ImportError(
            "CIFAR-10 loaders require torchvision. Install it with: pip install 'torchvision'"
        ) from e
    return _impl(*args, **kwargs)


def make_mnist_loaders(*args: Any, **kwargs: Any) -> Tuple[DataLoader, DataLoader]:
    try:
        from .mnist import make_mnist_loaders as _impl
    except ImportError as e:
        raise ImportError(
            "MNIST loaders require torchvision. Install it with: pip install 'torchvision'"
        ) from e
    return _impl(*args, **kwargs)
