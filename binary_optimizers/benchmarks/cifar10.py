import time
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional

import torch

from binary_optimizers.data.cifar10 import make_cifar10_loaders
from binary_optimizers.models.cifar import SmallBitConvNet, SmallConvNetSTE
from binary_optimizers.optimizers.signum import MomentumVotingOptimizer
from binary_optimizers.optimizers.ste import STEOptimizer
from binary_optimizers.optimizers.voting import VotingOptimizer
from binary_optimizers.training.loops import evaluate_accuracy, set_seed, train_one_epoch_classification


@dataclass
class RunResult:
    train_acc: List[float]
    test_acc: List[float]
    epoch_time_s: List[float]


def _make_optimizer(name: str, params):
    if name == "ste_sgd":
        return STEOptimizer(params, lr=0.1, momentum=0.9)
    if name == "voting":
        return VotingOptimizer(params, lr=0.1, momentum=0.9, push_rate=0.3, clip=1.0)
    if name == "signum":
        return MomentumVotingOptimizer(params, lr=2e-3, momentum=0.9, clip=1.2)
    if name == "adam":
        return torch.optim.Adam(params, lr=1e-3)

    raise ValueError(f"Unknown optimizer: {name}")


def _make_model(name: str) -> torch.nn.Module:
    if name == "small_convnet_ste":
        return SmallConvNetSTE(binary=True)
    if name == "small_bitconvnet":
        return SmallBitConvNet(binary=True, scale=True)

    raise ValueError(f"Unknown model: {name}")


def run_cifar10_benchmark(
    *,
    runs: Dict[str, Dict[str, str]],
    epochs: int = 5,
    seed: int = 42,
    device: Optional[str] = None,
    data_root: str = ".",
    num_workers: int = 2,
) -> Dict[str, RunResult]:
    set_seed(seed)

    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    pin_memory = device != "cpu"

    train_loader, test_loader = make_cifar10_loaders(
        root=data_root,
        batch_size_train=128,
        batch_size_test=256,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )

    results: Dict[str, RunResult] = {}

    for run_name, cfg in runs.items():
        model_name = cfg["model"]
        optimizer_name = cfg["optimizer"]

        model = _make_model(model_name).to(device)
        optimizer = _make_optimizer(optimizer_name, model.parameters())

        train_accs: List[float] = []
        test_accs: List[float] = []
        times: List[float] = []

        for epoch in range(epochs):
            t0 = time.time()
            train_acc = train_one_epoch_classification(model, optimizer, train_loader, device)
            t1 = time.time()
            test_acc = evaluate_accuracy(model, test_loader, device)

            train_accs.append(train_acc)
            test_accs.append(test_acc)
            times.append(t1 - t0)

            print(
                f"{run_name} | epoch {epoch + 1:02d}/{epochs} | "
                f"train={train_acc:.4f} test={test_acc:.4f} time={times[-1]:.1f}s"
            )

        results[run_name] = RunResult(train_acc=train_accs, test_acc=test_accs, epoch_time_s=times)

    return results
