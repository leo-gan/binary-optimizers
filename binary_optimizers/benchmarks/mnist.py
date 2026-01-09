import time
from dataclasses import dataclass
from typing import Dict, List, Optional

import torch

from binary_optimizers.data.mnist import make_mnist_loaders
from binary_optimizers.models.mnist import create_mnist_bit_mlp, create_mnist_swarm_mlp
from binary_optimizers.optimizers.signum import MomentumVotingOptimizer
from binary_optimizers.optimizers.swarm import SwarmOptimizer
from binary_optimizers.training.loops import evaluate_accuracy, set_seed, train_one_epoch_classification


@dataclass
class RunResult:
    test_acc: List[float]
    epoch_time_s: List[float]


def run_mnist_benchmark(
    *,
    epochs: int = 15,
    seed: int = 42,
    device: Optional[str] = None,
    data_root: str = "./data",
) -> Dict[str, RunResult]:
    set_seed(seed)

    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    pin_memory = device != "cpu"

    train_loader, test_loader = make_mnist_loaders(
        root=data_root,
        batch_size_train=100,
        batch_size_test=1000,
        num_workers=2,
        pin_memory=pin_memory,
    )

    runs = {
        "adam_ste": {
            "model": "bit_mlp",
            "optimizer": "adam",
        },
        "voting_mom": {
            "model": "bit_mlp",
            "optimizer": "signum",
        },
        "swarm": {
            "model": "swarm_mlp",
            "optimizer": "swarm",
        },
    }

    results: Dict[str, RunResult] = {}

    for run_name, cfg in runs.items():
        if cfg["model"] == "bit_mlp":
            model = create_mnist_bit_mlp(hidden_dim=128).to(device)
        elif cfg["model"] == "swarm_mlp":
            model = create_mnist_swarm_mlp(hidden_dim=128, swarm_size=32).to(device)
        else:
            raise ValueError(f"Unknown model: {cfg['model']}")

        if cfg["optimizer"] == "adam":
            optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
        elif cfg["optimizer"] == "signum":
            optimizer = MomentumVotingOptimizer(model.parameters(), lr=0.005, momentum=0.9)
        elif cfg["optimizer"] == "swarm":
            optimizer = SwarmOptimizer(model.parameters(), recruit_rate=50.0, bn_lr=0.01)
        else:
            raise ValueError(f"Unknown optimizer: {cfg['optimizer']}")

        test_accs: List[float] = []
        times: List[float] = []

        for epoch in range(epochs):
            t0 = time.time()
            train_one_epoch_classification(model, optimizer, train_loader, device)
            t1 = time.time()
            test_acc = evaluate_accuracy(model, test_loader, device)

            test_accs.append(test_acc)
            times.append(t1 - t0)

            print(f"{run_name} | epoch {epoch + 1:02d}/{epochs} | test={test_acc:.4f} time={times[-1]:.1f}s")

        results[run_name] = RunResult(test_acc=test_accs, epoch_time_s=times)

    return results
