"""
Scaffolding training sweep across optimizer × model combinations.

Designed for short runs that demonstrate convergence and analytics wiring —
not full-length training. Results are written as JSON.
"""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable, Optional

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset

from binary_optimizers.models.cifar import SmallBitConvNet, SmallConvNetSTE
from binary_optimizers.models.mnist import (
    create_mnist_bit_mlp,
    create_mnist_bit_mlp_large,
    create_mnist_swarm_mlp,
)
from binary_optimizers.optimizers.cosine_voting import CosineVotingOptimizer
from binary_optimizers.optimizers.ema_flip import EMAFlipOptimizer
from binary_optimizers.optimizers.hybrid_accumulator import HybridAccumulatorOptimizer
from binary_optimizers.optimizers.signum import MomentumVotingOptimizer
from binary_optimizers.optimizers.sparse_sign import SparseSignOptimizer
from binary_optimizers.optimizers.ste import STEOptimizer
from binary_optimizers.optimizers.swarm import SwarmOptimizer
from binary_optimizers.optimizers.swarm_log_optimizer import SwarmLogOptimizer
from binary_optimizers.optimizers.threshold_if import ThresholdedIntegrateFireOptimizer
from binary_optimizers.optimizers.voting import VotingOptimizer
from binary_optimizers.profiling.memory import profile_model_memory
from binary_optimizers.training.loops import (
    evaluate_accuracy,
    set_seed,
    train_one_epoch_classification,
)

# Active binary specialists in scaffold sweeps.
NEW_OPTIMIZER_NAMES = (
    "ema_flip",
    "cosine_voting",
    "sparse_sign",
)

# With potential (original ideas) — included in DEFAULT_SWEEP_CONFIGS again.
ACTIVE_RESEARCH_OPTIMIZERS = (
    "voting",
    "threshold_if",
    "hybrid_accumulator",
)

# Kept in code; excluded from DEFAULT_SWEEP_CONFIGS.
PAUSED_SWEEP_OPTIMIZERS = (
    "hybrid_v2",
)


@dataclass
class TrialResult:
    trial: int
    seed: int
    per_epoch_train_acc: list[float]
    per_epoch_test_acc: list[float]
    per_epoch_time_s: list[float]
    total_time_s: float
    final_test_acc: float
    memory: dict[str, Any]


@dataclass
class ConfigResult:
    name: str
    dataset: str
    model: str
    optimizer: str
    epochs: int
    trials: int
    trial_results: list[TrialResult] = field(default_factory=list)

    def summary(self) -> dict[str, Any]:
        finals = [t.final_test_acc for t in self.trial_results]
        times = [t.total_time_s for t in self.trial_results]
        mean = sum(finals) / len(finals) if finals else 0.0
        if len(finals) > 1:
            var = sum((x - mean) ** 2 for x in finals) / (len(finals) - 1)
            std = var**0.5
        else:
            std = 0.0
        mem = self.trial_results[0].memory if self.trial_results else {}
        return {
            "name": self.name,
            "dataset": self.dataset,
            "model": self.model,
            "optimizer": self.optimizer,
            "epochs": self.epochs,
            "trials": self.trials,
            "test_acc_mean": mean,
            "test_acc_std": std,
            "total_time_mean_s": sum(times) / len(times) if times else 0.0,
            "memory": mem,
            "per_epoch_test_acc_mean": _mean_curves(
                [t.per_epoch_test_acc for t in self.trial_results]
            ),
        }


def _mean_curves(curves: list[list[float]]) -> list[float]:
    if not curves:
        return []
    n = len(curves[0])
    return [sum(c[i] for c in curves) / len(curves) for i in range(n)]


def _make_optimizer(name: str, model: nn.Module) -> torch.optim.Optimizer:
    params = model.parameters()
    if name == "adam":
        return torch.optim.Adam(params, lr=1e-2)
    if name == "ste":
        return STEOptimizer(params, lr=0.1, momentum=0.9)
    if name == "voting":
        return VotingOptimizer(params, lr=0.1, momentum=0.9, push_rate=0.3, clip=1.0)
    if name == "signum":
        return MomentumVotingOptimizer(params, lr=5e-3, momentum=0.9, clip=1.5)
    if name == "threshold_if":
        return ThresholdedIntegrateFireOptimizer(
            params, lr=0.1, threshold=0.02, decay=0.99
        )
    if name == "swarm":
        return SwarmOptimizer(params, recruit_rate=50.0, bn_lr=0.01)
    if name == "swarm_log":
        return SwarmLogOptimizer(params, threshold=10, flip_prob=0.1, dynamic=False)
    if name == "swarm_log_dynamic":
        return SwarmLogOptimizer(params, threshold=10, flip_prob=0.1, dynamic=True)
    # New optimizers (sequential improvements over voting / IF / sign baselines)
    if name == "ema_flip":
        return EMAFlipOptimizer(
            params,
            lr=0.05,
            momentum=0.9,
            threshold_scale=0.5,
            clip=1.5,
            flip_mode=False,
        )
    if name == "cosine_voting":
        # lr scale matches MomentumVoting / Voting family for fair comparison
        return CosineVotingOptimizer(
            params,
            lr_max=0.1,
            lr_min=0.01,
            momentum=0.9,
            total_steps=400,
            clip=1.5,
            confidence_threshold=0.0,
        )
    if name == "sparse_sign":
        return SparseSignOptimizer(
            params, lr=0.05, momentum=0.9, density=0.6, clip=1.5
        )
    if name == "hybrid_accumulator":
        return HybridAccumulatorOptimizer(
            params,
            lr=0.1,
            momentum=0.9,
            init_threshold=0.05,
            target_fire_rate=0.05,
            decay=0.95,
            clip=1.5,
        )
    raise ValueError(f"Unknown optimizer: {name}")


def _make_model(name: str) -> nn.Module:
    if name == "bit_mlp_small":
        return create_mnist_bit_mlp(hidden_dim=64)
    if name == "bit_mlp_large":
        return create_mnist_bit_mlp_large(hidden_dim=128)
    if name == "swarm_mlp":
        return create_mnist_swarm_mlp(hidden_dim=64, swarm_size=8)
    if name == "small_convnet_ste":
        return SmallConvNetSTE(binary=True)
    if name == "small_bitconvnet":
        return SmallBitConvNet(binary=True, scale=True)
    raise ValueError(f"Unknown model: {name}")


# Scaffolding configs: active accuracy suite + research opts with potential.
DEFAULT_SWEEP_CONFIGS: list[dict[str, str]] = [
    # Small MLP
    {"name": "mnist_small_adam", "dataset": "mnist", "model": "bit_mlp_small", "optimizer": "adam"},
    {"name": "mnist_small_ste", "dataset": "mnist", "model": "bit_mlp_small", "optimizer": "ste"},
    {"name": "mnist_small_signum", "dataset": "mnist", "model": "bit_mlp_small", "optimizer": "signum"},
    {"name": "mnist_small_ema_flip", "dataset": "mnist", "model": "bit_mlp_small", "optimizer": "ema_flip"},
    {"name": "mnist_small_cosine_voting", "dataset": "mnist", "model": "bit_mlp_small", "optimizer": "cosine_voting"},
    {"name": "mnist_small_sparse_sign", "dataset": "mnist", "model": "bit_mlp_small", "optimizer": "sparse_sign"},
    {"name": "mnist_small_voting", "dataset": "mnist", "model": "bit_mlp_small", "optimizer": "voting"},
    {"name": "mnist_small_threshold_if", "dataset": "mnist", "model": "bit_mlp_small", "optimizer": "threshold_if"},
    {"name": "mnist_small_hybrid_accumulator", "dataset": "mnist", "model": "bit_mlp_small", "optimizer": "hybrid_accumulator"},
    # Large MLP
    {"name": "mnist_large_adam", "dataset": "mnist", "model": "bit_mlp_large", "optimizer": "adam"},
    {"name": "mnist_large_ste", "dataset": "mnist", "model": "bit_mlp_large", "optimizer": "ste"},
    {"name": "mnist_large_signum", "dataset": "mnist", "model": "bit_mlp_large", "optimizer": "signum"},
    {"name": "mnist_large_ema_flip", "dataset": "mnist", "model": "bit_mlp_large", "optimizer": "ema_flip"},
    {"name": "mnist_large_cosine_voting", "dataset": "mnist", "model": "bit_mlp_large", "optimizer": "cosine_voting"},
    {"name": "mnist_large_sparse_sign", "dataset": "mnist", "model": "bit_mlp_large", "optimizer": "sparse_sign"},
    {"name": "mnist_large_voting", "dataset": "mnist", "model": "bit_mlp_large", "optimizer": "voting"},
    {"name": "mnist_large_threshold_if", "dataset": "mnist", "model": "bit_mlp_large", "optimizer": "threshold_if"},
    {"name": "mnist_large_hybrid_accumulator", "dataset": "mnist", "model": "bit_mlp_large", "optimizer": "hybrid_accumulator"},
    # Swarm model
    {"name": "mnist_swarm", "dataset": "mnist", "model": "swarm_mlp", "optimizer": "swarm"},
    {"name": "mnist_swarm_log", "dataset": "mnist", "model": "swarm_mlp", "optimizer": "swarm_log"},
    {"name": "mnist_swarm_log_dynamic", "dataset": "mnist", "model": "swarm_mlp", "optimizer": "swarm_log_dynamic"},
    {"name": "mnist_swarm_adam", "dataset": "mnist", "model": "swarm_mlp", "optimizer": "adam"},
    # CIFAR-10 convnets
    {"name": "cifar_adam", "dataset": "cifar10", "model": "small_bitconvnet", "optimizer": "adam"},
    {"name": "cifar_signum", "dataset": "cifar10", "model": "small_bitconvnet", "optimizer": "signum"},
    {"name": "cifar_ste_sgd", "dataset": "cifar10", "model": "small_convnet_ste", "optimizer": "ste"},
]


def _subset_loader(loader: DataLoader, max_batches: int) -> DataLoader:
    """Limit dataset to first max_batches * batch_size samples for scaffolding."""
    if max_batches is None or max_batches <= 0:
        return loader
    ds = loader.dataset
    n = min(len(ds), max_batches * loader.batch_size)
    subset = Subset(ds, list(range(n)))
    return DataLoader(
        subset,
        batch_size=loader.batch_size,
        shuffle=getattr(loader, "shuffle", False) or True,
        num_workers=0,
        pin_memory=False,
    )


def _get_loaders(
    dataset: str,
    data_root: str,
    *,
    max_train_batches: int,
    max_test_batches: int,
):
    if dataset == "mnist":
        from binary_optimizers.data.mnist import make_mnist_loaders

        train_loader, test_loader = make_mnist_loaders(
            root=data_root, batch_size_train=64, batch_size_test=128, num_workers=0
        )
    elif dataset == "cifar10":
        from binary_optimizers.data.cifar10 import make_cifar10_loaders

        train_loader, test_loader = make_cifar10_loaders(
            root=data_root, batch_size_train=64, batch_size_test=128, num_workers=0
        )
    else:
        raise ValueError(f"Unknown dataset: {dataset}")

    return (
        _subset_loader(train_loader, max_train_batches),
        _subset_loader(test_loader, max_test_batches),
    )


def run_training_sweep(
    *,
    configs: list[dict[str, str]] | None = None,
    epochs: int = 2,
    num_trials: int = 2,
    seed: int = 42,
    device: Optional[str] = None,
    data_root: str = "./data",
    max_train_batches: int = 3,
    max_test_batches: int = 2,
    output_json: str | Path | None = "results/training_sweep.json",
) -> dict[str, Any]:
    """
    Run a scaffolding training sweep.

    Defaults keep work tiny (2 epochs, 2 trials, few batches) so the pipeline
    produces real analytics without full training.
    """
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    configs = configs or DEFAULT_SWEEP_CONFIGS
    all_results: list[ConfigResult] = []

    # Cache loaders per dataset
    loader_cache: dict[str, tuple] = {}

    for cfg in configs:
        print(f"\n=== {cfg['name']} ===")
        ds = cfg["dataset"]
        if ds not in loader_cache:
            loader_cache[ds] = _get_loaders(
                ds,
                data_root,
                max_train_batches=max_train_batches,
                max_test_batches=max_test_batches,
            )
        train_loader, test_loader = loader_cache[ds]

        config_result = ConfigResult(
            name=cfg["name"],
            dataset=ds,
            model=cfg["model"],
            optimizer=cfg["optimizer"],
            epochs=epochs,
            trials=num_trials,
        )

        for trial in range(num_trials):
            trial_seed = seed + trial
            set_seed(trial_seed)
            model = _make_model(cfg["model"]).to(device)
            optimizer = _make_optimizer(cfg["optimizer"], model)

            # Warm-up step so optimizer state exists for memory profiling
            model.train()
            try:
                xb, yb = next(iter(train_loader))
                xb, yb = xb.to(device), yb.to(device)
                optimizer.zero_grad()
                out = model(xb)
                out.sum().backward()
                optimizer.step()
                optimizer.zero_grad()
            except StopIteration:
                pass

            mem = profile_model_memory(model, optimizer).to_dict()

            per_train, per_test, per_time = [], [], []
            t_start = time.perf_counter()
            for epoch in range(epochs):
                t0 = time.perf_counter()
                tr_acc = train_one_epoch_classification(
                    model, optimizer, train_loader, device
                )
                te_acc = evaluate_accuracy(model, test_loader, device)
                dt = time.perf_counter() - t0
                per_train.append(tr_acc)
                per_test.append(te_acc)
                per_time.append(dt)
                print(
                    f"  trial {trial+1}/{num_trials} epoch {epoch+1}/{epochs} "
                    f"train={tr_acc:.3f} test={te_acc:.3f} ({dt:.2f}s)"
                )
            total_t = time.perf_counter() - t_start

            config_result.trial_results.append(
                TrialResult(
                    trial=trial,
                    seed=trial_seed,
                    per_epoch_train_acc=per_train,
                    per_epoch_test_acc=per_test,
                    per_epoch_time_s=per_time,
                    total_time_s=total_t,
                    final_test_acc=per_test[-1] if per_test else 0.0,
                    memory=mem,
                )
            )

        all_results.append(config_result)

    payload = {
        "meta": {
            "epochs": epochs,
            "num_trials": num_trials,
            "seed": seed,
            "device": device,
            "max_train_batches": max_train_batches,
            "max_test_batches": max_test_batches,
            "note": "Scaffolding run — short epochs/batches to demonstrate pipeline.",
        },
        "configs": [
            {
                **asdict(cr),
                "summary": cr.summary(),
            }
            for cr in all_results
        ],
        "summaries": [cr.summary() for cr in all_results],
    }

    if output_json is not None:
        path = Path(output_json)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload, indent=2))
        print(f"\nWrote {path}")

    return payload
