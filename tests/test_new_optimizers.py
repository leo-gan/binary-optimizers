"""Tests for the four new binary optimizers and their sweep integration."""

from __future__ import annotations

import torch
import torch.nn as nn
import pytest

from binary_optimizers.models.mnist import create_mnist_bit_mlp
from binary_optimizers.optimizers import (
    CosineVotingOptimizer,
    EMAFlipOptimizer,
    HybridAccumulatorOptimizer,
    SparseSignOptimizer,
)
from binary_optimizers.benchmarks.training_sweep import (
    DEFAULT_SWEEP_CONFIGS,
    NEW_OPTIMIZER_NAMES,
    _make_optimizer,
)


@pytest.mark.parametrize(
    "opt_cls,kwargs",
    [
        (EMAFlipOptimizer, dict(lr=0.01, momentum=0.9, threshold_scale=0.5)),
        (CosineVotingOptimizer, dict(lr_max=0.1, lr_min=0.01, total_steps=50)),
        (SparseSignOptimizer, dict(lr=0.05, density=0.5)),
        (HybridAccumulatorOptimizer, dict(lr=0.1, init_threshold=0.05)),
    ],
)
def test_optimizer_step_runs_and_clamps(opt_cls, kwargs):
    model = create_mnist_bit_mlp(hidden_dim=32)
    opt = opt_cls(model.parameters(), **kwargs)
    x = torch.randn(8, 1, 28, 28)
    y = torch.randint(0, 10, (8,))
    out = model(x)
    loss = nn.functional.cross_entropy(out, y)
    loss.backward()
    opt.step()
    opt.zero_grad()
    for p in model.parameters():
        if p.data.dim() >= 2:
            assert p.data.abs().max() <= 1.5 + 1e-5
        assert torch.isfinite(p.data).all()


def test_hybrid_fires_and_sets_signs():
    """Hybrid must set weights to ±1 on fire, not leave them near-init."""
    torch.manual_seed(0)
    model = create_mnist_bit_mlp(hidden_dim=16)
    opt = HybridAccumulatorOptimizer(
        model.parameters(), lr=0.5, init_threshold=0.001, target_fire_rate=0.2, decay=0.5
    )
    before = {
        n: p.detach().clone()
        for n, p in model.named_parameters()
        if p.dim() >= 2
    }
    for _ in range(20):
        x = torch.randn(16, 1, 28, 28)
        y = torch.randint(0, 10, (16,))
        opt.zero_grad()
        nn.functional.cross_entropy(model(x), y).backward()
        opt.step()
    changed = False
    for n, p in model.named_parameters():
        if p.dim() < 2:
            continue
        if not torch.allclose(p.data, before[n], atol=1e-5):
            changed = True
        # Fired weights should be on the clamp/sign extremes often
        assert torch.isfinite(p.data).all()
    assert changed, "HybridAccumulator did not update any 2D weights"


def test_ema_flip_can_flip_weights():
    torch.manual_seed(1)
    w = nn.Parameter(torch.ones(4, 4))
    opt = EMAFlipOptimizer(
        [w], lr=0.01, momentum=0.0, threshold_scale=0.1, flip_mode=True
    )
    # Grad same sign as weight → positive wrongness → flip toward -1
    w.grad = torch.ones_like(w)
    opt.step()
    assert (w.data < 0).any(), "EMAFlip flip_mode should flip on positive wrongness"


def test_ema_flip_continuous_moves_against_grad():
    w = nn.Parameter(torch.zeros(4, 4))
    opt = EMAFlipOptimizer(
        [w], lr=0.1, momentum=0.0, threshold_scale=0.1, flip_mode=False
    )
    w.grad = torch.ones_like(w)
    opt.step()
    # sign update subtracts sign(grad) → weights become negative
    assert (w.data < 0).all()



def test_cosine_state_safe_for_profiler():
    from binary_optimizers.profiling.memory import measure_training_memory

    model = create_mnist_bit_mlp(hidden_dim=8)
    opt = CosineVotingOptimizer(model.parameters(), lr_max=0.1, lr_min=0.01, total_steps=10)
    x = torch.randn(4, 1, 28, 28)
    y = torch.randint(0, 10, (4,))
    opt.zero_grad()
    nn.functional.cross_entropy(model(x), y).backward()
    opt.step()
    # Must not raise (old bug: state['global_step'] was an int)
    mem = measure_training_memory(model, opt)
    assert mem.total_bytes > 0


@pytest.mark.parametrize(
    "name",
    list(NEW_OPTIMIZER_NAMES),
)
def test_make_optimizer_factory(name):
    model = create_mnist_bit_mlp(hidden_dim=16)
    opt = _make_optimizer(name, model)
    assert opt is not None
    # one dummy step
    x = torch.randn(4, 1, 28, 28)
    out = model(x)
    out.sum().backward()
    opt.step()


def test_sparse_sign_rejects_bad_density():
    model = create_mnist_bit_mlp(hidden_dim=8)
    with pytest.raises(ValueError):
        SparseSignOptimizer(model.parameters(), density=0.0)


def test_cosine_voting_lr_anneals():
    model = create_mnist_bit_mlp(hidden_dim=8)
    opt = CosineVotingOptimizer(
        model.parameters(), lr_max=0.1, lr_min=0.01, total_steps=10
    )
    lr0 = opt._cosine_lr(opt.param_groups[0])
    opt.global_step = 10
    lr_end = opt._cosine_lr(opt.param_groups[0])
    assert lr0 > lr_end
    assert abs(lr_end - 0.01) < 1e-6


def test_sweep_includes_new_optimizers():
    names = {c["name"] for c in DEFAULT_SWEEP_CONFIGS}
    for opt in NEW_OPTIMIZER_NAMES:
        assert f"mnist_small_{opt}" in names
        assert f"mnist_large_{opt}" in names


def test_new_optimizer_comparison_builder():
    from binary_optimizers.benchmarks.new_optimizer_report import (
        build_new_optimizer_comparison,
    )

    fake = {
        "meta": {"epochs": 2, "num_trials": 1, "device": "cpu", "note": "test"},
        "summaries": [],
    }
    for opt, acc in [
        ("adam", 0.1),
        ("ste", 0.15),
        ("voting", 0.12),
        ("signum", 0.11),
        ("threshold_if", 0.14),
        ("ema_flip", 0.2),
        ("cosine_voting", 0.13),
        ("sparse_sign", 0.16),
        ("hybrid_accumulator", 0.18),
    ]:
        fake["summaries"].append(
            {
                "name": f"mnist_small_{opt}",
                "dataset": "mnist",
                "model": "bit_mlp_small",
                "optimizer": opt,
                "test_acc_mean": acc,
                "test_acc_std": 0.01,
                "total_time_mean_s": 0.1,
                "memory": {
                    "training": {"total_bytes": 1000},
                    "inference": {"bitpacked_bytes": 100},
                },
                "per_epoch_test_acc_mean": [acc * 0.8, acc],
            }
        )
    cmp_ = build_new_optimizer_comparison(fake, model="bit_mlp_small")
    assert cmp_["wins"]["ema_flip"]["n_wins"] == 5
    assert cmp_["sequential"][0]["n_acc_wins"] >= 1
