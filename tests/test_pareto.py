"""Tests for Pareto front computation."""

from binary_optimizers.benchmarks.pareto import compute_pareto_front


def test_pareto_front_simple():
    points = [
        {"name": "a", "test_acc_mean": 0.9, "inference_memory_bitpacked": 100},
        {"name": "b", "test_acc_mean": 0.8, "inference_memory_bitpacked": 50},
        {"name": "c", "test_acc_mean": 0.7, "inference_memory_bitpacked": 200},  # dominated
        {"name": "d", "test_acc_mean": 0.85, "inference_memory_bitpacked": 80},
    ]
    front = compute_pareto_front(
        points,
        maximize=["test_acc_mean"],
        minimize=["inference_memory_bitpacked"],
    )
    names = {p["name"] for p in front}
    assert "c" not in names
    assert "a" in names
    assert "b" in names


def test_sweep_config_coverage():
    from binary_optimizers.benchmarks.training_sweep import (
        DEFAULT_SWEEP_CONFIGS,
        NEW_OPTIMIZER_NAMES,
    )

    names = {c["name"] for c in DEFAULT_SWEEP_CONFIGS}
    # small + large STE optimizers
    for opt in ("adam", "ste", "voting", "signum", "threshold_if"):
        assert f"mnist_small_{opt}" in names
        assert f"mnist_large_{opt}" in names
    for opt in NEW_OPTIMIZER_NAMES:
        assert f"mnist_small_{opt}" in names
        assert f"mnist_large_{opt}" in names
    assert "mnist_swarm" in names
    assert "cifar_adam" in names
    assert "cifar_signum" in names
