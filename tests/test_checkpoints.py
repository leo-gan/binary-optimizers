"""Tests for train checkpoint cache."""

from __future__ import annotations

import torch
import torch.nn as nn

from binary_optimizers.models.mnist import create_mnist_bit_mlp
from binary_optimizers.training.checkpoints import (
    TrainSpec,
    checkpoint_exists,
    load_checkpoint,
    save_checkpoint,
)


def test_fingerprint_changes_with_optimizer():
    a = TrainSpec(model="bit_mlp", optimizer="adam", epochs=5, seed=0)
    b = TrainSpec(model="bit_mlp", optimizer="ema_flip", epochs=5, seed=0)
    c = TrainSpec(model="bit_mlp", optimizer="adam", epochs=5, seed=0, optimizer_kwargs={"lr": 1e-3})
    d = TrainSpec(model="bit_mlp", optimizer="adam", epochs=5, seed=0, optimizer_kwargs={"lr": 1e-2})
    assert a.fingerprint() != b.fingerprint()
    assert c.fingerprint() != d.fingerprint()
    assert a.fingerprint() == TrainSpec(model="bit_mlp", optimizer="adam", epochs=5, seed=0).fingerprint()


def test_save_load_roundtrip(tmp_path):
    spec = TrainSpec(
        model="bit_mlp",
        optimizer="ste",
        epochs=2,
        seed=1,
        optimizer_kwargs={"lr": 0.1},
        tag="test",
    )
    model = create_mnist_bit_mlp(hidden_dim=16)
    # mutate weights so load is detectable
    with torch.no_grad():
        for p in model.parameters():
            p.add_(0.123)

    paths = save_checkpoint(
        spec, model, metrics={"best_test_acc": 0.5, "final_test_acc": 0.4}, root=tmp_path
    )
    assert paths.weights.is_file()
    assert checkpoint_exists(spec, tmp_path)

    model2 = create_mnist_bit_mlp(hidden_dim=16)
    meta = load_checkpoint(spec, model2, root=tmp_path)
    assert meta["metrics"]["best_test_acc"] == 0.5
    for p1, p2 in zip(model.parameters(), model2.parameters()):
        assert torch.allclose(p1, p2)
