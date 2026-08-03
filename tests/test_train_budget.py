"""Unit tests for pure wall train budgets."""

from __future__ import annotations

import time

import pytest

from binary_optimizers.training.budget import (
    EarlyStopTracker,
    TrainBudget,
    resolve_budget,
)


def test_pure_wall_defaults():
    b = resolve_budget()
    assert b.max_wall_sec == 1200.0
    assert b.patience_wall_sec == 150.0
    assert b.use_epoch_stops is False
    assert b.patience_epochs is None


def test_pure_wall_requires_wall():
    with pytest.raises(ValueError, match="max_wall_sec"):
        resolve_budget(max_wall_sec=None, use_epoch_stops=False)


def test_legacy_epoch_mode():
    b = resolve_budget(
        max_wall_sec=0,
        use_epoch_stops=True,
        epochs=80,
        patience=5,
    )
    assert b.max_wall_sec is None
    assert b.use_epoch_stops
    assert b.patience_epochs == 5


def test_any_strict_gain_resets_wall_stall():
    b = resolve_budget(max_wall_sec=1200, patience_frac=0.125)
    tr = EarlyStopTracker(b)
    tr.observe(1, 0.90)
    tr.t_at_best = time.time() - 200  # would exceed 150s wall patience
    # but then improve slightly
    d = tr.observe(2, 0.9001)
    assert d.improved
    assert not d.stop
    assert tr.stall_epochs == 0


def test_stop_patience_wall():
    b = resolve_budget(max_wall_sec=1200, patience_frac=0.125)
    tr = EarlyStopTracker(b)
    tr.observe(1, 0.5)
    tr.t_at_best = time.time() - 151
    d = tr.observe(2, 0.4)
    assert d.stop and d.reason == "patience_wall"


def test_stop_max_wall():
    b = resolve_budget(max_wall_sec=10.0, patience_frac=0.5)
    tr = EarlyStopTracker(b)
    tr.t0 = time.time() - 11.0
    tr.t_at_best = tr.t0
    d = tr.observe(1, 0.1)
    assert d.stop and d.reason == "max_wall_sec"


def test_no_epoch_patience_stop_in_pure_wall():
    """Many non-improving epochs must not stop if wall patience not hit."""
    b = resolve_budget(max_wall_sec=1200, patience_frac=0.125)
    tr = EarlyStopTracker(b)
    tr.observe(1, 0.9)
    for ep in range(2, 50):
        d = tr.observe(ep, 0.5)
        assert not d.stop, f"stopped early at ep={ep} reason={d.reason}"
    assert tr.stall_epochs == 48
