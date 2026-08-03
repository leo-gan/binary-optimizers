"""Unit tests for wall/epoch train budgets."""

from __future__ import annotations

import time

from binary_optimizers.training.budget import (
    EarlyStopTracker,
    TrainBudget,
    resolve_budget,
)


def test_patience_frac_of_epochs():
    b = resolve_budget(epochs=80, max_wall_sec=1200, patience_frac=0.125)
    assert b.patience_epochs == 10  # round(0.125 * 80)
    assert b.patience_wall_sec == 150.0  # 0.125 * 1200


def test_patience_absolute_override():
    b = resolve_budget(epochs=80, max_wall_sec=1200, patience_frac=0.125, patience=5)
    assert b.patience_epochs == 5
    assert b.patience_wall_sec == 150.0  # wall still from frac


def test_wall_disabled():
    b = resolve_budget(epochs=80, max_wall_sec=0, patience_frac=0.1)
    assert b.max_wall_sec is None
    assert b.patience_wall_sec is None
    assert b.patience_epochs == 8


def test_any_strict_gain_resets_stall():
    b = TrainBudget(max_epochs=80, max_wall_sec=None, patience_frac=0.1, min_delta=0.0)
    # force patience_epochs
    b.patience_epochs = 3
    tr = EarlyStopTracker(b)
    d = tr.observe(1, 0.90)
    assert d.improved and not d.stop
    tr.observe(2, 0.90)  # no gain
    tr.observe(3, 0.9001)  # strict gain even if tiny
    assert tr.best == 0.9001
    assert tr.stall_epochs == 0


def test_stop_patience_epochs():
    b = TrainBudget(max_epochs=80, max_wall_sec=None, patience_frac=0.1, min_delta=0.0)
    b.patience_epochs = 2
    tr = EarlyStopTracker(b)
    tr.observe(1, 0.5)
    assert not tr.observe(2, 0.4).stop
    d = tr.observe(3, 0.4)
    assert d.stop and d.reason == "patience_epochs"


def test_stop_max_wall(monkeypatch):
    b = TrainBudget(max_epochs=1000, max_wall_sec=10.0, patience_frac=0.5, min_delta=0.0)
    tr = EarlyStopTracker(b)
    tr.t0 = time.time() - 11.0
    tr.t_at_best = tr.t0
    d = tr.observe(1, 0.1)
    assert d.stop and d.reason == "max_wall_sec"
