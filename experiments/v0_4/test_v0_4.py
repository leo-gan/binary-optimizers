"""Tests for experiment v0_4."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

_HERE = Path(__file__).resolve().parent
_REPO = _HERE.parents[1]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))


def _load(name: str, file: str):
    path = _HERE / file
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


_layers = _load("v0_4_layers", "layers.py")
sys.modules["layers"] = _layers
_model = _load("v0_4_model", "model.py")
_optimizer = _load("v0_4_optimizer", "optimizer.py")

Int8BalancedTernaryLinear = _layers.Int8BalancedTernaryLinear
BitNetTernaryPlaceMLP = _model.BitNetTernaryPlaceMLP
SwarmOptimizerV04 = _optimizer.SwarmOptimizerV04


def test_balanced_ternary_roundtrip():
    n = 6
    s_max = (3**n - 1) // 2
    layer = Int8BalancedTernaryLinear(4, 3, n_trits=n)
    for val in [-s_max, -10, -1, 0, 1, 10, s_max]:
        s = torch.full((3, 4), val, dtype=torch.int64)
        layer.set_from_signed_integer_(s)
        layer.enforce_ternary_()
        uniq = set(layer.population.unique().tolist())
        assert uniq.issubset({-1, 0, 1})
        got = layer.place_value_sum().to(torch.int64)
        assert torch.equal(got, s), (val, got[0, 0].item())


def test_train_step():
    m = BitNetTernaryPlaceMLP(hidden_dim=32, n_trits=6, ln_mode="none")
    opt = SwarmOptimizerV04(m.swarm_layers(), recruit_rate=1e5, max_step_prob=0.5)
    x = torch.randn(4, 1, 28, 28)
    y = torch.randint(0, 10, (4,))
    opt.zero_grad()
    F.cross_entropy(m(x), y).backward()
    opt.step()
    m.assert_ternary_invariants()


def test_ln_modes():
    for mode in ("none", "no_affine", "affine"):
        m = BitNetTernaryPlaceMLP(hidden_dim=16, n_trits=5, ln_mode=mode)
        opt = SwarmOptimizerV04(m.swarm_layers(), ln_params=m.ln_parameters())
        opt.zero_grad()
        F.cross_entropy(m(torch.randn(2, 1, 28, 28)), torch.tensor([0, 1])).backward()
        opt.step()
        m.assert_ternary_invariants()
