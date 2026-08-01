"""Tests for experiment v0_3."""

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


_layers = _load("v0_3_layers", "layers.py")
_model = _load("v0_3_model", "model.py")
# model imports "layers" — rebind after forcing local modules
sys.modules["layers"] = _layers
_optimizer = _load("v0_3_optimizer", "optimizer.py")

Int8CarrySafeLinear = _layers.Int8CarrySafeLinear
BitNetCarrySafeMLP = _model.BitNetCarrySafeMLP
SwarmOptimizerV03 = _optimizer.SwarmOptimizerV03


def test_integer_roundtrip():
    layer = Int8CarrySafeLinear(8, 4, n_bits=8)
    v = torch.tensor([[0, 1, 127, 255], [3, 16, 32, 64], [100, 200, 10, 5], [7, 8, 9, 10]])
    # shape [out=4, in=8] — fix
    v = torch.randint(0, 256, (4, 8))
    layer.set_from_integer_(v)
    layer.enforce_binary_()
    got = layer.integer_value().to(torch.int64)
    assert torch.equal(got, v.clamp(0, 255))


def test_train_step():
    m = BitNetCarrySafeMLP(hidden_dim=32, n_bits=8, ln_mode="none")
    opt = SwarmOptimizerV03(m.swarm_layers(), recruit_rate=1e5, max_step_prob=0.5)
    x = torch.randn(4, 1, 28, 28)
    y = torch.randint(0, 10, (4,))
    opt.zero_grad()
    loss = F.cross_entropy(m(x), y)
    loss.backward()
    opt.step()
    m.assert_binary_invariants()


def test_ln_modes():
    for mode in ("none", "no_affine", "affine"):
        m = BitNetCarrySafeMLP(hidden_dim=16, n_bits=6, ln_mode=mode)
        opt = SwarmOptimizerV03(m.swarm_layers(), ln_params=m.ln_parameters())
        opt.zero_grad()
        F.cross_entropy(m(torch.randn(2, 1, 28, 28)), torch.tensor([0, 1])).backward()
        opt.step()
        m.assert_binary_invariants()
