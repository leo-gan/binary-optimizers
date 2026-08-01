"""Sanity tests for experiment v0_2."""

from __future__ import annotations

import sys
from pathlib import Path

import torch
import torch.nn.functional as F

_THIS = Path(__file__).resolve().parent
_REPO = _THIS.parents[1]
sys.path.insert(0, str(_REPO))
sys.path.insert(0, str(_THIS))

from layers import Int8PlaceValueSwarmLinear
from model import BitNetPlaceValueSwarmMLP
from optimizer import SwarmOptimizerV02


def test_place_values_and_grad():
    layer = Int8PlaceValueSwarmLinear(16, 8, n_bits=8)
    assert list(layer.place_values.tolist()) == [2.0**i for i in range(8)]
    x = torch.randn(4, 16)
    y = layer(x)
    assert y.shape == (4, 8)
    y.sum().backward()
    assert layer.last_agent_grad is not None
    assert layer.last_agent_grad.shape == layer.population.shape
    # Higher bits get larger |grad| (place 2^i in the sum).
    g = layer.last_agent_grad.abs().mean(dim=(0, 1))
    assert g[-1] >= g[0] - 1e-6
    w = layer.effective_weight()
    assert w.min() >= -1.0 - 1e-5 and w.max() <= 1.0 + 1e-5


def test_optimizer_flip_and_invariants():
    layer = Int8PlaceValueSwarmLinear(12, 6, n_bits=8)
    opt = SwarmOptimizerV02([layer], recruit_rate=1e5, max_flip_prob=0.5, lsb_bias=True)
    x = torch.randn(3, 12)
    opt.zero_grad()
    loss = layer(x).sum()
    loss.backward()
    opt.step()
    layer.enforce_binary_()
    assert set(layer.population.unique().tolist()).issubset({-1, 1})


def test_model_ln_modes():
    for mode in ("none", "no_affine", "affine"):
        m = BitNetPlaceValueSwarmMLP(hidden_dim=32, n_bits=8, ln_mode=mode)
        x = torch.randn(2, 1, 28, 28)
        opt = SwarmOptimizerV02(
            m.swarm_layers(),
            recruit_rate=1e4,
            ln_params=m.ln_parameters(),
        )
        opt.zero_grad()
        out = m(x)
        assert out.shape == (2, 10)
        F.cross_entropy(out, torch.tensor([0, 1])).backward()
        opt.step()
        m.assert_binary_invariants()
