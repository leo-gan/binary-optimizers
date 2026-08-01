"""Sanity tests for experiment v0_1 (run: pytest experiments/v0_1/test_v0_1.py)."""

from __future__ import annotations

import sys
from pathlib import Path

import torch
import torch.nn.functional as F

_THIS = Path(__file__).resolve().parent
_REPO = _THIS.parents[1]
sys.path.insert(0, str(_REPO))
sys.path.insert(0, str(_THIS))

from layers import Int8SwarmLinear
from model import BitNetSwarmMLP
from optimizer import SwarmOptimizerV01


def test_int8_forward_backward_and_flip():
    layer = Int8SwarmLinear(16, 8, swarm_size=8)
    x = torch.randn(4, 16)
    y = layer(x)
    assert y.shape == (4, 8)
    y.sum().backward()
    assert layer.last_agent_grad is not None
    assert layer.last_agent_grad.shape == layer.population.shape

    opt = SwarmOptimizerV01([layer], recruit_rate=100.0)
    opt.step()
    layer.enforce_binary_()
    uniq = set(layer.population.unique().tolist())
    assert uniq.issubset({-1, 1})


def test_model_ln_modes_and_invariants():
    for mode in ("none", "no_affine", "affine"):
        m = BitNetSwarmMLP(hidden_dim=32, swarm_size=8, ln_mode=mode)
        x = torch.randn(2, 1, 28, 28)
        out = m(x)
        assert out.shape == (2, 10)
        loss = F.cross_entropy(out, torch.tensor([0, 1]))
        opt = SwarmOptimizerV01(
            m.swarm_layers(),
            recruit_rate=20.0,
            ln_params=m.ln_parameters(),
            ln_lr=1e-2,
        )
        opt.zero_grad()
        out = m(x)
        loss = F.cross_entropy(out, torch.tensor([0, 1]))
        loss.backward()
        opt.step()
        m.assert_binary_invariants()
        if mode == "affine":
            assert len(m.ln_parameters()) > 0
        else:
            assert len(m.ln_parameters()) == 0
