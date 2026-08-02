"""Smoke tests for v0_5_width_unary (no MNIST download)."""

from __future__ import annotations

import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO / "experiments" / "v0_1"))
sys.path.insert(0, str(_REPO))
for _n in ("layers", "model", "optimizer", "metrics"):
    sys.modules.pop(_n, None)

import torch
import torch.nn.functional as F

from model import BitNetSwarmMLP
from optimizer import SwarmOptimizerV01

sys.path.insert(0, str(_REPO / "experiments"))
from _width_atlas_common import approx_unary_state_bytes, parse_int_list


def test_parse_widths():
    assert parse_int_list("8,16,32") == [8, 16, 32]


def test_state_bytes_grows():
    assert approx_unary_state_bytes(128, 64) == 2 * approx_unary_state_bytes(128, 32)


def test_one_step_swarm_sizes():
    for S in (8, 32):
        m = BitNetSwarmMLP(hidden_dim=32, swarm_size=S, ln_mode="none")
        opt = SwarmOptimizerV01(m.swarm_layers(), recruit_rate=1e4)
        x = torch.randn(4, 1, 28, 28)
        y = torch.randint(0, 10, (4,))
        opt.zero_grad()
        F.cross_entropy(m(x), y).backward()
        opt.step()
        m.assert_binary_invariants()
