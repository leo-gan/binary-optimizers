"""Smoke tests for v0_5_width_register (no MNIST download)."""

from __future__ import annotations

import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO / "experiments" / "v0_3"))
sys.path.insert(0, str(_REPO))
for _n in ("layers", "model", "optimizer", "metrics"):
    sys.modules.pop(_n, None)

import torch
import torch.nn.functional as F

from model import BitNetCarrySafeMLP
from optimizer import SwarmOptimizerV03

sys.path.insert(0, str(_REPO / "experiments"))
from _width_atlas_common import approx_register_state_bytes, parse_int_list


def test_parse_widths():
    assert parse_int_list("8, 16,32") == [8, 16, 32]


def test_state_bytes_grows():
    assert approx_register_state_bytes(128, 32) > approx_register_state_bytes(128, 16)


def test_one_step_nbits():
    for n in (8, 16, 32):
        m = BitNetCarrySafeMLP(hidden_dim=32, n_bits=n, ln_mode="none")
        opt = SwarmOptimizerV03(m.swarm_layers(), recruit_rate=1e4, max_step_prob=0.5)
        x = torch.randn(4, 1, 28, 28)
        y = torch.randint(0, 10, (4,))
        opt.zero_grad()
        F.cross_entropy(m(x), y).backward()
        opt.step()
        m.assert_binary_invariants()


def test_scale_steps_for_width():
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from train import scale_steps_for_width

    m16, s16 = scale_steps_for_width(16, 512, 1e6)
    m32, s32 = scale_steps_for_width(32, 512, 1e6)
    assert m16 == 512
    assert m32 > m16
    assert s32 > s16


def test_load_result_rows_empty(tmp_path):
    from _width_atlas_common import load_result_rows

    assert load_result_rows(
        tmp_path, width_key="n_bits", prefix="nbits", ln_mode="none", seed=42
    ) == []
