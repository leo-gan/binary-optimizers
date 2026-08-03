"""Smoke tests for v0_6_encoding (no MNIST download)."""

from __future__ import annotations

import sys
from pathlib import Path

_THIS = Path(__file__).resolve().parent
sys.path.insert(0, str(_THIS))

import torch
import torch.nn.functional as F

from layers import EncodingLinear
from model import BitNetEncodingMLP
from optimizer import SwarmOptimizerV06
from train import Cell, default_cells, parse_cells, scale_steps_for_mant


def test_parse_cells():
    cells = parse_cells("fixed@8,exp_mant:2@8,block_scale:3@16,unary:256")
    assert cells[0].encoding == "fixed" and cells[0].n_bits == 8
    assert cells[1].encoding == "exp_mant" and cells[1].n_exp == 2
    assert cells[2].encoding == "block_scale" and cells[2].n_bits == 16
    assert cells[3].kind == "unary" and cells[3].swarm_size == 256


def test_default_cells_primary():
    cells = default_cells(include_rescue=False)
    tags = {c.tag() for c in cells}
    assert "fixed_n8" in tags
    assert "fixed_n16" in tags
    assert "unary_S256" in tags
    assert not any(c.n_bits == 32 for c in cells if c.kind == "encoding")


def test_default_cells_rescue():
    cells = default_cells(include_rescue=True)
    assert any(c.n_bits == 32 and c.encoding == "fixed" for c in cells)


def test_scale_steps():
    m8, s8 = scale_steps_for_mant(8, 512, 1e6)
    m16, s16 = scale_steps_for_mant(16, 512, 1e6)
    assert m16 == 512
    assert m8 < m16
    assert s8 < s16


def test_one_step_each_encoding():
    for enc, ne in (("fixed", 0), ("exp_mant", 2), ("block_scale", 3)):
        m = BitNetEncodingMLP(
            hidden_dim=32, n_bits=8, encoding=enc, n_exp=ne, ln_mode="none"
        )
        opt = SwarmOptimizerV06(m.swarm_layers(), recruit_rate=1e4, max_step=8)
        x = torch.randn(4, 1, 28, 28)
        y = torch.randint(0, 10, (4,))
        opt.zero_grad()
        F.cross_entropy(m(x), y).backward()
        opt.step()
        m.assert_binary_invariants()
        w = m.swarm_layers()[0].effective_weight()
        assert torch.isfinite(w).all()
        assert w.abs().max() <= 1.0 + 1e-5


def test_weight_in_range_exp_mant():
    layer = EncodingLinear(16, 8, n_bits=8, encoding="exp_mant", n_exp=3)
    w = layer.effective_weight()
    assert w.min() >= -1.0 - 1e-5
    assert w.max() <= 1.0 + 1e-5


def test_cell_tag():
    assert Cell(kind="encoding", n_bits=8, encoding="exp_mant", n_exp=2).tag() == "exp_mant2_n8"
