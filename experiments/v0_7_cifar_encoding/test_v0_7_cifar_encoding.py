"""Smoke tests for v0_7_cifar_encoding (no full CIFAR train)."""

from __future__ import annotations

import sys
from pathlib import Path

_THIS = Path(__file__).resolve().parent
sys.path.insert(0, str(_THIS))

import torch
import torch.nn.functional as F

from model import BitNetEncodingCIFARMLP, CIFAR_FLAT
from optimizer import SwarmOptimizerV06
from train import default_cells, parse_cells, scale_steps_for_mant


def test_default_four_cells():
    cells = default_cells()
    assert len(cells) == 4
    tags = {c.tag() for c in cells}
    assert tags == {"fixed_n8", "fixed_n16", "exp_mant2_n8", "exp_mant2_n16"}


def test_parse_cells():
    cells = parse_cells("fixed@8,exp_mant:2@16")
    assert cells[0].encoding == "fixed" and cells[0].n_bits == 8
    assert cells[1].n_exp == 2 and cells[1].n_bits == 16


def test_cifar_input_dim():
    assert CIFAR_FLAT == 3072
    m = BitNetEncodingCIFARMLP(hidden_dim=32, n_bits=8, encoding="fixed")
    x = torch.randn(2, 3, 32, 32)
    y = m(x)
    assert y.shape == (2, 10)


def test_one_step_fixed_and_exp():
    for enc, ne in (("fixed", 0), ("exp_mant", 2)):
        m = BitNetEncodingCIFARMLP(
            hidden_dim=32, n_bits=8, encoding=enc, n_exp=ne, ln_mode="none"
        )
        opt = SwarmOptimizerV06(m.swarm_layers(), recruit_rate=1e4, max_step=8)
        x = torch.randn(4, 3, 32, 32)
        y = torch.randint(0, 10, (4,))
        opt.zero_grad()
        F.cross_entropy(m(x), y).backward()
        opt.step()
        m.assert_binary_invariants()


def test_scale_steps():
    m8, _ = scale_steps_for_mant(8, 512, 1e6)
    m16, _ = scale_steps_for_mant(16, 512, 1e6)
    assert m16 == 512 and m8 < m16
