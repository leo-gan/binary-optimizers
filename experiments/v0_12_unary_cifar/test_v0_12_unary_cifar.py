"""Smoke tests for v0_12 CIFAR unary probe (no dataset download)."""

from __future__ import annotations

import sys
from pathlib import Path

import torch

_THIS = Path(__file__).resolve().parent
_REPO = _THIS.parents[1]
_V08 = _REPO / "experiments" / "v0_8_unary_link"
sys.path.insert(0, str(_V08))
sys.path.insert(0, str(_THIS))

from cifar_grid import DEFAULT_WIDTHS, parse_widths  # noqa: E402
from model import UnaryLinkMLP  # noqa: E402


def test_widths():
    assert parse_widths("64,256") == [64, 256]
    assert 256 in DEFAULT_WIDTHS


def test_cifar_input_dim_model():
    m = UnaryLinkMLP(
        hidden_dim=32,
        swarm_size=8,
        in_dim=3 * 32 * 32,
        n_classes=10,
    )
    x = torch.randn(2, 3, 32, 32)
    out = m(x)
    assert out.shape == (2, 10)
