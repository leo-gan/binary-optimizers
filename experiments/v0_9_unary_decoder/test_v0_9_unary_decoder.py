"""Smoke tests for v0_9 cell grid (no dataset / no full train)."""

from __future__ import annotations

import sys
from pathlib import Path

_THIS = Path(__file__).resolve().parent
sys.path.insert(0, str(_THIS))

from ablation_grid import Cell, default_cells


def test_default_cells_unique_tags():
    cells = default_cells()
    tags = [c.tag() for c in cells]
    assert len(tags) == len(set(tags))
    assert any(c.opt == "adam" for c in cells)
    assert any(c.decoder == "density" for c in cells)


def test_cell_tag_stable():
    c = Cell("sgd", "density", 0.001, 0.1)
    assert "sgd" in c.tag() and "density" in c.tag()
