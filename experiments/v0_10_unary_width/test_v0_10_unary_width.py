"""Smoke tests for v0_10 width list parsing."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from width_grid import DEFAULT_WIDTHS, parse_widths


def test_default_widths():
    assert DEFAULT_WIDTHS[0] == 8
    assert DEFAULT_WIDTHS[-1] == 1024
    assert 256 in DEFAULT_WIDTHS


def test_parse_widths():
    assert parse_widths("8, 16, 32") == [8, 16, 32]
