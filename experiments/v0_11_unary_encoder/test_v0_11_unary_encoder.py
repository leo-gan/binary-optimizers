"""Smoke tests for v0_11 encoder list."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from encoder_grid import DEFAULT_ENCODERS


def test_default_encoders():
    assert "fixed" in DEFAULT_ENCODERS
    assert "majority" in DEFAULT_ENCODERS
    assert len(DEFAULT_ENCODERS) == 4
