"""Resolve default experiment database path."""

from __future__ import annotations

import os
from pathlib import Path

# binary_optimizers/store/paths.py -> repo root is parents[2]
_REPO_ROOT = Path(__file__).resolve().parents[2]
_DEFAULT_REL = Path("results") / "experiments.duckdb"

ENV_DB = "BINARY_OPTIMIZERS_DB"


def repo_root() -> Path:
    return _REPO_ROOT


def default_db_path() -> Path:
    """Return DB path from env or ``results/experiments.duckdb`` under repo root."""
    override = os.environ.get(ENV_DB)
    if override:
        return Path(override).expanduser().resolve()
    return (_REPO_ROOT / _DEFAULT_REL).resolve()
