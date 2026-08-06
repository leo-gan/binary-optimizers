"""Width helpers for v0_12 CIFAR probe."""

from __future__ import annotations

from typing import List

DEFAULT_WIDTHS = [64, 256, 512]
CIFAR_DEFAULT_MAX_WALL = 2400.0


def parse_widths(s: str) -> List[int]:
    return [int(x.strip()) for x in s.split(",") if x.strip()]
