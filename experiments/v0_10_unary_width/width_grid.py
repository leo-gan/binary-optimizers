"""Width list helpers for v0_10."""

from __future__ import annotations

from typing import List

DEFAULT_WIDTHS = [8, 16, 32, 64, 128, 256, 512, 1024]


def parse_widths(s: str) -> List[int]:
    return [int(x.strip()) for x in s.split(",") if x.strip()]
