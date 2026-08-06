"""Sparse ablation grid for v0_9."""

from __future__ import annotations

from dataclasses import dataclass
from typing import List


@dataclass(frozen=True)
class Cell:
    opt: str
    decoder: str
    p_noise: float
    lr: float

    def tag(self) -> str:
        pn = f"p{self.p_noise:g}".replace(".", "p")
        return f"{self.opt}_{self.decoder}_{pn}"


def default_cells() -> List[Cell]:
    return [
        Cell("sgd", "density", 0.001, 0.1),
        Cell("sgd_m", "density", 0.001, 0.1),
        Cell("adam", "density", 0.001, 1e-3),
        Cell("sgd", "thresholded", 0.001, 0.1),
        Cell("sgd", "sign_noise", 0.001, 0.1),
        Cell("sgd", "density", 0.0, 0.1),
        Cell("sgd", "density", 0.01, 0.1),
    ]
