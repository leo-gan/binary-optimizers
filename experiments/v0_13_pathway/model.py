"""Binary sign MLP for experiment v0_13 (pathway optimizer)."""

from __future__ import annotations

from typing import Dict, List, Literal

import torch
import torch.nn as nn

from binary_optimizers.assembly import snap_binary_
from binary_optimizers.models.bit_layers import BitLinearSTE
from binary_optimizers.models.pathway_norm import PathwayNorm

NormName = Literal["none", "pathway", "layernorm", "bn"]


class PathwayMLP(nn.Module):
    """Flatten → BitLinearSTE(in→H) → [norm] → ReLU → BitLinearSTE(H→C).

    After init the linear weights are snapped to ``{+1, -1}``. Pathway
    training keeps them there.

    ``norm`` is applied on the hidden pre-activations (where ±1 matmuls
    explode). ``pathway`` is PathwayNorm; ``layernorm`` / ``bn`` are FP
    crutches for comparison.
    """

    def __init__(
        self,
        *,
        hidden_dim: int = 128,
        in_dim: int = 28 * 28,
        n_classes: int = 10,
        norm: NormName = "none",
        use_bn: bool = False,
    ):
        super().__init__()
        if use_bn and norm == "none":
            norm = "bn"
        if norm not in ("none", "pathway", "layernorm", "bn"):
            raise ValueError(f"unknown norm: {norm!r}")
        self.hidden_dim = hidden_dim
        self.in_dim = in_dim
        self.n_classes = n_classes
        self.norm_name: NormName = norm
        self.use_bn = norm == "bn"

        self.flatten = nn.Flatten()
        self.fc1 = BitLinearSTE(in_dim, hidden_dim, bias=False)
        if norm == "pathway":
            self.norm: nn.Module = PathwayNorm(hidden_dim)
        elif norm == "layernorm":
            self.norm = nn.LayerNorm(hidden_dim, elementwise_affine=False)
        elif norm == "bn":
            self.norm = nn.BatchNorm1d(hidden_dim)
        else:
            self.norm = nn.Identity()
        self.act = nn.ReLU()
        self.fc2 = BitLinearSTE(hidden_dim, n_classes, bias=False)
        self.snap_binary_()

    def binary_parameters(self) -> List[nn.Parameter]:
        return [self.fc1.weight, self.fc2.weight]

    def snap_binary_(self) -> None:
        for p in self.binary_parameters():
            snap_binary_(p.data)

    def assert_binary(self) -> None:
        for p in self.binary_parameters():
            uniq = set(p.detach().unique().tolist())
            if not uniq.issubset({-1.0, 1.0}):
                raise AssertionError(f"non-binary weights: {uniq}")

    def norm_stats(self) -> Dict[str, float]:
        n = self.norm
        if isinstance(n, PathwayNorm):
            return n.stats()
        return {}

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.flatten(x)
        x = self.fc1(x)
        x = self.norm(x)
        x = self.act(x)
        return self.fc2(x)
