"""Matched BitNet-style MLP with BitLinearSTE (latent FP + sign STE).

Topology mirrors experiments/v0_*: Flatten → [LN] → Linear → ReLU → [LN] → Linear.
"""

from __future__ import annotations

from typing import List, Literal

import torch
import torch.nn as nn

from binary_optimizers.models.bit_layers import BitLinearSTE

LNMode = Literal["none", "no_affine", "affine"]
ActName = Literal["squared_relu", "relu"]


class SquaredReLU(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.relu(x).square()


def _make_ln(num_features: int, mode: LNMode) -> nn.Module:
    if mode == "none":
        return nn.Identity()
    if mode == "no_affine":
        return nn.LayerNorm(num_features, elementwise_affine=False)
    if mode == "affine":
        return nn.LayerNorm(num_features, elementwise_affine=True)
    raise ValueError(f"Unknown ln_mode: {mode}")


class BitNetSTEMLP(nn.Module):
    """STE baseline MLP: FP latent weights, sign(W) in forward (straight-through)."""

    def __init__(
        self,
        *,
        hidden_dim: int = 128,
        ln_mode: LNMode = "none",
        depth: int = 1,
        activation: ActName = "relu",
        bias: bool = False,
    ):
        super().__init__()
        if depth < 1:
            raise ValueError(f"depth must be >= 1, got {depth}")
        self.hidden_dim = hidden_dim
        self.ln_mode = ln_mode
        self.flatten = nn.Flatten()
        self.act: nn.Module = (
            SquaredReLU() if activation == "squared_relu" else nn.ReLU()
        )
        self.lns = nn.ModuleList()
        self.linears = nn.ModuleList()
        in_dim = 28 * 28
        for _ in range(depth):
            self.lns.append(_make_ln(in_dim, ln_mode))
            self.linears.append(BitLinearSTE(in_dim, hidden_dim, bias=bias))
            in_dim = hidden_dim
        self.lns.append(_make_ln(in_dim, ln_mode))
        self.linears.append(BitLinearSTE(in_dim, 10, bias=bias))

    def ln_parameters(self) -> List[nn.Parameter]:
        return [p for ln in self.lns for p in ln.parameters()]

    def ste_parameters(self) -> List[nn.Parameter]:
        """Weight (and bias) params for the STE optimizer / SGD group."""
        params: List[nn.Parameter] = []
        for linear in self.linears:
            params.extend(list(linear.parameters()))
        return params

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.flatten(x)
        n = len(self.linears)
        for i, (ln, linear) in enumerate(zip(self.lns, self.linears)):
            x = ln(x)
            x = linear(x)
            if i < n - 1:
                x = self.act(x)
        return x
