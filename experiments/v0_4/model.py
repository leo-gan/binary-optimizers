"""BitNet-style MLP with balanced ternary place-value linears (v0_4)."""

from __future__ import annotations

from typing import List, Literal

import torch
import torch.nn as nn

from layers import Int8BalancedTernaryLinear, SquaredReLU

LNMode = Literal["none", "no_affine", "affine"]
ActName = Literal["squared_relu", "relu"]


def _make_ln(num_features: int, mode: LNMode) -> nn.Module:
    if mode == "none":
        return nn.Identity()
    if mode == "no_affine":
        return nn.LayerNorm(num_features, elementwise_affine=False)
    if mode == "affine":
        return nn.LayerNorm(num_features, elementwise_affine=True)
    raise ValueError(mode)


class BitNetTernaryPlaceMLP(nn.Module):
    def __init__(
        self,
        *,
        hidden_dim: int = 128,
        n_trits: int = 10,
        ln_mode: LNMode = "none",
        depth: int = 1,
        activation: ActName = "relu",
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.n_trits = n_trits
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
            self.linears.append(
                Int8BalancedTernaryLinear(in_dim, hidden_dim, n_trits=n_trits)
            )
            in_dim = hidden_dim
        self.lns.append(_make_ln(in_dim, ln_mode))
        self.linears.append(
            Int8BalancedTernaryLinear(in_dim, 10, n_trits=n_trits)
        )

    def swarm_layers(self) -> List[Int8BalancedTernaryLinear]:
        return list(self.linears)

    def ln_parameters(self) -> List[nn.Parameter]:
        return [p for ln in self.lns for p in ln.parameters()]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.flatten(x)
        n = len(self.linears)
        for i, (ln, linear) in enumerate(zip(self.lns, self.linears)):
            x = ln(x)
            x = linear(x)
            if i < n - 1:
                x = self.act(x)
        return x

    @torch.no_grad()
    def assert_ternary_invariants(self) -> None:
        for layer in self.linears:
            uniq = set(layer.population.unique().tolist())
            assert uniq.issubset({-1, 0, 1}), uniq
