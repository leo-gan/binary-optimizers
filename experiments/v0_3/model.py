"""BitNet-style MLP with carry-safe place-value linears (v0_3)."""

from __future__ import annotations

from typing import List, Literal

import torch
import torch.nn as nn

from layers import Int8CarrySafeLinear, SquaredReLU

LNMode = Literal["none", "no_affine", "affine"]
ActName = Literal["squared_relu", "relu"]


def _make_ln(num_features: int, mode: LNMode) -> nn.Module:
    if mode == "none":
        return nn.Identity()
    if mode == "no_affine":
        return nn.LayerNorm(num_features, elementwise_affine=False)
    if mode == "affine":
        return nn.LayerNorm(num_features, elementwise_affine=True)
    raise ValueError(f"Unknown ln_mode: {mode}")


class BitNetCarrySafeMLP(nn.Module):
    def __init__(
        self,
        *,
        hidden_dim: int = 128,
        n_bits: int = 16,
        ln_mode: LNMode = "none",
        depth: int = 1,
        activation: ActName = "relu",
    ):
        super().__init__()
        if depth < 1:
            raise ValueError(f"depth must be >= 1, got {depth}")
        self.hidden_dim = hidden_dim
        self.n_bits = n_bits
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
            self.linears.append(Int8CarrySafeLinear(in_dim, hidden_dim, n_bits=n_bits))
            in_dim = hidden_dim
        self.lns.append(_make_ln(in_dim, ln_mode))
        self.linears.append(Int8CarrySafeLinear(in_dim, 10, n_bits=n_bits))

    def swarm_layers(self) -> List[Int8CarrySafeLinear]:
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
    def assert_binary_invariants(self) -> None:
        for layer in self.linears:
            p = layer.population
            assert p.dtype == torch.int8
            uniq = set(p.unique().tolist())
            assert uniq.issubset({0, 1}), uniq
