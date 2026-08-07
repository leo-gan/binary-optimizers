"""MNIST MLP with UnaryLinkLinear layers."""

from __future__ import annotations

from typing import List, Literal

import torch
import torch.nn as nn

from layers import EncoderName, SquaredReLU, UnaryLinkLinear

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


class UnaryLinkMLP(nn.Module):
    """Flatten → [LN] → UnaryLink 784→H → Act → [LN] → UnaryLink H→10."""

    def __init__(
        self,
        *,
        hidden_dim: int = 128,
        swarm_size: int = 256,
        encoder: EncoderName = "fixed",
        tanh_tau: float = 0.0,
        ln_mode: LNMode = "none",
        depth: int = 1,
        activation: ActName = "relu",
        in_dim: int = 28 * 28,
        n_classes: int = 10,
    ):
        super().__init__()
        if depth < 1:
            raise ValueError(f"depth must be >= 1, got {depth}")
        self.hidden_dim = hidden_dim
        self.swarm_size = swarm_size
        self.encoder = encoder
        self.ln_mode = ln_mode
        self.in_dim = in_dim
        self.n_classes = n_classes

        self.flatten = nn.Flatten()
        self.act: nn.Module = (
            SquaredReLU() if activation == "squared_relu" else nn.ReLU()
        )
        self.lns = nn.ModuleList()
        self.linears = nn.ModuleList()

        d = in_dim
        for _ in range(depth):
            self.lns.append(_make_ln(d, ln_mode))
            self.linears.append(
                UnaryLinkLinear(
                    d, hidden_dim, swarm_size, encoder=encoder, tanh_tau=tanh_tau
                )
            )
            d = hidden_dim
        self.lns.append(_make_ln(d, ln_mode))
        self.linears.append(
            UnaryLinkLinear(
                d, n_classes, swarm_size, encoder=encoder, tanh_tau=tanh_tau
            )
        )

    def swarm_layers(self) -> List[UnaryLinkLinear]:
        return [m for m in self.linears if isinstance(m, UnaryLinkLinear)]

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
        for layer in self.swarm_layers():
            p = layer.swarm
            assert p.dtype == torch.int8, p.dtype
            uniq = set(torch.unique(p).tolist())
            assert uniq.issubset({-1, 1}), uniq
