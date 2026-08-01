"""BitNet-style binary swarm MLP for experiment v0_1."""

from __future__ import annotations

from typing import List, Literal

import torch
import torch.nn as nn

from layers import Int8SwarmLinear, SquaredReLU

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


class BitNetSwarmMLP(nn.Module):
    """Small BitNet-inspired MLP with latent-free int8 swarm linears.

    Architecture::

        Flatten
        → [LN?] → Int8SwarmLinear(784 → hidden) → SquaredReLU
        → [LN?] → Int8SwarmLinear(hidden → 10)
    """

    def __init__(
        self,
        *,
        hidden_dim: int = 128,
        swarm_size: int = 32,
        ln_mode: LNMode = "no_affine",
        depth: int = 1,
        activation: ActName = "relu",
    ):
        super().__init__()
        if depth < 1:
            raise ValueError(f"depth must be >= 1, got {depth}")
        if ln_mode not in ("none", "no_affine", "affine"):
            raise ValueError(f"ln_mode must be none|no_affine|affine, got {ln_mode}")
        if activation not in ("squared_relu", "relu"):
            raise ValueError(f"activation must be squared_relu|relu, got {activation}")

        self.hidden_dim = hidden_dim
        self.swarm_size = swarm_size
        self.ln_mode = ln_mode
        self.depth = depth
        self.activation = activation

        self.flatten = nn.Flatten()
        # Default ReLU for stable latent-free training; SquaredReLU is BitNet-faithful.
        self.act: nn.Module = SquaredReLU() if activation == "squared_relu" else nn.ReLU()

        self.lns = nn.ModuleList()
        self.linears = nn.ModuleList()

        in_dim = 28 * 28
        for _ in range(depth):
            self.lns.append(_make_ln(in_dim, ln_mode))
            self.linears.append(Int8SwarmLinear(in_dim, hidden_dim, swarm_size))
            in_dim = hidden_dim

        self.lns.append(_make_ln(in_dim, ln_mode))
        self.linears.append(Int8SwarmLinear(in_dim, 10, swarm_size))

    def swarm_layers(self) -> List[Int8SwarmLinear]:
        # isinstance narrows ModuleList elements for type checkers.
        return [m for m in self.linears if isinstance(m, Int8SwarmLinear)]

    def ln_parameters(self) -> List[nn.Parameter]:
        """Trainable LayerNorm affine params (empty unless ln_mode=affine)."""
        params: List[nn.Parameter] = []
        for ln in self.lns:
            for p in ln.parameters():
                params.append(p)
        return params

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.flatten(x)
        n_blocks = len(self.linears)
        for i, (ln, linear) in enumerate(zip(self.lns, self.linears)):
            x = ln(x)
            x = linear(x)
            if i < n_blocks - 1:
                x = self.act(x)
        return x

    @torch.no_grad()
    def assert_binary_invariants(self) -> None:
        for layer in self.swarm_layers():
            p = layer.population
            assert p.dtype == torch.int8, p.dtype
            # torch.unique(...) type-checks cleanly; Tensor.unique() does not
            # under current PyTorch stubs (seen as non-callable Tensor).
            uniq = set(torch.unique(p).tolist())
            assert uniq.issubset({-1, 1}), uniq


def collect_swarm_layers(model: nn.Module) -> List[Int8SwarmLinear]:
    return [m for m in model.modules() if isinstance(m, Int8SwarmLinear)]
