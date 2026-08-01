"""Latent-free place-value (exponential) swarm layers for experiment v0_2."""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class SquaredReLU(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.relu(x).square()


class Int8PlaceValueSwarmLinear(nn.Module):
    """BitLinear-shaped layer: int8 ±1 agents with exponential place values.

    Storage
    -------
    ``population`` : int8 ``[out, in, n_bits]``, values in {-1, +1}.
    Agent index ``i`` has place value ``2**i`` (LSB = i=0).

    Forward
    -------
    ``s = sum_i a_i * 2^i``, ``s_norm = s / sum(2^i) ∈ [-1, 1]``.
    Matmul uses **multi-level** ``s_norm`` (place-value decode), not majority
    sign — so LSB flips change the weight by ``2^{1-n}`` (exponential steps).
    Agents remain discrete ±1; no FP master weight. Gradients flow through
    ``s_norm`` (fully differentiable in the float view).
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        n_bits: int = 16,
        weight_scale: bool = True,
    ):
        super().__init__()
        if n_bits < 1:
            raise ValueError(f"n_bits must be >= 1, got {n_bits}")
        self.in_features = in_features
        self.out_features = out_features
        self.n_bits = n_bits
        # Alias for metrics / optimizers that expect swarm_size.
        self.swarm_size = n_bits

        gain = 1.0 / math.sqrt(in_features) if weight_scale else 1.0
        self.register_buffer("gain", torch.tensor(gain, dtype=torch.float32))

        # Place values 2^0 .. 2^{n_bits-1}
        places = torch.tensor([2.0**i for i in range(n_bits)], dtype=torch.float32)
        self.register_buffer("place_values", places)
        self.register_buffer(
            "place_norm", torch.tensor(float(places.sum().item()), dtype=torch.float32)
        )

        init = torch.randint(0, 2, (out_features, in_features, n_bits))
        init = (init * 2 - 1).to(torch.int8)
        self.register_buffer("population", init)

        self._pop_f: torch.Tensor | None = None

    @property
    def last_agent_grad(self) -> torch.Tensor | None:
        if self._pop_f is None:
            return None
        return self._pop_f.grad

    def place_value_sum(self, pop_f: torch.Tensor | None = None) -> torch.Tensor:
        """Signed place-value sum ``[out, in]``."""
        if pop_f is None:
            pop_f = self.population.float()
        # pop_f: [out, in, n_bits], place_values: [n_bits]
        return (pop_f * self.place_values).sum(dim=-1)

    def effective_weight(self) -> torch.Tensor:
        """Normalized place-value weight in [-1, 1] (multi-level)."""
        return self.place_value_sum() / self.place_norm

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        pop_f = self.population.detach().float().requires_grad_(True)
        self._pop_f = pop_f

        s = self.place_value_sum(pop_f)
        s_norm = s / self.place_norm
        return F.linear(x, s_norm * self.gain)

    @torch.no_grad()
    def enforce_binary_(self) -> None:
        p = self.population
        p.copy_(torch.where(p >= 0, torch.ones_like(p), -torch.ones_like(p)))

    def extra_repr(self) -> str:
        return (
            f"in_features={self.in_features}, out_features={self.out_features}, "
            f"n_bits={self.n_bits}, place=2^i, dtype=int8, gain={float(self.gain):.5f}"
        )
