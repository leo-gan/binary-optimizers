"""Carry-safe binary place-value layers (experiment v0_3)."""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class SquaredReLU(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.relu(x).square()


class Int8CarrySafeLinear(nn.Module):
    """Binary register weight: bits → integer v → w ∈ [-1, 1].

    Storage: int8 bits in {0, 1}, shape ``[out, in, n_bits]``.
    ``v = sum_i b_i * 2^i``, ``w = 2v/vmax - 1`` with ``vmax = 2^n - 1``.
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
        self.swarm_size = n_bits

        gain = 1.0 / math.sqrt(in_features) if weight_scale else 1.0
        self.register_buffer("gain", torch.tensor(gain, dtype=torch.float32))

        places = torch.tensor([2.0**i for i in range(n_bits)], dtype=torch.float32)
        vmax = float(2**n_bits - 1)
        self.register_buffer("place_values", places)
        self.register_buffer("vmax", torch.tensor(vmax, dtype=torch.float32))

        init = torch.randint(0, 2, (out_features, in_features, n_bits), dtype=torch.int8)
        self.register_buffer("population", init)
        self._pop_f: torch.Tensor | None = None

    @property
    def last_agent_grad(self) -> torch.Tensor | None:
        if self._pop_f is None:
            return None
        return self._pop_f.grad

    def integer_value(self, pop_f: torch.Tensor | None = None) -> torch.Tensor:
        if pop_f is None:
            pop_f = self.population.float()
        return (pop_f * self.place_values).sum(dim=-1)

    def effective_weight(self, pop_f: torch.Tensor | None = None) -> torch.Tensor:
        v = self.integer_value(pop_f)
        return 2.0 * v / self.vmax - 1.0

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        pop_f = self.population.detach().float().requires_grad_(True)
        self._pop_f = pop_f
        w = self.effective_weight(pop_f)
        return F.linear(x, w * self.gain)

    @torch.no_grad()
    def set_from_integer_(self, v: torch.Tensor) -> None:
        """Write bits from integer register ``v`` (carry-safe encode)."""
        v = v.round().clamp(0, int(self.vmax.item())).to(torch.int64)
        bits = []
        tmp = v
        for _ in range(self.n_bits):
            bits.append((tmp & 1).to(torch.int8))
            tmp = tmp >> 1
        # bits[0] is LSB → stack on last dim
        encoded = torch.stack(bits, dim=-1)
        self.population.copy_(encoded)

    @torch.no_grad()
    def enforce_binary_(self) -> None:
        p = self.population
        p.copy_(torch.where(p > 0, torch.ones_like(p), torch.zeros_like(p)))

    def extra_repr(self) -> str:
        return (
            f"in_features={self.in_features}, out_features={self.out_features}, "
            f"n_bits={self.n_bits}, carry_safe=True, dtype=int8"
        )
