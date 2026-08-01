"""Balanced ternary place-value layers (experiment v0_4)."""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class SquaredReLU(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.relu(x).square()


def _int_to_balanced_ternary(v: torch.Tensor, n_trits: int) -> torch.Tensor:
    """Encode signed integers to digits {-1,0,1} with places 3^i."""
    digits = []
    tmp = v.to(torch.int64).clone()
    for _ in range(n_trits):
        r = torch.remainder(tmp, 3)  # 0,1,2 (PyTorch remainder ≥ 0)
        digit = r.clone()
        is_two = digit == 2
        digit = torch.where(is_two, -torch.ones_like(digit), digit)
        tmp = torch.where(is_two, tmp // 3 + 1, tmp // 3)
        digits.append(digit.to(torch.int8))
    return torch.stack(digits, dim=-1)


class Int8BalancedTernaryLinear(nn.Module):
    """Balanced ternary digits with places 3^i → multi-level weight in [-1,1]."""

    def __init__(
        self,
        in_features: int,
        out_features: int,
        n_trits: int = 10,
        weight_scale: bool = True,
    ):
        super().__init__()
        if n_trits < 1:
            raise ValueError(f"n_trits must be >= 1, got {n_trits}")
        self.in_features = in_features
        self.out_features = out_features
        self.n_trits = n_trits
        self.swarm_size = n_trits

        gain = 1.0 / math.sqrt(in_features) if weight_scale else 1.0
        self.register_buffer("gain", torch.tensor(gain, dtype=torch.float32))

        places = torch.tensor([3.0**i for i in range(n_trits)], dtype=torch.float32)
        s_max = float((3**n_trits - 1) // 2)
        self.register_buffer("place_values", places)
        self.register_buffer("s_max", torch.tensor(s_max, dtype=torch.float32))
        self.register_buffer(
            "vmax_u", torch.tensor(float(3**n_trits - 1), dtype=torch.float32)
        )

        # Init random ternary digits
        init = torch.randint(-1, 2, (out_features, in_features, n_trits), dtype=torch.int8)
        self.register_buffer("population", init)
        self._pop_f: torch.Tensor | None = None

    @property
    def last_agent_grad(self) -> torch.Tensor | None:
        if self._pop_f is None:
            return None
        return self._pop_f.grad

    def place_value_sum(self, pop_f: torch.Tensor | None = None) -> torch.Tensor:
        if pop_f is None:
            pop_f = self.population.float()
        return (pop_f * self.place_values).sum(dim=-1)

    def effective_weight(self, pop_f: torch.Tensor | None = None) -> torch.Tensor:
        return self.place_value_sum(pop_f) / self.s_max

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        pop_f = self.population.detach().float().requires_grad_(True)
        self._pop_f = pop_f
        w = self.effective_weight(pop_f)
        return F.linear(x, w * self.gain)

    @torch.no_grad()
    def set_from_signed_integer_(self, s: torch.Tensor) -> None:
        """``s`` signed integer in [-s_max, s_max]; write balanced ternary digits."""
        s_max = int(self.s_max.item())
        s = s.round().clamp(-s_max, s_max).to(torch.int64)
        encoded = _int_to_balanced_ternary(s, self.n_trits)
        self.population.copy_(encoded)

    @torch.no_grad()
    def enforce_ternary_(self) -> None:
        p = self.population.clamp(-1, 1)
        self.population.copy_(p.to(torch.int8))

    def extra_repr(self) -> str:
        return (
            f"in_features={self.in_features}, out_features={self.out_features}, "
            f"n_trits={self.n_trits}, place=3^i, balanced_ternary"
        )
