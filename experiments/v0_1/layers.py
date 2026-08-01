"""Latent-free binary swarm layers (design B: int8 agents + manual STE)."""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class SquaredReLU(nn.Module):
    """BitNet-style FFN activation: ReLU²."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.relu(x).square()


class Int8SwarmLinear(nn.Module):
    """BitLinear-shaped layer with int8 ±1 agents (no FP master weight).

    Storage
    -------
    ``population`` : int8 buffer shaped ``[out, in, swarm_size]``, values in {-1, +1}.

    Forward
    -------
    Decode effective binary weight by majority of agents (ties → +1).
    Manual STE: matmul uses binary ``w_eff`` in forward; backward flows through
    the **normalized** swarm mean ``sum/S ∈ [-1,1]`` so each agent gets equal
    pressure. Output is scaled by ``1/sqrt(fan_in)`` (standard binary-net gain)
    so ReLU² / no-LN stacks stay numerically stable.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        swarm_size: int = 32,
        weight_scale: bool = True,
    ):
        super().__init__()
        if swarm_size < 1:
            raise ValueError(f"swarm_size must be >= 1, got {swarm_size}")
        self.in_features = in_features
        self.out_features = out_features
        self.swarm_size = swarm_size
        self.weight_scale = weight_scale
        # Fixed gain (not a trainable latent weight).
        gain = 1.0 / math.sqrt(in_features) if weight_scale else 1.0
        self.register_buffer("gain", torch.tensor(gain, dtype=torch.float32))

        init = torch.randint(0, 2, (out_features, in_features, swarm_size))
        init = (init * 2 - 1).to(torch.int8)
        self.register_buffer("population", init)

        # Populated during forward; holds grad after backward.
        self._pop_f: torch.Tensor | None = None

    @property
    def last_agent_grad(self) -> torch.Tensor | None:
        if self._pop_f is None:
            return None
        return self._pop_f.grad

    def effective_weight(self) -> torch.Tensor:
        """Binary effective weights from current agents (no STE), unscaled."""
        s = self.population.float().sum(dim=-1)
        w = torch.sign(s)
        return torch.where(w == 0, torch.ones_like(w), w)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Leaf float view of int8 agents for manual STE (not an nn.Parameter).
        pop_f = self.population.detach().float().requires_grad_(True)
        self._pop_f = pop_f

        swarm_sum = pop_f.sum(dim=-1)
        s_norm = swarm_sum / float(self.swarm_size)
        w_eff = torch.sign(swarm_sum)
        w_eff = torch.where(w_eff == 0, torch.ones_like(w_eff), w_eff)
        # Manual STE: forward binary ±1, backward through normalized mean.
        w_proxy = s_norm + (w_eff - s_norm).detach()
        return F.linear(x, w_proxy * self.gain)

    @torch.no_grad()
    def enforce_binary_(self) -> None:
        """Project agents onto {-1, +1} int8 (invariant)."""
        p = self.population
        p.copy_(torch.where(p >= 0, torch.ones_like(p), -torch.ones_like(p)))

    def extra_repr(self) -> str:
        return (
            f"in_features={self.in_features}, out_features={self.out_features}, "
            f"swarm_size={self.swarm_size}, dtype=int8, gain={float(self.gain):.5f}"
        )
