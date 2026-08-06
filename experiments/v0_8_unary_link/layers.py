"""Unary link linear: swarm of ±1 weights → link value via sum encoder."""

from __future__ import annotations

import math
from typing import Literal

import torch
import torch.nn as nn
import torch.nn.functional as F

EncoderName = Literal["fixed", "tanh", "signed_sqrt", "majority"]


class SquaredReLU(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.relu(x).square()


def encode_sum(s: torch.Tensor, swarm_size: int, encoder: EncoderName, tanh_tau: float) -> torch.Tensor:
    """Map sum s = Σ a_k to link value. Encoder input is sum only."""
    S = float(swarm_size)
    if encoder == "fixed":
        return s / S
    if encoder == "tanh":
        tau = tanh_tau if tanh_tau > 0 else max(S / 2.0, 1.0)
        return torch.tanh(s / tau)
    if encoder == "signed_sqrt":
        return torch.sign(s) * torch.sqrt(s.abs() / S)
    if encoder == "majority":
        w = torch.sign(s)
        return torch.where(w == 0, torch.ones_like(w), w)
    raise ValueError(f"unknown encoder: {encoder}")


class UnaryLinkLinear(nn.Module):
    """Latent-free linear: int8 swarm [out, in, S], link value from sum encoder.

    Forward uses multi-level (or majority) link values. Gradients attach to
    ``w_link`` via ``retain_grad`` for the per-link continuous optimizer.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        swarm_size: int = 256,
        encoder: EncoderName = "fixed",
        tanh_tau: float = 0.0,
        weight_scale: bool = True,
    ):
        super().__init__()
        if swarm_size < 1:
            raise ValueError(f"swarm_size must be >= 1, got {swarm_size}")
        self.in_features = in_features
        self.out_features = out_features
        self.swarm_size = swarm_size
        self.encoder: EncoderName = encoder
        self.tanh_tau = float(tanh_tau)
        gain = 1.0 / math.sqrt(in_features) if weight_scale else 1.0
        self.register_buffer("gain", torch.tensor(gain, dtype=torch.float32))

        init = torch.randint(0, 2, (out_features, in_features, swarm_size))
        init = (init * 2 - 1).to(torch.int8)
        self.register_buffer("swarm", init)

        self._w_link: torch.Tensor | None = None

    @property
    def population(self) -> torch.Tensor:
        """Alias for tests / legacy naming."""
        return self.swarm

    @property
    def last_link_grad(self) -> torch.Tensor | None:
        if self._w_link is None:
            return None
        return self._w_link.grad

    def link_value(self, swarm_f: torch.Tensor | None = None) -> torch.Tensor:
        if swarm_f is None:
            swarm_f = self.swarm.float()
        s = swarm_f.sum(dim=-1)
        return encode_sum(s, self.swarm_size, self.encoder, self.tanh_tau)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Detach storage; recompute link value as differentiable graph node.
        swarm_f = self.swarm.detach().float()
        s = swarm_f.sum(dim=-1)
        # Connect STE: treat soft sum path as if swarm_f required grad for encoder
        # by rebuilding from a leaf that tracks w_link only.
        w_link = encode_sum(s, self.swarm_size, self.encoder, self.tanh_tau)
        # Leaf for optimizer: same values, receives ∂L/∂w_link.
        w_param = w_link.detach().requires_grad_(True)
        self._w_link = w_param
        return F.linear(x, w_param * self.gain)

    @torch.no_grad()
    def enforce_binary_(self) -> None:
        p = self.swarm
        p.copy_(torch.where(p >= 0, torch.ones_like(p), -torch.ones_like(p)))

    def extra_repr(self) -> str:
        return (
            f"in_features={self.in_features}, out_features={self.out_features}, "
            f"swarm_size={self.swarm_size}, encoder={self.encoder}"
        )
