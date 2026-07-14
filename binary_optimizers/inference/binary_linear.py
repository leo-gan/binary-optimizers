"""XNOR + popcount binary linear forward pass (no float multiplies in matmul)."""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .packing import pack_activations, pack_binary_weights, unpack_binary_weights


def _popcount_uint8(t: torch.Tensor) -> torch.Tensor:
    """Population count for each uint8 element. Returns int32 tensor same shape."""
    # Software popcount via bit iteration — pure torch, no float matmul.
    x = t.to(torch.int32)
    count = torch.zeros_like(x)
    for i in range(8):
        count = count + ((x >> i) & 1)
    return count


def _valid_bits_mask(in_features: int, device: torch.device) -> torch.Tensor:
    """
    Per-byte valid-bit mask (uint8) so padding bits beyond ``in_features``
    do not contribute to the popcount.
    """
    n_bytes = math.ceil(in_features / 8)
    mask = torch.full((n_bytes,), 0xFF, dtype=torch.uint8, device=device)
    rem = in_features % 8
    if rem != 0:
        # Keep only the first ``rem`` bits (LSB side, matching pack order).
        mask[-1] = (1 << rem) - 1
    return mask


def binary_linear_forward(
    x: torch.Tensor,
    packed_w: torch.Tensor,
    in_features: int,
    bias: torch.Tensor | None = None,
) -> torch.Tensor:
    """
    Binary linear layer: sign(x) @ sign(w)^T via XNOR + popcount.

    Parameters
    ----------
    x : (batch, in_features) float — will be signed/packed
    packed_w : (out_features, ceil(in_features/8)) uint8
    in_features : original input dimension
    bias : optional (out_features,) float bias added after the binary matmul

    Returns
    -------
    out : (batch, out_features) float32
        Each output equals sum_i sign(x_i) * sign(w_ji)  ∈ [-in_features, +in_features]
    """
    if x.dim() != 2:
        raise ValueError(f"Expected 2D input, got {tuple(x.shape)}")
    if x.size(1) != in_features:
        raise ValueError(f"x has {x.size(1)} features, expected {in_features}")

    x_packed = pack_activations(x, in_features)  # (B, K)
    mask = _valid_bits_mask(in_features, x.device)  # (K,)

    # XNOR: ~(x XOR w); mask padding bits so they contribute 0 agreements.
    # Shape broadcast: (B, 1, K) xnor (1, O, K) -> (B, O, K)
    xnor = torch.bitwise_not(
        torch.bitwise_xor(x_packed.unsqueeze(1), packed_w.unsqueeze(0))
    )
    xnor = torch.bitwise_and(xnor, mask.view(1, 1, -1))

    agreements = _popcount_uint8(xnor).sum(dim=-1)  # (B, O) int32
    # sum of ±1 products = 2 * agreements - in_features
    out = (2 * agreements - in_features).to(torch.float32)

    if bias is not None:
        out = out + bias
    return out


class BinaryLinear(nn.Module):
    """Inference-only binary linear layer with packed uint8 weights."""

    def __init__(
        self,
        packed_w: torch.Tensor,
        in_features: int,
        bias: torch.Tensor | None = None,
    ):
        super().__init__()
        self.register_buffer("packed_w", packed_w.to(torch.uint8))
        self.in_features = in_features
        self.out_features = packed_w.size(0)
        if bias is not None:
            self.register_buffer("bias", bias.to(torch.float32))
        else:
            self.bias = None

    @classmethod
    def from_weight(
        cls,
        weight: torch.Tensor,
        bias: torch.Tensor | None = None,
    ) -> "BinaryLinear":
        """Build from a float weight matrix; weights are signed then packed."""
        packed, in_features = pack_binary_weights(torch.sign(weight))
        # zeros -> pack as +1; align with sign convention used in packing
        return cls(packed, in_features, bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return binary_linear_forward(x, self.packed_w, self.in_features, self.bias)

    def unpacked_weight(self) -> torch.Tensor:
        return unpack_binary_weights(self.packed_w, self.in_features)


def reference_binary_linear(
    x: torch.Tensor,
    w: torch.Tensor,
    bias: torch.Tensor | None = None,
) -> torch.Tensor:
    """Reference: F.linear(sign(x), sign(w), bias) with zeros treated as +1."""
    x_s = torch.where(x >= 0, torch.ones_like(x), -torch.ones_like(x))
    w_s = torch.where(w >= 0, torch.ones_like(w), -torch.ones_like(w))
    return F.linear(x_s, w_s, bias)
