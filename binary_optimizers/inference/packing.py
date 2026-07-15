"""Bit-packing utilities for binary (±1) tensors."""

from __future__ import annotations

import math

import torch


# Bit encoding: +1 -> 1, -1 -> 0. Zero is treated as +1.
def _to_bits(x: torch.Tensor) -> torch.Tensor:
    """Map real values to bits: +1 -> 1, -1/else -> 0 (zeros map to 1)."""
    bits = (x >= 0).to(torch.uint8)
    return bits


def pack_binary_weights(w: torch.Tensor) -> tuple[torch.Tensor, int]:
    """
    Pack a weight tensor of shape (out_features, in_features) with values in
    {-1, +1} into a uint8 tensor with 8 weights per byte.

    Returns
    -------
    packed : Tensor[uint8] of shape (out_features, ceil(in_features / 8))
    in_features : original in_features (needed for unpack / popcount correction)
    """
    if w.dim() != 2:
        raise ValueError(f"Expected 2D weight matrix, got shape {tuple(w.shape)}")

    out_features, in_features = w.shape
    bits = _to_bits(w)  # (O, I) uint8 in {0, 1}

    pad = (-in_features) % 8
    if pad:
        bits = torch.nn.functional.pad(bits, (0, pad), value=0)

    bits = bits.view(out_features, -1, 8)
    # Pack little-endian within each byte: bit 0 = least significant
    powers = (2 ** torch.arange(8, device=w.device, dtype=torch.int64)).view(1, 1, 8)
    packed = (bits.to(torch.int64) * powers).sum(dim=-1).to(torch.uint8)
    return packed.contiguous(), in_features


def unpack_binary_weights(packed: torch.Tensor, in_features: int) -> torch.Tensor:
    """
    Unpack a bitpacked uint8 weight tensor back to ±1 floats.

    Parameters
    ----------
    packed : Tensor[uint8] of shape (out_features, ceil(in_features / 8))
    in_features : original feature count
    """
    if packed.dim() != 2:
        raise ValueError(f"Expected 2D packed tensor, got shape {tuple(packed.shape)}")

    out_features, n_bytes = packed.shape
    expected_bytes = math.ceil(in_features / 8)
    if n_bytes != expected_bytes:
        raise ValueError(
            f"Packed width {n_bytes} does not match in_features={in_features} "
            f"(expected {expected_bytes} bytes)"
        )

    powers = (2 ** torch.arange(8, device=packed.device, dtype=torch.int64)).view(1, 1, 8)
    bits = ((packed.to(torch.int64).unsqueeze(-1) & powers) != 0).to(torch.float32)
    bits = bits.view(out_features, n_bytes * 8)[:, :in_features]
    # bit 1 -> +1, bit 0 -> -1
    return bits * 2.0 - 1.0


def pack_activations(x: torch.Tensor, in_features: int | None = None) -> torch.Tensor:
    """
    Pack a batch of activations of shape (batch, in_features) to uint8 bitpacked form.

    Values are signed first (x >= 0 -> +1 bit).
    """
    if x.dim() != 2:
        raise ValueError(f"Expected 2D activations, got shape {tuple(x.shape)}")

    batch, feat = x.shape
    if in_features is not None and feat != in_features:
        raise ValueError(f"Expected in_features={in_features}, got {feat}")

    bits = _to_bits(x)
    pad = (-feat) % 8
    if pad:
        bits = torch.nn.functional.pad(bits, (0, pad), value=0)

    bits = bits.view(batch, -1, 8)
    powers = (2 ** torch.arange(8, device=x.device, dtype=torch.int64)).view(1, 1, 8)
    packed = (bits.to(torch.int64) * powers).sum(dim=-1).to(torch.uint8)
    return packed.contiguous()


def packed_nbytes(in_features: int) -> int:
    """Number of bytes needed to store ``in_features`` binary weights."""
    return math.ceil(in_features / 8)
