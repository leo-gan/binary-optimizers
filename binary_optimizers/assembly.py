"""Compression-based assembly-index proxy for binary tensors.

Exact Assembly Index is NP-hard. This module uses packed signs + zlib, which
is the LZ-style proxy described in ``docs/PATHWAY_OPTIMIZER.md``.

Random ±1 tensors pack to incompressible bits (ratio near 1). Repeated or
tiled patterns compress. Do **not** compress raw float32 ``±1.0`` bytes: the
IEEE payload is highly redundant and even random signs look "structured".
"""

from __future__ import annotations

import zlib
from typing import Iterable

import numpy as np
import torch


def pack_signs(tensor: torch.Tensor) -> bytes:
    """Pack one bit per element (``>= 0`` → 1) into a big-endian bitstring."""
    bits = (tensor.detach().reshape(-1) >= 0).to(dtype=torch.uint8).cpu().numpy()
    packed = np.packbits(bits, bitorder="big")
    return packed.tobytes()


def approximate_assembly_index(tensor: torch.Tensor, *, level: int = 6) -> float:
    """Return ``len(zlib(pack_signs(t))) / len(pack_signs(t))``.

    Small values mean more repetition (lower assembly index). Random signs
    sit near 1. Very small tensors can exceed 1 because of the zlib header.
    """
    packed = pack_signs(tensor)
    n = max(1, len(packed))
    compressed = zlib.compress(packed, level=level)
    return len(compressed) / n


def mean_assembly_index(
    tensors: Iterable[torch.Tensor],
    *,
    level: int = 6,
) -> float:
    """Size-weighted mean of ``approximate_assembly_index`` over non-empty tensors."""
    num = 0.0
    den = 0.0
    for t in tensors:
        if t is None or t.numel() == 0:
            continue
        w = float(t.numel())
        num += approximate_assembly_index(t, level=level) * w
        den += w
    if den <= 0.0:
        return 0.0
    return num / den


def snap_binary_(tensor: torch.Tensor) -> torch.Tensor:
    """In-place project onto ``{+1, -1}`` (zeros become ``+1``)."""
    s = tensor.sign()
    tensor.copy_(torch.where(s == 0, torch.ones_like(s), s))
    return tensor
