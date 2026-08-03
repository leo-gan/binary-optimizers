"""Encoding-atlas layers: fixed-point, exp+mant, block-scale (register lineage)."""

from __future__ import annotations

import math
from typing import Literal

import torch
import torch.nn as nn
import torch.nn.functional as F

EncodingName = Literal["fixed", "exp_mant", "block_scale"]


class SquaredReLU(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.relu(x).square()


def _exp_bias(n_exp: int) -> int:
    """Center exponent so scales straddle 1 when possible."""
    if n_exp < 1:
        return 0
    return (2**n_exp - 1) // 2


class EncodingLinear(nn.Module):
    """Discrete weight with selectable coding of a fixed bit budget.

    All modes store **binary digits** only (no FP master W). Forward builds
    ``w ∈ [-1, 1]`` (approximately) and applies STE through soft bits.

    Modes
    -----
    fixed
        Classic place-value: ``w = 2v/(2^n-1) - 1`` (v0.3).
    exp_mant
        Per-weight split: ``n_exp`` exponent bits + ``n_mant`` mantissa bits.
        ``w = m_norm * 2^(e - bias) / 2^(e_max - bias)`` so |w| ≤ 1.
    block_scale
        Mantissa bits per weight (full ``n_bits``) + shared exponent per **output
        row** (``n_exp`` bits). Microscaling-style.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        *,
        n_bits: int = 8,
        encoding: EncodingName = "fixed",
        n_exp: int = 0,
        weight_scale: bool = True,
    ):
        super().__init__()
        if n_bits < 1:
            raise ValueError(f"n_bits must be >= 1, got {n_bits}")
        if encoding not in ("fixed", "exp_mant", "block_scale"):
            raise ValueError(f"unknown encoding: {encoding}")
        self.in_features = in_features
        self.out_features = out_features
        self.n_bits = int(n_bits)
        self.encoding: EncodingName = encoding
        self.weight_scale = weight_scale

        gain = 1.0 / math.sqrt(in_features) if weight_scale else 1.0
        self.register_buffer("gain", torch.tensor(gain, dtype=torch.float32))

        if encoding == "fixed":
            self.n_exp = 0
            self.n_mant = self.n_bits
            places = torch.tensor(
                [2.0**i for i in range(self.n_mant)], dtype=torch.float32
            )
            vmax = float(2**self.n_mant - 1)
            self.register_buffer("mant_places", places)
            self.register_buffer("mant_vmax", torch.tensor(vmax, dtype=torch.float32))
            self.register_buffer("exp_places", torch.zeros(0))
            self.register_buffer("exp_vmax", torch.tensor(0.0))
            self.e_bias = 0
            self.scale_max = 1.0
            init = torch.randint(
                0, 2, (out_features, in_features, self.n_mant), dtype=torch.int8
            )
            self.register_buffer("mant_pop", init)
            self.register_buffer(
                "exp_pop", torch.zeros(out_features, in_features, 0, dtype=torch.int8)
            )
            self.register_buffer("row_exp_pop", torch.zeros(out_features, 0, dtype=torch.int8))
        elif encoding == "exp_mant":
            if n_exp < 1 or n_exp >= n_bits:
                raise ValueError(
                    f"exp_mant needs 1 <= n_exp < n_bits, got n_exp={n_exp}, n_bits={n_bits}"
                )
            self.n_exp = int(n_exp)
            self.n_mant = self.n_bits - self.n_exp
            if self.n_mant < 1:
                raise ValueError("need at least 1 mantissa bit")
            m_places = torch.tensor(
                [2.0**i for i in range(self.n_mant)], dtype=torch.float32
            )
            e_places = torch.tensor(
                [2.0**i for i in range(self.n_exp)], dtype=torch.float32
            )
            self.register_buffer("mant_places", m_places)
            self.register_buffer(
                "mant_vmax", torch.tensor(float(2**self.n_mant - 1), dtype=torch.float32)
            )
            self.register_buffer("exp_places", e_places)
            self.register_buffer(
                "exp_vmax", torch.tensor(float(2**self.n_exp - 1), dtype=torch.float32)
            )
            self.e_bias = _exp_bias(self.n_exp)
            e_max = int(self.exp_vmax.item())
            self.scale_max = float(2 ** (e_max - self.e_bias))
            self.register_buffer(
                "mant_pop",
                torch.randint(
                    0, 2, (out_features, in_features, self.n_mant), dtype=torch.int8
                ),
            )
            self.register_buffer(
                "exp_pop",
                torch.randint(
                    0, 2, (out_features, in_features, self.n_exp), dtype=torch.int8
                ),
            )
            self.register_buffer("row_exp_pop", torch.zeros(out_features, 0, dtype=torch.int8))
        else:  # block_scale
            if n_exp < 1:
                raise ValueError("block_scale needs n_exp >= 1")
            self.n_exp = int(n_exp)
            self.n_mant = self.n_bits  # full budget on mantissa digits
            m_places = torch.tensor(
                [2.0**i for i in range(self.n_mant)], dtype=torch.float32
            )
            e_places = torch.tensor(
                [2.0**i for i in range(self.n_exp)], dtype=torch.float32
            )
            self.register_buffer("mant_places", m_places)
            self.register_buffer(
                "mant_vmax", torch.tensor(float(2**self.n_mant - 1), dtype=torch.float32)
            )
            self.register_buffer("exp_places", e_places)
            self.register_buffer(
                "exp_vmax", torch.tensor(float(2**self.n_exp - 1), dtype=torch.float32)
            )
            self.e_bias = _exp_bias(self.n_exp)
            e_max = int(self.exp_vmax.item())
            self.scale_max = float(2 ** (e_max - self.e_bias))
            self.register_buffer(
                "mant_pop",
                torch.randint(
                    0, 2, (out_features, in_features, self.n_mant), dtype=torch.int8
                ),
            )
            self.register_buffer(
                "exp_pop", torch.zeros(out_features, in_features, 0, dtype=torch.int8)
            )
            self.register_buffer(
                "row_exp_pop",
                torch.randint(0, 2, (out_features, self.n_exp), dtype=torch.int8),
            )

        # Alias for optimizer / width bookkeeping
        self.swarm_size = self.n_bits
        self._mant_f: torch.Tensor | None = None
        self._exp_f: torch.Tensor | None = None
        self._row_exp_f: torch.Tensor | None = None

    @property
    def last_mant_grad(self) -> torch.Tensor | None:
        if self._mant_f is None:
            return None
        return self._mant_f.grad

    @property
    def last_exp_grad(self) -> torch.Tensor | None:
        if self._exp_f is None:
            return None
        return self._exp_f.grad

    @property
    def last_row_exp_grad(self) -> torch.Tensor | None:
        if self._row_exp_f is None:
            return None
        return self._row_exp_f.grad

    def _mant_int(self, mant_f: torch.Tensor) -> torch.Tensor:
        return (mant_f * self.mant_places).sum(dim=-1)

    def _exp_int(self, exp_f: torch.Tensor) -> torch.Tensor:
        return (exp_f * self.exp_places).sum(dim=-1)

    def effective_weight(
        self,
        mant_f: torch.Tensor | None = None,
        exp_f: torch.Tensor | None = None,
        row_exp_f: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if mant_f is None:
            mant_f = self.mant_pop.float()
        m = self._mant_int(mant_f)
        m_norm = 2.0 * m / self.mant_vmax - 1.0

        if self.encoding == "fixed":
            return m_norm

        if self.encoding == "exp_mant":
            if exp_f is None:
                exp_f = self.exp_pop.float()
            e = self._exp_int(exp_f)
            scale = torch.pow(2.0, e - float(self.e_bias)) / self.scale_max
            return m_norm * scale

        # block_scale
        if row_exp_f is None:
            row_exp_f = self.row_exp_pop.float()
        e_row = self._exp_int(row_exp_f)  # [out]
        scale = torch.pow(2.0, e_row - float(self.e_bias)) / self.scale_max
        return m_norm * scale.unsqueeze(-1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        mant_f = self.mant_pop.detach().float().requires_grad_(True)
        self._mant_f = mant_f
        exp_f = row_exp_f = None
        if self.encoding == "exp_mant":
            exp_f = self.exp_pop.detach().float().requires_grad_(True)
            self._exp_f = exp_f
            self._row_exp_f = None
        elif self.encoding == "block_scale":
            row_exp_f = self.row_exp_pop.detach().float().requires_grad_(True)
            self._row_exp_f = row_exp_f
            self._exp_f = None
        else:
            self._exp_f = None
            self._row_exp_f = None
        w = self.effective_weight(mant_f, exp_f, row_exp_f)
        return F.linear(x, w * self.gain)

    @staticmethod
    def _encode_int(v: torch.Tensor, n_bits: int, vmax: int) -> torch.Tensor:
        v = v.round().clamp(0, vmax).to(torch.int64)
        bits = []
        tmp = v
        for _ in range(n_bits):
            bits.append((tmp & 1).to(torch.int8))
            tmp = tmp >> 1
        return torch.stack(bits, dim=-1)

    @torch.no_grad()
    def set_mant_from_integer_(self, v: torch.Tensor) -> None:
        vmax = int(self.mant_vmax.item())
        self.mant_pop.copy_(self._encode_int(v, self.n_mant, vmax))

    @torch.no_grad()
    def set_exp_from_integer_(self, v: torch.Tensor) -> None:
        if self.encoding != "exp_mant":
            raise RuntimeError("set_exp_from_integer_ only for exp_mant")
        vmax = int(self.exp_vmax.item())
        self.exp_pop.copy_(self._encode_int(v, self.n_exp, vmax))

    @torch.no_grad()
    def set_row_exp_from_integer_(self, v: torch.Tensor) -> None:
        if self.encoding != "block_scale":
            raise RuntimeError("set_row_exp_from_integer_ only for block_scale")
        vmax = int(self.exp_vmax.item())
        self.row_exp_pop.copy_(self._encode_int(v, self.n_exp, vmax))

    @torch.no_grad()
    def enforce_binary_(self) -> None:
        for buf in (self.mant_pop, self.exp_pop, self.row_exp_pop):
            if buf.numel() == 0:
                continue
            buf.copy_(torch.where(buf > 0, torch.ones_like(buf), torch.zeros_like(buf)))

    def extra_repr(self) -> str:
        return (
            f"in={self.in_features}, out={self.out_features}, n_bits={self.n_bits}, "
            f"encoding={self.encoding}, n_exp={self.n_exp}, n_mant={self.n_mant}"
        )
