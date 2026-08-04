"""Flat CIFAR-10 MLP with encoding-atlas linears (WP3 sparse probe)."""

from __future__ import annotations

from typing import List, Literal

import torch
import torch.nn as nn

from layers import EncodingLinear, EncodingName, SquaredReLU

LNMode = Literal["none", "no_affine", "affine"]
ActName = Literal["squared_relu", "relu"]

CIFAR_FLAT = 32 * 32 * 3  # 3072


def _make_ln(num_features: int, mode: LNMode) -> nn.Module:
    if mode == "none":
        return nn.Identity()
    if mode == "no_affine":
        return nn.LayerNorm(num_features, elementwise_affine=False)
    if mode == "affine":
        return nn.LayerNorm(num_features, elementwise_affine=True)
    raise ValueError(f"Unknown ln_mode: {mode}")


class BitNetEncodingCIFARMLP(nn.Module):
    """Same BitNet-style MLP family as v0.6, but 3072-d CIFAR flatten input."""

    def __init__(
        self,
        *,
        hidden_dim: int = 128,
        n_bits: int = 8,
        encoding: EncodingName = "fixed",
        n_exp: int = 0,
        ln_mode: LNMode = "none",
        depth: int = 1,
        activation: ActName = "relu",
    ):
        super().__init__()
        if depth < 1:
            raise ValueError(f"depth must be >= 1, got {depth}")
        self.hidden_dim = hidden_dim
        self.n_bits = n_bits
        self.encoding = encoding
        self.n_exp = n_exp
        self.ln_mode = ln_mode
        self.flatten = nn.Flatten()
        self.act: nn.Module = (
            SquaredReLU() if activation == "squared_relu" else nn.ReLU()
        )
        self.lns = nn.ModuleList()
        self.linears = nn.ModuleList()
        in_dim = CIFAR_FLAT
        for _ in range(depth):
            self.lns.append(_make_ln(in_dim, ln_mode))
            self.linears.append(
                EncodingLinear(
                    in_dim,
                    hidden_dim,
                    n_bits=n_bits,
                    encoding=encoding,
                    n_exp=n_exp,
                )
            )
            in_dim = hidden_dim
        self.lns.append(_make_ln(in_dim, ln_mode))
        self.linears.append(
            EncodingLinear(in_dim, 10, n_bits=n_bits, encoding=encoding, n_exp=n_exp)
        )

    def swarm_layers(self) -> List[EncodingLinear]:
        return list(self.linears)

    def ln_parameters(self) -> List[nn.Parameter]:
        return [p for ln in self.lns for p in ln.parameters()]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.flatten(x)
        n = len(self.linears)
        for i, (ln, linear) in enumerate(zip(self.lns, self.linears)):
            x = ln(x)
            x = linear(x)
            if i < n - 1:
                x = self.act(x)
        return x

    @torch.no_grad()
    def assert_binary_invariants(self) -> None:
        for layer in self.linears:
            for buf in (layer.mant_pop, layer.exp_pop, layer.row_exp_pop):
                if buf.numel() == 0:
                    continue
                assert buf.dtype == torch.int8
                uniq = set(buf.unique().tolist())
                assert uniq.issubset({0, 1}), uniq

    @torch.no_grad()
    def weight_stats(self) -> dict:
        ws = []
        es = []
        for layer in self.linears:
            w = layer.effective_weight()
            ws.append(w.reshape(-1))
            if layer.encoding == "exp_mant":
                es.append(layer._exp_int(layer.exp_pop.float()).reshape(-1))
            elif layer.encoding == "block_scale":
                es.append(layer._exp_int(layer.row_exp_pop.float()).reshape(-1))
        w_all = torch.cat(ws)
        out = {
            "mean_abs_w": float(w_all.abs().mean().item()),
            "std_w": float(w_all.std().item()),
            "frac_near_pm1": float((w_all.abs() > 0.9).float().mean().item()),
        }
        if es:
            e_all = torch.cat(es)
            out["mean_exp"] = float(e_all.mean().item())
            out["std_exp"] = float(e_all.std().item())
        return out
