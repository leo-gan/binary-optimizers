"""Split BatchNorm / bias (continuous) vs weight matrices (binary-ish)."""

from __future__ import annotations

from typing import Iterable

import torch.nn as nn


def split_binary_and_bn_params(
    model: nn.Module,
) -> tuple[list[nn.Parameter], list[nn.Parameter]]:
    """
    Returns (binary_weight_params, continuous_params).

    Heuristic: 2D+ tensors are binary layer weights; 1D (BN scale/bias, bias) continuous.
    """
    binary: list[nn.Parameter] = []
    continuous: list[nn.Parameter] = []
    for p in model.parameters():
        if not p.requires_grad:
            continue
        if p.dim() >= 2:
            binary.append(p)
        else:
            continuous.append(p)
    return binary, continuous


def adam_bn_param_group(
    continuous_params: Iterable[nn.Parameter],
    lr: float = 1e-3,
) -> dict:
    params = list(continuous_params)
    return {"params": params, "lr": lr, "is_bn": True}
