"""Metrics for experiment v0_4."""

from __future__ import annotations

from typing import Dict

import torch.nn as nn

from layers import Int8BalancedTernaryLinear


def swarm_stats(model: nn.Module) -> Dict[str, float]:
    n = n_zero = 0
    abs_w = 0.0
    n_w = 0
    for m in model.modules():
        if not isinstance(m, Int8BalancedTernaryLinear):
            continue
        p = m.population
        n += p.numel()
        n_zero += int((p == 0).sum().item())
        w = m.effective_weight()
        abs_w += float(w.abs().sum().item())
        n_w += w.numel()
    return {
        "n_digits": float(n),
        "zero_frac": n_zero / max(1, n),
        "mean_abs_w": abs_w / max(1, n_w),
    }
