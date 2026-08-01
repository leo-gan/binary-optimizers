"""Metrics for experiment v0_3."""

from __future__ import annotations

from typing import Dict

import torch.nn as nn

from layers import Int8CarrySafeLinear


def swarm_stats(model: nn.Module) -> Dict[str, float]:
    n_bits_on = 0
    n_agents = 0
    abs_w = 0.0
    n_w = 0
    for m in model.modules():
        if not isinstance(m, Int8CarrySafeLinear):
            continue
        p = m.population
        n_agents += p.numel()
        n_bits_on += int((p > 0).sum().item())
        w = m.effective_weight()
        abs_w += float(w.abs().sum().item())
        n_w += w.numel()
    return {
        "n_agents": float(n_agents),
        "frac_bits_on": n_bits_on / max(1, n_agents),
        "mean_abs_w": abs_w / max(1, n_w),
    }
