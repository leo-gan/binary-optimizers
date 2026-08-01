"""Metrics for experiment v0_1."""

from __future__ import annotations

from typing import Any, Dict, List

import torch
import torch.nn as nn

from layers import Int8SwarmLinear


@torch.no_grad()
def swarm_stats(model: nn.Module) -> Dict[str, float]:
    """Population diagnostics over all Int8SwarmLinear modules."""
    n_agents = 0
    n_pos = 0
    margin_abs_sum = 0.0
    n_weights = 0

    for m in model.modules():
        if not isinstance(m, Int8SwarmLinear):
            continue
        pop = m.population.float()
        n_agents += pop.numel()
        n_pos += int((pop > 0).sum().item())
        s = pop.sum(dim=-1)
        margin_abs_sum += s.abs().sum().item()
        n_weights += s.numel()

    swarm_size = 1
    for m in model.modules():
        if isinstance(m, Int8SwarmLinear):
            swarm_size = m.swarm_size
            break
    mean_margin = margin_abs_sum / max(1, n_weights)
    return {
        "n_agents": float(n_agents),
        "frac_plus": n_pos / max(1, n_agents),
        "mean_abs_margin": mean_margin,
        "mean_abs_margin_norm": mean_margin / float(swarm_size),
    }


def history_to_rows(history: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    return list(history)
