"""Metrics for experiment v0_2."""

from __future__ import annotations

from typing import Dict

import torch
import torch.nn as nn

from layers import Int8PlaceValueSwarmLinear


@torch.no_grad()
def swarm_stats(model: nn.Module) -> Dict[str, float]:
    n_agents = 0
    n_pos = 0
    abs_s_sum = 0.0
    n_weights = 0
    place_norm = 1.0

    for m in model.modules():
        if not isinstance(m, Int8PlaceValueSwarmLinear):
            continue
        pop = m.population.float()
        n_agents += pop.numel()
        n_pos += int((pop > 0).sum().item())
        s = m.place_value_sum(pop)
        abs_s_sum += s.abs().sum().item()
        n_weights += s.numel()
        place_norm = float(m.place_norm.item())

    mean_abs_s = abs_s_sum / max(1, n_weights)
    return {
        "n_agents": float(n_agents),
        "frac_plus": n_pos / max(1, n_agents),
        "mean_abs_place_sum": mean_abs_s,
        "mean_abs_place_sum_norm": mean_abs_s / max(place_norm, 1e-8),
    }
