"""Diagnostics for UnaryLinkMLP."""

from __future__ import annotations

from typing import Any, Dict

import torch
import torch.nn as nn

from layers import UnaryLinkLinear


@torch.no_grad()
def swarm_stats(model: nn.Module) -> Dict[str, float]:
    n_bits = 0
    n_pos = 0
    abs_s_sum = 0.0
    abs_w_sum = 0.0
    n_links = 0
    swarm_size = 1

    for m in model.modules():
        if not isinstance(m, UnaryLinkLinear):
            continue
        swarm_size = m.swarm_size
        pop = m.swarm.float()
        n_bits += pop.numel()
        n_pos += int((pop > 0).sum().item())
        s = pop.sum(dim=-1)
        abs_s_sum += s.abs().sum().item()
        w = m.link_value()
        abs_w_sum += w.abs().sum().item()
        n_links += s.numel()

    return {
        "n_weights": float(n_bits),
        "frac_plus": n_pos / max(1, n_bits),
        "mean_abs_sum": abs_s_sum / max(1, n_links),
        "mean_abs_sum_norm": (abs_s_sum / max(1, n_links)) / float(swarm_size),
        "mean_abs_link_value": abs_w_sum / max(1, n_links),
    }


def history_to_rows(history: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return list(history)
