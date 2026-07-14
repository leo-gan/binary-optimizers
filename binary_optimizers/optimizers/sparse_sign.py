"""
SparseSignOptimizer: Sign-based updates on a (possibly annealed) sparse subset.

Improvements over fixed-density sparse updates:
  - density can anneal from density_start → density_end over total_steps
  - optional magnitude-biased sampling (∝ |momentum|) instead of uniform
"""

from __future__ import annotations

from typing import Optional

import torch


class SparseSignOptimizer(torch.optim.Optimizer):
    """
    Each step, select a ``density`` fraction of weights and apply sign(momentum).

    Parameters
    ----------
    density : float
        Base density when annealing is disabled (density_end is None).
    density_start, density_end : float | None
        If both set, density anneals linearly start→end over total_steps.
    total_steps : int
        Annealing horizon (also used by effective_lr compensation).
    magnitude_biased : bool
        If True, sample proportional to |momentum_buffer| (soft top-k).
    """

    def __init__(
        self,
        params,
        lr: float = 0.05,
        momentum: float = 0.9,
        density: float = 0.6,
        density_start: Optional[float] = None,
        density_end: Optional[float] = None,
        total_steps: int = 1000,
        clip: float = 1.5,
        confidence_threshold: float = 0.001,
        bn_lr: Optional[float] = None,
        magnitude_biased: bool = False,
    ):
        if density_start is None:
            density_start = density
        if density_end is None:
            density_end = density
        for d, name in ((density_start, "density_start"), (density_end, "density_end")):
            if not 0 < d <= 1:
                raise ValueError(f"{name} must be in (0, 1], got {d}")
        if bn_lr is None:
            bn_lr = lr
        defaults = dict(
            lr=lr,
            momentum=momentum,
            density=density,
            density_start=density_start,
            density_end=density_end,
            total_steps=max(1, total_steps),
            clip=clip,
            confidence_threshold=confidence_threshold,
            bn_lr=bn_lr,
            magnitude_biased=magnitude_biased,
        )
        super().__init__(params, defaults)
        self.global_step = 0

    def _current_density(self, group) -> float:
        t = self.global_step
        T = group["total_steps"]
        d0 = group["density_start"]
        d1 = group["density_end"]
        if t >= T:
            return float(d1)
        alpha = t / T
        return float(d0 + alpha * (d1 - d0))

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            lr = group["lr"]
            momentum = group["momentum"]
            density = self._current_density(group)
            clip = group["clip"]
            conf_thresh = group["confidence_threshold"]
            bn_lr = group["bn_lr"]
            mag_bias = group["magnitude_biased"]

            for p in group["params"]:
                if p.grad is None:
                    continue

                if p.data.dim() < 2:
                    p.data.add_(p.grad.data, alpha=-bn_lr)
                    continue

                state = self.state[p]
                if "momentum_buffer" not in state:
                    state["momentum_buffer"] = torch.zeros_like(p.data)

                buf = state["momentum_buffer"]
                buf.mul_(momentum).add_(p.grad.data, alpha=1.0 - momentum)

                confident = buf.abs() > conf_thresh

                if mag_bias:
                    # Sample without replacement-ish: keep top fraction by |buf|
                    # approximated by Bernoulli with p ∝ |buf| scaled to mean density
                    w = buf.abs()
                    mean_w = w.mean().clamp(min=1e-12)
                    probs = (w / mean_w * density).clamp(0, 1)
                    sparse_mask = torch.rand_like(p.data) < probs
                else:
                    sparse_mask = torch.rand_like(p.data) < density

                active = confident & sparse_mask
                effective_lr = lr / max(density, 1e-3)
                vote = torch.sign(buf)
                p.data[active] -= effective_lr * vote[active]
                p.data.clamp_(-clip, clip)

        self.global_step += 1
        return loss
