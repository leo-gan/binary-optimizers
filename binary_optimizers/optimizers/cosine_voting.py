"""
CosineVotingOptimizer: Voting with cosine-annealed learning rate.

Improvement rationale over VotingOptimizer and MomentumVotingOptimizer:
  - VotingOptimizer uses a fixed lr, which means it either explores too
    aggressively late in training or converges too slowly early.
  - CosineVoting starts with a high lr for fast initial convergence, then
    anneals following a cosine schedule for fine-grained refinement.
  - Step counter is an instance attribute (not optimizer.state) so it does
    not break memory profilers that expect Parameter → dict state.
"""

from __future__ import annotations

import math
from typing import Optional

import torch


class CosineVotingOptimizer(torch.optim.Optimizer):
    """
    Binary-weight voting optimizer with built-in cosine annealing.

    Parameters
    ----------
    params : iterable
        Model parameters.
    lr_max : float
        Maximum (initial) learning rate for sign updates.
    lr_min : float
        Minimum learning rate at the end of annealing.
    momentum : float
        Momentum coefficient for gradient smoothing.
    total_steps : int
        Number of optimizer steps over which to anneal. After this,
        lr stays at lr_min.
    clip : float
        Weight clamp range.
    confidence_threshold : float
        Only apply a vote when |momentum buffer| exceeds this value.
    bn_lr : float | None
        Learning rate for batch-norm / 1-D parameters. Defaults to ``lr_max``.
    """

    def __init__(
        self,
        params,
        lr_max: float = 0.1,
        lr_min: float = 0.01,
        momentum: float = 0.9,
        total_steps: int = 200,
        clip: float = 1.5,
        confidence_threshold: float = 0.0,
        bn_lr: Optional[float] = None,
        restart_period: int = 0,
    ):
        if lr_max < lr_min:
            raise ValueError(f"lr_max ({lr_max}) must be >= lr_min ({lr_min})")
        if total_steps < 1:
            raise ValueError(f"total_steps must be >= 1, got {total_steps}")
        if bn_lr is None:
            bn_lr = lr_max
        defaults = dict(
            lr_max=lr_max,
            lr_min=lr_min,
            momentum=momentum,
            total_steps=total_steps,
            clip=clip,
            confidence_threshold=confidence_threshold,
            bn_lr=bn_lr,
            restart_period=max(0, int(restart_period)),
        )
        super().__init__(params, defaults)
        # Instance attribute: optimizer.state maps Parameter → dict only.
        self.global_step = 0

    def _cosine_lr(self, group) -> float:
        t = self.global_step
        T = group["total_steps"]
        lr_max = group["lr_max"]
        lr_min = group["lr_min"]
        period = group.get("restart_period") or 0
        # SGDR-style restarts within total_steps when period > 0
        if period > 0:
            cycle = t // period
            t_in = t % period
            # decay max lr each restart
            lr_max_c = lr_max * (0.7 ** cycle)
            lr_max_c = max(lr_max_c, lr_min)
            return float(
                lr_min + 0.5 * (lr_max_c - lr_min) * (1.0 + math.cos(math.pi * t_in / period))
            )
        if t >= T:
            return float(lr_min)
        return float(lr_min + 0.5 * (lr_max - lr_min) * (1.0 + math.cos(math.pi * t / T)))

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            lr = self._cosine_lr(group)
            mom = group["momentum"]
            clip = group["clip"]
            bn_lr = group["bn_lr"]
            conf_thresh = group["confidence_threshold"]

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
                buf.mul_(mom).add_(p.grad.data, alpha=1.0 - mom)

                if conf_thresh > 0:
                    vote = torch.zeros_like(buf)
                    confident = buf.abs() > conf_thresh
                    vote[confident] = torch.sign(buf[confident])
                else:
                    vote = torch.sign(buf)
                    # sign(0) → 0; treat zeros as no-op already

                p.data.add_(vote, alpha=-lr)
                p.data.clamp_(-clip, clip)

        self.global_step += 1
        return loss
