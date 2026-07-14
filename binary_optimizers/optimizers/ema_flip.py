"""
EMAFlipOptimizer: EMA-smoothed gradient with adaptive confidence gating + optional LR anneal.
"""

from __future__ import annotations

import math
from typing import Optional

import torch


class EMAFlipOptimizer(torch.optim.Optimizer):
    """
    EMA of gradient; update weights where |EMA| exceeds adaptive threshold.

    Parameters
    ----------
    threshold_scale : float
        Multiplier on running mean |grad EMA|. Lower = more aggressive (try 0.25–0.5).
    lr_min : float | None
        If set with total_steps, cosine-anneal lr from lr → lr_min.
    flip_mode : bool
        If True, confident positive-wrongness weights flip; else continuous sign step.
    """

    def __init__(
        self,
        params,
        lr: float = 0.05,
        lr_min: Optional[float] = None,
        momentum: float = 0.9,
        threshold_scale: float = 0.35,
        clip: float = 1.5,
        bn_lr: Optional[float] = None,
        flip_mode: bool = False,
        total_steps: int = 1000,
    ):
        if bn_lr is None:
            bn_lr = lr
        if threshold_scale <= 0:
            raise ValueError(f"threshold_scale must be > 0, got {threshold_scale}")
        if lr_min is None:
            lr_min = lr
        defaults = dict(
            lr=lr,
            lr_min=lr_min,
            momentum=momentum,
            threshold_scale=threshold_scale,
            clip=clip,
            bn_lr=bn_lr,
            flip_mode=flip_mode,
            total_steps=max(1, total_steps),
        )
        super().__init__(params, defaults)
        self.global_step = 0

    def _lr(self, group) -> float:
        lr0 = group["lr"]
        lr1 = group["lr_min"]
        if lr1 >= lr0:
            return float(lr0)
        t = self.global_step
        T = group["total_steps"]
        if t >= T:
            return float(lr1)
        return float(lr1 + 0.5 * (lr0 - lr1) * (1.0 + math.cos(math.pi * t / T)))

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            lr = self._lr(group)
            momentum = group["momentum"]
            threshold_scale = group["threshold_scale"]
            clip = group["clip"]
            bn_lr = group["bn_lr"]
            flip_mode = group["flip_mode"]

            for p in group["params"]:
                if p.grad is None:
                    continue

                if p.data.dim() < 2:
                    p.data.add_(p.grad.data, alpha=-bn_lr)
                    continue

                grad = p.grad.data
                state = self.state[p]
                if "grad_ema" not in state:
                    state["grad_ema"] = torch.zeros_like(p.data)
                    state["running_mean"] = torch.zeros(
                        (), device=p.device, dtype=p.dtype
                    )

                ema = state["grad_ema"]
                ema.mul_(momentum).add_(grad, alpha=1.0 - momentum)

                ema_abs = ema.abs()
                mean_abs = ema_abs.mean()
                rm = state["running_mean"]
                rm.mul_(0.95).add_(mean_abs, alpha=0.05)
                threshold = torch.clamp(rm * threshold_scale, min=1e-8)

                confident = ema_abs > threshold
                if not confident.any():
                    p.data.clamp_(-clip, clip)
                    continue

                if flip_mode:
                    wrongness = grad * p.data
                    flip = confident & (wrongness > threshold)
                    cont = confident & ~flip
                    if flip.any():
                        p.data[flip] *= -1
                        ema[flip] = 0.0
                    if cont.any():
                        p.data[cont] -= lr * torch.sign(ema[cont])
                else:
                    vote = torch.sign(ema)
                    p.data.add_(
                        torch.where(confident, vote, torch.zeros_like(vote)), alpha=-lr
                    )

                p.data.clamp_(-clip, clip)

        self.global_step += 1
        return loss
