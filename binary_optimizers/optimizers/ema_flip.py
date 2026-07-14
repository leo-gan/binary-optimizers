"""
EMAFlipOptimizer: EMA-smoothed gradient with adaptive confidence gating.

Improvement rationale over existing optimizers:
  - MomentumRankOptimizer uses expensive torch.topk every step.
    EMAFlip replaces topk with a threshold comparison (O(n)).
  - EMA smooths the directional signal; an adaptive threshold on |EMA|
    self-regulates how many weights update each step.
  - Optional flip_mode: when wrongness (grad * weight) is high, flip ±1
    weights instead of a continuous step (binary-native update).
"""

from __future__ import annotations

from typing import Optional

import torch


class EMAFlipOptimizer(torch.optim.Optimizer):
    """
    Maintains an EMA of the gradient. Updates weights where the smoothed
    signal exceeds an adaptive threshold (running mean of |EMA| * scale).

    Parameters
    ----------
    params : iterable
        Model parameters.
    lr : float
        Sign-update step size for continuous mode. Default 0.05.
    momentum : float
        EMA decay factor. Default 0.9.
    threshold_scale : float
        Multiplier on running mean |grad EMA| for the confidence gate.
        Lower = more aggressive updates. Default 0.5.
    clip : float
        Weight clamp range. Default 1.5.
    bn_lr : float | None
        Learning rate for 1-D params. Defaults to ``lr``.
    flip_mode : bool
        If True, confident weights with positive wrongness (grad * w) are
        flipped; remaining confident weights get a continuous sign step.
        If False (default), all confident weights get a continuous sign step.
    """

    def __init__(
        self,
        params,
        lr: float = 0.05,
        momentum: float = 0.9,
        threshold_scale: float = 0.5,
        clip: float = 1.5,
        bn_lr: Optional[float] = None,
        flip_mode: bool = False,
    ):
        if bn_lr is None:
            bn_lr = lr
        if threshold_scale <= 0:
            raise ValueError(f"threshold_scale must be > 0, got {threshold_scale}")
        defaults = dict(
            lr=lr,
            momentum=momentum,
            threshold_scale=threshold_scale,
            clip=clip,
            bn_lr=bn_lr,
            flip_mode=flip_mode,
        )
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            lr = group["lr"]
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
                    # Flip when weight is aligned with gradient (positive wrongness)
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
                    p.data.add_(torch.where(confident, vote, torch.zeros_like(vote)), alpha=-lr)

                p.data.clamp_(-clip, clip)

        return loss
