"""
HybridV2Optimizer: stack of proven binary pieces to close the gap to Adam.

Components (ablation-friendly flags):
  1. Momentum-smoothed gradient (Signum / MomentumVoting)
  2. Cosine-annealed step size (CosineVoting)
  3. Optional sparse active set (SparseSign)
  4. Integrate-and-fire hard/soft update (ThresholdIF / Hybrid)
"""

from __future__ import annotations

import math
from typing import Optional

import torch


class HybridV2Optimizer(torch.optim.Optimizer):
    """
    Binary-weight trainer combining momentum voting, cosine LR, sparse masks,
    and accumulate-then-fire updates.

    Parameters
    ----------
    lr_max, lr_min : float
        Cosine schedule endpoints for the integrate scale.
    total_steps : int
        Cosine horizon.
    momentum : float
        EMA of gradients.
    threshold : float
        Fire when |accumulator| exceeds this.
    decay : float
        Accumulator decay each step.
    density : float
        Fraction of weights that integrate each step (1.0 = dense).
    soft_fire : bool
        Soft blend toward sign(acc) vs hard assign.
    soft_alpha : float
        Blend weight when soft_fire is True.
    use_fire : bool
        If False, apply continuous sign(momentum) updates (no IF).
    clip : float
        Weight clamp.
    bn_lr : float
        SGD lr for 1-D params (unused if BN handled by outer Adam).
    """

    def __init__(
        self,
        params,
        lr_max: float = 0.12,
        lr_min: float = 0.008,
        total_steps: int = 7000,
        momentum: float = 0.9,
        threshold: float = 0.04,
        decay: float = 0.97,
        density: float = 0.85,
        soft_fire: bool = True,
        soft_alpha: float = 0.65,
        use_fire: bool = True,
        clip: float = 1.5,
        bn_lr: Optional[float] = None,
        confidence_threshold: float = 0.0,
    ):
        if bn_lr is None:
            bn_lr = lr_max
        if not 0 < density <= 1:
            raise ValueError(f"density must be in (0, 1], got {density}")
        defaults = dict(
            lr_max=lr_max,
            lr_min=lr_min,
            total_steps=max(1, total_steps),
            momentum=momentum,
            threshold=threshold,
            decay=decay,
            density=density,
            soft_fire=soft_fire,
            soft_alpha=soft_alpha,
            use_fire=use_fire,
            clip=clip,
            bn_lr=bn_lr,
            confidence_threshold=confidence_threshold,
        )
        super().__init__(params, defaults)
        self.global_step = 0

    def _cosine_lr(self, group) -> float:
        t = self.global_step
        T = group["total_steps"]
        lr_max = group["lr_max"]
        lr_min = group["lr_min"]
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
            thr = group["threshold"]
            decay = group["decay"]
            density = group["density"]
            soft_fire = group["soft_fire"]
            soft_alpha = group["soft_alpha"]
            use_fire = group["use_fire"]
            clip = group["clip"]
            bn_lr = group["bn_lr"]
            conf = group["confidence_threshold"]

            for p in group["params"]:
                if p.grad is None:
                    continue

                if p.data.dim() < 2:
                    p.data.add_(p.grad.data, alpha=-bn_lr)
                    continue

                state = self.state[p]
                if "grad_ema" not in state:
                    state["grad_ema"] = torch.zeros_like(p.data)
                    state["accumulator"] = torch.zeros_like(p.data)

                ema = state["grad_ema"]
                ema.mul_(mom).add_(p.grad.data, alpha=1.0 - mom)

                # Confidence gate
                if conf > 0:
                    active = ema.abs() > conf
                else:
                    active = torch.ones_like(ema, dtype=torch.bool)

                # Sparse subset of active weights
                if density < 1.0:
                    sparse = torch.rand_like(p.data) < density
                    active = active & sparse

                if use_fire:
                    acc = state["accumulator"]
                    # Integrate anti-gradient pressure only on active set
                    pressure = torch.zeros_like(ema)
                    pressure[active] = -lr * ema[active]
                    acc.mul_(decay).add_(pressure)

                    fire = acc.abs() > thr
                    if fire.any():
                        target = torch.sign(acc[fire])
                        target = torch.where(target == 0, torch.ones_like(target), target)
                        if soft_fire:
                            a = soft_alpha
                            p.data[fire] = (1 - a) * p.data[fire] + a * target
                        else:
                            p.data[fire] = target
                        acc[fire] = 0.0
                else:
                    # Continuous momentum sign (Signum-like) with cosine lr
                    vote = torch.sign(ema)
                    p.data.add_(
                        torch.where(active, vote, torch.zeros_like(vote)), alpha=-lr
                    )

                p.data.clamp_(-clip, clip)

        self.global_step += 1
        return loss
