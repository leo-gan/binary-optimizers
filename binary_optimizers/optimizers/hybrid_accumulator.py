"""
HybridAccumulatorOptimizer: per-parameter adaptive IF with optional soft fire.
"""

from __future__ import annotations

from typing import Optional

import torch


class HybridAccumulatorOptimizer(torch.optim.Optimizer):
    """
    EMA gradient + accumulate-then-fire with **per-tensor** adaptive thresholds.

    Parameters
    ----------
    soft_fire : bool
        If True, blend toward sign(acc) instead of hard assign.
    soft_alpha : float
        Blend factor when soft_fire is True.
    min_fire_rate : float
        Floor for actual fire rate used when adapting threshold.
    """

    def __init__(
        self,
        params,
        lr: float = 0.15,
        momentum: float = 0.9,
        init_threshold: float = 0.03,
        target_fire_rate: float = 0.08,
        decay: float = 0.95,
        clip: float = 1.5,
        bn_lr: Optional[float] = None,
        soft_fire: bool = True,
        soft_alpha: float = 0.5,
        min_fire_rate: float = 1e-3,
    ):
        if bn_lr is None:
            bn_lr = lr
        if not 0.0 < target_fire_rate <= 1.0:
            raise ValueError(f"target_fire_rate must be in (0, 1], got {target_fire_rate}")
        if init_threshold <= 0:
            raise ValueError(f"init_threshold must be > 0, got {init_threshold}")
        defaults = dict(
            lr=lr,
            momentum=momentum,
            init_threshold=init_threshold,
            target_fire_rate=target_fire_rate,
            decay=decay,
            clip=clip,
            bn_lr=bn_lr,
            soft_fire=soft_fire,
            soft_alpha=soft_alpha,
            min_fire_rate=min_fire_rate,
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
            target = group["target_fire_rate"]
            decay = group["decay"]
            clip = group["clip"]
            bn_lr = group["bn_lr"]
            soft_fire = group["soft_fire"]
            soft_alpha = group["soft_alpha"]
            min_fr = group["min_fire_rate"]
            init_thr = group["init_threshold"]

            for p in group["params"]:
                if p.grad is None:
                    continue

                if p.data.dim() < 2:
                    p.data.add_(p.grad.data, alpha=-bn_lr)
                    continue

                state = self.state[p]
                if "ema_grad" not in state:
                    state["ema_grad"] = torch.zeros_like(p.data)
                    state["accumulator"] = torch.zeros_like(p.data)
                    state["threshold"] = float(init_thr)

                threshold = float(state["threshold"])
                ema = state["ema_grad"]
                ema.mul_(momentum).add_(p.grad.data, alpha=1.0 - momentum)

                acc = state["accumulator"]
                acc.mul_(decay).add_(ema, alpha=-lr)

                fire_mask = acc.abs() > threshold
                n_fired = int(fire_mask.sum().item())
                n_params = p.numel()
                if n_fired > 0:
                    target_sign = torch.sign(acc[fire_mask])
                    target_sign = torch.where(
                        target_sign == 0,
                        torch.ones_like(target_sign),
                        target_sign,
                    )
                    if soft_fire:
                        a = soft_alpha
                        p.data[fire_mask] = (1 - a) * p.data[fire_mask] + a * target_sign
                    else:
                        p.data[fire_mask] = target_sign
                    acc[fire_mask] = 0.0

                # Per-tensor threshold adaptation
                actual_rate = max(n_fired / max(n_params, 1), min_fr * 0.1)
                if actual_rate < target * 0.5:
                    state["threshold"] = max(1e-4, threshold * 0.9)
                elif actual_rate > target * 2.0:
                    state["threshold"] = min(10.0, threshold * 1.1)

                p.data.clamp_(-clip, clip)

        return loss
