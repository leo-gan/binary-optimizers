"""
HybridAccumulatorOptimizer: EMA gradient tracking + adaptive threshold integrate-and-fire.

Combines the best ideas from:
  - MomentumVotingOptimizer (momentum-smoothed gradient signal)
  - ThresholdedIntegrateFireOptimizer (accumulate-then-fire to ±1)
  - AdaptiveIntegrateFireOptimizer (auto-tuning threshold)

Improvement rationale:
  - MomentumVoting applies updates every step, causing noisy oscillations
    in the binary regime. HybridAccumulator only fires when the
    accumulated evidence for a flip is strong enough.
  - ThresholdIF uses a fixed threshold. HybridAccumulator adapts the
    threshold per group to maintain a target fire rate.
  - On fire, weights are *set* to sign(accumulator) (not added), matching
    the proven ThresholdIF discrete update.
"""

from __future__ import annotations

from typing import Optional

import torch


class HybridAccumulatorOptimizer(torch.optim.Optimizer):
    """
    Maintains:
      1. An EMA of the gradient (directional signal).
      2. An accumulator that integrates the anti-gradient pressure.
      3. When |accumulator| exceeds threshold, the weight is set to
         sign(accumulator) and the accumulator resets.
      4. The threshold adapts per group to maintain a target fire rate.

    Parameters
    ----------
    params : iterable
        Model parameters.
    lr : float
        Scale for accumulation (integrate step). Default 0.1.
    momentum : float
        EMA coefficient for gradient smoothing. Default 0.9.
    init_threshold : float
        Initial fire threshold. Default 0.05.
    target_fire_rate : float
        Desired fraction of weights fired per step. Default 0.05.
    decay : float
        Accumulator decay each step (like ThresholdIF). Default 0.95.
    clip : float
        Weight clamp range. Default 1.5.
    bn_lr : float | None
        Learning rate for 1-D params. Defaults to ``lr``.
    """

    def __init__(
        self,
        params,
        lr: float = 0.1,
        momentum: float = 0.9,
        init_threshold: float = 0.05,
        target_fire_rate: float = 0.05,
        decay: float = 0.95,
        clip: float = 1.5,
        bn_lr: Optional[float] = None,
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
            threshold=init_threshold,
            target_fire_rate=target_fire_rate,
            decay=decay,
            clip=clip,
            bn_lr=bn_lr,
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
            threshold = group["threshold"]
            target = group["target_fire_rate"]
            decay = group["decay"]
            clip = group["clip"]
            bn_lr = group["bn_lr"]

            group_fires = 0
            group_params = 0

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

                # Step 1: EMA of gradient
                ema = state["ema_grad"]
                ema.mul_(momentum).add_(p.grad.data, alpha=1.0 - momentum)

                # Step 2: Integrate anti-gradient pressure with decay (ThresholdIF style)
                acc = state["accumulator"]
                acc.mul_(decay).add_(ema, alpha=-lr)

                # Step 3: Fire → set weight to consensus sign, reset accumulator
                fire_mask = acc.abs() > threshold
                n_fired = int(fire_mask.sum().item())
                if n_fired > 0:
                    p.data[fire_mask] = torch.sign(acc[fire_mask])
                    # Avoid zeros from sign(0)
                    p.data[fire_mask] = torch.where(
                        p.data[fire_mask] == 0,
                        torch.ones_like(p.data[fire_mask]),
                        p.data[fire_mask],
                    )
                    acc[fire_mask] = 0.0
                    group_fires += n_fired

                group_params += p.numel()
                p.data.clamp_(-clip, clip)

            # Step 4: Adapt threshold to maintain target fire rate
            if group_params > 0:
                actual_rate = group_fires / group_params
                if actual_rate < target * 0.5:
                    group["threshold"] = max(1e-4, group["threshold"] * 0.9)
                elif actual_rate > target * 2.0:
                    group["threshold"] = min(10.0, group["threshold"] * 1.1)

        return loss
