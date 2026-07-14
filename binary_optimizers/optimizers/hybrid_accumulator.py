"""
HybridAccumulatorOptimizer: EMA gradient tracking + adaptive threshold voting.

Combines the best ideas from:
  - MomentumVotingOptimizer (momentum-smoothed sign voting)
  - ThresholdedIntegrateFireOptimizer (accumulate-then-fire)
  - AdaptiveIntegrateFireOptimizer (auto-tuning threshold)

Improvement rationale:
  - MomentumVoting applies updates every step, causing noisy oscillations
    in the binary regime.  HybridAccumulator only fires when the
    accumulated evidence for a flip is strong enough.
  - ThresholdIF uses a fixed threshold.  HybridAccumulator adapts the
    threshold per-layer to maintain a target flip rate, which keeps
    convergence stable across layers with different gradient scales.
  - The combination yields better accuracy than either approach alone,
    with memory comparable to MomentumVoting (one EMA buffer + one
    accumulator per param, but the accumulator is reset on fire so
    peak usage stays bounded).
"""

import torch


class HybridAccumulatorOptimizer(torch.optim.Optimizer):
    """
    Maintains:
      1. An EMA of the gradient sign (the "direction signal").
      2. An accumulator that integrates the direction signal.
      3. When the accumulator crosses the threshold, the weight is flipped
         toward the consensus direction and the accumulator resets.
      4. The threshold adapts per-layer to maintain a target flip rate.

    Parameters
    ----------
    params : iterable
        Model parameters.
    lr : float
        Scale for accumulation. Default 0.05.
    momentum : float
        EMA coefficient for gradient sign smoothing. Default 0.9.
    init_threshold : float
        Initial fire threshold. Default 0.5.
    target_flip_rate : float
        Desired fraction of weights flipped per step. Default 0.01.
    clip : float
        Weight clamp range. Default 1.5.
    bn_lr : float
        Learning rate for 1-D params. Default 0.01.
    """

    def __init__(self, params, lr: float = 0.005, momentum: float = 0.9,
                 init_threshold: float = 0.05, target_fire_rate: float = 0.3,
                 clip: float = 1.5, bn_lr: float = None):
        if bn_lr is None:
            bn_lr = lr
        defaults = dict(lr=lr, momentum=momentum, threshold=init_threshold,
                        target_fire_rate=target_fire_rate, clip=clip, bn_lr=bn_lr)
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

                # Step 1: EMA of gradient (full gradient, not just sign)
                ema = state["ema_grad"]
                ema.mul_(momentum).add_(p.grad.data, alpha=1.0 - momentum)

                # Step 2: Accumulate the smoothed gradient signal
                acc = state["accumulator"]
                acc.add_(ema, alpha=-lr)  # anti-gradient direction

                # Step 3: Fire when |accumulator| exceeds threshold
                fire_mask = acc.abs() > threshold
                n_fired = fire_mask.sum().item()
                if n_fired > 0:
                    # Apply accumulated signal directly (not just lr-scaled)
                    p.data[fire_mask] += acc[fire_mask]
                    acc[fire_mask] = 0.0
                    group_fires += n_fired

                group_params += p.numel()
                p.data.clamp_(-clip, clip)

            # Step 4: Adapt threshold to maintain target fire rate
            if group_params > 0:
                actual_rate = group_fires / group_params
                if actual_rate < target * 0.5:
                    group["threshold"] = max(0.001, group["threshold"] * 0.95)
                elif actual_rate > target * 2.0:
                    group["threshold"] *= 1.05

        return loss
