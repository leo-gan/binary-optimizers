"""
EMAFlipOptimizer: Exponential Moving Average + threshold-gated bit flipping.

Improvement rationale over existing optimizers:
  - MomentumRankOptimizer uses expensive torch.topk every step.
    EMAFlip replaces topk with a simple threshold comparison, making it
    O(n) instead of O(n log k).
  - The EMA provides temporal smoothing like MomentumRank, reducing
    noisy gradient-driven oscillations.
  - Adaptive threshold tracks the running mean of wrongness so the flip
    rate self-regulates without hyperparameter tuning.
  - Lower memory than Adam (one buffer vs two), competitive with
    momentum-based optimizers.
"""

import torch


class EMAFlipOptimizer(torch.optim.Optimizer):
    """
    Maintains an EMA of per-weight "wrongness" (grad * weight).
    Flips weights whose smoothed wrongness exceeds an adaptive threshold.

    Parameters
    ----------
    params : iterable
        Model parameters.
    momentum : float
        EMA decay factor (higher = more smoothing). Default 0.95.
    threshold_scale : float
        Multiplier on the running mean wrongness to set the flip threshold.
        Lower values = more aggressive flipping. Default 1.5.
    bn_lr : float
        Learning rate for batch-norm / 1-D parameters. Default 0.01.
    """

    def __init__(self, params, lr: float = 0.005, momentum: float = 0.9,
                 threshold_scale: float = 1.0, clip: float = 1.5,
                 bn_lr: float = None):
        if bn_lr is None:
            bn_lr = lr
        defaults = dict(lr=lr, momentum=momentum, threshold_scale=threshold_scale,
                        clip=clip, bn_lr=bn_lr)
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

            for p in group["params"]:
                if p.grad is None:
                    continue

                # 1-D params (batch-norm, bias): plain SGD
                if p.data.dim() < 2:
                    p.data.add_(p.grad.data, alpha=-bn_lr)
                    continue

                grad = p.grad.data
                state = self.state[p]
                if "grad_ema" not in state:
                    state["grad_ema"] = torch.zeros_like(p.data)
                    state["running_mean"] = torch.tensor(0.0, device=p.device)

                # EMA of gradient (directional signal)
                ema = state["grad_ema"]
                ema.mul_(momentum).add_(grad, alpha=1.0 - momentum)

                # Adaptive threshold: only update weights with strong signal
                ema_abs = ema.abs()
                mean_abs = ema_abs.mean()
                rm = state["running_mean"]
                rm.mul_(0.95).add_(mean_abs, alpha=0.05)
                threshold = rm * threshold_scale

                # Sign-based update, gated by confidence
                confident = ema_abs > threshold
                vote = torch.sign(ema)
                # Apply sign update only where signal is confident
                update = torch.where(confident, vote, torch.zeros_like(vote))
                p.data.add_(update, alpha=-lr)

                p.data.clamp_(-clip, clip)

        return loss
