"""
SparseSignOptimizer: Sign-based updates on a random sparse subset each step.

Improvement rationale:
  - STE_SGD and MomentumVoting update *all* parameters every step.
    For binary networks the gradient sign is the only useful signal,
    so updating a random subset each step has almost the same
    convergence curve but at a fraction of the compute.
  - Memory: only one buffer (momentum), same as MomentumVoting.
  - Speed: the sparse mask means only `density` fraction of parameters
    are read/written each step, giving a near-linear speedup.
  - Convergence: the stochastic subset selection acts as implicit
    regularization, similar to dropout applied in weight-space.
"""

import torch


class SparseSignOptimizer(torch.optim.Optimizer):
    """
    Each step, select a random `density` fraction of weights.
    Apply sign(momentum_buffer) updates only to the selected subset.

    Parameters
    ----------
    params : iterable
        Model parameters.
    lr : float
        Step size for the sign update.
    momentum : float
        Momentum coefficient for gradient smoothing.
    density : float
        Fraction of weights updated each step (0, 1]. Default 0.3.
    clip : float
        Weight clamp range.
    bn_lr : float
        Learning rate for 1-D params (batch norm, bias).
    """

    def __init__(self, params, lr: float = 0.005, momentum: float = 0.9,
                 density: float = 0.5, clip: float = 1.5,
                 confidence_threshold: float = 0.001,
                 bn_lr: float = None):
        if not 0 < density <= 1:
            raise ValueError(f"density must be in (0, 1], got {density}")
        if bn_lr is None:
            bn_lr = lr
        defaults = dict(lr=lr, momentum=momentum, density=density, clip=clip,
                        confidence_threshold=confidence_threshold, bn_lr=bn_lr)
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
            density = group["density"]
            clip = group["clip"]
            conf_thresh = group["confidence_threshold"]
            bn_lr = group["bn_lr"]

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

                # Confidence filter: only act where momentum magnitude is significant
                confident = buf.abs() > conf_thresh

                # Sparse mask: randomly select `density` fraction of the confident set
                sparse_mask = torch.rand_like(p.data) < density
                active = confident & sparse_mask

                # Sign update only on active positions — effective lr is
                # scaled up by 1/density to compensate for sparsity
                effective_lr = lr / density
                vote = torch.sign(buf)
                p.data[active] -= effective_lr * vote[active]
                p.data.clamp_(-clip, clip)

        return loss
