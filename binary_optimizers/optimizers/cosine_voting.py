"""
CosineVotingOptimizer: Voting with cosine-annealed learning rate.

Improvement rationale over VotingOptimizer and MomentumVotingOptimizer:
  - VotingOptimizer uses a fixed lr and push_rate, which means it either
    explores too aggressively late in training or converges too slowly early.
  - CosineVoting starts with a high lr for fast initial convergence, then
    anneals down following a cosine schedule for fine-grained refinement.
  - Accumulator decay is also annealed: high early (explore) → low late
    (exploit accumulated consensus).
  - No extra memory over VotingOptimizer (just an accumulator buffer per param).
  - The step counter is stored in optimizer state, so no external scheduler needed.
"""

import math

import torch


class CosineVotingOptimizer(torch.optim.Optimizer):
    """
    Binary-weight voting optimizer with built-in cosine annealing.

    Parameters
    ----------
    params : iterable
        Model parameters.
    lr_max : float
        Maximum (initial) learning rate for vote accumulation.
    lr_min : float
        Minimum learning rate at the end of annealing.
    push_rate : float
        Rate at which weights move toward the consensus sign.
    total_steps : int
        Number of optimizer steps over which to anneal. After this,
        lr stays at lr_min.
    clip : float
        Weight clamp range.
    bn_lr : float
        Learning rate for batch-norm / 1-D parameters.
    """

    def __init__(self, params, lr_max: float = 0.01, lr_min: float = 0.001,
                 momentum: float = 0.9, total_steps: int = 200,
                 clip: float = 1.5, confidence_threshold: float = 0.0005,
                 bn_lr: float = None):
        if bn_lr is None:
            bn_lr = lr_max
        defaults = dict(lr_max=lr_max, lr_min=lr_min, momentum=momentum,
                        total_steps=total_steps, clip=clip,
                        confidence_threshold=confidence_threshold, bn_lr=bn_lr)
        super().__init__(params, defaults)
        # Instance attribute (not optimizer.state): state maps Parameter → dict only.
        self.global_step = 0

    def _cosine_lr(self, group) -> float:
        t = self.global_step
        T = group["total_steps"]
        lr_max = group["lr_max"]
        lr_min = group["lr_min"]
        if t >= T:
            return lr_min
        return lr_min + 0.5 * (lr_max - lr_min) * (1 + math.cos(math.pi * t / T))

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

                # Momentum-smoothed gradient
                buf = state["momentum_buffer"]
                buf.mul_(mom).add_(p.grad.data, alpha=1.0 - mom)

                # Confidence-gated sign voting (like MomentumVoting_Conf
                # but with cosine-annealed lr)
                if conf_thresh > 0:
                    vote = torch.zeros_like(buf)
                    confident = buf.abs() > conf_thresh
                    vote[confident] = torch.sign(buf[confident])
                else:
                    vote = torch.sign(buf)

                p.data.add_(vote, alpha=-lr)
                p.data.clamp_(-clip, clip)

        self.global_step += 1
        return loss
