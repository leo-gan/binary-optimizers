import torch
from torch.optim import Optimizer


class IntegerVotingOptimizer(Optimizer):
    """
    Optimizer that uses integer-valued accumulators to trigger bit flips.
    No floating point gradients are applied directly to parameters.
    """

    def __init__(self, params, threshold: int = 10, lr: float = 1.0):
        if threshold < 1:
            raise ValueError("Threshold must be at least 1")
        defaults = dict(threshold=threshold, lr=lr)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            threshold = group["threshold"]

            for p in group["params"]:
                if p.grad is None:
                    continue

                state = self.state[p]

                # Initialize integer accumulator
                if "accumulator" not in state:
                    state["accumulator"] = torch.zeros_like(p.data, dtype=torch.int32)

                acc = state["accumulator"]

                # Vote based on sign of gradient
                # batch_consensus is -1, 0, or 1
                batch_consensus = -torch.sign(p.grad).to(torch.int32)

                # Accumulate integer votes
                acc.add_(batch_consensus)

                # Check for "fire" condition (threshold reached)
                # If accumulator exceeds threshold, flip the weight and reset accumulator
                fire_positive = acc >= threshold
                fire_negative = acc <= -threshold

                # Flip weights that reached consensus
                p.data[fire_positive] = 1.0
                p.data[fire_negative] = -1.0

                # Refractory reset (reset voted bits)
                acc[fire_positive] = 0
                acc[fire_negative] = 0

        return loss
