import torch


class ThresholdedIntegrateFireOptimizer(torch.optim.Optimizer):
    """
    Thresholded Integrate-Fire Optimizer.
    Accumulates gradients (Integrate) and flips weights (Fire) when a threshold is reached.
    """

    def __init__(self, params, lr=0.1, threshold=0.1, decay=0.9, bn_lr=None):
        defaults = dict(lr=lr, threshold=threshold, decay=decay, bn_lr=bn_lr)
        super().__init__(params, defaults)

    def step(self, closure=None):
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            lr = group["lr"]
            threshold = group["threshold"]
            decay = group["decay"]
            bn_lr = group.get("bn_lr", None)
            if bn_lr is None:
                bn_lr = lr

            for p in group["params"]:
                if p.grad is None:
                    continue

                if p.data.dim() < 2:
                    p.data.add_(p.grad.data, alpha=-bn_lr)
                    continue

                # For binary weights (assumed to be ±1 or clamped around ±1)
                grad = p.grad.data
                state = self.state[p]

                if "accumulator" not in state:
                    state["accumulator"] = torch.zeros_like(p.data)

                acc = state["accumulator"]

                # Integration: Accumulate gradient signal (with decay)
                # Pressure is directed AWAY from the current sign to encourage flips
                # or TOWARDS it to stabilize. Here we use the gradient directly.
                acc.mul_(decay).add_(grad, alpha=-lr)

                # Fire: If accumulation exceeds threshold, flip and reset
                fire_mask = torch.abs(acc) > threshold
                if fire_mask.any():
                    # Flip the sign of the data to match the direction of accumulated pressure
                    p.data[fire_mask] = torch.sign(acc[fire_mask])
                    # Reset the accumulator for fired neurons
                    acc[fire_mask] = 0.0

                # Optional: clamp data to ensure it stays close to ±1 if not purely binary
                p.data.clamp_(-1.0, 1.0)

        return loss
