import torch


class IntegrateFireOptimizer(torch.optim.Optimizer):
    def __init__(self, params, threshold=10.0, decay=0.9):
        defaults = dict(threshold=threshold, decay=decay)
        super().__init__(params, defaults)

    def step(self, closure=None):
        loss = None
        if closure is not None:
            loss = closure()

        total_flips = 0
        total_params = 0

        for group in self.param_groups:
            threshold = group["threshold"]
            decay = group["decay"]

            for p in group["params"]:
                if p.grad is None:
                    continue

                if p.dim() == 3:
                    state = self.state[p]
                    if "accumulator" not in state:
                        state["accumulator"] = torch.zeros_like(p.data)

                    pressure = p.grad * p.data * 100.0

                    acc = state["accumulator"]
                    acc.mul_(decay).add_(pressure)

                    fire_mask = acc > threshold
                    if fire_mask.any():
                        p.data[fire_mask] *= -1
                        acc[fire_mask] = 0.0
                        total_flips += fire_mask.sum().item()

                    total_params += p.numel()
                else:
                    p.data.add_(p.grad.clamp(-1, 1), alpha=-0.01)

        if total_params == 0:
            return 0.0
        return total_flips / total_params


class AdaptiveIntegrateFireOptimizer(torch.optim.Optimizer):
    def __init__(self, params, init_threshold=1.0, decay=0.9, target_rate=0.005):
        defaults = dict(threshold=init_threshold, decay=decay, target_rate=target_rate)
        super().__init__(params, defaults)

    def step(self, closure=None):
        loss = None
        if closure is not None:
            loss = closure()

        total_flips = 0
        total_params = 0

        for group in self.param_groups:
            threshold = group["threshold"]
            decay = group["decay"]
            target = group["target_rate"]

            group_flips = 0
            group_total_params = 0

            for p in group["params"]:
                if p.grad is None:
                    continue

                if p.dim() == 3:
                    state = self.state[p]
                    if "accumulator" not in state:
                        state["accumulator"] = torch.zeros_like(p.data)

                    pressure = (p.grad * p.data) * 1000.0

                    acc = state["accumulator"]
                    acc.mul_(decay).add_(pressure)

                    fire_mask = acc > threshold
                    if fire_mask.any():
                        p.data[fire_mask] *= -1
                        acc[fire_mask] = 0.0

                        count = fire_mask.sum().item()
                        group_flips += count
                        total_flips += count

                    group_total_params += p.numel()
                    total_params += p.numel()
                else:
                    p.data.add_(p.grad.clamp(-1, 1), alpha=-0.01)

            if group_total_params > 0:
                current_rate = group_flips / group_total_params
                if current_rate < target / 2:
                    group["threshold"] *= 0.95
                    group["threshold"] = max(0.001, group["threshold"])
                elif current_rate > target * 2:
                    group["threshold"] *= 1.05

        if total_params == 0:
            return 0.0
        return total_flips / total_params
