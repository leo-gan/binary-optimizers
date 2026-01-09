import torch


class BitLogicOptimizer(torch.optim.Optimizer):
    def __init__(self, params, sensitivity=1.0, mutation_rate=0.0, max_flip_prob=0.2):
        defaults = dict(sensitivity=sensitivity, mutation_rate=mutation_rate, max_flip_prob=max_flip_prob)
        super().__init__(params, defaults)

    def step(self, closure=None):
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            sensitivity = group["sensitivity"]
            mutation_rate = group["mutation_rate"]
            max_flip_prob = group["max_flip_prob"]

            for p in group["params"]:
                if p.grad is None:
                    continue

                if p.dim() == 3:
                    grad_pressure = p.grad.mean(dim=2)
                    flip_probability = (torch.abs(grad_pressure) * sensitivity).unsqueeze(-1)
                    flip_probability = flip_probability.clamp(0.0, max_flip_prob)

                    noise_grad = torch.rand_like(p.data)
                    should_flip_grad = noise_grad < flip_probability

                    target = -torch.sign(grad_pressure).unsqueeze(-1)
                    target[target == 0] = 1

                    mask = (p.data != target) & should_flip_grad

                    if mutation_rate > 0.0:
                        noise_mut = torch.rand_like(p.data)
                        mask = mask | (noise_mut < mutation_rate)

                    p.data[mask] *= -1
                else:
                    p.data.add_(torch.sign(p.grad), alpha=-0.01)

        return loss


class RankBasedBitOptimizer(torch.optim.Optimizer):
    def __init__(self, params, flip_rate=0.01):
        defaults = dict(flip_rate=flip_rate)
        super().__init__(params, defaults)

    def step(self, closure=None):
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            flip_rate = group["flip_rate"]

            for p in group["params"]:
                if p.grad is None:
                    continue

                if p.dim() == 3:
                    wrongness_score = p.grad * p.data
                    scores_flat = wrongness_score.view(-1)
                    k = int(scores_flat.numel() * flip_rate)
                    if k <= 0:
                        continue

                    _, top_indices = torch.topk(scores_flat, k)
                    mask_flat = torch.zeros_like(scores_flat, dtype=torch.bool)
                    mask_flat[top_indices] = True
                    mask = mask_flat.view_as(p.data)
                    p.data[mask] *= -1
                else:
                    p.data.add_(torch.sign(p.grad), alpha=-0.01)

        return loss


class AdaptiveBitOptimizer(torch.optim.Optimizer):
    def __init__(self, params, base_flip_rate=0.02):
        defaults = dict(base_flip_rate=base_flip_rate)
        super().__init__(params, defaults)

    def step(self, closure=None):
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            base_rate = group["base_flip_rate"]

            for p in group["params"]:
                if p.grad is None:
                    continue

                if p.dim() == 3:
                    grad_pressure = p.grad.mean(dim=2)
                    layer_scale = torch.mean(torch.abs(grad_pressure)) + 1e-8
                    norm_pressure = grad_pressure / layer_scale

                    flip_prob = (torch.abs(norm_pressure) * base_rate).clamp(0.0, 1.0)
                    flip_prob = flip_prob.unsqueeze(-1).expand_as(p.data)

                    noise = torch.rand_like(p.data)
                    should_flip = noise < flip_prob

                    target = -torch.sign(grad_pressure).unsqueeze(-1)
                    target[target == 0] = 1

                    mask = (p.data != target) & should_flip
                    p.data[mask] *= -1
                else:
                    p.data.add_(torch.sign(p.grad), alpha=-0.01)

        return loss


class MomentumRankOptimizer(torch.optim.Optimizer):
    def __init__(self, params, base_flip_rate=0.02, momentum=0.9):
        defaults = dict(flip_rate=base_flip_rate, momentum=momentum)
        super().__init__(params, defaults)

    def step(self, closure=None):
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            flip_rate = group["flip_rate"]
            momentum = group["momentum"]

            for p in group["params"]:
                if p.grad is None:
                    continue

                if p.dim() == 3:
                    state = self.state[p]
                    if "wrongness_buffer" not in state:
                        state["wrongness_buffer"] = torch.zeros_like(p.data)

                    current_wrongness = p.grad * p.data

                    buf = state["wrongness_buffer"]
                    buf.mul_(momentum).add_(current_wrongness, alpha=1.0 - momentum)

                    scores_flat = buf.view(-1)
                    k = int(scores_flat.numel() * flip_rate)
                    if k <= 0:
                        continue

                    _, top_indices = torch.topk(scores_flat, k)
                    mask_flat = torch.zeros_like(scores_flat, dtype=torch.bool)
                    mask_flat[top_indices] = True
                    mask = mask_flat.view_as(p.data)

                    p.data[mask] *= -1
                    buf[mask] = 0
                else:
                    p.data.add_(torch.sign(p.grad), alpha=-0.005)

        return loss
