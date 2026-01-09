import torch


class SwarmOptimizer(torch.optim.Optimizer):
    def __init__(self, params, recruit_rate=10.0, bn_lr=0.01):
        defaults = dict(recruit_rate=recruit_rate, bn_lr=bn_lr)
        super().__init__(params, defaults)

    def step(self, closure=None):
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            recruit_rate = group["recruit_rate"]
            bn_lr = group["bn_lr"]

            for p in group["params"]:
                if p.grad is None:
                    continue

                if p.dim() == 3:
                    grad_pressure = p.grad.mean(dim=2)
                    probs = (torch.abs(grad_pressure) * recruit_rate).clamp(0, 0.5)
                    probs = probs.unsqueeze(-1).expand_as(p.data)

                    target = -torch.sign(grad_pressure).unsqueeze(-1).expand_as(p.data)
                    target[target == 0] = 1

                    random_roll = torch.rand_like(p.data)
                    should_flip = (p.data != target) & (random_roll < probs)
                    p.data[should_flip] *= -1
                else:
                    p.data.add_(p.grad.data, alpha=-bn_lr)

        return loss
