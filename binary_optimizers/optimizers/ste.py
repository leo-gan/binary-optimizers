import torch


class STEOptimizer(torch.optim.SGD):
    def step(self, closure=None):
        super().step(closure)
        for group in self.param_groups:
            for p in group["params"]:
                if p is None or p.grad is None:
                    continue
                if p.data.dim() >= 2:
                    p.data.clamp_(-1.0, 1.0)
