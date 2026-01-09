import torch


class MomentumVotingOptimizer(torch.optim.Optimizer):
    def __init__(self, params, lr=1e-3, momentum=0.9, clip=1.5, weight_decay=0.0, bn_lr=None):
        defaults = dict(lr=lr, momentum=momentum, clip=clip, weight_decay=weight_decay, bn_lr=bn_lr)
        super().__init__(params, defaults)

    def step(self, closure=None):
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            lr = group["lr"]
            momentum = group["momentum"]
            clip = group["clip"]
            weight_decay = group["weight_decay"]
            bn_lr = group.get("bn_lr", None)
            if bn_lr is None:
                bn_lr = lr

            for p in group["params"]:
                if p.grad is None:
                    continue

                if p.data.dim() < 2:
                    p.data.add_(p.grad.data, alpha=-bn_lr)
                    continue

                grad = p.grad.data
                if weight_decay != 0.0:
                    grad = grad.add(p.data, alpha=weight_decay)

                state = self.state[p]
                if "momentum_buffer" not in state:
                    state["momentum_buffer"] = torch.zeros_like(p.data)

                buf = state["momentum_buffer"]
                buf.mul_(momentum).add_(grad, alpha=1.0 - momentum)

                vote = torch.sign(buf)
                p.data.add_(vote, alpha=-lr)
                p.data.clamp_(-clip, clip)

        return loss
