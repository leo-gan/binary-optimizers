import torch


class VotingOptimizer(torch.optim.Optimizer):
    def __init__(self, params, lr=0.1, momentum=0.9, push_rate=0.3, clip=1.0, bn_lr=None):
        defaults = dict(lr=lr, momentum=momentum, push_rate=push_rate, clip=clip, bn_lr=bn_lr)
        super().__init__(params, defaults)

    def step(self, closure=None):
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            lr = group["lr"]
            push_rate = group["push_rate"]
            clip = group["clip"]
            bn_lr = group.get("bn_lr", None)
            if bn_lr is None:
                bn_lr = lr

            for p in group["params"]:
                if p is None or p.grad is None:
                    continue

                g = p.grad.data

                if p.data.dim() < 2:
                    p.data.add_(g, alpha=-bn_lr)
                    continue

                batch_consensus = -torch.sign(g)

                state = self.state[p]
                if "accumulator" not in state:
                    state["accumulator"] = torch.zeros_like(p.data)

                a = state["accumulator"]
                a.add_(lr * batch_consensus)
                a.clamp_(-1.0, 1.0)

                w_target = a.sign()
                p.data.add_(push_rate * (w_target - p.data))
                if p.data.dim() >= 2:
                    p.data.clamp_(-clip, clip)

        return loss
