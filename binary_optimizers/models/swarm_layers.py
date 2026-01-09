import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class BitSwarmLinear(nn.Module):
    def __init__(self, in_features: int, out_features: int, swarm_size: int = 32):
        super().__init__()
        self.swarm_size = swarm_size
        self.population = nn.Parameter(
            torch.randint(0, 2, (out_features, in_features, swarm_size)).float() * 2 - 1,
            requires_grad=True,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        swarm_sum = self.population.sum(dim=2)
        w_eff = torch.sign(swarm_sum)
        w_eff[w_eff == 0] = 1
        weight_proxy = swarm_sum + (w_eff - swarm_sum).detach()
        return F.linear(x, weight_proxy)


class BitSwarmLogicLinear(nn.Module):
    def __init__(self, in_features: int, out_features: int, swarm_size: int = 32, normalize: bool = True):
        super().__init__()
        self.swarm_size = swarm_size
        self.normalize = normalize
        self.population = nn.Parameter(
            torch.randint(0, 2, (out_features, in_features, swarm_size)).float() * 2 - 1,
            requires_grad=True,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        swarm_sum = self.population.sum(dim=2)

        if not self.normalize:
            return F.linear(x, swarm_sum)

        w_eff = torch.sign(swarm_sum)
        w_eff[w_eff == 0] = 1
        w_norm = swarm_sum / self.swarm_size
        weight_proxy = w_norm + (w_eff - w_norm).detach()
        return F.linear(x, weight_proxy)


class HomeostaticThreshold(nn.Module):
    def __init__(self, num_features: int):
        super().__init__()
        self.threshold = nn.Parameter(torch.zeros(num_features))
        self.register_buffer("running_activity", torch.ones(num_features) * 0.5)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = x - self.threshold.view(1, -1)

        if self.training:
            with torch.no_grad():
                current_activity = (out.detach() > 0).float().mean(dim=0)
                self.running_activity = 0.8 * self.running_activity + 0.2 * current_activity
                error = self.running_activity - 0.5
                update = (error * 0.1).clamp(-0.5, 0.5)
                self.threshold.data += update

        return out


class SimpleCentering(nn.Module):
    def __init__(self, num_features: int):
        super().__init__()
        self.bias = nn.Parameter(torch.zeros(num_features))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x - self.bias


class HomeostaticScaleShift(nn.Module):
    def __init__(self, num_features: int, fan_in: int, start_lr: float = 0.1):
        super().__init__()
        self.threshold = nn.Parameter(torch.zeros(num_features))
        init_gain = 1.0 / math.sqrt(fan_in)
        self.gain = nn.Parameter(torch.ones(num_features) * init_gain)
        self.lr = start_lr

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = x - self.threshold.view(1, -1)
        out = out * self.gain.view(1, -1)

        if self.training:
            with torch.no_grad():
                batch_mean = out.mean(dim=0)
                self.threshold.data += batch_mean * self.lr

        return out
