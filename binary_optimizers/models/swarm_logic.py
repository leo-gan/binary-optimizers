import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class LogSwarmLinear(nn.Module):
    """
    Weight is represented as a population of 32 binary agents.
    Effective weight is the sum of the population.
    """

    def __init__(self, in_features: int, out_features: int, swarm_size: int = 32):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.swarm_size = swarm_size

        # Population of bits {-1, 1}
        # Initializing half-half or random
        pop = (
            torch.randint(0, 2, (out_features, in_features, swarm_size)).float() * 2 - 1
        )
        self.population = nn.Parameter(pop, requires_grad=True)

        # Scaling factor to keep activations in range
        # In pure hardware, this would be a bit-shift or fixed-point gain
        self.scale = 1.0 / math.sqrt(in_features)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Sum agents: range [-32, 32]
        weight_eff = self.population.sum(dim=2)

        # Scale for stability
        weight_scaled = weight_eff * self.scale

        return F.linear(x, weight_scaled)


class LogSwarmConv2d(nn.Module):
    """
    Weight is represented as a population of 32 binary agents per kernel element.
    Effective weight is the sum of the population.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        padding: int = 0,
        swarm_size: int = 32,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.swarm_size = swarm_size

        # Population of bits {-1, 1}
        # Shape: [out_channels, in_channels, kernel_h, kernel_w, swarm_size]
        pop = (
            torch.randint(
                0, 2, (out_channels, in_channels, kernel_size, kernel_size, swarm_size)
            ).float()
            * 2
            - 1
        )
        self.population = nn.Parameter(pop, requires_grad=True)

        # Scaling factor
        fan_in = in_channels * kernel_size * kernel_size
        self.scale = 1.0 / math.sqrt(fan_in)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Sum agents: range [-32, 32]
        # Sum over the last dimension (swarm_size)
        weight_eff = self.population.sum(dim=-1)

        # Scale for stability
        weight_scaled = weight_eff * self.scale

        return F.conv2d(x, weight_scaled, stride=self.stride, padding=self.padding)
