import torch.nn as nn

from binary_optimizers.models.bit_layers import BitLinearSTE
from binary_optimizers.models.swarm_layers import BitSwarmLogicLinear


def create_mnist_bit_mlp(hidden_dim: int = 128) -> nn.Module:
    return nn.Sequential(
        nn.Flatten(),
        BitLinearSTE(28 * 28, hidden_dim),
        nn.BatchNorm1d(hidden_dim),
        nn.ReLU(),
        BitLinearSTE(hidden_dim, 10),
    )


def create_mnist_swarm_mlp(hidden_dim: int = 128, swarm_size: int = 32) -> nn.Module:
    return nn.Sequential(
        nn.Flatten(),
        BitSwarmLogicLinear(28 * 28, hidden_dim, swarm_size=swarm_size, normalize=True),
        nn.BatchNorm1d(hidden_dim),
        nn.ReLU(),
        BitSwarmLogicLinear(hidden_dim, 10, swarm_size=swarm_size, normalize=True),
        nn.BatchNorm1d(10),
    )
