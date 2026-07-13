import torch
import torch.nn as nn
import torch.nn.functional as F


class IntegerLogicLinear(nn.Module):
    """
    Linear layer that uses binary weights {-1, 1}.
    Forward pass simulates integer accumulation logic.
    """

    def __init__(self, in_features: int, out_features: int, bias: bool = False):
        super().__init__()
        # Initializing with random signs (-1 or 1)
        self.weight = nn.Parameter(
            torch.randint(0, 2, (out_features, in_features)).float() * 2 - 1
        )
        self.bias = nn.Parameter(torch.zeros(out_features)) if bias else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Weights are already binary in this prototype.
        # Gradients will be handled by the IntegerVotingOptimizer.
        return F.linear(x, self.weight, self.bias)


class LogicHomeostasis(nn.Module):
    """
    Counter-based threshold adjustment layer.
    Replaces BatchNorm with logic-friendly frequency control.
    """

    def __init__(
        self, num_features: int, threshold_init: float = 0.0, target_rate: float = 0.5
    ):
        super().__init__()
        self.threshold = nn.Parameter(torch.full((num_features,), threshold_init))
        self.target_rate = target_rate
        # Integer-like accumulation of activity samples
        self.register_buffer("activity_counter", torch.zeros(num_features))
        self.register_buffer("sample_count", torch.tensor(0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim == 4:
            # Conv2d path: [N, C, H, W]
            threshold_view = self.threshold.view(1, -1, 1, 1)
            reduce_dims = (0, 2, 3)
        else:
            # Linear path: [N, C]
            threshold_view = self.threshold.view(1, -1)
            reduce_dims = (0,)

        out = x - threshold_view

        if self.training:
            with torch.no_grad():
                # Count activations (firing rate)
                fired = (out > 0).float()
                batch_activity = fired.mean(dim=reduce_dims)

                # Update running statistics (could be done with pure integer counters)
                self.activity_counter.mul_(0.9).add_(batch_activity, alpha=0.1)

                # Nudge threshold to maintain target firing rate
                # If too active, increase threshold. If too quiet, decrease.
                error = self.activity_counter - self.target_rate
                self.threshold.data.add_(error, alpha=0.1)

        return out
