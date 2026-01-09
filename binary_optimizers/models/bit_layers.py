import torch
import torch.nn as nn
import torch.nn.functional as F

from binary_optimizers.binarization import ste_binarize_weight


class BitConv2d(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size=3,
        stride=1,
        padding=1,
        bias=False,
        scale: bool = True,
    ):
        super().__init__()
        self.scale = scale
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            bias=bias,
        )
        nn.init.kaiming_normal_(self.conv.weight, nonlinearity="relu")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        w = self.conv.weight
        w_b = ste_binarize_weight(w, scale=self.scale, dim=(1, 2, 3))
        return F.conv2d(x, w_b, bias=self.conv.bias, stride=self.conv.stride, padding=self.conv.padding)


class BitLinear(nn.Module):
    def __init__(self, in_features: int, out_features: int, bias=False, scale: bool = True):
        super().__init__()
        self.scale = scale
        self.fc = nn.Linear(in_features, out_features, bias=bias)
        nn.init.kaiming_normal_(self.fc.weight, nonlinearity="linear")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        w = self.fc.weight
        w_b = ste_binarize_weight(w, scale=self.scale, dim=(1,))
        return F.linear(x, w_b, bias=self.fc.bias)


class BitLinearSTE(nn.Module):
    def __init__(self, in_features: int, out_features: int, bias: bool = False):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(out_features, in_features) * 0.01)
        self.bias = nn.Parameter(torch.zeros(out_features)) if bias else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        w_q = torch.sign(self.weight)
        weight_proxy = self.weight + (w_q - self.weight).detach()
        return F.linear(x, weight_proxy, self.bias)
