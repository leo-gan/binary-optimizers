import torch
import torch.nn as nn
import torch.nn.functional as F

from binary_optimizers.binarization import ste_binarize
from binary_optimizers.models.bit_layers import BitConv2d, BitLinear


class SmallConvNetSTE(nn.Module):
    def __init__(self, binary: bool = True):
        super().__init__()
        self.binary = binary
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(64)
        self.fc1 = nn.Linear(64 * 8 * 8, 256, bias=False)
        self.bn3 = nn.BatchNorm1d(256)
        self.fc2 = nn.Linear(256, 10, bias=False)

        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.binary:
            w1 = ste_binarize(self.conv1.weight)
            w2 = ste_binarize(self.conv2.weight)
            w_fc1 = ste_binarize(self.fc1.weight)
            w_fc2 = ste_binarize(self.fc2.weight)
        else:
            w1 = self.conv1.weight
            w2 = self.conv2.weight
            w_fc1 = self.fc1.weight
            w_fc2 = self.fc2.weight

        x = F.conv2d(x, w1, bias=None, stride=self.conv1.stride, padding=self.conv1.padding)
        x = self.bn1(x)
        x = F.relu(x, inplace=False)
        x = F.max_pool2d(x, 2)

        x = F.conv2d(x, w2, bias=None, stride=self.conv2.stride, padding=self.conv2.padding)
        x = self.bn2(x)
        x = F.relu(x, inplace=False)
        x = F.max_pool2d(x, 2)

        x = x.view(x.size(0), -1)

        x = F.linear(x, w_fc1, bias=None)
        x = self.bn3(x)
        x = F.relu(x, inplace=False)

        x = F.linear(x, w_fc2, bias=None)
        return x


class SmallBitConvNet(nn.Module):
    def __init__(self, binary: bool = True, scale: bool = True):
        super().__init__()
        self.binary = binary
        self.conv1 = BitConv2d(3, 64, kernel_size=3, padding=1, bias=False, scale=scale)
        self.bn1 = nn.BatchNorm2d(64)
        self.conv2 = BitConv2d(64, 128, kernel_size=3, padding=1, bias=False, scale=scale)
        self.bn2 = nn.BatchNorm2d(128)
        self.fc1 = BitLinear(128 * 8 * 8, 256, bias=False, scale=scale)
        self.bn3 = nn.BatchNorm1d(256)
        self.fc2 = BitLinear(256, 10, bias=False, scale=scale)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.binary:
            x = self.conv1(x)
        else:
            x = self.conv1.conv(x)
        x = self.bn1(x)
        x = F.relu(x, inplace=False)
        x = F.max_pool2d(x, 2)

        if self.binary:
            x = self.conv2(x)
        else:
            x = self.conv2.conv(x)
        x = self.bn2(x)
        x = F.relu(x, inplace=False)
        x = F.max_pool2d(x, 2)

        x = x.view(x.size(0), -1)

        if self.binary:
            x = self.fc1(x)
        else:
            x = self.fc1.fc(x)
        x = self.bn3(x)
        x = F.relu(x, inplace=False)

        if self.binary:
            x = self.fc2(x)
        else:
            x = self.fc2.fc(x)
        return x
