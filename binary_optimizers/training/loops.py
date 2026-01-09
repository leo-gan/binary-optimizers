import random
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def train_one_epoch_classification(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    loader,
    device: str,
) -> float:
    model.train()
    total, correct = 0, 0
    for x, y in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        optimizer.zero_grad()
        out = model(x)
        loss = F.cross_entropy(out, y)
        loss.backward()
        optimizer.step()

        pred = out.argmax(dim=1)
        correct += pred.eq(y).sum().item()
        total += y.size(0)

    return correct / max(1, total)


@torch.no_grad()
def evaluate_accuracy(model: nn.Module, loader, device: str) -> float:
    model.eval()
    total, correct = 0, 0
    for x, y in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        out = model(x)
        pred = out.argmax(dim=1)
        correct += pred.eq(y).sum().item()
        total += y.size(0)
    return correct / max(1, total)
