"""Wrap a binary-weight optimizer + Adam on continuous (BN/bias) params."""

from __future__ import annotations

from typing import Optional

import torch
from torch.optim import Optimizer


class DualOptimizer:
    """
    Mimics the Optimizer zero_grad/step interface used by training loops.

    ``binary_opt`` updates 2D weights; ``bn_opt`` (typically Adam) updates 1D.
    """

    def __init__(self, binary_opt: Optimizer, bn_opt: Optional[Optimizer] = None):
        self.binary_opt = binary_opt
        self.bn_opt = bn_opt
        # Expose state for memory profilers that inspect .state
        self.state = binary_opt.state
        self.param_groups = list(binary_opt.param_groups)
        if bn_opt is not None:
            self.param_groups = list(binary_opt.param_groups) + list(bn_opt.param_groups)

    def zero_grad(self, set_to_none: bool = False):
        self.binary_opt.zero_grad(set_to_none=set_to_none)
        if self.bn_opt is not None:
            self.bn_opt.zero_grad(set_to_none=set_to_none)

    def step(self, closure=None):
        loss = self.binary_opt.step(closure)
        if self.bn_opt is not None:
            self.bn_opt.step()
        return loss
