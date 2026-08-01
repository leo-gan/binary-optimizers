"""Swarm optimizer for int8 agent populations (experiment v0_1)."""

from __future__ import annotations

from typing import Iterable, List, Optional, Sequence

import torch
import torch.nn as nn

from layers import Int8SwarmLinear


class SwarmOptimizerV01:
    """Flip int8 agents from gradient pressure; optional FP SGD for LN affine.

    Not a ``torch.optim.Optimizer`` subclass: works with buffer-backed agents
    and manual-STE grads attached to the per-forward float view.
    """

    def __init__(
        self,
        swarm_layers: Sequence[Int8SwarmLinear],
        *,
        recruit_rate: float = 1e4,
        max_flip_prob: float = 0.15,
        grad_momentum: float = 0.9,
        ln_params: Optional[Iterable[nn.Parameter]] = None,
        ln_lr: float = 1e-2,
    ):
        self.swarm_layers: List[Int8SwarmLinear] = list(swarm_layers)
        self.recruit_rate = float(recruit_rate)
        self.max_flip_prob = float(max_flip_prob)
        self.grad_momentum = float(grad_momentum)
        ln_list = list(ln_params) if ln_params is not None else []
        self._ln_params = ln_list
        self.ln_optimizer: torch.optim.Optimizer | None
        if ln_list:
            self.ln_optimizer = torch.optim.SGD(ln_list, lr=ln_lr)
        else:
            self.ln_optimizer = None
        # Per-layer EMA of grad pressure [out, in], keyed by id(layer).
        self._pressure_ema: dict[int, torch.Tensor] = {}
        self.last_flip_frac: float = 0.0
        self.last_grad_abs_mean: float = 0.0

    def zero_grad(self) -> None:
        for layer in self.swarm_layers:
            layer._pop_f = None
        if self.ln_optimizer is not None:
            self.ln_optimizer.zero_grad(set_to_none=True)

    @torch.no_grad()
    def step(self) -> float:
        """Apply swarm flips; return fraction of agents flipped this step."""
        flipped = 0
        total = 0
        grad_abs_sum = 0.0
        grad_abs_n = 0
        m = self.grad_momentum

        for layer in self.swarm_layers:
            g = layer.last_agent_grad
            if g is None:
                continue

            # Mean pressure over swarm dim → one scalar per logical weight.
            # (STE assigns equal grad to every agent of a weight.)
            grad_pressure = g.mean(dim=-1)
            grad_abs_sum += float(grad_pressure.abs().sum().item())
            grad_abs_n += grad_pressure.numel()

            # EMA of pressure (BOP-like inertia) — still discrete flips only.
            key = id(layer)
            if key not in self._pressure_ema:
                self._pressure_ema[key] = torch.zeros_like(grad_pressure)
            ema = self._pressure_ema[key]
            ema.mul_(m).add_(grad_pressure, alpha=1.0 - m)
            pressure = ema

            probs = (pressure.abs() * self.recruit_rate).clamp(0.0, self.max_flip_prob)
            target = -torch.sign(pressure)
            target = torch.where(target == 0, torch.ones_like(target), target)

            pop = layer.population  # int8 [out, in, S]
            pop_f = pop.float()
            target_e = target.unsqueeze(-1).expand_as(pop_f)
            probs_e = probs.unsqueeze(-1).expand_as(pop_f)

            disagree = pop_f != target_e
            roll = torch.rand_like(pop_f)
            should_flip = disagree & (roll < probs_e)

            n_flip = int(should_flip.sum().item())
            flipped += n_flip
            total += pop.numel()

            if n_flip:
                new_f = pop_f.clone()
                new_f[should_flip] *= -1.0
                new_i = torch.where(
                    new_f >= 0,
                    torch.ones_like(pop),
                    -torch.ones_like(pop),
                ).to(torch.int8)
                pop.copy_(new_i)

            layer.enforce_binary_()

        if self.ln_optimizer is not None:
            self.ln_optimizer.step()

        self.last_flip_frac = flipped / max(1, total)
        self.last_grad_abs_mean = grad_abs_sum / max(1, grad_abs_n)
        return self.last_flip_frac
