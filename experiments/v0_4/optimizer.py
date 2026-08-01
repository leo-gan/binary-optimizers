"""Carry-safe ±1 updates on balanced ternary registers (v0_4)."""

from __future__ import annotations

from typing import Iterable, List, Optional, Sequence

import torch
import torch.nn as nn

from layers import Int8BalancedTernaryLinear


class SwarmOptimizerV04:
    def __init__(
        self,
        swarm_layers: Sequence[Int8BalancedTernaryLinear],
        *,
        recruit_rate: float = 1e4,
        max_step_prob: float = 0.5,
        grad_momentum: float = 0.9,
        max_step: int = 64,
        step_scale: float = 1e6,
        ln_params: Optional[Iterable[nn.Parameter]] = None,
        ln_lr: float = 1e-2,
    ):
        self.swarm_layers = list(swarm_layers)
        self.recruit_rate = float(recruit_rate)
        self.max_step_prob = float(max_step_prob)
        self.grad_momentum = float(grad_momentum)
        self.max_step = int(max_step)
        self.step_scale = float(step_scale)
        ln_list = list(ln_params) if ln_params is not None else []
        self.ln_optimizer = (
            torch.optim.SGD(ln_list, lr=ln_lr) if ln_list else None
        )
        self._pressure_ema: dict[int, torch.Tensor] = {}
        self.last_flip_frac = 0.0
        self.last_step_frac = 0.0
        self.last_grad_abs_mean = 0.0

    def zero_grad(self) -> None:
        for layer in self.swarm_layers:
            layer._pop_f = None
        if self.ln_optimizer is not None:
            self.ln_optimizer.zero_grad(set_to_none=True)

    @torch.no_grad()
    def step(self) -> float:
        n_stepped = n_weights = 0
        dig_changed = dig_total = 0
        grad_abs_sum = grad_abs_n = 0.0
        m = self.grad_momentum

        for layer in self.swarm_layers:
            g = layer.last_agent_grad
            if g is None:
                continue
            # w = s / s_max, s = sum d_i 3^i
            # sum_i ∂L/∂d_i = ∂L/∂w * (1/s_max) * sum 3^i
            # sum 3^i = (3^n-1)/2 = s_max ⇒ sum g_i = ∂L/∂w
            grad_w = g.sum(dim=-1)
            grad_abs_sum += float(grad_w.abs().sum().item())
            grad_abs_n += grad_w.numel()

            key = id(layer)
            if key not in self._pressure_ema:
                self._pressure_ema[key] = torch.zeros_like(grad_w)
            ema = self._pressure_ema[key]
            ema.mul_(m).add_(grad_w, alpha=1.0 - m)
            pressure = ema

            direction = -torch.sign(pressure)
            direction = torch.where(
                direction == 0, torch.ones_like(direction), direction
            )
            probs = (pressure.abs() * self.recruit_rate).clamp(0.0, self.max_step_prob)
            do_step = torch.rand_like(probs) < probs

            mag = (pressure.abs() * self.step_scale).clamp(1.0, float(self.max_step))
            mag = mag.round().to(torch.int64)
            delta = direction.to(torch.int64) * mag

            s = layer.place_value_sum().to(torch.int64)
            s_max = int(layer.s_max.item())
            s_new = s.clone()
            s_new = torch.where(do_step, (s + delta).clamp(-s_max, s_max), s)

            n_stepped += int(do_step.sum().item())
            n_weights += do_step.numel()

            before = layer.population.clone()
            layer.set_from_signed_integer_(s_new)
            layer.enforce_ternary_()
            ch = before != layer.population
            dig_changed += int(ch.sum().item())
            dig_total += ch.numel()

        if self.ln_optimizer is not None:
            self.ln_optimizer.step()

        self.last_step_frac = n_stepped / max(1, n_weights)
        self.last_flip_frac = dig_changed / max(1, dig_total)
        self.last_grad_abs_mean = grad_abs_sum / max(1, grad_abs_n)
        return self.last_flip_frac
