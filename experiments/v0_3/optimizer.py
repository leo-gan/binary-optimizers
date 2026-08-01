"""Carry-safe ±1 integer updates for experiment v0_3."""

from __future__ import annotations

from typing import Iterable, List, Optional, Sequence

import torch
import torch.nn as nn

from layers import Int8CarrySafeLinear


class SwarmOptimizerV03:
    """EMA grad pressure → probabilistic ±1 on integer register (with carry)."""

    def __init__(
        self,
        swarm_layers: Sequence[Int8CarrySafeLinear],
        *,
        recruit_rate: float = 1e4,
        max_step_prob: float = 0.5,
        grad_momentum: float = 0.9,
        max_step: int = 512,
        step_scale: float = 1e6,
        ln_params: Optional[Iterable[nn.Parameter]] = None,
        ln_lr: float = 1e-2,
    ):
        self.swarm_layers: List[Int8CarrySafeLinear] = list(swarm_layers)
        self.recruit_rate = float(recruit_rate)
        self.max_step_prob = float(max_step_prob)
        self.grad_momentum = float(grad_momentum)
        self.max_step = int(max_step)
        self.step_scale = float(step_scale)
        ln_list = list(ln_params) if ln_params is not None else []
        self.ln_optimizer: torch.optim.Optimizer | None
        if ln_list:
            self.ln_optimizer = torch.optim.SGD(ln_list, lr=ln_lr)
        else:
            self.ln_optimizer = None
        self._pressure_ema: dict[int, torch.Tensor] = {}
        self.last_flip_frac: float = 0.0
        self.last_grad_abs_mean: float = 0.0
        self.last_step_frac: float = 0.0

    def zero_grad(self) -> None:
        for layer in self.swarm_layers:
            layer._pop_f = None
        if self.ln_optimizer is not None:
            self.ln_optimizer.zero_grad(set_to_none=True)

    @torch.no_grad()
    def step(self) -> float:
        n_stepped = 0
        n_weights = 0
        bits_changed = 0
        bits_total = 0
        grad_abs_sum = 0.0
        grad_abs_n = 0
        m = self.grad_momentum

        for layer in self.swarm_layers:
            g = layer.last_agent_grad
            if g is None:
                continue

            # w = 2/vmax * sum b_i 2^i - 1
            # sum_i ∂L/∂b_i = ∂L/∂w * (2/vmax) * vmax = 2 ∂L/∂w
            # => ∂L/∂w = 0.5 * sum_i g_i
            grad_w = 0.5 * g.sum(dim=-1)
            grad_abs_sum += float(grad_w.abs().sum().item())
            grad_abs_n += grad_w.numel()

            key = id(layer)
            if key not in self._pressure_ema:
                self._pressure_ema[key] = torch.zeros_like(grad_w)
            ema = self._pressure_ema[key]
            ema.mul_(m).add_(grad_w, alpha=1.0 - m)
            pressure = ema

            # Move v opposite to gradient (descent on w).
            # Adaptive integer step: larger |g| → larger carry-safe Δv.
            direction = -torch.sign(pressure)
            direction = torch.where(direction == 0, torch.ones_like(direction), direction)
            probs = (pressure.abs() * self.recruit_rate).clamp(0.0, self.max_step_prob)
            roll = torch.rand_like(probs)
            do_step = roll < probs

            mag = (pressure.abs() * self.step_scale).clamp(1.0, float(self.max_step))
            mag = mag.round().to(torch.int64)
            delta = direction.to(torch.int64) * mag

            v = layer.integer_value().to(torch.int64)
            vmax = int(layer.vmax.item())
            v_new = v.clone()
            v_new = torch.where(do_step, (v + delta).clamp(0, vmax), v)

            n_stepped += int(do_step.sum().item())
            n_weights += do_step.numel()

            before = layer.population.clone()
            layer.set_from_integer_(v_new)
            layer.enforce_binary_()
            changed = before != layer.population
            bits_changed += int(changed.sum().item())
            bits_total += changed.numel()

        if self.ln_optimizer is not None:
            self.ln_optimizer.step()

        self.last_step_frac = n_stepped / max(1, n_weights)
        self.last_flip_frac = bits_changed / max(1, bits_total)
        self.last_grad_abs_mean = grad_abs_sum / max(1, grad_abs_n)
        return self.last_flip_frac
