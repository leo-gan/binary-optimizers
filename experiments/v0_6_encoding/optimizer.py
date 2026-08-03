"""Carry-safe integer updates for encoding-atlas layers."""

from __future__ import annotations

from typing import Iterable, List, Optional, Sequence

import torch
import torch.nn as nn

from layers import EncodingLinear


class SwarmOptimizerV06:
    """EMA pressure → probabilistic ±Δ on mantissa / exponent registers."""

    def __init__(
        self,
        swarm_layers: Sequence[EncodingLinear],
        *,
        recruit_rate: float = 1e4,
        max_step_prob: float = 0.5,
        grad_momentum: float = 0.9,
        max_step: int = 512,
        step_scale: float = 1e6,
        exp_max_step: int = 1,
        exp_step_scale: float = 1.0,
        ln_params: Optional[Iterable[nn.Parameter]] = None,
        ln_lr: float = 1e-2,
    ):
        self.swarm_layers: List[EncodingLinear] = list(swarm_layers)
        self.recruit_rate = float(recruit_rate)
        self.max_step_prob = float(max_step_prob)
        self.grad_momentum = float(grad_momentum)
        self.max_step = int(max_step)
        self.step_scale = float(step_scale)
        self.exp_max_step = int(exp_max_step)
        self.exp_step_scale = float(exp_step_scale)
        ln_list = list(ln_params) if ln_params is not None else []
        self.ln_optimizer: torch.optim.Optimizer | None
        if ln_list:
            self.ln_optimizer = torch.optim.SGD(ln_list, lr=ln_lr)
        else:
            self.ln_optimizer = None
        self._mant_ema: dict[int, torch.Tensor] = {}
        self._exp_ema: dict[int, torch.Tensor] = {}
        self._row_ema: dict[int, torch.Tensor] = {}
        self.last_flip_frac: float = 0.0
        self.last_step_frac: float = 0.0
        self.last_grad_abs_mean: float = 0.0

    def zero_grad(self) -> None:
        for layer in self.swarm_layers:
            layer._mant_f = None
            layer._exp_f = None
            layer._row_exp_f = None
        if self.ln_optimizer is not None:
            self.ln_optimizer.zero_grad(set_to_none=True)

    def _step_register(
        self,
        *,
        pressure: torch.Tensor,
        v: torch.Tensor,
        vmax: int,
        max_step: int,
        step_scale: float,
        set_fn,
        layer: EncodingLinear,
        bits_before: torch.Tensor,
    ) -> tuple[int, int, int, int]:
        """One carry-safe update; returns stepped, weights, bits_changed, bits_total."""
        direction = -torch.sign(pressure)
        direction = torch.where(direction == 0, torch.ones_like(direction), direction)
        probs = (pressure.abs() * self.recruit_rate).clamp(0.0, self.max_step_prob)
        do_step = torch.rand_like(probs) < probs
        mag = (pressure.abs() * step_scale).clamp(1.0, float(max_step))
        mag = mag.round().to(torch.int64)
        delta = direction.to(torch.int64) * mag
        v_new = torch.where(do_step, (v + delta).clamp(0, vmax), v)
        set_fn(v_new)
        after = (
            layer.mant_pop
            if bits_before.shape == layer.mant_pop.shape
            else (
                layer.exp_pop
                if bits_before.shape == layer.exp_pop.shape
                else layer.row_exp_pop
            )
        )
        changed = bits_before != after
        return (
            int(do_step.sum().item()),
            int(do_step.numel()),
            int(changed.sum().item()),
            int(changed.numel()),
        )

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
            g_m = layer.last_mant_grad
            if g_m is None:
                continue

            # Mantissa: ∂L/∂m_bits ~ place-value; pressure on w via sum of bit grads.
            # Same identity as v0.3 for pure fixed-point: grad_w ≈ 0.5 * sum g_bits
            # when w = 2v/vmax-1. For scaled encodings this is approximate STE pressure.
            grad_m = 0.5 * g_m.sum(dim=-1)
            grad_abs_sum += float(grad_m.abs().sum().item())
            grad_abs_n += grad_m.numel()

            key = id(layer)
            if key not in self._mant_ema:
                self._mant_ema[key] = torch.zeros_like(grad_m)
            ema_m = self._mant_ema[key]
            ema_m.mul_(m).add_(grad_m, alpha=1.0 - m)

            vmax_m = int(layer.mant_vmax.item())
            v_m = layer._mant_int(layer.mant_pop.float()).to(torch.int64)
            before_m = layer.mant_pop.clone()
            s, w, bc, bt = self._step_register(
                pressure=ema_m,
                v=v_m,
                vmax=vmax_m,
                max_step=self.max_step,
                step_scale=self.step_scale,
                set_fn=layer.set_mant_from_integer_,
                layer=layer,
                bits_before=before_m,
            )
            n_stepped += s
            n_weights += w
            bits_changed += bc
            bits_total += bt

            if layer.encoding == "exp_mant":
                g_e = layer.last_exp_grad
                if g_e is not None:
                    grad_e = 0.5 * g_e.sum(dim=-1)
                    if key not in self._exp_ema:
                        self._exp_ema[key] = torch.zeros_like(grad_e)
                    ema_e = self._exp_ema[key]
                    ema_e.mul_(m).add_(grad_e, alpha=1.0 - m)
                    vmax_e = int(layer.exp_vmax.item())
                    v_e = layer._exp_int(layer.exp_pop.float()).to(torch.int64)
                    before_e = layer.exp_pop.clone()
                    s, w, bc, bt = self._step_register(
                        pressure=ema_e,
                        v=v_e,
                        vmax=vmax_e,
                        max_step=self.exp_max_step,
                        step_scale=self.exp_step_scale,
                        set_fn=layer.set_exp_from_integer_,
                        layer=layer,
                        bits_before=before_e,
                    )
                    n_stepped += s
                    n_weights += w
                    bits_changed += bc
                    bits_total += bt

            if layer.encoding == "block_scale":
                g_r = layer.last_row_exp_grad
                if g_r is not None:
                    grad_r = 0.5 * g_r.sum(dim=-1)
                    if key not in self._row_ema:
                        self._row_ema[key] = torch.zeros_like(grad_r)
                    ema_r = self._row_ema[key]
                    ema_r.mul_(m).add_(grad_r, alpha=1.0 - m)
                    vmax_e = int(layer.exp_vmax.item())
                    v_r = layer._exp_int(layer.row_exp_pop.float()).to(torch.int64)
                    before_r = layer.row_exp_pop.clone()
                    s, w, bc, bt = self._step_register(
                        pressure=ema_r,
                        v=v_r,
                        vmax=vmax_e,
                        max_step=self.exp_max_step,
                        step_scale=self.exp_step_scale,
                        set_fn=layer.set_row_exp_from_integer_,
                        layer=layer,
                        bits_before=before_r,
                    )
                    n_stepped += s
                    n_weights += w
                    bits_changed += bc
                    bits_total += bt

            layer.enforce_binary_()

        if self.ln_optimizer is not None:
            self.ln_optimizer.step()

        self.last_step_frac = n_stepped / max(1, n_weights)
        self.last_flip_frac = bits_changed / max(1, bits_total)
        self.last_grad_abs_mean = grad_abs_sum / max(1, grad_abs_n)
        return self.last_flip_frac
