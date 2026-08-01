"""Swarm optimizer for place-value int8 agents (experiment v0_2)."""

from __future__ import annotations

from typing import Iterable, List, Optional, Sequence

import torch
import torch.nn as nn

from layers import Int8PlaceValueSwarmLinear


class SwarmOptimizerV02:
    """Flip place-value agents; LSB-biased probs; optional LN SGD.

    Per-agent gradient from STE already weights by place value (∂s/∂a_i = 2^i).
    EMA is taken **per agent bit plane** (not mean over bits), so MSB and LSB
    keep distinct pressure. Flip probability is further scaled by ``2^{-i}`` so
    fine updates hit low bits first (exponential step sizes).
    """

    def __init__(
        self,
        swarm_layers: Sequence[Int8PlaceValueSwarmLinear],
        *,
        recruit_rate: float = 1e4,
        max_flip_prob: float = 0.15,
        grad_momentum: float = 0.9,
        lsb_bias: bool = True,
        ln_params: Optional[Iterable[nn.Parameter]] = None,
        ln_lr: float = 1e-2,
    ):
        self.swarm_layers: List[Int8PlaceValueSwarmLinear] = list(swarm_layers)
        self.recruit_rate = float(recruit_rate)
        self.max_flip_prob = float(max_flip_prob)
        self.grad_momentum = float(grad_momentum)
        self.lsb_bias = bool(lsb_bias)
        ln_list = list(ln_params) if ln_params is not None else []
        self.ln_optimizer: torch.optim.Optimizer | None
        if ln_list:
            self.ln_optimizer = torch.optim.SGD(ln_list, lr=ln_lr)
        else:
            self.ln_optimizer = None
        # EMA of full agent grad [out, in, n_bits]
        self._pressure_ema: dict[int, torch.Tensor] = {}
        self.last_flip_frac: float = 0.0
        self.last_grad_abs_mean: float = 0.0
        self.last_flip_frac_by_bit: List[float] = []

    def zero_grad(self) -> None:
        for layer in self.swarm_layers:
            layer._pop_f = None
        if self.ln_optimizer is not None:
            self.ln_optimizer.zero_grad(set_to_none=True)

    @torch.no_grad()
    def step(self) -> float:
        flipped = 0
        total = 0
        grad_abs_sum = 0.0
        grad_abs_n = 0
        bit_flips: List[int] = []
        bit_totals: List[int] = []
        m = self.grad_momentum

        for layer in self.swarm_layers:
            g = layer.last_agent_grad
            if g is None:
                continue

            # g: [out, in, n_bits]. Under STE through s_norm,
            # sum_i ∂L/∂a_i = ∂L/∂s_norm (place values cancel in the sum).
            # Use that scalar pressure for direction/magnitude (v0.1-like),
            # then LSB bias only chooses *which* bits flip.
            grad_s = g.sum(dim=-1)  # [out, in]
            grad_abs_sum += float(grad_s.abs().sum().item())
            grad_abs_n += grad_s.numel()

            key = id(layer)
            if key not in self._pressure_ema:
                self._pressure_ema[key] = torch.zeros_like(grad_s)
            ema = self._pressure_ema[key]
            ema.mul_(m).add_(grad_s, alpha=1.0 - m)
            pressure = ema  # [out, in]

            n_bits = layer.n_bits
            if self.lsb_bias:
                # Soft LSB preference: scale = 0.5^(i / (n_bits/4)).
                # Hard 2^{-i} starves mid/MSB bits; soft bias still favors fine
                # updates while allowing coarse bits to move.
                denom = max(n_bits / 4.0, 1.0)
                bit_scale = torch.tensor(
                    [0.5 ** (i / denom) for i in range(n_bits)],
                    device=pressure.device,
                    dtype=pressure.dtype,
                )
            else:
                bit_scale = torch.ones(
                    n_bits, device=pressure.device, dtype=pressure.dtype
                )

            # Base prob from |pressure|; broadcast × bit_scale over last dim.
            base = (pressure.abs() * self.recruit_rate).clamp(0.0, self.max_flip_prob)
            probs = base.unsqueeze(-1) * bit_scale  # [out, in, n_bits]
            probs = probs.clamp(0.0, self.max_flip_prob)

            target = -torch.sign(pressure)
            target = torch.where(target == 0, torch.ones_like(target), target)
            target_e = target.unsqueeze(-1).expand_as(g)

            pop = layer.population
            pop_f = pop.float()
            disagree = pop_f != target_e
            roll = torch.rand_like(pop_f)
            should_flip = disagree & (roll < probs)

            if bit_flips == []:
                bit_flips = [0] * n_bits
                bit_totals = [0] * n_bits
            for i in range(n_bits):
                bit_flips[i] += int(should_flip[..., i].sum().item())
                bit_totals[i] += int(should_flip[..., i].numel())

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
        if bit_totals:
            self.last_flip_frac_by_bit = [
                bit_flips[i] / max(1, bit_totals[i]) for i in range(len(bit_totals))
            ]
        else:
            self.last_flip_frac_by_bit = []
        return self.last_flip_frac
