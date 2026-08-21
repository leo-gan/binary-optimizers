"""PathwayNorm: center activations on a running assembled ±1 motif."""

from __future__ import annotations

import torch
import torch.nn as nn


def _sign_pm1(t: torch.Tensor) -> torch.Tensor:
    s = t.sign()
    return torch.where(s == 0, torch.ones_like(s), s)


class PathwayNorm(nn.Module):
    """Normalize activations against a remembered assembled motif.

    Per channel the layer stores:

    - ``motif_acc`` — EMA of signs (direction of the assembled block)
    - ``abs_acc`` — EMA of mean |x| (magnitude of that block)

    The forward residual is ``x - strength * sign(motif_acc) * abs_acc``,
    then a **per-sample** L1 scale (eval-safe, no batch σ). Buffers are not
    parameters; optional affine γ,β are the only trainable 1-D state.

    See ``docs/PATHWAY_NORM.md``.
    """

    def __init__(
        self,
        num_features: int,
        *,
        momentum: float = 0.1,
        eps: float = 1e-5,
        strength: float = 1.0,
        affine: bool = False,
        binarize: bool = False,
    ):
        super().__init__()
        if num_features < 1:
            raise ValueError(f"num_features must be >= 1, got {num_features}")
        if not 0.0 < momentum <= 1.0:
            raise ValueError(f"momentum must be in (0, 1], got {momentum}")
        if strength < 0.0:
            raise ValueError(f"strength must be >= 0, got {strength}")
        self.num_features = int(num_features)
        self.momentum = float(momentum)
        self.eps = float(eps)
        self.strength = float(strength)
        self.binarize = bool(binarize)
        # When False, forward does not touch EMAs. Flip-search closures must
        # leave this False; call update_stats() once on the accepted state.
        self.track = True

        self.register_buffer("motif_acc", torch.zeros(num_features))
        self.register_buffer("abs_acc", torch.ones(num_features))
        self.register_buffer("initialized", torch.zeros((), dtype=torch.bool))

        if affine:
            self.weight = nn.Parameter(torch.ones(num_features))
            self.bias = nn.Parameter(torch.zeros(num_features))
        else:
            self.register_parameter("weight", None)
            self.register_parameter("bias", None)

        self.last_agreement: float = 0.0
        self.last_scale: float = 0.0
        self.last_motif_frac_plus: float = 0.5

    def extra_repr(self) -> str:
        return (
            f"{self.num_features}, momentum={self.momentum}, "
            f"strength={self.strength}, affine={self.weight is not None}, "
            f"binarize={self.binarize}"
        )

    def motif_dir(self) -> torch.Tensor:
        return _sign_pm1(self.motif_acc)

    def _channel_view(self, channel_vec: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        shape = [1, -1] + [1] * (x.ndim - 2)
        return channel_vec.view(*shape)

    def _batch_spatial_mean(self, t: torch.Tensor) -> torch.Tensor:
        dims = [0] + list(range(2, t.ndim))
        return t.mean(dim=dims)

    @torch.no_grad()
    def update_stats(self, x: torch.Tensor) -> None:
        """EMA update from an activation batch (accepted state only)."""
        if x.ndim < 2 or x.shape[1] != self.num_features:
            raise ValueError(
                f"update_stats expected [N, {self.num_features}, …], got {tuple(x.shape)}"
            )
        signs = _sign_pm1(x)
        batch_sign = self._batch_spatial_mean(signs)
        batch_abs = self._batch_spatial_mean(x.abs())
        if not bool(self.initialized.item()):
            self.motif_acc.copy_(batch_sign)
            self.abs_acc.copy_(batch_abs)
            self.initialized.fill_(True)
        else:
            m = self.momentum
            self.motif_acc.mul_(1.0 - m).add_(batch_sign, alpha=m)
            self.abs_acc.mul_(1.0 - m).add_(batch_abs, alpha=m)

    def stats(self) -> dict[str, float]:
        return {
            "agreement": self.last_agreement,
            "scale": self.last_scale,
            "motif_frac_plus": self.last_motif_frac_plus,
        }

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim < 2:
            raise ValueError(f"PathwayNorm expected [N, C, …], got shape {tuple(x.shape)}")
        if x.shape[1] != self.num_features:
            raise ValueError(
                f"PathwayNorm expected {self.num_features} channels, got {x.shape[1]}"
            )

        signs = _sign_pm1(x)
        if self.training and self.track:
            self.update_stats(x.detach())

        motif_dir = self.motif_dir()
        assembled = self._channel_view(motif_dir * self.abs_acc, x).detach()
        residual = x - self.strength * assembled

        reduce_dims = tuple(range(1, residual.ndim))
        scale = residual.abs().mean(dim=reduce_dims, keepdim=True).clamp_min(self.eps)
        y = residual / scale

        if self.weight is not None:
            y = y * self._channel_view(self.weight, y) + self._channel_view(self.bias, y)

        if self.binarize:
            y_s = _sign_pm1(y)
            y = y + (y_s - y).detach()

        with torch.no_grad():
            agree = (signs * self._channel_view(motif_dir, signs)).mean()
            self.last_agreement = float(agree.item())
            self.last_scale = float(scale.mean().item())
            self.last_motif_frac_plus = float((motif_dir > 0).float().mean().item())
        return y
