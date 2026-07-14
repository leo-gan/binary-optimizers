"""Extract packed binary weights from trained models."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from .binary_linear import BinaryLinear, binary_linear_forward
from .packing import pack_binary_weights


def _sign_pm1(w: torch.Tensor) -> torch.Tensor:
    """Map reals to ±1 (zeros / non-negatives -> +1)."""
    return torch.where(w >= 0, torch.ones_like(w), -torch.ones_like(w))


def _get_bias_tensor(module: nn.Module) -> torch.Tensor | None:
    bias = getattr(module, "bias", None)
    if isinstance(bias, nn.Parameter):
        return bias.data
    if hasattr(module, "fc") and getattr(module.fc, "bias", None) is not None:
        return module.fc.bias.data
    return None


@dataclass
class PackedLayer:
    name: str
    packed_w: torch.Tensor
    in_features: int
    out_features: int
    bias: torch.Tensor | None = None
    source_kind: str = "linear"  # linear | swarm_majority | swarm_full


@dataclass
class PackedModel:
    """Collection of packed binary linear layers extracted from a model."""

    layers: list[PackedLayer] = field(default_factory=list)
    meta: dict[str, Any] = field(default_factory=dict)

    def total_packed_bytes(self) -> int:
        return sum(int(layer.packed_w.numel()) for layer in self.layers)

    def total_weight_bits(self) -> int:
        return sum(layer.in_features * layer.out_features for layer in self.layers)

    def as_binary_linears(self) -> list[BinaryLinear]:
        return [
            BinaryLinear(layer.packed_w, layer.in_features, layer.bias)
            for layer in self.layers
        ]


def _try_pack_module(
    name: str,
    module: nn.Module,
    *,
    use_swarm_majority: bool,
    handled: set[str],
) -> PackedLayer | None:
    """Attempt to extract one packed layer from a module. Returns None if N/A."""
    if name in handled:
        return None

    # --- Swarm population ---
    if hasattr(module, "population") and isinstance(module.population, nn.Parameter):
        pop = module.population.data
        if pop.dim() == 3:
            if use_swarm_majority:
                swarm_sum = pop.sum(dim=-1)
                w = _sign_pm1(swarm_sum)
                pw, in_f = pack_binary_weights(w)
                handled.add(name)
                return PackedLayer(
                    name=name or type(module).__name__,
                    packed_w=pw,
                    in_features=in_f,
                    out_features=w.size(0),
                    bias=_get_bias_tensor(module),
                    source_kind="swarm_majority",
                )
            o, i, s = pop.shape
            w = _sign_pm1(pop.permute(0, 2, 1).reshape(o * s, i))
            pw, in_f = pack_binary_weights(w)
            handled.add(name)
            return PackedLayer(
                name=name or type(module).__name__,
                packed_w=pw,
                in_features=in_f,
                out_features=o * s,
                bias=None,
                source_kind="swarm_full",
            )

    # --- BitLinear wrapper (has .fc) ---
    if hasattr(module, "fc") and isinstance(getattr(module, "fc", None), nn.Linear):
        w = _sign_pm1(module.fc.weight.data)
        bias = module.fc.bias.data if module.fc.bias is not None else None
        pw, in_f = pack_binary_weights(w)
        handled.add(name)
        if name:
            handled.add(f"{name}.fc")
        else:
            handled.add("fc")
        return PackedLayer(
            name=name or type(module).__name__,
            packed_w=pw,
            in_features=in_f,
            out_features=w.size(0),
            bias=bias,
            source_kind="linear",
        )

    # --- Modules with direct 2D .weight (BitLinearSTE, nn.Linear, …) ---
    weight = getattr(module, "weight", None)
    if (
        isinstance(weight, nn.Parameter)
        and weight.dim() == 2
        and not isinstance(module, (nn.BatchNorm1d, nn.BatchNorm2d, nn.Embedding))
    ):
        # Skip nested .fc children counted via wrapper
        if name.endswith(".fc"):
            return None
        w = _sign_pm1(weight.data)
        pw, in_f = pack_binary_weights(w)
        handled.add(name)
        return PackedLayer(
            name=name or type(module).__name__,
            packed_w=pw,
            in_features=in_f,
            out_features=w.size(0),
            bias=_get_bias_tensor(module),
            source_kind="linear",
        )

    return None


def extract_packed_weights(
    model: nn.Module,
    *,
    use_swarm_majority: bool = True,
) -> PackedModel:
    """
    Walk ``model`` and pack every 2D weight matrix.

    Swarm modules with a ``population`` of shape (out, in, swarm_size) are
    reduced by majority vote (default) or packed member-wise.

    The root module itself is considered (so a bare ``nn.Linear`` / ``BitLinearSTE``
    extracts correctly).
    """
    packed = PackedModel(meta={"use_swarm_majority": use_swarm_majority})
    handled: set[str] = set()

    # Root first (empty name)
    layer = _try_pack_module(
        "", model, use_swarm_majority=use_swarm_majority, handled=handled
    )
    if layer is not None:
        packed.layers.append(layer)

    for name, module in model.named_modules():
        if not name:
            continue
        layer = _try_pack_module(
            name, module, use_swarm_majority=use_swarm_majority, handled=handled
        )
        if layer is not None:
            packed.layers.append(layer)

    return packed


def packed_mlp_forward(
    x: torch.Tensor,
    packed_layers: list[PackedLayer],
    *,
    apply_relu: bool = False,
) -> torch.Tensor:
    """Stack binary linear layers (optional ReLU between). For testing only."""
    h = x
    for i, layer in enumerate(packed_layers):
        h = binary_linear_forward(h, layer.packed_w, layer.in_features, layer.bias)
        if apply_relu and i < len(packed_layers) - 1:
            h = F.relu(h)
    return h
