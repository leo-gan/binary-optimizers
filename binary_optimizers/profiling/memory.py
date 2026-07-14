"""Memory profiler for binary network training and inference representations."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

import torch
import torch.nn as nn

from binary_optimizers.inference.extract import extract_packed_weights


def tensor_nbytes(t: torch.Tensor) -> int:
    """Number of bytes occupied by a tensor's storage (numel * element_size)."""
    return int(t.numel() * t.element_size())


def _param_nbytes(model: nn.Module) -> dict[str, Any]:
    total = 0
    numel = 0
    per_param: dict[str, int] = {}
    for name, p in model.named_parameters():
        nb = tensor_nbytes(p.data)
        per_param[name] = nb
        total += nb
        numel += p.numel()
    return {"bytes": total, "numel": numel, "per_param": per_param}


def _optimizer_state_nbytes(optimizer: torch.optim.Optimizer) -> dict[str, Any]:
    total = 0
    n_tensors = 0
    details: list[dict[str, Any]] = []
    for p, state in optimizer.state.items():
        for key, val in state.items():
            if torch.is_tensor(val):
                nb = tensor_nbytes(val)
                total += nb
                n_tensors += 1
                details.append({"key": key, "shape": list(val.shape), "bytes": nb})
    return {"bytes": total, "n_tensors": n_tensors, "details": details}


@dataclass
class InferenceMemory:
    float_bytes: int
    int8_bytes: int
    bitpacked_bytes: int
    n_weights: int
    compression_float_to_int8: float
    compression_float_to_bitpacked: float
    compression_int8_to_bitpacked: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class TrainingMemory:
    param_bytes: int
    optimizer_state_bytes: int
    total_bytes: int
    param_numel: int
    optimizer_n_tensors: int

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class MemoryReport:
    training: TrainingMemory | None = None
    inference: InferenceMemory | None = None
    extra: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "training": self.training.to_dict() if self.training else None,
            "inference": self.inference.to_dict() if self.inference else None,
            "extra": self.extra,
        }


def measure_training_memory(
    model: nn.Module,
    optimizer: torch.optim.Optimizer | None = None,
) -> TrainingMemory:
    """
    Measure training memory: parameter storage + optimizer state tensors.

    Note: optimizer state is only populated after at least one ``step()``
    (or after buffers are allocated). Call after a warm-up step for realistic
    numbers.
    """
    pinfo = _param_nbytes(model)
    if optimizer is not None:
        oinfo = _optimizer_state_nbytes(optimizer)
        opt_bytes = oinfo["bytes"]
        opt_n = oinfo["n_tensors"]
    else:
        opt_bytes = 0
        opt_n = 0

    return TrainingMemory(
        param_bytes=pinfo["bytes"],
        optimizer_state_bytes=opt_bytes,
        total_bytes=pinfo["bytes"] + opt_bytes,
        param_numel=pinfo["numel"],
        optimizer_n_tensors=opt_n,
    )


def measure_inference_memory(
    model: nn.Module,
    *,
    use_swarm_majority: bool = True,
) -> InferenceMemory:
    """
    Measure inference memory in three representations:

    - **float**: as-trained parameter storage (all parameters)
    - **int8**: one byte per binary weight (linear/swarm majority weights only)
    - **bitpacked**: one bit per binary weight (8 weights per byte), plus
      residual non-binary params counted at float size in ``extra`` via
      :func:`profile_model_memory` if needed.

    Weight count for int8/bitpacked is the number of binary weight elements
    extractable by :func:`extract_packed_weights`.
    """
    param_bytes = sum(tensor_nbytes(p.data) for p in model.parameters())
    packed = extract_packed_weights(model, use_swarm_majority=use_swarm_majority)
    n_weights = packed.total_weight_bits()
    bitpacked_bytes = packed.total_packed_bytes()
    int8_bytes = n_weights  # 1 byte per weight

    def _ratio(a: int, b: int) -> float:
        return float(a) / float(b) if b > 0 else float("inf")

    return InferenceMemory(
        float_bytes=param_bytes,
        int8_bytes=int8_bytes,
        bitpacked_bytes=bitpacked_bytes,
        n_weights=n_weights,
        compression_float_to_int8=_ratio(param_bytes, int8_bytes),
        compression_float_to_bitpacked=_ratio(param_bytes, bitpacked_bytes),
        compression_int8_to_bitpacked=_ratio(int8_bytes, bitpacked_bytes),
    )


def profile_model_memory(
    model: nn.Module,
    optimizer: torch.optim.Optimizer | None = None,
    *,
    use_swarm_majority: bool = True,
) -> MemoryReport:
    """Combined training + inference memory report."""
    return MemoryReport(
        training=measure_training_memory(model, optimizer),
        inference=measure_inference_memory(
            model, use_swarm_majority=use_swarm_majority
        ),
    )
