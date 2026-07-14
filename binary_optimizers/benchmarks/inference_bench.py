"""
Inference benchmarks: float vs sign-weight float vs packed binary (XNOR+popcount).

Scaffolding mode: few warmup/measure iters, small models.
"""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from binary_optimizers.inference.binary_linear import (
    binary_linear_forward,
    reference_binary_linear,
)
from binary_optimizers.inference.extract import extract_packed_weights
from binary_optimizers.inference.packing import pack_binary_weights
from binary_optimizers.models.bit_layers import BitLinearSTE
from binary_optimizers.models.mnist import create_mnist_bit_mlp, create_mnist_swarm_mlp
from binary_optimizers.profiling.memory import measure_inference_memory


@dataclass
class LatencyResult:
    mode: str
    batch_size: int
    mean_ms: float
    std_ms: float
    iters: int


def _sign_pm1(t: torch.Tensor) -> torch.Tensor:
    return torch.where(t >= 0, torch.ones_like(t), -torch.ones_like(t))


def _bench_fn(
    fn: Callable[[], None],
    *,
    warmup: int = 3,
    iters: int = 10,
) -> tuple[float, float]:
    for _ in range(warmup):
        fn()
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    times = []
    for _ in range(iters):
        t0 = time.perf_counter()
        fn()
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        times.append((time.perf_counter() - t0) * 1000.0)
    mean = sum(times) / len(times)
    var = sum((t - mean) ** 2 for t in times) / max(1, len(times) - 1)
    return mean, var**0.5


def _extract_linear_weights(model: nn.Module) -> list[tuple[torch.Tensor, torch.Tensor | None]]:
    """Collect (weight, bias) for BitLinearSTE / Linear layers in order."""
    layers = []
    for m in model.modules():
        if isinstance(m, BitLinearSTE):
            layers.append((m.weight.data, m.bias.data if m.bias is not None else None))
        elif isinstance(m, nn.Linear):
            layers.append((m.weight.data, m.bias.data if m.bias is not None else None))
    return layers


def float_forward(model: nn.Module, x: torch.Tensor) -> torch.Tensor:
    return model(x)


def sign_weight_forward(model: nn.Module, x: torch.Tensor) -> torch.Tensor:
    """Forward with ±1 weights still stored as float (arithmetic effect only)."""
    weights = _extract_linear_weights(model)
    if not weights:
        # Fallback: run model but that won't sign internal weights; use extract path
        return model(x)

    # Manual 2-layer MLP-style path matching create_mnist_bit_mlp structure
    h = x.view(x.size(0), -1)
    # Apply sequential modules but replace linear with signed weights
    for module in model.children() if not isinstance(model, nn.Sequential) else model:
        if isinstance(module, BitLinearSTE):
            w = _sign_pm1(module.weight.data)
            b = module.bias
            h = F.linear(h, w, b)
        elif isinstance(module, nn.Linear):
            w = _sign_pm1(module.weight.data)
            h = F.linear(h, w, module.bias)
        elif isinstance(module, nn.Flatten):
            h = module(h) if h.dim() > 2 else h
        else:
            h = module(h)
    return h


def packed_binary_mlp_forward(
    x: torch.Tensor,
    packed_layers,
    bn_relu_modules: list | None = None,
) -> torch.Tensor:
    """
    Packed binary linear layers only (no BN). Useful for pure binary throughput.
    """
    h = x.view(x.size(0), -1)
    for layer in packed_layers:
        h = binary_linear_forward(h, layer.packed_w, layer.in_features, layer.bias)
    return h


def benchmark_linear_modes(
    weight: torch.Tensor,
    bias: torch.Tensor | None,
    batch_sizes: list[int],
    *,
    warmup: int = 3,
    iters: int = 10,
    device: str = "cpu",
) -> list[LatencyResult]:
    """Benchmark a single linear layer in three modes across batch sizes."""
    out_f, in_f = weight.shape
    w = weight.to(device)
    b = bias.to(device) if bias is not None else None
    packed, _ = pack_binary_weights(_sign_pm1(w.cpu()))
    packed = packed.to(device)
    w_s = _sign_pm1(w)

    results: list[LatencyResult] = []
    for bs in batch_sizes:
        x = torch.randn(bs, in_f, device=device)

        def float_fn(x=x, w=w, b=b):
            F.linear(x, w, b)

        def sign_fn(x=x, w=w_s, b=b):
            F.linear(_sign_pm1(x), w, b)

        def packed_fn(x=x, packed=packed, in_f=in_f, b=b):
            binary_linear_forward(x, packed, in_f, b)

        for mode, fn in [
            ("float", float_fn),
            ("sign_weight_float", sign_fn),
            ("packed_binary", packed_fn),
        ]:
            mean, std = _bench_fn(fn, warmup=warmup, iters=iters)
            results.append(
                LatencyResult(mode=mode, batch_size=bs, mean_ms=mean, std_ms=std, iters=iters)
            )
    return results


def benchmark_model_inference(
    model: nn.Module,
    batch_sizes: list[int],
    input_shape: tuple[int, ...],
    *,
    warmup: int = 3,
    iters: int = 10,
    device: str = "cpu",
) -> dict[str, Any]:
    model = model.to(device).eval()
    packed = extract_packed_weights(model, use_swarm_majority=True)
    mem = measure_inference_memory(model, use_swarm_majority=True).to_dict()
    mem_full = measure_inference_memory(model, use_swarm_majority=False).to_dict()

    latencies: list[dict] = []
    for bs in batch_sizes:
        x = torch.randn(bs, *input_shape, device=device)

        def float_fn(x=x):
            model(x)

        def sign_fn(x=x):
            sign_weight_forward(model, x)

        def packed_fn(x=x):
            packed_binary_mlp_forward(x, packed.layers)

        for mode, fn in [
            ("float", float_fn),
            ("sign_weight_float", sign_fn),
            ("packed_binary", packed_fn),
        ]:
            mean, std = _bench_fn(fn, warmup=warmup, iters=iters)
            latencies.append(
                asdict(
                    LatencyResult(
                        mode=mode, batch_size=bs, mean_ms=mean, std_ms=std, iters=iters
                    )
                )
            )

    return {
        "memory_majority": mem,
        "memory_swarm_full": mem_full,
        "latencies": latencies,
        "n_packed_layers": len(packed.layers),
        "packed_bytes": packed.total_packed_bytes(),
    }


def benchmark_swarm_inference_cost(
    *,
    hidden_dim: int = 64,
    swarm_size: int = 16,
    batch_size: int = 32,
    warmup: int = 3,
    iters: int = 10,
    device: str = "cpu",
) -> dict[str, Any]:
    """Compare full population forward vs majority-vote cached weights."""
    model = create_mnist_swarm_mlp(hidden_dim=hidden_dim, swarm_size=swarm_size).to(device)
    model.eval()
    x = torch.randn(batch_size, 1, 28, 28, device=device)

    packed_maj = extract_packed_weights(model, use_swarm_majority=True)
    packed_full = extract_packed_weights(model, use_swarm_majority=False)

    def full_pop_fn():
        model(x)

    def majority_cache_fn():
        packed_binary_mlp_forward(x, packed_maj.layers)

    mean_full, std_full = _bench_fn(full_pop_fn, warmup=warmup, iters=iters)
    mean_maj, std_maj = _bench_fn(majority_cache_fn, warmup=warmup, iters=iters)

    mem_maj = measure_inference_memory(model, use_swarm_majority=True)
    mem_full = measure_inference_memory(model, use_swarm_majority=False)

    return {
        "swarm_size": swarm_size,
        "hidden_dim": hidden_dim,
        "batch_size": batch_size,
        "full_population_ms": {"mean": mean_full, "std": std_full},
        "majority_cached_ms": {"mean": mean_maj, "std": std_maj},
        "speedup_majority_vs_full": mean_full / mean_maj if mean_maj > 0 else float("inf"),
        "memory_majority_bitpacked": mem_maj.bitpacked_bytes,
        "memory_full_bitpacked": mem_full.bitpacked_bytes,
        "memory_ratio_full_to_majority": (
            mem_full.bitpacked_bytes / mem_maj.bitpacked_bytes
            if mem_maj.bitpacked_bytes
            else float("inf")
        ),
        "packed_majority_bytes": packed_maj.total_packed_bytes(),
        "packed_full_bytes": packed_full.total_packed_bytes(),
    }


def run_inference_benchmark(
    *,
    batch_sizes: list[int] | None = None,
    warmup: int = 2,
    iters: int = 5,
    device: Optional[str] = None,
    output_md: str | Path | None = "results/inference_benchmark.md",
    output_json: str | Path | None = "results/inference_benchmark.json",
) -> dict[str, Any]:
    """Scaffolding inference benchmark with markdown + JSON output."""
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    batch_sizes = batch_sizes or [1, 8, 32]

    torch.manual_seed(0)
    # Single layer microbench
    w = torch.randn(64, 256)
    bias = torch.randn(64)
    linear_results = benchmark_linear_modes(
        w, bias, batch_sizes, warmup=warmup, iters=iters, device=device
    )

    # Full small MLP
    mlp = create_mnist_bit_mlp(hidden_dim=64)
    mlp_results = benchmark_model_inference(
        mlp,
        batch_sizes,
        (1, 28, 28),
        warmup=warmup,
        iters=iters,
        device=device,
    )

    # Swarm cost study
    swarm_results = benchmark_swarm_inference_cost(
        hidden_dim=64,
        swarm_size=16,
        batch_size=32,
        warmup=warmup,
        iters=iters,
        device=device,
    )

    payload = {
        "meta": {
            "device": device,
            "warmup": warmup,
            "iters": iters,
            "batch_sizes": batch_sizes,
            "note": "Scaffolding benchmark — few iterations for pipeline validation.",
        },
        "linear_layer": [asdict(r) for r in linear_results],
        "mnist_bit_mlp": mlp_results,
        "swarm_comparison": swarm_results,
    }

    md = _format_inference_report(payload)

    if output_json is not None:
        path = Path(output_json)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload, indent=2))
        print(f"Wrote {path}")

    if output_md is not None:
        path = Path(output_md)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(md)
        print(f"Wrote {path}")

    return payload


def _format_inference_report(payload: dict[str, Any]) -> str:
    lines = [
        "# Inference Benchmark Report",
        "",
        f"**Device:** {payload['meta']['device']}  ",
        f"**Warmup / Iters:** {payload['meta']['warmup']} / {payload['meta']['iters']}  ",
        f"**Note:** {payload['meta']['note']}",
        "",
        "## Single linear layer (out=64, in=256)",
        "",
        "| Mode | Batch | Mean (ms) | Std (ms) | Speedup vs float |",
        "| :--- | :---: | --------: | -------: | ---------------: |",
    ]

    linear = payload["linear_layer"]
    float_means = {
        r["batch_size"]: r["mean_ms"] for r in linear if r["mode"] == "float"
    }
    for r in linear:
        base = float_means.get(r["batch_size"], r["mean_ms"])
        speedup = base / r["mean_ms"] if r["mean_ms"] > 0 else float("inf")
        lines.append(
            f"| {r['mode']} | {r['batch_size']} | {r['mean_ms']:.4f} | "
            f"{r['std_ms']:.4f} | {speedup:.2f}x |"
        )

    lines += [
        "",
        "## MNIST Bit-MLP (hidden=64)",
        "",
        "### Memory",
        "",
        f"- Float params: **{payload['mnist_bit_mlp']['memory_majority']['float_bytes']}** bytes",
        f"- Int8 weights: **{payload['mnist_bit_mlp']['memory_majority']['int8_bytes']}** bytes",
        f"- Bitpacked: **{payload['mnist_bit_mlp']['memory_majority']['bitpacked_bytes']}** bytes",
        f"- Compression float→bitpacked: "
        f"**{payload['mnist_bit_mlp']['memory_majority']['compression_float_to_bitpacked']:.1f}x**",
        "",
        "### Latency",
        "",
        "| Mode | Batch | Mean (ms) | Speedup vs float |",
        "| :--- | :---: | --------: | ---------------: |",
    ]

    mlp_lat = payload["mnist_bit_mlp"]["latencies"]
    float_mlp = {r["batch_size"]: r["mean_ms"] for r in mlp_lat if r["mode"] == "float"}
    for r in mlp_lat:
        base = float_mlp.get(r["batch_size"], r["mean_ms"])
        speedup = base / r["mean_ms"] if r["mean_ms"] > 0 else float("inf")
        lines.append(
            f"| {r['mode']} | {r['batch_size']} | {r['mean_ms']:.4f} | {speedup:.2f}x |"
        )

    sc = payload["swarm_comparison"]
    lines += [
        "",
        "## Swarm: full population vs majority-vote cache",
        "",
        f"- Swarm size: {sc['swarm_size']}, hidden: {sc['hidden_dim']}, batch: {sc['batch_size']}",
        f"- Full population forward: **{sc['full_population_ms']['mean']:.4f} ms**",
        f"- Majority-cached packed: **{sc['majority_cached_ms']['mean']:.4f} ms**",
        f"- Speedup (majority vs full): **{sc['speedup_majority_vs_full']:.2f}x**",
        f"- Bitpacked memory full / majority: "
        f"**{sc['memory_full_bitpacked']}** / **{sc['memory_majority_bitpacked']}** bytes "
        f"(ratio **{sc['memory_ratio_full_to_majority']:.1f}x**)",
        "",
        "## Takeaways",
        "",
        "- **Packed binary** replaces float matmul with XNOR+popcount on uint8 bitplanes.",
        "  The reference engine is pure PyTorch (software popcount) for *correctness* and",
        "  API completeness — wall-clock on CPU may lag highly optimized float GEMM.",
        "  Memory compression (float → bitpacked) is the primary measurable win here.",
        "- **Sign-weight float** isolates the arithmetic effect of ±1 weights without packing.",
        "- **Swarm majority cache** collapses population tensors to a single ±1 matrix,",
        "  cutting inference *memory* by ~swarm_size vs carrying the full population.",
        "",
    ]
    return "\n".join(lines)
