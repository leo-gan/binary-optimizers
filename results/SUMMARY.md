# Binary Optimizers — End-to-End Summary

## Why this infrastructure

Binary neural networks train with floating-point proxies (STE, voting accumulators,
swarm populations) but **deploy** as ±1 weights. Without a bitpacked inference
engine and memory profiler we cannot measure the real memory/speed gains that
justify binary training.

This work adds:

1. **Binary inference engine** — pack ±1 weights into uint8 (8 weights/byte),
   unpack for verification, and run linear layers via **XNOR + popcount**
   (no float multiplies in the matmul).
2. **Memory profiler** — training footprint (parameters + optimizer state) and
   inference footprint in float / int8 / bitpacked form with compression ratios.
3. **Inference benchmarks** — float vs sign-weight float vs packed binary across
   batch sizes, plus swarm full-population vs majority-vote cache.
4. **Training sweep (scaffold)** — all optimizer × model combinations with
   multi-trial stats, epoch curves, wall time, and memory.
5. **Pareto analysis** — accuracy vs inference memory/speed and training efficiency.

> **Scaffolding note:** epochs, batches, and benchmark iterations are intentionally
> small so the analytics pipeline is end-to-end without full training cost.
> Numbers show *relative structure*, not final published accuracy.

## Trade-offs

| Approach | Training memory | Inference memory | Notes |
| :--- | :--- | :--- | :--- |
| Adam + STE layers | High (moments) | Bitpacked | Strong baseline; float optimizer state |
| STE SGD | Medium | Bitpacked | Clamps weights to [-1,1] |
| Voting / Signum | Medium–high (accumulators / momentum) | Bitpacked | Discrete flip dynamics |
| Threshold IF | Medium (accumulators) | Bitpacked | Event-driven flips |
| Swarm | High (population × S) | Bitpacked via majority | Training stores S bits/weight |

**Key trade-off:** swarm training multiplies parameter memory by population size,
but inference can cache the majority vote and match STE bitpacked size.

## Results

### Inference (see `results/inference_benchmark.md`)

**Device:** cpu  
**Warmup / Iters:** 2 / 5  
**Note:** Scaffolding benchmark — few iterations for pipeline validation.

## Single linear layer (out=64, in=256)

| Mode | Batch | Mean (ms) | Std (ms) | Speedup vs float |
| :--- | :---: | --------: | -------: | ---------------: |
| float | 1 | 0.0043 | 0.0009 | 1.00x |
| sign_weight_float | 1 | 0.0135 | 0.0005 | 0.32x |
| packed_binary | 1 | 0.1151 | 0.0105 | 0.04x |
| float | 8 | 0.0150 | 0.0003 | 1.00x |
| sign_weight_float | 8 | 0.0396 | 0.0027 | 0.38x |
| packed_binary | 8 | 0.2866 | 0.0256 | 0.05x |
| float | 32 | 0.0187 | 0.0003 | 1.00x |
| sign_weight_float | 32 | 0.0788 | 0.0012 | 0.24x |
| packed_binary | 32 | 0.5560 | 0.2028 | 0.03x |

## MNIST Bit-MLP (hidden=64)

### Memory

- Float params: **203776** bytes
- Int8 weights: **50816** bytes
- Bitpacked: **6352** bytes
- Compression float→bitpacked: **32.1x**

### Latency

| Mode | Batch | Mean (ms) | Speedup vs float |
| :--- | :---: | --------: | ---------------: |
| float | 1 | 0.1038 | 1.00x |
| sign_weight_float | 1 | 0.2776 | 0.37x |
| packed_binary | 1 | 0.4308 | 0.24x |

### Training sweep (see `results/training_sweep.json`)

Configs run: all STE small/large MLP optimizers, swarm, and CIFAR Adam/Signum/STE.

### Pareto (see `results/pareto_analysis.md`)

Combines scaffolding training-sweep results with inference benchmarks.

**Note:** Scaffolding run — short epochs/batches to demonstrate pipeline.

## Full comparison

| Config | Dataset | Test Acc (mean ± std) | Infer mem (bitpacked) | Infer latency (ms) | Train mem | Train time (s) |
| :--- | :--- | ---: | ---: | ---: | ---: | ---: |
| mnist_small_threshold_if | mnist | 0.3086 ± 0.0829 | 6.4 KB | 0.4308 | 407.0 KB | 0.11 |
| mnist_swarm | mnist | 0.2695 ± 0.0387 | 6.4 KB | 0.8973 | 1.63 MB | 0.08 |
| mnist_small_voting | mnist | 0.2246 ± 0.0470 | 6.4 KB | 0.4308 | 407.0 KB | 0.06 |
| mnist_small_ste | mnist | 0.2168 ± 0.1464 | 6.4 KB | 0.4308 | 407.6 KB | 0.06 |
| mnist_large_ste | mnist | 0.1992 ± 0.0552 | 14.8 KB | 0.4308 | 948.2 KB | 0.11 |
| mnist_large_threshold_if | mnist | 0.1992 ± 0.0442 | 14.8 KB | 0.4308 | 946.2 KB | 0.07 |
| mnist_large_voting | mnist | 0.1953 ± 0.0055 | 14.8 KB | 0.4308 | 946.2 KB | 0.09 |
| mnist_small_adam | mnist | 0.1426 ± 0.0967 | 6.4 KB | 0.4308 | 611.3 KB | 0.07 |
| mnist_small_signum | mnist | 0.1426 ± 0.0967 | 6.4 KB | 0.4308 | 407.0 KB | 0.06 |
| cifar_adam | cifar10 | 0.1348 ± 0.0083 | 262.5 KB | 0.1151 | 26.11 MB | 0.49 |
| cifar_signum | cifar10 | 0.1328 ± 0.0000 | 262.5 KB | 0.1151 | 17.40 MB | 0.53 |
| cifar_ste_sgd | cifar10 | 0.1172 ± 0.0000 | 131.4 KB | 0.1151 | 8.57 MB | 0.36 |
| mnist_large_signum | mnist | 0.0723 ± 0.0028 | 14.8 KB | 0.4308 | 946.2 KB | 0.06 |
| mnist_large_adam | mnist | 0.0703 ± 0.0055 | 14.8 KB | 0.4308 | 1.42 MB | 0.07 |

## Accuracy vs inference memory

| Config | Acc | Bitpacked mem |
| :--- | ---: | ---: |
| mnist_small_adam | 0.1426 | 6.4 KB |
| mnist_small_ste | 0.2168 | 6.4 KB |
| mnist_small_voting | 0.2246 | 6.4 KB |
| mnist_small_signum | 0.1426 | 6.4 KB |
| mnist_small_threshold_if ★ | 0.3086 | 6.4 KB |
| mnist_swarm | 0.2695 | 6.4 KB |
| mnist_large_adam | 0.0703 | 14.8 KB |
| mnist_large_ste | 0.1992 | 14.8 KB |
| mnist_large_voting | 0.1953 | 14.8 KB |
| mnist_large_signum | 0.0723 | 14.8 KB |
| mnist_large_threshold_if | 0.1992 | 14.8 KB |
| cifar_ste_sgd | 0.1172 | 131.4 KB |
| cifar_adam | 0.1348 | 262.5 KB |
| cifar_signum | 0.1328 | 262.5 KB |

## Accuracy vs inference speed

| Config | Acc | Latency (ms) |
| :--- | ---: | ---: |
| cifar_adam ★ | 0.1348 | 0.1151 |
| cifar_signum | 0.1328 | 0.1151 |
| cifar_ste_sgd | 0.1172 | 0.1151 |

## How to reproduce

```bash
# Unit tests
uv run pytest tests/ -q

# Scaffolding pipeline
uv run python experiments/run_full_pipeline.py
# or stepwise:
uv run python experiments/run_inference_benchmark.py
uv run python experiments/run_training_sweep.py
uv run python experiments/run_pareto_analysis.py
```

## File map

| Path | Role |
| :--- | :--- |
| `binary_optimizers/inference/` | Pack/unpack, XNOR+popcount linear, weight extract |
| `binary_optimizers/profiling/` | Training + inference memory profiler |
| `binary_optimizers/benchmarks/training_sweep.py` | Multi-config training harness |
| `binary_optimizers/benchmarks/inference_bench.py` | Inference mode benchmarks |
| `binary_optimizers/benchmarks/pareto.py` | Combined Pareto analysis |
| `tests/test_packing.py` | Packing round-trip tests |
| `tests/test_binary_linear.py` | Binary forward ≡ F.linear(sign x, sign w) |
| `tests/test_memory_profiler.py` | Profiler tests |
| `results/` | JSON + markdown artifacts |
