#!/usr/bin/env python3
"""Run scaffolding pipeline: inference bench → training sweep → Pareto → summary doc."""

from pathlib import Path

from binary_optimizers.benchmarks.inference_bench import run_inference_benchmark
from binary_optimizers.benchmarks.new_optimizer_report import write_new_optimizer_report
from binary_optimizers.benchmarks.pareto import run_pareto_analysis
from binary_optimizers.benchmarks.training_sweep import run_training_sweep


SUMMARY_TEMPLATE = '''# Binary Optimizers — End-to-End Summary

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
4. **Training sweep (scaffold)** — optimizer × model combinations with multi-trial
   stats, epoch curves, wall time, and memory.
5. **Pareto analysis** — accuracy vs inference memory/speed and training efficiency.

> **Scaffolding note:** epochs, batches, and benchmark iterations are intentionally
> small so the analytics pipeline is end-to-end without full training cost.
> Numbers show *relative structure*, not final published accuracy.
> For fit-scale ranking and the **default Bit-MLP optimizer (`ema_flip`)**, see
> `docs/optimizers.md` and `docs/FIT_TRAINING_ANALYSIS.md`.

## Trade-offs

| Approach | Training memory | Inference memory | Notes |
| :--- | :--- | :--- | :--- |
| Adam + STE layers | High (moments) | Bitpacked | Strong baseline; float optimizer state |
| STE SGD | Medium | Bitpacked | Clamps weights to [-1,1] |
| Voting / Signum / CosineVoting / EMAFlip | Medium (one buffer) | Bitpacked | Sign / vote dynamics; **ema_flip recommended for Bit-MLP** |
| Threshold IF / hybrids | Medium–high | Bitpacked | Event-driven; often lower FC accuracy |
| Swarm | High (population × S) | Bitpacked via majority | Training stores S bits/weight |

**Key trade-off:** swarm training multiplies parameter memory by population size,
but inference can cache the majority vote and match STE bitpacked size.

## Results

### Inference (see `results/inference_benchmark.md`)

{inference_excerpt}

### Training sweep (see `results/training_sweep.json`)

Configs: adam/ste/voting/signum/threshold_if/swarm and binary specialists
(`ema_flip`, `cosine_voting`, `sparse_sign`, `hybrid_accumulator`, …)
on MNIST Bit-MLP and CIFAR variants.

### Pareto (see `results/pareto_analysis.md`)

{pareto_excerpt}

### Optimizer guide

See **`docs/optimizers.md`** for mechanisms, reasons, trade-offs, and the Bit-MLP default.

## How to reproduce

```bash
# Unit tests
uv run pytest tests/ -q

# Scaffolding pipeline
uv run python experiments/run_full_pipeline.py
# Fit-scale (cached checkpoints):
uv run python experiments/run_fit_training.py
```

## File map

| Path | Role |
| :--- | :--- |
| `binary_optimizers/inference/` | Pack/unpack, XNOR+popcount linear, weight extract |
| `binary_optimizers/profiling/` | Training + inference memory profiler |
| `binary_optimizers/optimizers/` | All optimizers (see docs/optimizers.md) |
| `binary_optimizers/benchmarks/training_sweep.py` | Multi-config training harness |
| `binary_optimizers/benchmarks/inference_bench.py` | Inference mode benchmarks |
| `binary_optimizers/benchmarks/pareto.py` | Combined Pareto analysis |
| `docs/optimizers.md` | Unified optimizer design + trade-offs |
| `tests/` | Packing, inference, optimizers, checkpoints |
| `results/` | JSON + markdown artifacts |
'''


def write_summary(
    pareto_md: Path,
    inference_md: Path,
    out: Path,
) -> None:
    inf_text = inference_md.read_text() if inference_md.exists() else "_pending_"
    par_text = pareto_md.read_text() if pareto_md.exists() else "_pending_"

    # Pull key tables (skip titles)
    def excerpt(text: str, max_lines: int = 40) -> str:
        lines = text.strip().splitlines()
        # drop first H1 if present
        if lines and lines[0].startswith("#"):
            lines = lines[1:]
        return "\n".join(lines[:max_lines]).strip()

    body = SUMMARY_TEMPLATE.format(
        inference_excerpt=excerpt(inf_text, 35),
        pareto_excerpt=excerpt(par_text, 55),
    )
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(body)
    print(f"Wrote {out}")


def main() -> None:
    print("=== 1/4 Inference benchmark (scaffold) ===")
    run_inference_benchmark(
        batch_sizes=[1, 8, 32],
        warmup=2,
        iters=5,
        output_md="results/inference_benchmark.md",
        output_json="results/inference_benchmark.json",
    )

    print("\n=== 2/4 Training sweep (scaffold, incl. new optimizers) ===")
    # Slightly longer than micro-scaffold so EMA warm-up / voting dynamics show up,
    # while remaining far short of full training.
    run_training_sweep(
        epochs=6,
        num_trials=2,
        max_train_batches=8,
        max_test_batches=4,
        output_json="results/training_sweep.json",
    )

    print("\n=== 3/4 Pareto analysis ===")
    run_pareto_analysis(
        training_json="results/training_sweep.json",
        inference_json="results/inference_benchmark.json",
        output_md="results/pareto_analysis.md",
        output_json="results/pareto_analysis.json",
    )

    print("\n=== 4/4 Scaffold comparison report (results only) ===")
    write_new_optimizer_report(
        training_json="results/training_sweep.json",
        output_md="results/new_optimizers_report.md",
        output_json="results/new_optimizers_report.json",
        model="bit_mlp_small",
    )
    # Optimizer design docs live in docs/optimizers.md (unified); not overwritten here.

    write_summary(
        Path("results/pareto_analysis.md"),
        Path("results/inference_benchmark.md"),
        Path("docs/END_TO_END_SUMMARY.md"),
    )
    # also copy key report
    Path("results/SUMMARY.md").write_text(Path("docs/END_TO_END_SUMMARY.md").read_text())
    print("Wrote results/SUMMARY.md")


if __name__ == "__main__":
    main()
