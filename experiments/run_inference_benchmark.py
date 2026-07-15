#!/usr/bin/env python3
"""Scaffolding inference benchmark entrypoint."""

from binary_optimizers.benchmarks.inference_bench import run_inference_benchmark


if __name__ == "__main__":
    run_inference_benchmark(
        batch_sizes=[1, 8, 32],
        warmup=2,
        iters=5,
        output_md="results/inference_benchmark.md",
        output_json="results/inference_benchmark.json",
    )
