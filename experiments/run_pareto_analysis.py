#!/usr/bin/env python3
"""Pareto analysis over training + inference results."""

from binary_optimizers.benchmarks.pareto import run_pareto_analysis


if __name__ == "__main__":
    run_pareto_analysis(
        training_json="results/training_sweep.json",
        inference_json="results/inference_benchmark.json",
        output_md="results/pareto_analysis.md",
        output_json="results/pareto_analysis.json",
    )
