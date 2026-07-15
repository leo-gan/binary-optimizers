#!/usr/bin/env python3
"""Scaffolding training sweep entrypoint."""

from binary_optimizers.benchmarks.training_sweep import run_training_sweep


if __name__ == "__main__":
    run_training_sweep(
        epochs=2,
        num_trials=2,
        max_train_batches=3,
        max_test_batches=2,
        output_json="results/training_sweep.json",
    )
