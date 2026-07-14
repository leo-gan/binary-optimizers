#!/usr/bin/env python3
"""Fit-scale MNIST training with checkpoint cache."""

from __future__ import annotations

import argparse
from pathlib import Path

from binary_optimizers.benchmarks.fit_training import (
    FIT_OPTIMIZERS,
    run_fit_training,
    write_fit_report,
)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--epochs", type=int, default=15)
    p.add_argument("--trials", type=int, default=1)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--model", default="bit_mlp", choices=("bit_mlp", "bit_mlp_large"))
    p.add_argument("--optimizers", nargs="*", default=None)
    p.add_argument("--data-root", default="./data")
    p.add_argument("--checkpoint-root", default="checkpoints")
    p.add_argument(
        "--force-retrain",
        action="store_true",
        help="Ignore cache and retrain all selected configs",
    )
    p.add_argument("--no-packed-eval", action="store_true")
    p.add_argument("--output", default="results/fit_training.json")
    args = p.parse_args()

    payload = run_fit_training(
        optimizers=args.optimizers or FIT_OPTIMIZERS,
        model_name=args.model,
        epochs=args.epochs,
        num_trials=args.trials,
        seed=args.seed,
        batch_size_train=args.batch_size,
        data_root=args.data_root,
        checkpoint_root=args.checkpoint_root,
        force_retrain=args.force_retrain,
        output_json=args.output,
        eval_packed=not args.no_packed_eval,
    )
    analysis = write_fit_report(payload)
    Path("docs/FIT_TRAINING_ANALYSIS.md").write_text(
        Path("results/fit_training_report.md").read_text()
    )
    print("Wrote docs/FIT_TRAINING_ANALYSIS.md")
    print("\n=== Ranking ===")
    for r in analysis["ranking"]:
        flag = " ★" if r["is_new"] else ""
        cache = " [cache]" if r.get("from_cache") else " [trained]"
        print(
            f"  {r['rank']:2d}. {r['optimizer']:20s}{flag}{cache}  "
            f"best={r['best_test_acc_mean']:.4f} final={r['final_test_acc_mean']:.4f}"
        )


if __name__ == "__main__":
    main()
