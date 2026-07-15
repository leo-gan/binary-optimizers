#!/usr/bin/env python3
"""Fit-scale MNIST training with checkpoint cache (Bit-MLP + swarm combinations)."""

from __future__ import annotations

import argparse
from pathlib import Path

from binary_optimizers.benchmarks.fit_training import (
    FIT_OPTIMIZERS,
    FIT_SWARM_COMBINATIONS,
    default_fit_combinations,
    run_fit_training,
    write_fit_report,
)


def main() -> None:
    p = argparse.ArgumentParser(
        description=__doc__
        + " Default Bit-MLP trainer: ema_flip. Swarm paths compare swarm/swarm_log."
    )
    p.add_argument("--epochs", type=int, default=15)
    p.add_argument("--trials", type=int, default=1)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument(
        "--model",
        default="bit_mlp",
        choices=("bit_mlp", "bit_mlp_large", "swarm_mlp"),
        help="Bit-MLP model for --optimizers list (swarm combos via --include-swarm)",
    )
    p.add_argument(
        "--optimizers",
        nargs="*",
        default=None,
        help="Bit-MLP optimizer keys (default: full FIT_OPTIMIZERS). "
        "Example: --optimizers ema_flip",
    )
    p.add_argument(
        "--default-only",
        action="store_true",
        help="Train only bit_mlp + ema_flip (no swarm)",
    )
    p.add_argument(
        "--include-swarm",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Include swarm_mlp × (swarm, swarm_log, swarm_log_dynamic, adam) (default: true)",
    )
    p.add_argument(
        "--swarm-only",
        action="store_true",
        help="Only swarm model combinations",
    )
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

    if args.default_only:
        combos = [("bit_mlp", "ema_flip")]
    elif args.swarm_only:
        combos = list(FIT_SWARM_COMBINATIONS)
    else:
        opts = args.optimizers if args.optimizers else list(FIT_OPTIMIZERS)
        combos = default_fit_combinations(
            include_bit_mlp=True,
            include_swarm=args.include_swarm and args.model != "swarm_mlp",
            bit_mlp_optimizers=opts,
            model_name=args.model if args.model != "swarm_mlp" else "bit_mlp",
        )
        if args.model == "swarm_mlp":
            # User selected swarm model + optimizers list
            if args.optimizers:
                combos = [("swarm_mlp", o) for o in args.optimizers]
            else:
                combos = list(FIT_SWARM_COMBINATIONS)

    payload = run_fit_training(
        combinations=combos,
        include_swarm=False,  # already baked into combinations
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
    print("\n=== Ranking (model + optimizer) ===")
    for r in analysis["ranking"]:
        dflt = (
            " (default)"
            if r["optimizer"] == "ema_flip" and r.get("model") == "bit_mlp"
            else ""
        )
        cache = " [cache]" if r.get("from_cache") else " [trained]"
        print(
            f"  {r['rank']:2d}. {r.get('model', '?'):12s} {r['optimizer']:20s}"
            f"{dflt}{cache}  "
            f"best={r['best_test_acc_mean']:.4f} final={r['final_test_acc_mean']:.4f}"
        )


if __name__ == "__main__":
    main()
