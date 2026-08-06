#!/usr/bin/env python3
"""WP-U4: 1D sum-encoder ablations for Unary link Swarm."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, List

import torch

_THIS = Path(__file__).resolve().parent
_REPO = _THIS.parents[1]
_V08 = _REPO / "experiments" / "v0_8_unary_link"
sys.path.insert(0, str(_REPO))
sys.path.insert(0, str(_V08))

from binary_optimizers.data.mnist import make_mnist_loaders  # noqa: E402
from binary_optimizers.training.budget import add_budget_args, budget_from_args  # noqa: E402
from runner import train_run  # noqa: E402

from encoder_grid import DEFAULT_ENCODERS  # noqa: E402

EXPERIMENT_ID = "v0_11_unary_encoder"


def main() -> None:
    parser = argparse.ArgumentParser(description="v0_11 unary encoder atlas")
    add_budget_args(parser)
    parser.add_argument("--hidden", type=int, default=128)
    parser.add_argument("--swarm-size", type=int, default=256)
    parser.add_argument(
        "--encoders",
        type=str,
        default=",".join(DEFAULT_ENCODERS),
    )
    parser.add_argument("--opt", default="sgd", choices=("sgd", "sgd_m", "adam"))
    parser.add_argument("--lr", type=float, default=0.1)
    parser.add_argument("--decoder", default="density")
    parser.add_argument("--alpha", type=float, default=10.0)
    parser.add_argument("--p-max", type=float, default=0.25)
    parser.add_argument("--p-noise", type=float, default=0.001)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--data-root", type=str, default=None)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--results-dir", type=str, default=None)
    args = parser.parse_args()

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    data_root = args.data_root or str(_REPO / "data")
    results_dir = Path(args.results_dir or (_REPO / "results" / EXPERIMENT_ID))
    budget = budget_from_args(args)
    encoders = [e.strip() for e in args.encoders.split(",") if e.strip()]

    lr = args.lr
    if args.opt == "adam" and abs(lr - 0.1) < 1e-12:
        lr = 1e-3

    train_loader, test_loader = make_mnist_loaders(
        root=data_root,
        batch_size_train=args.batch_size,
        batch_size_test=1000,
        num_workers=0,
        pin_memory=device != "cpu",
    )

    ranking: list[dict[str, Any]] = []
    for enc in encoders:
        result = train_run(
            experiment_id=EXPERIMENT_ID,
            budget=budget,
            hidden=args.hidden,
            swarm_size=args.swarm_size,
            encoder=enc,
            tanh_tau=0.0,
            opt_name=args.opt,
            lr=lr,
            momentum=0.9,
            decoder=args.decoder,
            alpha=args.alpha,
            p_min=0.0,
            p_max=args.p_max,
            p_noise=args.p_noise,
            threshold=1e-3,
            freeze_swarm=False,
            ln_mode="none",
            activation="relu",
            ln_lr=1e-2,
            seed=args.seed,
            device=device,
            train_loader=train_loader,
            test_loader=test_loader,
            results_dir=results_dir,
            run_tag=f"enc_{enc}",
        )
        ranking.append(
            {
                "encoder": enc,
                "best_test_acc": result["best_test_acc"],
                "best_epoch": result["best_epoch"],
                "wall_sec": result["wall_sec"],
                "json_path": result["json_path"],
            }
        )

    ranking.sort(key=lambda r: -float(r["best_test_acc"] or 0.0))
    results_dir.mkdir(parents=True, exist_ok=True)
    summary_path = results_dir / f"summary_seed{args.seed}.json"
    with open(summary_path, "w") as f:
        json.dump(
            {
                "experiment": EXPERIMENT_ID,
                "seed": args.seed,
                "S": args.swarm_size,
                "recipe": {
                    "opt": args.opt,
                    "lr": lr,
                    "decoder": args.decoder,
                },
                "ranking": ranking,
            },
            f,
            indent=2,
        )

    print("\n===== v0_11 encoder ranking =====")
    for row in ranking:
        print(f"  {row['encoder']:12s}  best_test={row['best_test_acc']:.4f}")
    print(f"Summary: {summary_path}")


if __name__ == "__main__":
    main()
