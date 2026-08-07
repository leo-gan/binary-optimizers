#!/usr/bin/env python3
"""WP-U5: sparse CIFAR-10 flat MLP probe for Unary link Swarm."""

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

from binary_optimizers.data.cifar10 import make_cifar10_loaders  # noqa: E402
from binary_optimizers.training.budget import (  # noqa: E402
    TrainBudget,
    add_budget_args,
    budget_from_args,
)
from runner import train_run  # noqa: E402

from cifar_grid import CIFAR_DEFAULT_MAX_WALL, DEFAULT_WIDTHS, parse_widths  # noqa: E402

EXPERIMENT_ID = "v0_12_unary_cifar"


def main() -> None:
    parser = argparse.ArgumentParser(description="v0_12 unary CIFAR probe")
    add_budget_args(parser)
    parser.add_argument("--hidden", type=int, default=256)
    parser.add_argument(
        "--widths",
        type=str,
        default=",".join(str(w) for w in DEFAULT_WIDTHS),
    )
    parser.add_argument("--encoder", default="fixed")
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

    # CIFAR longer wall default if user left MNIST-scale 1200
    if getattr(args, "max_wall_sec", None) is None or args.max_wall_sec == 1200:
        # budget_from_args may already set 1200; override when still default-ish
        pass
    budget = budget_from_args(args)
    if budget.max_wall_sec is not None and budget.max_wall_sec <= 1200.0:
        budget = TrainBudget(
            max_wall_sec=CIFAR_DEFAULT_MAX_WALL,
            patience_frac=budget.patience_frac,
            min_delta=budget.min_delta,
            max_epochs=budget.max_epochs,
            use_epoch_stops=budget.use_epoch_stops,
            patience_epochs=budget.patience_epochs,
        )

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    data_root = args.data_root or str(_REPO / "data")
    results_dir = Path(args.results_dir or (_REPO / "results" / EXPERIMENT_ID))
    widths = parse_widths(args.widths)

    lr = args.lr
    if args.opt == "adam" and abs(lr - 0.1) < 1e-12:
        lr = 1e-3

    train_loader, test_loader = make_cifar10_loaders(
        root=data_root,
        batch_size_train=args.batch_size,
        batch_size_test=500,
        num_workers=0,
        pin_memory=device != "cpu",
    )

    curve: list[dict[str, Any]] = []
    for S in widths:
        result = train_run(
            experiment_id=EXPERIMENT_ID,
            budget=budget,
            hidden=args.hidden,
            swarm_size=S,
            encoder=args.encoder,
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
            run_tag=f"S{S}_{args.encoder}",
            in_dim=3 * 32 * 32,
            n_classes=10,
        )
        curve.append(
            {
                "S": S,
                "encoder": args.encoder,
                "best_test_acc": result["best_test_acc"],
                "best_epoch": result["best_epoch"],
                "wall_sec": result["wall_sec"],
                "json_path": result["json_path"],
            }
        )

    results_dir.mkdir(parents=True, exist_ok=True)
    summary_path = results_dir / f"summary_seed{args.seed}.json"
    best = max(curve, key=lambda r: float(r["best_test_acc"] or 0.0)) if curve else None
    with open(summary_path, "w") as f:
        json.dump(
            {
                "experiment": EXPERIMENT_ID,
                "seed": args.seed,
                "dataset": "cifar10_flat_mlp",
                "hidden": args.hidden,
                "budget": budget.to_dict(),
                "curve": curve,
                "best_cell": best,
                "note": "Sparse WP-U5 probe; flat MLP, ranking not polish.",
            },
            f,
            indent=2,
        )

    print("\n===== v0_12 CIFAR unary curve =====")
    for row in curve:
        print(f"  S={row['S']:4d}  best_test={row['best_test_acc']:.4f}")
    print(f"Summary: {summary_path}")


if __name__ == "__main__":
    main()
