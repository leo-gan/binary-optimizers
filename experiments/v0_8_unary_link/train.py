#!/usr/bin/env python3
"""CLI for experiment v0_8 Unary link Swarm (WP-U1)."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import List, Optional

import torch

_THIS = Path(__file__).resolve().parent
_REPO = _THIS.parents[1]
sys.path.insert(0, str(_REPO))
sys.path.insert(0, str(_THIS))

from binary_optimizers.data.mnist import make_mnist_loaders  # noqa: E402
from binary_optimizers.training.budget import add_budget_args, budget_from_args  # noqa: E402

from runner import EXPERIMENT_ID, train_run  # noqa: E402


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="v0_8 Unary link Swarm training")
    add_budget_args(p)
    p.add_argument("--hidden", type=int, default=128)
    p.add_argument("--swarm-size", type=int, default=256)
    p.add_argument(
        "--encoder",
        choices=("fixed", "tanh", "signed_sqrt", "majority"),
        default="fixed",
    )
    p.add_argument("--tanh-tau", type=float, default=0.0, help="0 → S/2 for tanh")
    p.add_argument("--opt", choices=("sgd", "sgd_m", "adam"), default="sgd")
    p.add_argument("--lr", type=float, default=0.1)
    p.add_argument("--momentum", type=float, default=0.9)
    p.add_argument(
        "--decoder",
        choices=("density", "thresholded", "sign_noise"),
        default="density",
    )
    p.add_argument("--alpha", type=float, default=10.0)
    p.add_argument("--p-min", type=float, default=0.0)
    p.add_argument("--p-max", type=float, default=0.25)
    p.add_argument("--p-noise", type=float, default=0.001)
    p.add_argument("--threshold", type=float, default=1e-3)
    p.add_argument(
        "--freeze-swarm",
        action="store_true",
        help="Baseline: no XOR updates (frozen random swarm)",
    )
    p.add_argument("--ln-mode", choices=("none", "no_affine", "affine"), default="none")
    p.add_argument("--activation", choices=("relu", "squared_relu"), default="relu")
    p.add_argument("--ln-lr", type=float, default=1e-2)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--data-root", type=str, default=None)
    p.add_argument("--device", type=str, default=None)
    p.add_argument("--results-dir", type=str, default=None)
    p.add_argument("--run-tag", type=str, default="default")
    p.add_argument("--experiment-id", type=str, default=EXPERIMENT_ID)
    return p


def main(argv: Optional[List[str]] = None) -> None:
    args = build_arg_parser().parse_args(argv)
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    data_root = args.data_root or str(_REPO / "data")
    results_dir = Path(args.results_dir or (_REPO / "results" / args.experiment_id))
    budget = budget_from_args(args)

    train_loader, test_loader = make_mnist_loaders(
        root=data_root,
        batch_size_train=args.batch_size,
        batch_size_test=1000,
        num_workers=0,
        pin_memory=device != "cpu",
    )

    train_run(
        experiment_id=args.experiment_id,
        budget=budget,
        hidden=args.hidden,
        swarm_size=args.swarm_size,
        encoder=args.encoder,
        tanh_tau=args.tanh_tau,
        opt_name=args.opt,
        lr=args.lr,
        momentum=args.momentum,
        decoder=args.decoder,
        alpha=args.alpha,
        p_min=args.p_min,
        p_max=args.p_max,
        p_noise=args.p_noise,
        threshold=args.threshold,
        freeze_swarm=args.freeze_swarm,
        ln_mode=args.ln_mode,
        activation=args.activation,
        ln_lr=args.ln_lr,
        seed=args.seed,
        device=device,
        train_loader=train_loader,
        test_loader=test_loader,
        results_dir=results_dir,
        run_tag=args.run_tag,
    )


if __name__ == "__main__":
    main()
