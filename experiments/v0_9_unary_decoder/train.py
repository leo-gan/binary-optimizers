#!/usr/bin/env python3
"""WP-U2: sparse optimizer × decoder ablations for Unary link Swarm."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import torch

_THIS = Path(__file__).resolve().parent
_REPO = _THIS.parents[1]
_V08 = _REPO / "experiments" / "v0_8_unary_link"
sys.path.insert(0, str(_REPO))
sys.path.insert(0, str(_V08))

from binary_optimizers.data.mnist import make_mnist_loaders  # noqa: E402
from binary_optimizers.training.budget import add_budget_args, budget_from_args  # noqa: E402
from runner import train_run  # noqa: E402  # v0_8 runner

from ablation_grid import default_cells  # noqa: E402

EXPERIMENT_ID = "v0_9_unary_decoder"


def main() -> None:
    parser = argparse.ArgumentParser(description="v0_9 unary decoder ablations")
    add_budget_args(parser)
    parser.add_argument("--hidden", type=int, default=128)
    parser.add_argument("--swarm-size", type=int, default=256)
    parser.add_argument("--alpha", type=float, default=10.0)
    parser.add_argument("--p-max", type=float, default=0.25)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--data-root", type=str, default=None)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--results-dir", type=str, default=None)
    parser.add_argument(
        "--cells",
        type=str,
        default="all",
        help="Comma tags or 'all' (default sparse grid)",
    )
    args = parser.parse_args()

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    data_root = args.data_root or str(_REPO / "data")
    results_dir = Path(args.results_dir or (_REPO / "results" / EXPERIMENT_ID))
    budget = budget_from_args(args)

    train_loader, test_loader = make_mnist_loaders(
        root=data_root,
        batch_size_train=args.batch_size,
        batch_size_test=1000,
        num_workers=0,
        pin_memory=device != "cpu",
    )

    cells = default_cells()
    if args.cells != "all":
        want = {t.strip() for t in args.cells.split(",") if t.strip()}
        cells = [c for c in cells if c.tag() in want]
        if not cells:
            raise SystemExit(f"no cells matched {want}")

    summaries: list[dict[str, Any]] = []
    for cell in cells:
        result = train_run(
            experiment_id=EXPERIMENT_ID,
            budget=budget,
            hidden=args.hidden,
            swarm_size=args.swarm_size,
            encoder="fixed",
            tanh_tau=0.0,
            opt_name=cell.opt,
            lr=cell.lr,
            momentum=0.9,
            decoder=cell.decoder,
            alpha=args.alpha,
            p_min=0.0,
            p_max=args.p_max,
            p_noise=cell.p_noise,
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
            run_tag=cell.tag(),
        )
        summaries.append(
            {
                "tag": cell.tag(),
                "opt": cell.opt,
                "decoder": cell.decoder,
                "p_noise": cell.p_noise,
                "lr": cell.lr,
                "best_test_acc": result["best_test_acc"],
                "best_epoch": result["best_epoch"],
                "wall_sec": result["wall_sec"],
                "json_path": result["json_path"],
            }
        )

    summaries.sort(key=lambda r: -float(r["best_test_acc"] or 0.0))
    summary_path = results_dir / f"summary_seed{args.seed}.json"
    results_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "experiment": EXPERIMENT_ID,
        "seed": args.seed,
        "default_recipe_hint": (
            "Prefer top-ranked density cell; lock after inspecting this summary."
        ),
        "ranking": summaries,
    }
    with open(summary_path, "w") as f:
        json.dump(payload, f, indent=2)

    print("\n===== v0_9 ranking (best test acc) =====")
    for s in summaries:
        print(
            f"  {s['tag']:28s}  best={s['best_test_acc']:.4f}  "
            f"@ep {s['best_epoch']}  ({s['wall_sec']:.0f}s)"
        )
    print(f"Summary: {summary_path}")


if __name__ == "__main__":
    main()
