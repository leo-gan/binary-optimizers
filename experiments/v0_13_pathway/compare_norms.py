#!/usr/bin/env python3
"""Matched-wall comparison: PathwayNorm vs LayerNorm (no affine).

Same optimizer, data, seed, and ``pure_wall_budget_v1``. Patience is the
full wall so one arm cannot be cut after a 4-epoch test stall.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import torch

JsonRun = Dict[str, Any]

_THIS = Path(__file__).resolve().parent
_REPO = _THIS.parents[1]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))
if str(_THIS) not in sys.path:
    sys.path.insert(0, str(_THIS))

from binary_optimizers.data.mnist import make_mnist_loaders  # noqa: E402
from binary_optimizers.training.budget import add_budget_args, budget_from_args  # noqa: E402

from train import EXPERIMENT_ID, train_run  # noqa: E402

COMPARE_NORMS: tuple[str, ...] = ("pathway", "layernorm")


def summarize(runs: Sequence[JsonRun]) -> Dict[str, Any]:
    rows = []
    for r in runs:
        rows.append(
            {
                "norm": r.get("norm"),
                "run_tag": r.get("run_tag"),
                "best_test_acc": r.get("best_test_acc"),
                "best_epoch": r.get("best_epoch"),
                "final_test_acc": r.get("final_test_acc"),
                "final_test_loss": r.get("final_test_loss"),
                "epochs_ran": r.get("epochs_ran"),
                "wall_sec": r.get("wall_sec"),
            }
        )
    best = max(rows, key=lambda x: float(x["best_test_acc"] or 0.0))
    return {
        "arms": rows,
        "winner_norm": best["norm"],
        "winner_acc": best["best_test_acc"],
    }


def format_markdown(summary: Dict[str, Any]) -> str:
    lines = [
        "# PathwayNorm vs LayerNorm",
        "",
        "| norm | best test | best ep | epochs | wall s | final test | final loss |",
        "|------|----------:|--------:|-------:|-------:|-----------:|-----------:|",
    ]
    for a in summary["arms"]:
        lines.append(
            f"| `{a['norm']}` | {a['best_test_acc']:.4f} | {a['best_epoch']} | "
            f"{a['epochs_ran']} | {a['wall_sec']:.0f} | "
            f"{a['final_test_acc']:.4f} | {a['final_test_loss']:.3f} |"
        )
    lines.append("")
    lines.append(
        f"**Winner (best test):** `{summary['winner_norm']}` "
        f"({summary['winner_acc']:.4f})."
    )
    lines.append("")
    return "\n".join(lines)


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="v0_13 PathwayNorm vs LayerNorm")
    add_budget_args(p, patience_frac=1.0)
    p.add_argument("--hidden", type=int, default=128)
    p.add_argument("--mode", choices=("pathway", "hybrid"), default="pathway")
    p.add_argument("--candidates", type=int, default=8)
    p.add_argument("--lambda-a", type=float, default=0.1)
    p.add_argument("--ranking", choices=("ste_topk", "random"), default="ste_topk")
    p.add_argument(
        "--assembly-of",
        choices=("weights", "logits", "both"),
        default="both",
    )
    p.add_argument("--bn-lr", type=float, default=1e-2)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--data-root", type=str, default=None)
    p.add_argument("--device", type=str, default=None)
    p.add_argument("--results-dir", type=str, default=None)
    p.add_argument("--experiment-id", type=str, default=EXPERIMENT_ID)
    p.add_argument(
        "--norms",
        type=str,
        default=",".join(COMPARE_NORMS),
        help="Comma-separated norms (default: pathway,layernorm)",
    )
    return p


def main(argv: Optional[List[str]] = None) -> None:
    args = build_arg_parser().parse_args(argv)
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    data_root = args.data_root or str(_REPO / "data")
    results_dir = Path(args.results_dir or (_REPO / "results" / args.experiment_id))
    budget = budget_from_args(args)
    norms = tuple(n.strip() for n in args.norms.split(",") if n.strip())
    for n in norms:
        if n not in ("none", "pathway", "layernorm", "bn"):
            raise SystemExit(f"unknown norm {n!r}")

    train_loader, test_loader = make_mnist_loaders(
        root=data_root,
        batch_size_train=args.batch_size,
        batch_size_test=1000,
        num_workers=0,
        pin_memory=device != "cpu",
    )

    runs: List[Dict[str, Any]] = []
    for norm in norms:
        tag = f"cmp_{norm}"
        print(f"\n######## compare arm norm={norm} tag={tag} ########", flush=True)
        out = train_run(
            experiment_id=args.experiment_id,
            budget=budget,
            hidden=args.hidden,
            mode=args.mode,
            candidates=args.candidates,
            lambda_a=args.lambda_a,
            ranking=args.ranking,
            assembly_of=args.assembly_of,
            bn_lr=args.bn_lr,
            use_bn=False,
            norm=norm,
            seed=args.seed,
            device=device,
            train_loader=train_loader,
            test_loader=test_loader,
            results_dir=results_dir,
            run_tag=tag,
        )
        runs.append(out)

    summary = summarize(runs)
    results_dir.mkdir(parents=True, exist_ok=True)
    json_path = results_dir / f"compare_norms_seed{args.seed}.json"
    md_path = results_dir / f"compare_norms_seed{args.seed}.md"
    with open(json_path, "w") as f:
        json.dump(summary, f, indent=2)
    md_path.write_text(format_markdown(summary), encoding="utf-8")
    print(format_markdown(summary), flush=True)
    print(f"Saved {json_path}", flush=True)
    print(f"Saved {md_path}", flush=True)


if __name__ == "__main__":
    main()
