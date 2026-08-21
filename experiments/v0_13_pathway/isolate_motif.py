#!/usr/bin/env python3
"""Isolation panel: motif vs L1-only vs LayerNorm, seeds {42, 0, 1}.

Pass rule (pre-registered in RESULTS.md §5):

- mean(pathway) - mean(scale_only) >= 0.02
- mean(pathway) - mean(layernorm) >= 0.02
- pathway > scale_only on at least 2 of 3 seeds
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

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

ARMS: tuple[Tuple[str, str, float], ...] = (
    ("layernorm", "layernorm", 1.0),
    ("scale_only", "pathway", 0.0),
    ("pathway", "pathway", 1.0),
)
DEFAULT_SEEDS: tuple[int, ...] = (42, 0, 1)
PASS_DELTA = 0.02

# Seed 42 already ran under the isolation protocol (full wall, tracking fix).
REUSE_SEED42: Dict[str, str] = {
    "layernorm": "cmp_layernorm_seed42.json",
    "scale_only": "cmp_pathway_l1_seed42.json",
    "pathway": "cmp_pathway_fix_seed42.json",
}


def evaluate_isolation(
    cells: Sequence[Dict[str, Any]],
    *,
    pass_delta: float = PASS_DELTA,
) -> Dict[str, Any]:
    """cells: [{seed, arm, best_test_acc}, ...]"""
    by_arm: Dict[str, List[float]] = {a[0]: [] for a in ARMS}
    by_seed: Dict[int, Dict[str, float]] = {}
    for c in cells:
        arm = str(c["arm"])
        seed = int(c["seed"])
        acc = float(c["best_test_acc"])
        by_arm.setdefault(arm, []).append(acc)
        by_seed.setdefault(seed, {})[arm] = acc

    means = {arm: (sum(v) / len(v) if v else None) for arm, v in by_arm.items()}
    d_scale = None
    d_ln = None
    if means["pathway"] is not None and means["scale_only"] is not None:
        d_scale = means["pathway"] - means["scale_only"]
    if means["pathway"] is not None and means["layernorm"] is not None:
        d_ln = means["pathway"] - means["layernorm"]

    seed_motif_wins = 0
    seed_signs: Dict[int, Optional[bool]] = {}
    for seed, accs in sorted(by_seed.items()):
        if "pathway" in accs and "scale_only" in accs:
            win = accs["pathway"] > accs["scale_only"]
            seed_signs[seed] = win
            if win:
                seed_motif_wins += 1
        else:
            seed_signs[seed] = None

    n_compared = sum(1 for v in seed_signs.values() if v is not None)
    passed = (
        n_compared >= 3
        and d_scale is not None
        and d_ln is not None
        and d_scale >= pass_delta
        and d_ln >= pass_delta
        and seed_motif_wins >= 2
    )
    return {
        "cells": list(cells),
        "means": means,
        "delta_pathway_minus_scale_only": d_scale,
        "delta_pathway_minus_layernorm": d_ln,
        "seed_motif_gt_scale": seed_signs,
        "seed_motif_wins": seed_motif_wins,
        "n_seeds_compared": n_compared,
        "pass_delta": pass_delta,
        "passed": passed,
    }


def _fmt(x: Optional[float]) -> str:
    return f"{x:.4f}" if x is not None else "—"


def format_markdown(verdict: Dict[str, Any]) -> str:
    lines = [
        "# Motif isolation (PathwayNorm vs L1 vs LayerNorm)",
        "",
        "| seed | layernorm | scale_only | pathway | pathway − scale | pathway − LN |",
        "|-----:|----------:|-----------:|--------:|----------------:|-------------:|",
    ]
    by_seed: Dict[int, Dict[str, float]] = {}
    for c in verdict["cells"]:
        by_seed.setdefault(int(c["seed"]), {})[str(c["arm"])] = float(c["best_test_acc"])
    for seed in sorted(by_seed):
        a = by_seed[seed]
        ln = a.get("layernorm")
        sc = a.get("scale_only")
        pw = a.get("pathway")
        d_sc = (pw - sc) if pw is not None and sc is not None else None
        d_ln = (pw - ln) if pw is not None and ln is not None else None
        lines.append(
            f"| {seed} | {_fmt(ln)} | {_fmt(sc)} | {_fmt(pw)} | "
            f"{_fmt(d_sc)} | {_fmt(d_ln)} |"
        )
    means = verdict["means"]
    lines.append(
        f"| **mean** | {_fmt(means.get('layernorm'))} | "
        f"{_fmt(means.get('scale_only'))} | {_fmt(means.get('pathway'))} | "
        f"{_fmt(verdict.get('delta_pathway_minus_scale_only'))} | "
        f"{_fmt(verdict.get('delta_pathway_minus_layernorm'))} |"
    )
    lines.append("")
    status = "PASS" if verdict["passed"] else "FAIL"
    lines.append(
        f"**Verdict: {status}** "
        f"(need Δscale ≥ {verdict['pass_delta']:.2f}, "
        f"ΔLN ≥ {verdict['pass_delta']:.2f}, "
        f"motif>L1 on ≥2 seeds; "
        f"motif>L1 on {verdict['seed_motif_wins']}/{verdict['n_seeds_compared']} seeds)."
    )
    lines.append("")
    return "\n".join(lines)


def _load_run(path: Path) -> JsonRun:
    with open(path) as f:
        return json.load(f)


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="v0_13 motif isolation panel")
    add_budget_args(p, patience_frac=1.0)
    p.add_argument("--hidden", type=int, default=128)
    p.add_argument("--candidates", type=int, default=8)
    p.add_argument("--lambda-a", type=float, default=0.1)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--data-root", type=str, default=None)
    p.add_argument("--device", type=str, default=None)
    p.add_argument("--results-dir", type=str, default=None)
    p.add_argument("--experiment-id", type=str, default=EXPERIMENT_ID)
    p.add_argument("--seeds", type=str, default="42,0,1")
    p.add_argument(
        "--no-reuse-seed42",
        action="store_true",
        help="Retrain seed 42 instead of loading cmp_* JSONs",
    )
    return p


def main(argv: Optional[List[str]] = None) -> None:
    args = build_arg_parser().parse_args(argv)
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    data_root = args.data_root or str(_REPO / "data")
    results_dir = Path(args.results_dir or (_REPO / "results" / args.experiment_id))
    budget = budget_from_args(args)
    seeds = tuple(int(s) for s in args.seeds.split(",") if s.strip())

    train_loader = test_loader = None
    cells: List[Dict[str, Any]] = []

    for seed in seeds:
        for arm, norm, strength in ARMS:
            reuse_name = REUSE_SEED42.get(arm) if seed == 42 and not args.no_reuse_seed42 else None
            reuse_path = results_dir / reuse_name if reuse_name else None
            if reuse_path is not None and reuse_path.is_file():
                run = _load_run(reuse_path)
                print(f"Reuse {reuse_path.name} as seed={seed} arm={arm}", flush=True)
            else:
                if train_loader is None:
                    train_loader, test_loader = make_mnist_loaders(
                        root=data_root,
                        batch_size_train=args.batch_size,
                        batch_size_test=1000,
                        num_workers=0,
                        pin_memory=device != "cpu",
                    )
                tag = f"iso_{arm}_s{seed}"
                print(f"\n######## isolate seed={seed} arm={arm} tag={tag} ########", flush=True)
                run = train_run(
                    experiment_id=args.experiment_id,
                    budget=budget,
                    hidden=args.hidden,
                    mode="pathway",
                    candidates=args.candidates,
                    lambda_a=args.lambda_a,
                    ranking="ste_topk",
                    assembly_of="both",
                    bn_lr=1e-2,
                    use_bn=False,
                    norm=norm,
                    norm_strength=strength,
                    seed=seed,
                    device=device,
                    train_loader=train_loader,
                    test_loader=test_loader,
                    results_dir=results_dir,
                    run_tag=tag,
                )
            cells.append(
                {
                    "seed": seed,
                    "arm": arm,
                    "best_test_acc": run["best_test_acc"],
                    "best_epoch": run.get("best_epoch"),
                    "run_tag": run.get("run_tag"),
                    "reused": bool(reuse_path is not None and reuse_path.is_file()),
                }
            )

    verdict = evaluate_isolation(cells)
    results_dir.mkdir(parents=True, exist_ok=True)
    json_path = results_dir / "isolate_motif_summary.json"
    md_path = results_dir / "isolate_motif_summary.md"
    with open(json_path, "w") as f:
        json.dump(verdict, f, indent=2)
    md_path.write_text(format_markdown(verdict), encoding="utf-8")
    print(format_markdown(verdict), flush=True)
    print(f"Saved {json_path}", flush=True)
    print(f"Saved {md_path}", flush=True)
    if not verdict["passed"]:
        print("ISOLATION CLAIM: FAIL", flush=True)
    else:
        print("ISOLATION CLAIM: PASS", flush=True)


if __name__ == "__main__":
    main()
