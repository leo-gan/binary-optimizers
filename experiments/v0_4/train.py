#!/usr/bin/env python3
"""Experiment v0_4: balanced ternary place-value Swarm on MNIST."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict

import torch
import torch.nn.functional as F

_THIS_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _THIS_DIR.parents[1]
sys.path.insert(0, str(_REPO_ROOT))
sys.path.insert(0, str(_THIS_DIR))

from binary_optimizers.data.mnist import make_mnist_loaders
from binary_optimizers.store import soft_record_completed_run
from binary_optimizers.training.budget import (
    EarlyStopTracker,
    add_budget_args,
    budget_from_args,
)
from binary_optimizers.training.loops import set_seed

from metrics import swarm_stats
from model import BitNetTernaryPlaceMLP, LNMode
from optimizer import SwarmOptimizerV04

LN_MODES: tuple[LNMode, ...] = ("none", "no_affine", "affine")


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    total = correct = 0
    loss_sum = 0.0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        logits = model(x)
        loss_sum += F.cross_entropy(logits, y, reduction="sum").item()
        correct += (logits.argmax(1) == y).sum().item()
        total += y.size(0)
    return correct / max(1, total), loss_sum / max(1, total)


def train_one_epoch(model, opt, loader, device):
    model.train()
    total = correct = 0
    loss_sum = flip_sum = 0.0
    n_steps = 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        opt.zero_grad()
        logits = model(x)
        loss = F.cross_entropy(logits, y)
        loss.backward()
        flip = opt.step()
        loss_sum += loss.item() * y.size(0)
        correct += (logits.argmax(1) == y).sum().item()
        total += y.size(0)
        flip_sum += flip
        n_steps += 1
    return correct / max(1, total), loss_sum / max(1, total), flip_sum / max(1, n_steps)


def train_one_mode(**kw) -> Dict[str, Any]:
    ln_mode = kw["ln_mode"]
    set_seed(kw["seed"])
    model = BitNetTernaryPlaceMLP(
        hidden_dim=kw["hidden"],
        n_trits=kw["n_trits"],
        ln_mode=ln_mode,
        activation=kw["activation"],
    ).to(kw["device"])
    opt = SwarmOptimizerV04(
        model.swarm_layers(),
        recruit_rate=kw["recruit_rate"],
        max_step_prob=kw["max_step_prob"],
        grad_momentum=kw["grad_momentum"],
        max_step=kw.get("max_step", 64),
        step_scale=kw.get("step_scale", 1e6),
        ln_params=model.ln_parameters(),
        ln_lr=kw["ln_lr"],
    )
    history = []
    best_state = None
    budget = kw["budget"]
    tracker = EarlyStopTracker(budget)
    print(
        f"\n===== v0_4 | ln_mode={ln_mode} | n_trits={kw['n_trits']} | "
        f"seed={kw['seed']} =====",
        flush=True,
    )
    for epoch in range(1, budget.max_epochs + 1):
        t0 = time.time()
        tr_acc, tr_loss, flip = train_one_epoch(
            model, opt, kw["train_loader"], kw["device"]
        )
        te_acc, te_loss = evaluate(model, kw["test_loader"], kw["device"])
        model.assert_ternary_invariants()
        stats = swarm_stats(model)
        row = {
            "epoch": epoch,
            "train_acc": tr_acc,
            "train_loss": tr_loss,
            "test_acc": te_acc,
            "test_loss": te_loss,
            "flip_frac": flip,
            "step_frac": opt.last_step_frac,
            "zero_frac": stats["zero_frac"],
            "mean_abs_w": stats["mean_abs_w"],
            "epoch_sec": time.time() - t0,
        }
        history.append(row)
        decision = tracker.observe(epoch, te_acc)
        if decision.improved:
            best_state = {
                k: v.detach().cpu().clone() for k, v in model.state_dict().items()
            }
        print(
            f"epoch {epoch:03d}/{budget.max_epochs} | train={tr_acc:.4f} loss={tr_loss:.4f} | "
            f"test={te_acc:.4f} loss={te_loss:.4f} | flip={flip:.4f} step={opt.last_step_frac:.4f} "
            f"zero={stats['zero_frac']:.3f} | {tracker.status_str()} | {row['epoch_sec']:.1f}s",
            flush=True,
        )
        if decision.stop:
            print(f"Stop: {decision.reason}", flush=True)
            break
    if best_state is not None:
        model.load_state_dict(best_state)
        model.to(kw["device"])
    final_acc, final_loss = evaluate(model, kw["test_loader"], kw["device"])
    out = {
        "experiment": "v0_4",
        "coding": "balanced_ternary_place",
        "ln_mode": ln_mode,
        "seed": kw["seed"],
        "hidden": kw["hidden"],
        "n_trits": kw["n_trits"],
        "recruit_rate": kw["recruit_rate"],
        "max_step_prob": kw["max_step_prob"],
        "grad_momentum": kw["grad_momentum"],
        "activation": kw["activation"],
        "device": kw["device"],
        "epochs_ran": len(history),
        "budget": budget.to_dict(),
        "stop_meta": tracker.meta_dict(),
        "best_test_acc": tracker.best,
        "best_epoch": tracker.best_epoch,
        "final_test_acc": final_acc,
        "final_test_loss": final_loss,
        "wall_sec": tracker.wall_sec,
        "history": history,
        "baseline_v0_2_best": 0.9365,
        "baseline_v0_3_note": "see results/v0_3",
        "swarm_stats_final": swarm_stats(model),
    }
    rd = kw["results_dir"]
    rd.mkdir(parents=True, exist_ok=True)
    jp = rd / f"ln_{ln_mode}_seed{kw['seed']}.json"
    with open(jp, "w") as f:
        json.dump(out, f, indent=2)
    ck = _REPO_ROOT / "checkpoints" / "v0_4"
    ck.mkdir(parents=True, exist_ok=True)
    ck_path = ck / f"ln_{ln_mode}_seed{kw['seed']}.pt"
    torch.save(
        {
            "model_state": model.state_dict(),
            "meta": {k: v for k, v in out.items() if k != "history"},
        },
        ck_path,
    )
    out["json_path"] = str(jp)
    print(f"Saved {jp}", flush=True)

    # Dual-write to DuckDB experiment store (soft-fail: never abort training).
    config = {
        "coding": out["coding"],
        "ln_mode": ln_mode,
        "hidden": out["hidden"],
        "n_trits": out["n_trits"],
        "recruit_rate": out["recruit_rate"],
        "max_step_prob": out["max_step_prob"],
        "grad_momentum": out["grad_momentum"],
        "activation": out["activation"],
        "device": out["device"],
        "max_step": kw.get("max_step", 64),
        "step_scale": kw.get("step_scale", 1e6),
        "ln_lr": kw.get("ln_lr"),
        "epochs": kw.get("epochs"),
        "patience": kw.get("patience"),
        "min_delta": kw.get("min_delta"),
    }
    summary = {
        "epochs_ran": out["epochs_ran"],
        "baseline_v0_2_best": out.get("baseline_v0_2_best"),
        "baseline_v0_3_note": out.get("baseline_v0_3_note"),
        "swarm_stats_final": out.get("swarm_stats_final"),
        "source_json": str(jp),
    }
    run_id = soft_record_completed_run(
        experiment="v0_4",
        name=f"ln_{ln_mode}",
        config=config,
        history=history,
        seed=kw["seed"],
        wall_sec=out["wall_sec"],
        best_test_acc=out["best_test_acc"],
        best_epoch=out["best_epoch"],
        final_test_acc=out["final_test_acc"],
        final_test_loss=out["final_test_loss"],
        summary=summary,
        checkpoint_path=ck_path,
    )
    if run_id:
        out["run_id"] = run_id
        print(f"Stored run_id={run_id}", flush=True)
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ln-mode", choices=["all", *LN_MODES], default="all")
    add_budget_args(p)
    p.add_argument("--hidden", type=int, default=128)
    p.add_argument("--n-trits", type=int, default=10)
    p.add_argument("--recruit-rate", type=float, default=1e4)
    p.add_argument("--max-step-prob", type=float, default=0.5)
    p.add_argument("--max-step", type=int, default=64)
    p.add_argument("--step-scale", type=float, default=1e6)
    p.add_argument("--grad-momentum", type=float, default=0.9)
    p.add_argument("--activation", choices=("relu", "squared_relu"), default="relu")
    p.add_argument("--ln-lr", type=float, default=1e-2)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--data-root", type=str, default=None)
    p.add_argument("--device", type=str, default=None)
    p.add_argument("--results-dir", type=str, default=None)
    args = p.parse_args()
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    data_root = args.data_root or str(_REPO_ROOT / "data")
    results_dir = Path(args.results_dir or (_REPO_ROOT / "results" / "v0_4"))
    train_loader, test_loader = make_mnist_loaders(
        root=data_root,
        batch_size_train=args.batch_size,
        batch_size_test=1000,
        num_workers=0,
    )
    modes = list(LN_MODES) if args.ln_mode == "all" else [args.ln_mode]
    budget = budget_from_args(args)
    summaries = []
    for mode in modes:
        r = train_one_mode(
            ln_mode=mode,
            budget=budget,
            hidden=args.hidden,
            n_trits=args.n_trits,
            recruit_rate=args.recruit_rate,
            max_step_prob=args.max_step_prob,
            max_step=args.max_step,
            step_scale=args.step_scale,
            grad_momentum=args.grad_momentum,
            activation=args.activation,
            ln_lr=args.ln_lr,
            seed=args.seed,
            device=device,
            train_loader=train_loader,
            test_loader=test_loader,
            results_dir=results_dir,
        )
        summaries.append(
            {
                "ln_mode": mode,
                "best_test_acc": r["best_test_acc"],
                "best_epoch": r["best_epoch"],
                "epochs_ran": r["epochs_ran"],
                "wall_sec": r["wall_sec"],
            }
        )
    sp = results_dir / f"summary_seed{args.seed}.json"
    results_dir.mkdir(parents=True, exist_ok=True)
    with open(sp, "w") as f:
        json.dump({"experiment": "v0_4", "seed": args.seed, "runs": summaries}, f, indent=2)
    print("\n===== v0_4 summary =====", flush=True)
    for s in summaries:
        print(
            f"  ln={s['ln_mode']:10s} best_test={s['best_test_acc']:.4f} "
            f"@ {s['best_epoch']} ({s['epochs_ran']} ep)",
            flush=True,
        )


if __name__ == "__main__":
    main()
