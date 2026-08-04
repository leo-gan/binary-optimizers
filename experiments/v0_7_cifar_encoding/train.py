#!/usr/bin/env python3
"""WP3 sparse CIFAR-10 encoding probe: fixed vs exp_mant:2 at n=8,16.

Pure wall option B: max_wall_sec=2400, patience_frac=0.125.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F

_THIS = Path(__file__).resolve().parent
_REPO = _THIS.parents[1]
sys.path.insert(0, str(_THIS))
sys.path.insert(0, str(_REPO))

from binary_optimizers.data.cifar10 import make_cifar10_loaders  # noqa: E402
from binary_optimizers.store import db_notes, enrich_config, soft_record_completed_run  # noqa: E402
from binary_optimizers.training.budget import (  # noqa: E402
    EarlyStopTracker,
    TrainBudget,
    add_budget_args,
    budget_from_args,
)
from binary_optimizers.training.loops import set_seed  # noqa: E402

from model import BitNetEncodingCIFARMLP  # noqa: E402
from optimizer import SwarmOptimizerV06  # noqa: E402

EXPERIMENT_ID = "v0_7_cifar_encoding"

# CIFAR option B defaults
DEFAULT_MAX_WALL_SEC = 2400.0
DEFAULT_PATIENCE_FRAC = 0.125


def scale_steps_for_mant(n_mant: int, base_max_step: int, base_step_scale: float) -> tuple[int, float]:
    ref_n = 16
    ref_vmax = float(2**ref_n - 1)
    vmax = float(2**n_mant - 1)
    scale = vmax / ref_vmax
    max_step = max(1, int(round(base_max_step * scale)))
    max_step = min(max_step, int(2**62 - 1))
    return max_step, base_step_scale * scale


@dataclass(frozen=True)
class Cell:
    n_bits: int
    encoding: str
    n_exp: int = 0

    def tag(self) -> str:
        if self.encoding == "fixed":
            return f"fixed_n{self.n_bits}"
        return f"{self.encoding}{self.n_exp}_n{self.n_bits}"

    def to_dict(self) -> dict[str, Any]:
        return {
            "n_bits": self.n_bits,
            "encoding": self.encoding,
            "n_exp": self.n_exp,
            "tag": self.tag(),
        }


def default_cells() -> list[Cell]:
    """Max learning / min runs: 4 cells."""
    return [
        Cell(n_bits=8, encoding="fixed", n_exp=0),
        Cell(n_bits=16, encoding="fixed", n_exp=0),
        Cell(n_bits=8, encoding="exp_mant", n_exp=2),
        Cell(n_bits=16, encoding="exp_mant", n_exp=2),
    ]


def parse_cells(s: str) -> list[Cell]:
    out: list[Cell] = []
    for part in s.split(","):
        part = part.strip()
        if not part:
            continue
        if "@" not in part:
            raise ValueError(f"cell needs @n_bits: {part}")
        left, n_s = part.rsplit("@", 1)
        n = int(n_s)
        if left == "fixed":
            out.append(Cell(n_bits=n, encoding="fixed", n_exp=0))
        elif left.startswith("exp_mant:"):
            ne = int(left.split(":")[1])
            out.append(Cell(n_bits=n, encoding="exp_mant", n_exp=ne))
        else:
            raise ValueError(f"unknown cell: {part}")
    return out


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


def run_cell(
    cell: Cell,
    *,
    budget: TrainBudget,
    args,
    device: str,
    train_loader,
    test_loader,
    results_dir: Path,
) -> dict[str, Any]:
    set_seed(args.seed)
    n_mant = cell.n_bits if cell.encoding != "exp_mant" else cell.n_bits - cell.n_exp
    max_step_w, step_scale_w = scale_steps_for_mant(n_mant, args.max_step, args.step_scale)
    model = BitNetEncodingCIFARMLP(
        hidden_dim=args.hidden,
        n_bits=cell.n_bits,
        encoding=cell.encoding,  # type: ignore[arg-type]
        n_exp=cell.n_exp,
        ln_mode=args.ln_mode,  # type: ignore[arg-type]
        activation=args.activation,  # type: ignore[arg-type]
    ).to(device)
    opt = SwarmOptimizerV06(
        model.swarm_layers(),
        recruit_rate=args.recruit_rate,
        max_step_prob=args.max_step_prob,
        max_step=max_step_w,
        step_scale=step_scale_w,
        exp_max_step=args.exp_max_step,
        exp_step_scale=args.exp_step_scale,
        grad_momentum=args.grad_momentum,
        ln_params=model.ln_parameters(),
        ln_lr=args.ln_lr,
    )
    history: list[dict[str, Any]] = []
    best_state = None
    tracker = EarlyStopTracker(budget)
    print(
        f"\n===== {EXPERIMENT_ID} | {cell.tag()} | ln={args.ln_mode} | seed={args.seed} =====",
        flush=True,
    )
    for epoch in range(1, budget.max_epochs + 1):
        t0 = time.time()
        tr_acc, tr_loss, flip = train_one_epoch(model, opt, train_loader, device)
        te_acc, te_loss = evaluate(model, test_loader, device)
        model.assert_binary_invariants()
        stats = model.weight_stats()
        row = {
            "epoch": epoch,
            "train_acc": tr_acc,
            "train_loss": tr_loss,
            "test_acc": te_acc,
            "test_loss": te_loss,
            "flip_frac": flip,
            "step_frac": opt.last_step_frac,
            "epoch_sec": time.time() - t0,
            **{f"stat_{k}": v for k, v in stats.items()},
        }
        history.append(row)
        decision = tracker.observe(epoch, te_acc)
        if decision.improved:
            best_state = {
                k: v.detach().cpu().clone() for k, v in model.state_dict().items()
            }
        print(
            f"  ep {epoch:04d} train={tr_acc:.4f} test={te_acc:.4f} "
            f"{tracker.status_str()} ({row['epoch_sec']:.1f}s)",
            flush=True,
        )
        if decision.stop:
            print(f"  Stop: {decision.reason}", flush=True)
            break
    if best_state is not None:
        model.load_state_dict(best_state)
        model.to(device)
    final_acc, final_loss = evaluate(model, test_loader, device)
    out = {
        "experiment": EXPERIMENT_ID,
        "dataset": "cifar10",
        "net": "flat_mlp_3072_128_10",
        **cell.to_dict(),
        "ln_mode": args.ln_mode,
        "seed": args.seed,
        "hidden": args.hidden,
        "best_test_acc": tracker.best,
        "best_epoch": tracker.best_epoch,
        "epochs_ran": len(history),
        "budget": budget.to_dict(),
        "stop_meta": tracker.meta_dict(),
        "final_test_acc": final_acc,
        "final_test_loss": final_loss,
        "wall_sec": tracker.wall_sec,
        "final_stats": model.weight_stats(),
        "history": history,
        "status": "completed",
        "max_step_used": max_step_w,
        "step_scale_used": step_scale_w,
        "n_mant": n_mant,
    }
    results_dir.mkdir(parents=True, exist_ok=True)
    jp = results_dir / f"{cell.tag()}_ln_{args.ln_mode}_seed{args.seed}.json"
    with open(jp, "w") as f:
        json.dump(out, f, indent=2)
    print(f"  Saved {jp}", flush=True)
    cfg = enrich_config(
        EXPERIMENT_ID,
        {
            **cell.to_dict(),
            "dataset": "cifar10",
            "ln_mode": args.ln_mode,
            "hidden": args.hidden,
            "budget": budget.to_dict(),
        },
    )
    rid = soft_record_completed_run(
        experiment=EXPERIMENT_ID,
        name=cell.tag(),
        config=cfg,
        history=history,
        seed=args.seed,
        wall_sec=out["wall_sec"],
        best_test_acc=out["best_test_acc"],
        best_epoch=out["best_epoch"],
        final_test_acc=final_acc,
        final_test_loss=final_loss,
        summary={"epochs_ran": len(history), "source_json": str(jp)},
        notes=db_notes(EXPERIMENT_ID),
    )
    if rid:
        out["run_id"] = rid
        print(f"  Stored run_id={rid}", flush=True)
    return out


def main() -> None:
    p = argparse.ArgumentParser(description="v0.7 CIFAR encoding probe (4 cells)")
    p.add_argument(
        "--cells",
        type=str,
        default=None,
        help="Default: fixed@8,fixed@16,exp_mant:2@8,exp_mant:2@16",
    )
    p.add_argument("--ln-mode", type=str, default="none")
    add_budget_args(
        p,
        max_wall_sec=DEFAULT_MAX_WALL_SEC,
        patience_frac=DEFAULT_PATIENCE_FRAC,
    )
    p.add_argument("--hidden", type=int, default=128)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--device", type=str, default=None)
    p.add_argument("--data-root", type=str, default=None)
    p.add_argument("--results-dir", type=str, default=None)
    p.add_argument("--recruit-rate", type=float, default=1e4)
    p.add_argument("--max-step-prob", type=float, default=0.5)
    p.add_argument("--max-step", type=int, default=512)
    p.add_argument("--step-scale", type=float, default=1e6)
    p.add_argument("--exp-max-step", type=int, default=1)
    p.add_argument("--exp-step-scale", type=float, default=1.0)
    p.add_argument("--grad-momentum", type=float, default=0.9)
    p.add_argument("--ln-lr", type=float, default=1e-2)
    p.add_argument("--activation", type=str, default="relu")
    args = p.parse_args()
    budget = budget_from_args(args)

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    data_root = args.data_root or str(_REPO / "data")
    results_dir = Path(args.results_dir or (_REPO / "results" / EXPERIMENT_ID))
    cells = parse_cells(args.cells) if args.cells else default_cells()

    print(
        f"Budget: {budget.to_dict()} | device={device} | cells={[c.tag() for c in cells]}",
        flush=True,
    )
    train_loader, test_loader = make_cifar10_loaders(
        root=data_root,
        batch_size_train=args.batch_size,
        batch_size_test=256,
        num_workers=0,
        pin_memory=device.startswith("cuda"),
    )

    curve: list[dict[str, Any]] = []
    for cell in cells:
        try:
            out = run_cell(
                cell,
                budget=budget,
                args=args,
                device=device,
                train_loader=train_loader,
                test_loader=test_loader,
                results_dir=results_dir,
            )
            curve.append(
                {
                    **cell.to_dict(),
                    "best_test_acc": out["best_test_acc"],
                    "best_epoch": out["best_epoch"],
                    "epochs_ran": out["epochs_ran"],
                    "wall_sec": out["wall_sec"],
                    "status": "completed",
                }
            )
        except Exception as e:
            print(f"FAILED {cell.tag()}: {e}", flush=True)
            traceback.print_exc()
            curve.append({**cell.to_dict(), "best_test_acc": None, "status": f"error:{type(e).__name__}"})

    summary = {
        "experiment": EXPERIMENT_ID,
        "dataset": "cifar10",
        "seed": args.seed,
        "ln_mode": args.ln_mode,
        "budget": budget.to_dict(),
        "note": (
            "WP3 sparse: fixed vs exp_mant:2 at n=8,16 on CIFAR flat MLP; "
            "pure wall option B (2400s default)"
        ),
        "curve": curve,
    }
    results_dir.mkdir(parents=True, exist_ok=True)
    sp = results_dir / f"summary_ln_{args.ln_mode}_seed{args.seed}.json"
    with open(sp, "w") as f:
        json.dump(summary, f, indent=2)
    csv_path = results_dir / f"curve_ln_{args.ln_mode}_seed{args.seed}.csv"
    keys = ["tag", "n_bits", "encoding", "n_exp", "best_test_acc", "best_epoch", "epochs_ran", "wall_sec", "status"]
    with open(csv_path, "w") as f:
        f.write(",".join(keys) + "\n")
        for r in curve:
            f.write(",".join(str(r.get(k, "")) for k in keys) + "\n")
    print("\n===== v0_7_cifar_encoding curve =====", flush=True)
    for r in sorted(curve, key=lambda x: (-(x.get("best_test_acc") or -1), x.get("tag", ""))):
        print(
            f"  {r.get('tag', '?'):20s}  best={r.get('best_test_acc')}  "
            f"wall={r.get('wall_sec')}  status={r.get('status')}",
            flush=True,
        )
    print(f"Summary: {sp}", flush=True)


if __name__ == "__main__":
    main()
