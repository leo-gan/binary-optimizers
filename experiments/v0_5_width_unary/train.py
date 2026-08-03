#!/usr/bin/env python3
"""Width atlas for unary Swarm population size S (v0.1 scaffold).

Varies only swarm_size; default train settings fixed.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import traceback
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F

_THIS = Path(__file__).resolve().parent
_REPO = _THIS.parents[1]
_V01 = _REPO / "experiments" / "v0_1"
sys.path.insert(0, str(_V01))
sys.path.insert(0, str(_REPO))
sys.path.insert(0, str(_THIS.parent))

for _name in ("layers", "model", "optimizer", "metrics"):
    sys.modules.pop(_name, None)

from binary_optimizers.data.mnist import make_mnist_loaders  # noqa: E402
from binary_optimizers.training.budget import (  # noqa: E402
    EarlyStopTracker,
    TrainBudget,
    add_budget_args,
    budget_from_args,
)
from binary_optimizers.training.loops import set_seed  # noqa: E402

from model import BitNetSwarmMLP  # noqa: E402
from optimizer import SwarmOptimizerV01  # noqa: E402
from metrics import swarm_stats  # noqa: E402

from _width_atlas_common import (  # noqa: E402
    approx_unary_state_bytes,
    estimate_ok,
    parse_int_list,
    write_summary,
)

DEFAULT_WIDTHS = [8, 16, 32, 64, 128, 256, 512, 1024]


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


def run_width(
    *,
    swarm_size: int,
    ln_mode: str,
    budget: TrainBudget,
    hidden: int,
    seed: int,
    device: str,
    train_loader,
    test_loader,
    results_dir: Path,
    recruit_rate: float,
    max_flip_prob: float,
    grad_momentum: float,
    ln_lr: float,
    activation: str,
) -> dict[str, Any]:
    set_seed(seed)
    model = BitNetSwarmMLP(
        hidden_dim=hidden,
        swarm_size=swarm_size,
        ln_mode=ln_mode,  # type: ignore[arg-type]
        activation=activation,  # type: ignore[arg-type]
    ).to(device)
    opt = SwarmOptimizerV01(
        model.swarm_layers(),
        recruit_rate=recruit_rate,
        max_flip_prob=max_flip_prob,
        grad_momentum=grad_momentum,
        ln_params=model.ln_parameters(),
        ln_lr=ln_lr,
    )
    history = []
    best_state = None
    tracker = EarlyStopTracker(budget)
    print(
        f"\n===== v0_5_width_unary | S={swarm_size} | ln={ln_mode} | seed={seed} =====",
        flush=True,
    )
    for epoch in range(1, budget.max_epochs + 1):
        t0 = time.time()
        tr_acc, tr_loss, flip = train_one_epoch(model, opt, train_loader, device)
        te_acc, te_loss = evaluate(model, test_loader, device)
        model.assert_binary_invariants()
        stats = swarm_stats(model)
        row = {
            "epoch": epoch,
            "train_acc": tr_acc,
            "train_loss": tr_loss,
            "test_acc": te_acc,
            "test_loss": te_loss,
            "flip_frac": flip,
            "frac_plus": stats.get("frac_plus"),
            "epoch_sec": time.time() - t0,
        }
        history.append(row)
        decision = tracker.observe(epoch, te_acc)
        if decision.improved:
            best_state = {
                k: v.detach().cpu().clone() for k, v in model.state_dict().items()
            }
        print(
            f"  ep {epoch:03d}/{budget.max_epochs} train={tr_acc:.4f} test={te_acc:.4f} "
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
    state_b = approx_unary_state_bytes(hidden, swarm_size)
    out = {
        "experiment": "v0_5_width_unary",
        "coding": "unary_majority",
        "swarm_size": swarm_size,
        "ln_mode": ln_mode,
        "seed": seed,
        "hidden": hidden,
        "best_test_acc": tracker.best,
        "best_epoch": tracker.best_epoch,
        "epochs_ran": len(history),
        "budget": budget.to_dict(),
        "stop_meta": tracker.meta_dict(),
        "final_test_acc": final_acc,
        "final_test_loss": final_loss,
        "wall_sec": tracker.wall_sec,
        "approx_state_bytes": state_b,
        "history": history,
        "status": "completed",
    }
    results_dir.mkdir(parents=True, exist_ok=True)
    jp = results_dir / f"S{swarm_size}_ln_{ln_mode}_seed{seed}.json"
    with open(jp, "w") as f:
        json.dump(out, f, indent=2)
    print(f"  Saved {jp}", flush=True)
    return out


def main() -> None:
    p = argparse.ArgumentParser(description="v0.5 width atlas — unary swarm size")
    p.add_argument(
        "--widths",
        type=str,
        default=",".join(str(w) for w in DEFAULT_WIDTHS),
        help=f"Comma-separated swarm sizes (default {DEFAULT_WIDTHS})",
    )
    p.add_argument("--ln-mode", type=str, default="none")
    add_budget_args(p)
    p.add_argument("--hidden", type=int, default=128)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--device", type=str, default=None)
    p.add_argument("--data-root", type=str, default=None)
    p.add_argument("--results-dir", type=str, default=None)
    p.add_argument("--recruit-rate", type=float, default=1e4)
    p.add_argument("--max-flip-prob", type=float, default=0.15)
    p.add_argument("--grad-momentum", type=float, default=0.9)
    p.add_argument("--ln-lr", type=float, default=1e-2)
    p.add_argument("--activation", type=str, default="relu")
    p.add_argument("--mem-limit-gb", type=float, default=4.0)
    args = p.parse_args()

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    data_root = args.data_root or str(_REPO / "data")
    results_dir = Path(args.results_dir or (_REPO / "results" / "v0_5_width_unary"))
    widths = parse_int_list(args.widths)
    budget = budget_from_args(args)

    train_loader, test_loader = make_mnist_loaders(
        root=data_root,
        batch_size_train=args.batch_size,
        batch_size_test=1000,
        num_workers=0,
    )

    curve: list[dict[str, Any]] = []
    for S in widths:
        state_b = approx_unary_state_bytes(args.hidden, S)
        row_base = {
            "swarm_size": S,
            "ln_mode": args.ln_mode,
            "approx_state_bytes": state_b,
        }
        if S < 1:
            curve.append({**row_base, "status": "skipped_invalid", "best_test_acc": None,
                          "best_epoch": None, "epochs_ran": 0, "wall_sec": 0.0})
            continue
        if not estimate_ok(state_b, args.mem_limit_gb):
            print(f"Skip S={S}: approx state {state_b/1e6:.1f}MB soft limit.", flush=True)
            curve.append(
                {
                    **row_base,
                    "best_test_acc": None,
                    "best_epoch": None,
                    "epochs_ran": 0,
                    "wall_sec": 0.0,
                    "status": "skipped_memory",
                }
            )
            continue
        try:
            out = run_width(
                swarm_size=S,
                ln_mode=args.ln_mode,
                budget=budget,
                hidden=args.hidden,
                seed=args.seed,
                device=device,
                train_loader=train_loader,
                test_loader=test_loader,
                results_dir=results_dir,
                recruit_rate=args.recruit_rate,
                max_flip_prob=args.max_flip_prob,
                grad_momentum=args.grad_momentum,
                ln_lr=args.ln_lr,
                activation=args.activation,
            )
            curve.append(
                {
                    **row_base,
                    "best_test_acc": out["best_test_acc"],
                    "best_epoch": out["best_epoch"],
                    "epochs_ran": out["epochs_ran"],
                    "wall_sec": out["wall_sec"],
                    "status": "completed",
                }
            )
        except Exception as e:
            print(f"FAILED S={S}: {e}", flush=True)
            traceback.print_exc()
            curve.append(
                {
                    **row_base,
                    "best_test_acc": None,
                    "best_epoch": None,
                    "epochs_ran": 0,
                    "wall_sec": 0.0,
                    "status": f"error:{type(e).__name__}",
                }
            )

    sp = write_summary(
        results_dir,
        experiment="v0_5_width_unary",
        seed=args.seed,
        ln_mode=args.ln_mode,
        hidden=args.hidden,
        widths_requested=widths,
        note="unary majority coding; fixed train defaults (no per-S tuning)",
        width_key="swarm_size",
        this_run_curve=curve,
        disk_prefix="S",
    )
    with open(sp) as f:
        full = json.load(f)["curve"]
    print("\n===== v0_5_width_unary curve (merged disk) =====", flush=True)
    for r in full:
        print(
            f"  S={r['swarm_size']:4d}  best={r.get('best_test_acc')}  "
            f"status={r['status']}",
            flush=True,
        )
    print(f"Summary: {sp}", flush=True)


if __name__ == "__main__":
    main()
