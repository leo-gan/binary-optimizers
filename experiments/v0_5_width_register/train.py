#!/usr/bin/env python3
"""Width atlas for carry-safe binary register (v0.3 scaffold).

Varies only n_bits; default train settings fixed (no per-width hyperparam search).
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
_V03 = _REPO / "experiments" / "v0_3"
sys.path.insert(0, str(_V03))
sys.path.insert(0, str(_REPO))
sys.path.insert(0, str(_THIS.parent))  # experiments/ for _width_atlas_common

# Fresh imports from v0_3
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

from model import BitNetCarrySafeMLP  # noqa: E402
from optimizer import SwarmOptimizerV03  # noqa: E402
from metrics import swarm_stats  # noqa: E402

from _width_atlas_common import (  # noqa: E402
    approx_register_state_bytes,
    estimate_ok,
    parse_int_list,
    write_summary,
)

# int64-safe vmax = 2^n-1 requires n <= 62
DEFAULT_WIDTHS = [8, 16, 32, 48, 62]
MAX_SAFE_N_BITS = 62


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


def scale_steps_for_width(n_bits: int, base_max_step: int, base_step_scale: float) -> tuple[int, float]:
    """Keep Δv / vmax similar to the n_bits=16 reference (not accuracy tuning).

    At n=16, vmax≈65535 and max_step=512 ≈ 0.78% of the register range.
    Without this, wide registers barely move and look 'broken'.
    """
    ref_n = 16
    ref_vmax = float(2**ref_n - 1)
    vmax = float(2**n_bits - 1)
    scale = vmax / ref_vmax
    # Keep Δv/vmax ≈ 512/65535; stay within signed int64 when added to v.
    max_step = max(1, int(round(base_max_step * scale)))
    max_step = min(max_step, int(2**62 - 1))
    step_scale = base_step_scale * scale
    return max_step, step_scale


def run_width(
    *,
    n_bits: int,
    ln_mode: str,
    budget: TrainBudget,
    hidden: int,
    seed: int,
    device: str,
    train_loader,
    test_loader,
    results_dir: Path,
    recruit_rate: float,
    max_step_prob: float,
    max_step: int,
    step_scale: float,
    grad_momentum: float,
    ln_lr: float,
    activation: str,
) -> dict[str, Any]:
    set_seed(seed)
    max_step_w, step_scale_w = scale_steps_for_width(n_bits, max_step, step_scale)
    model = BitNetCarrySafeMLP(
        hidden_dim=hidden,
        n_bits=n_bits,
        ln_mode=ln_mode,  # type: ignore[arg-type]
        activation=activation,  # type: ignore[arg-type]
    ).to(device)
    opt = SwarmOptimizerV03(
        model.swarm_layers(),
        recruit_rate=recruit_rate,
        max_step_prob=max_step_prob,
        max_step=max_step_w,
        step_scale=step_scale_w,
        grad_momentum=grad_momentum,
        ln_params=model.ln_parameters(),
        ln_lr=ln_lr,
    )
    history = []
    best_state = None
    tracker = EarlyStopTracker(budget)
    print(
        f"\n===== v0_5_width_register | n_bits={n_bits} | ln={ln_mode} | "
        f"max_step={max_step_w} | seed={seed} =====",
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
            "step_frac": opt.last_step_frac,
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
    state_b = approx_register_state_bytes(hidden, n_bits)
    out = {
        "experiment": "v0_5_width_register",
        "coding": "carry_safe_integer",
        "n_bits": n_bits,
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
        "max_step_used": max_step_w,
        "step_scale_used": step_scale_w,
        "history": history,
        "status": "completed",
    }
    results_dir.mkdir(parents=True, exist_ok=True)
    jp = results_dir / f"nbits{n_bits}_ln_{ln_mode}_seed{seed}.json"
    with open(jp, "w") as f:
        json.dump(out, f, indent=2)
    print(f"  Saved {jp}", flush=True)
    return out


def main() -> None:
    p = argparse.ArgumentParser(description="v0.5 width atlas — binary register")
    p.add_argument(
        "--widths",
        type=str,
        default=",".join(str(w) for w in DEFAULT_WIDTHS),
        help=f"Comma-separated n_bits (default {DEFAULT_WIDTHS}; max safe {MAX_SAFE_N_BITS})",
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
    p.add_argument("--max-step-prob", type=float, default=0.5)
    p.add_argument("--max-step", type=int, default=512)
    p.add_argument("--step-scale", type=float, default=1e6)
    p.add_argument("--grad-momentum", type=float, default=0.9)
    p.add_argument("--ln-lr", type=float, default=1e-2)
    p.add_argument("--activation", type=str, default="relu")
    p.add_argument("--mem-limit-gb", type=float, default=4.0)
    args = p.parse_args()

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    data_root = args.data_root or str(_REPO / "data")
    results_dir = Path(
        args.results_dir or (_REPO / "results" / "v0_5_width_register")
    )
    widths = parse_int_list(args.widths)
    budget = budget_from_args(args)

    train_loader, test_loader = make_mnist_loaders(
        root=data_root,
        batch_size_train=args.batch_size,
        batch_size_test=1000,
        num_workers=0,
    )

    curve: list[dict[str, Any]] = []
    for n_bits in widths:
        state_b = approx_register_state_bytes(args.hidden, n_bits)
        row_base = {
            "n_bits": n_bits,
            "ln_mode": args.ln_mode,
            "approx_state_bytes": state_b,
        }
        if n_bits < 1 or n_bits > MAX_SAFE_N_BITS:
            print(
                f"Skip n_bits={n_bits}: need 1..{MAX_SAFE_N_BITS} for int64 register.",
                flush=True,
            )
            curve.append(
                {
                    **row_base,
                    "best_test_acc": None,
                    "best_epoch": None,
                    "epochs_ran": 0,
                    "wall_sec": 0.0,
                    "status": "skipped_n_bits_limit",
                }
            )
            continue
        if not estimate_ok(state_b, args.mem_limit_gb):
            print(
                f"Skip n_bits={n_bits}: approx state {state_b/1e6:.1f}MB "
                f"exceeds soft limit.",
                flush=True,
            )
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
                n_bits=n_bits,
                ln_mode=args.ln_mode,
                budget=budget,
                hidden=args.hidden,
                seed=args.seed,
                device=device,
                train_loader=train_loader,
                test_loader=test_loader,
                results_dir=results_dir,
                recruit_rate=args.recruit_rate,
                max_step_prob=args.max_step_prob,
                max_step=args.max_step,
                step_scale=args.step_scale,
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
            print(f"FAILED n_bits={n_bits}: {e}", flush=True)
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
        experiment="v0_5_width_register",
        seed=args.seed,
        ln_mode=args.ln_mode,
        hidden=args.hidden,
        widths_requested=widths,
        note=(
            f"int64-safe n_bits <= {MAX_SAFE_N_BITS}; "
            "fixed train defaults (no per-width tuning); "
            "max_step/step_scale scale with vmax so Δv/vmax ≈ n=16 ref"
        ),
        width_key="n_bits",
        this_run_curve=curve,
        disk_prefix="nbits",
    )
    with open(sp) as f:
        full = json.load(f)["curve"]
    print("\n===== v0_5_width_register curve (merged disk) =====", flush=True)
    for r in full:
        print(
            f"  n_bits={r['n_bits']:4d}  best={r.get('best_test_acc')}  "
            f"status={r['status']}",
            flush=True,
        )
    print(f"Summary: {sp}", flush=True)


if __name__ == "__main__":
    main()
