#!/usr/bin/env python3
"""Experiment v0_2: place-value (exponential) binary Swarm on MNIST."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch
import torch.nn.functional as F

_THIS_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _THIS_DIR.parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))
if str(_THIS_DIR) not in sys.path:
    sys.path.insert(0, str(_THIS_DIR))

from binary_optimizers.data.mnist import make_mnist_loaders  # noqa: E402
from binary_optimizers.training.budget import (  # noqa: E402
    EarlyStopTracker,
    TrainBudget,
    add_budget_args,
    budget_from_args,
)
from binary_optimizers.training.loops import set_seed  # noqa: E402

from metrics import swarm_stats  # noqa: E402
from model import BitNetPlaceValueSwarmMLP, LNMode  # noqa: E402
from optimizer import SwarmOptimizerV02  # noqa: E402

LN_MODES: tuple[LNMode, ...] = ("none", "no_affine", "affine")


@torch.no_grad()
def evaluate(model: torch.nn.Module, loader, device: str) -> tuple[float, float]:
    model.eval()
    total = 0
    correct = 0
    loss_sum = 0.0
    for x, y in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        logits = model(x)
        loss_sum += F.cross_entropy(logits, y, reduction="sum").item()
        correct += (logits.argmax(1) == y).sum().item()
        total += y.size(0)
    return correct / max(1, total), loss_sum / max(1, total)


def train_one_epoch(
    model: BitNetPlaceValueSwarmMLP,
    opt: SwarmOptimizerV02,
    loader,
    device: str,
) -> tuple[float, float, float]:
    model.train()
    total = 0
    correct = 0
    loss_sum = 0.0
    flip_sum = 0.0
    n_steps = 0

    for x, y in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        opt.zero_grad()
        logits = model(x)
        loss = F.cross_entropy(logits, y)
        loss.backward()
        flip_frac = opt.step()

        loss_sum += loss.item() * y.size(0)
        correct += (logits.argmax(1) == y).sum().item()
        total += y.size(0)
        flip_sum += flip_frac
        n_steps += 1

    return correct / max(1, total), loss_sum / max(1, total), flip_sum / max(1, n_steps)


def train_one_mode(
    *,
    ln_mode: LNMode,
    budget: TrainBudget,
    hidden: int,
    n_bits: int,
    recruit_rate: float,
    max_flip_prob: float,
    grad_momentum: float,
    lsb_bias: bool,
    activation: str,
    ln_lr: float,
    seed: int,
    device: str,
    train_loader,
    test_loader,
    results_dir: Path,
) -> Dict[str, Any]:
    set_seed(seed)
    model = BitNetPlaceValueSwarmMLP(
        hidden_dim=hidden,
        n_bits=n_bits,
        ln_mode=ln_mode,
        depth=1,
        activation=activation,  # type: ignore[arg-type]
    ).to(device)

    opt = SwarmOptimizerV02(
        model.swarm_layers(),
        recruit_rate=recruit_rate,
        max_flip_prob=max_flip_prob,
        grad_momentum=grad_momentum,
        lsb_bias=lsb_bias,
        ln_params=model.ln_parameters(),
        ln_lr=ln_lr,
    )

    history: List[Dict[str, Any]] = []
    best_state: Optional[Dict[str, torch.Tensor]] = None
    tracker = EarlyStopTracker(budget)

    print(
        f"\n===== v0_2 | ln_mode={ln_mode} | n_bits={n_bits} | "
        f"seed={seed} | recruit_rate={recruit_rate} =====",
        flush=True,
    )

    for epoch in range(1, budget.max_epochs + 1):
        t0 = time.time()
        train_acc, train_loss, flip_frac = train_one_epoch(
            model, opt, train_loader, device
        )
        test_acc, test_loss = evaluate(model, test_loader, device)
        model.assert_binary_invariants()
        stats = swarm_stats(model)
        dt = time.time() - t0

        row: Dict[str, Any] = {
            "epoch": epoch,
            "train_acc": train_acc,
            "train_loss": train_loss,
            "test_acc": test_acc,
            "test_loss": test_loss,
            "flip_frac": flip_frac,
            "frac_plus": stats["frac_plus"],
            "mean_abs_place_sum_norm": stats["mean_abs_place_sum_norm"],
            "flip_frac_by_bit": list(opt.last_flip_frac_by_bit),
            "epoch_sec": dt,
        }
        history.append(row)

        decision = tracker.observe(epoch, test_acc)
        if decision.improved:
            best_state = {
                k: v.detach().cpu().clone() for k, v in model.state_dict().items()
            }

        bit_hint = ""
        if opt.last_flip_frac_by_bit:
            fb = opt.last_flip_frac_by_bit
            bit_hint = f" bits[0/mid/last]={fb[0]:.4f}/{fb[len(fb)//2]:.4f}/{fb[-1]:.4f}"

        print(
            f"epoch {epoch:03d}/{budget.max_epochs} | "
            f"train={train_acc:.4f} loss={train_loss:.4f} | "
            f"test={test_acc:.4f} loss={test_loss:.4f} | "
            f"flip={flip_frac:.4f} |g|={opt.last_grad_abs_mean:.3e} "
            f"s_norm={stats['mean_abs_place_sum_norm']:.3f} | "
            f"{tracker.status_str()} | {dt:.1f}s{bit_hint}",
            flush=True,
        )

        if decision.stop:
            print(f"Stop: {decision.reason}", flush=True)
            break

    if best_state is not None:
        model.load_state_dict(best_state)
        model.to(device)

    final_test_acc, final_test_loss = evaluate(model, test_loader, device)
    model.assert_binary_invariants()

    out = {
        "experiment": "v0_2",
        "ln_mode": ln_mode,
        "seed": seed,
        "hidden": hidden,
        "n_bits": n_bits,
        "coding": "place_value_2i",
        "recruit_rate": recruit_rate,
        "max_flip_prob": max_flip_prob,
        "grad_momentum": grad_momentum,
        "lsb_bias": lsb_bias,
        "activation": activation,
        "ln_lr": ln_lr,
        "device": device,
        "epochs_ran": len(history),
        "budget": budget.to_dict(),
        "stop_meta": tracker.meta_dict(),
        "best_test_acc": tracker.best,
        "best_epoch": tracker.best_epoch,
        "final_test_acc": final_test_acc,
        "final_test_loss": final_test_loss,
        "wall_sec": tracker.wall_sec,
        "history": history,
        "swarm_stats_final": swarm_stats(model),
        "baseline_v0_1_none_best": 0.9239,
    }

    results_dir.mkdir(parents=True, exist_ok=True)
    json_path = results_dir / f"ln_{ln_mode}_seed{seed}.json"
    with open(json_path, "w") as f:
        json.dump(out, f, indent=2)

    ckpt_dir = _REPO_ROOT / "checkpoints" / "v0_2"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = ckpt_dir / f"ln_{ln_mode}_seed{seed}.pt"
    torch.save(
        {
            "model_state": model.state_dict(),
            "meta": {k: v for k, v in out.items() if k != "history"},
        },
        ckpt_path,
    )
    out["json_path"] = str(json_path)
    out["ckpt_path"] = str(ckpt_path)
    print(f"Saved {json_path}", flush=True)
    print(f"Saved {ckpt_path}", flush=True)
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Experiment v0_2 training")
    parser.add_argument(
        "--ln-mode",
        choices=["all", *LN_MODES],
        default="all",
    )
    add_budget_args(parser)
    parser.add_argument("--hidden", type=int, default=128)
    parser.add_argument("--n-bits", type=int, default=16)
    parser.add_argument("--recruit-rate", type=float, default=1e4)
    parser.add_argument("--max-flip-prob", type=float, default=0.15)
    parser.add_argument("--grad-momentum", type=float, default=0.9)
    parser.add_argument(
        "--lsb-bias",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Scale flip prob by 2^{-bit_index} (default: on)",
    )
    parser.add_argument(
        "--activation",
        choices=("relu", "squared_relu"),
        default="relu",
    )
    parser.add_argument("--ln-lr", type=float, default=1e-2)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--data-root", type=str, default=None)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--results-dir", type=str, default=None)
    args = parser.parse_args()

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    data_root = args.data_root or str(_REPO_ROOT / "data")
    results_dir = Path(args.results_dir or (_REPO_ROOT / "results" / "v0_2"))

    train_loader, test_loader = make_mnist_loaders(
        root=data_root,
        batch_size_train=args.batch_size,
        batch_size_test=1000,
        num_workers=0,
        pin_memory=device != "cpu",
    )

    modes: List[LNMode]
    if args.ln_mode == "all":
        modes = list(LN_MODES)
    else:
        modes = [args.ln_mode]  # type: ignore[list-item]

    budget = budget_from_args(args)
    summaries = []
    for mode in modes:
        result = train_one_mode(
            ln_mode=mode,
            budget=budget,
            hidden=args.hidden,
            n_bits=args.n_bits,
            recruit_rate=args.recruit_rate,
            max_flip_prob=args.max_flip_prob,
            grad_momentum=args.grad_momentum,
            lsb_bias=args.lsb_bias,
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
                "best_test_acc": result["best_test_acc"],
                "best_epoch": result["best_epoch"],
                "epochs_ran": result["epochs_ran"],
                "wall_sec": result["wall_sec"],
                "n_bits": args.n_bits,
                "json_path": result["json_path"],
            }
        )

    summary_path = results_dir / f"summary_seed{args.seed}.json"
    summary = {
        "experiment": "v0_2",
        "seed": args.seed,
        "device": device,
        "coding": "place_value_2i",
        "baseline_v0_1_none_best": 0.9239,
        "hyper": {
            "hidden": args.hidden,
            "n_bits": args.n_bits,
            "recruit_rate": args.recruit_rate,
            "max_flip_prob": args.max_flip_prob,
            "grad_momentum": args.grad_momentum,
            "lsb_bias": args.lsb_bias,
            "activation": args.activation,
            "ln_lr": args.ln_lr,
            "budget": budget.to_dict(),
            "batch_size": args.batch_size,
        },
        "runs": summaries,
    }
    results_dir.mkdir(parents=True, exist_ok=True)
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    print("\n===== v0_2 summary =====", flush=True)
    print(f"  v0.1 baseline (none): 0.9239", flush=True)
    for s in summaries:
        print(
            f"  ln={s['ln_mode']:10s}  best_test={s['best_test_acc']:.4f} "
            f"@ epoch {s['best_epoch']}  (ran {s['epochs_ran']} epochs, "
            f"{s['wall_sec']:.0f}s) n_bits={s['n_bits']}",
            flush=True,
        )
    print(f"Summary written to {summary_path}", flush=True)


if __name__ == "__main__":
    main()
