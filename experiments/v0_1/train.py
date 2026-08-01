#!/usr/bin/env python3
"""Experiment v0_1: latent-free binary Swarm, BitNet-style MLP, MNIST.

Trains all three LayerNorm modes to early-stop convergence by default.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch
import torch.nn.functional as F

# Repo root on sys.path for binary_optimizers.*; this dir for local modules.
_THIS_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _THIS_DIR.parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))
if str(_THIS_DIR) not in sys.path:
    sys.path.insert(0, str(_THIS_DIR))

from binary_optimizers.data.mnist import make_mnist_loaders  # noqa: E402
from binary_optimizers.training.loops import set_seed  # noqa: E402

from metrics import swarm_stats  # noqa: E402
from model import BitNetSwarmMLP, LNMode  # noqa: E402
from optimizer import SwarmOptimizerV01  # noqa: E402

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
    model: BitNetSwarmMLP,
    opt: SwarmOptimizerV01,
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
    epochs: int,
    patience: int,
    min_delta: float,
    hidden: int,
    swarm_size: int,
    recruit_rate: float,
    max_flip_prob: float,
    grad_momentum: float,
    activation: str,
    ln_lr: float,
    seed: int,
    device: str,
    train_loader,
    test_loader,
    results_dir: Path,
) -> Dict[str, Any]:
    set_seed(seed)
    model = BitNetSwarmMLP(
        hidden_dim=hidden,
        swarm_size=swarm_size,
        ln_mode=ln_mode,
        depth=1,
        activation=activation,
    ).to(device)

    opt = SwarmOptimizerV01(
        model.swarm_layers(),
        recruit_rate=recruit_rate,
        max_flip_prob=max_flip_prob,
        grad_momentum=grad_momentum,
        ln_params=model.ln_parameters(),
        ln_lr=ln_lr,
    )

    history: List[Dict[str, Any]] = []
    best_test = -1.0
    best_epoch = 0
    best_state: Optional[Dict[str, torch.Tensor]] = None
    stalled = 0

    print(
        f"\n===== v0_1 | ln_mode={ln_mode} | seed={seed} | "
        f"recruit_rate={recruit_rate} =====",
        flush=True,
    )
    t_run0 = time.time()

    for epoch in range(1, epochs + 1):
        t0 = time.time()
        train_acc, train_loss, flip_frac = train_one_epoch(
            model, opt, train_loader, device
        )
        test_acc, test_loss = evaluate(model, test_loader, device)
        model.assert_binary_invariants()
        stats = swarm_stats(model)
        dt = time.time() - t0

        row = {
            "epoch": epoch,
            "train_acc": train_acc,
            "train_loss": train_loss,
            "test_acc": test_acc,
            "test_loss": test_loss,
            "flip_frac": flip_frac,
            "frac_plus": stats["frac_plus"],
            "mean_abs_margin": stats["mean_abs_margin"],
            "epoch_sec": dt,
        }
        history.append(row)

        improved = test_acc > best_test + min_delta
        if improved:
            best_test = test_acc
            best_epoch = epoch
            stalled = 0
            best_state = {
                k: v.detach().cpu().clone() for k, v in model.state_dict().items()
            }
        else:
            stalled += 1

        print(
            f"epoch {epoch:03d}/{epochs} | "
            f"train={train_acc:.4f} loss={train_loss:.4f} | "
            f"test={test_acc:.4f} loss={test_loss:.4f} | "
            f"flip={flip_frac:.4f} |g|={opt.last_grad_abs_mean:.3e} "
            f"margin={stats['mean_abs_margin']:.2f} | "
            f"best={best_test:.4f}@{best_epoch} stall={stalled}/{patience} | "
            f"{dt:.1f}s",
            flush=True,
        )

        if stalled >= patience:
            print(f"Early stop: no test_acc gain > {min_delta} for {patience} epochs.")
            break

    if best_state is not None:
        model.load_state_dict(best_state)
        model.to(device)

    final_test_acc, final_test_loss = evaluate(model, test_loader, device)
    model.assert_binary_invariants()

    out = {
        "experiment": "v0_1",
        "ln_mode": ln_mode,
        "seed": seed,
        "hidden": hidden,
        "swarm_size": swarm_size,
        "recruit_rate": recruit_rate,
        "max_flip_prob": max_flip_prob,
        "grad_momentum": grad_momentum,
        "activation": activation,
        "ln_lr": ln_lr,
        "device": device,
        "epochs_ran": len(history),
        "max_epochs": epochs,
        "patience": patience,
        "min_delta": min_delta,
        "best_test_acc": best_test,
        "best_epoch": best_epoch,
        "final_test_acc": final_test_acc,
        "final_test_loss": final_test_loss,
        "wall_sec": time.time() - t_run0,
        "history": history,
        "swarm_stats_final": swarm_stats(model),
    }

    results_dir.mkdir(parents=True, exist_ok=True)
    json_path = results_dir / f"ln_{ln_mode}_seed{seed}.json"
    with open(json_path, "w") as f:
        json.dump(out, f, indent=2)

    ckpt_dir = _REPO_ROOT / "checkpoints" / "v0_1"
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
    print(f"Saved {json_path}")
    print(f"Saved {ckpt_path}")
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Experiment v0_1 training")
    parser.add_argument(
        "--ln-mode",
        choices=["all", *LN_MODES],
        default="all",
        help="LayerNorm mode, or all three",
    )
    parser.add_argument("--epochs", type=int, default=80, help="Max epochs")
    parser.add_argument(
        "--patience",
        type=int,
        default=10,
        help="Early-stop patience on test acc",
    )
    parser.add_argument(
        "--min-delta",
        type=float,
        default=5e-4,
        help="Min test_acc improvement to reset patience",
    )
    parser.add_argument("--hidden", type=int, default=128)
    parser.add_argument("--swarm-size", type=int, default=32)
    parser.add_argument(
        "--recruit-rate",
        type=float,
        default=1e4,
        help="Flip prob = min(max_flip_prob, |ema_grad| * recruit_rate)",
    )
    parser.add_argument("--max-flip-prob", type=float, default=0.15)
    parser.add_argument(
        "--grad-momentum",
        type=float,
        default=0.9,
        help="EMA momentum on grad pressure before flip decisions",
    )
    parser.add_argument(
        "--activation",
        choices=("relu", "squared_relu"),
        default="relu",
        help="Hidden activation (relu more stable; squared_relu = BitNet-style)",
    )
    parser.add_argument("--ln-lr", type=float, default=1e-2)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--data-root", type=str, default=None)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument(
        "--results-dir",
        type=str,
        default=None,
        help="Default: <repo>/results/v0_1",
    )
    args = parser.parse_args()

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    data_root = args.data_root or str(_REPO_ROOT / "data")
    results_dir = Path(args.results_dir or (_REPO_ROOT / "results" / "v0_1"))

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

    summaries = []
    for mode in modes:
        result = train_one_mode(
            ln_mode=mode,
            epochs=args.epochs,
            patience=args.patience,
            min_delta=args.min_delta,
            hidden=args.hidden,
            swarm_size=args.swarm_size,
            recruit_rate=args.recruit_rate,
            max_flip_prob=args.max_flip_prob,
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
                "best_test_acc": result["best_test_acc"],
                "best_epoch": result["best_epoch"],
                "epochs_ran": result["epochs_ran"],
                "wall_sec": result["wall_sec"],
                "json_path": result["json_path"],
            }
        )

    summary_path = results_dir / f"summary_seed{args.seed}.json"
    summary = {
        "experiment": "v0_1",
        "seed": args.seed,
        "device": device,
        "hyper": {
            "hidden": args.hidden,
            "swarm_size": args.swarm_size,
            "recruit_rate": args.recruit_rate,
            "max_flip_prob": args.max_flip_prob,
            "grad_momentum": args.grad_momentum,
            "activation": args.activation,
            "ln_lr": args.ln_lr,
            "epochs": args.epochs,
            "patience": args.patience,
            "min_delta": args.min_delta,
            "batch_size": args.batch_size,
        },
        "runs": summaries,
    }
    results_dir.mkdir(parents=True, exist_ok=True)
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    print("\n===== v0_1 summary =====")
    for s in summaries:
        print(
            f"  ln={s['ln_mode']:10s}  best_test={s['best_test_acc']:.4f} "
            f"@ epoch {s['best_epoch']}  (ran {s['epochs_ran']} epochs, "
            f"{s['wall_sec']:.0f}s)"
        )
    print(f"Summary written to {summary_path}")


if __name__ == "__main__":
    main()
