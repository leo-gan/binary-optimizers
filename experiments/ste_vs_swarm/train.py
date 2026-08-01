#!/usr/bin/env python3
"""Shared-protocol STE vs Swarm comparison on MNIST; logs runs to DuckDB."""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Sequence

import torch
import torch.nn.functional as F

_THIS_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _THIS_DIR.parents[1]
sys.path.insert(0, str(_REPO_ROOT))
sys.path.insert(0, str(_THIS_DIR))

from binary_optimizers.data.mnist import make_mnist_loaders
from binary_optimizers.optimizers.ste import STEOptimizer
from binary_optimizers.store import soft_record_completed_run
from binary_optimizers.training.loops import set_seed

from ste_model import BitNetSTEMLP, LNMode

EXPERIMENT_ID = "ste_vs_swarm"
METHODS: tuple[str, ...] = ("ste_sgd", "swarm_v0_3", "swarm_v0_4")
LN_MODES: tuple[LNMode, ...] = ("none", "no_affine", "affine")


_SWARM_SHORT = ("layers", "model", "optimizer", "metrics")


def _load_swarm_stack(version: str) -> tuple[type, type, Callable]:
    """Return (ModelCls, OptimizerCls, stats_fn) for v0_3 or v0_4.

    Sibling experiment dirs are plain scripts (not packages). We import them
    with their directory first on ``sys.path`` and clear short module names
    so v0_3 and v0_4 ``layers`` do not collide.
    """
    import importlib

    vdir = _REPO_ROOT / "experiments" / version
    if not vdir.is_dir():
        raise FileNotFoundError(vdir)
    vdir_s = str(vdir)
    # Prefer this version's modules for bare imports (layers, model, …).
    while vdir_s in sys.path:
        sys.path.remove(vdir_s)
    sys.path.insert(0, vdir_s)
    for name in _SWARM_SHORT:
        sys.modules.pop(name, None)
    model_mod = importlib.import_module("model")
    opt_mod = importlib.import_module("optimizer")
    metrics_mod = importlib.import_module("metrics")
    if version == "v0_3":
        return model_mod.BitNetCarrySafeMLP, opt_mod.SwarmOptimizerV03, metrics_mod.swarm_stats
    if version == "v0_4":
        return (
            model_mod.BitNetTernaryPlaceMLP,
            opt_mod.SwarmOptimizerV04,
            metrics_mod.swarm_stats,
        )
    raise ValueError(version)


@torch.no_grad()
def evaluate(model, loader, device) -> tuple[float, float]:
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


def train_one_epoch_ste(model, opt, loader, device, ln_opt=None) -> tuple[float, float]:
    model.train()
    total = correct = 0
    loss_sum = 0.0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        opt.zero_grad()
        if ln_opt is not None:
            ln_opt.zero_grad()
        logits = model(x)
        loss = F.cross_entropy(logits, y)
        loss.backward()
        opt.step()
        if ln_opt is not None:
            ln_opt.step()
        loss_sum += loss.item() * y.size(0)
        correct += (logits.argmax(1) == y).sum().item()
        total += y.size(0)
    return correct / max(1, total), loss_sum / max(1, total)


def train_one_epoch_swarm(model, opt, loader, device) -> tuple[float, float, float]:
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
        flip_sum += float(flip)
        n_steps += 1
    return (
        correct / max(1, total),
        loss_sum / max(1, total),
        flip_sum / max(1, n_steps),
    )


def _build_method(
    method: str,
    *,
    ln_mode: LNMode,
    hidden: int,
    activation: str,
    device: str,
    ste_lr: float,
    ste_momentum: float,
    ln_lr: float,
    n_bits: int,
    n_trits: int,
    recruit_rate: float,
    max_step_prob: float,
    grad_momentum: float,
    max_step: int | None,
    step_scale: float,
) -> tuple[Any, Any, dict[str, Any], Callable | None]:
    """Return model, optimizer_bundle, config extras, optional post_step_assert."""
    config: dict[str, Any] = {
        "method": method,
        "ln_mode": ln_mode,
        "hidden": hidden,
        "activation": activation,
        "protocol": "ste_vs_swarm_v1",
    }

    if method == "ste_sgd":
        model = BitNetSTEMLP(
            hidden_dim=hidden, ln_mode=ln_mode, activation=activation  # type: ignore[arg-type]
        ).to(device)
        ste_params = model.ste_parameters()
        opt = STEOptimizer(ste_params, lr=ste_lr, momentum=ste_momentum)
        ln_params = model.ln_parameters()
        ln_opt = torch.optim.SGD(ln_params, lr=ln_lr) if ln_params else None
        config.update(
            {
                "coding": "ste_sign_latent_fp",
                "optimizer": "STEOptimizer",
                "ste_lr": ste_lr,
                "ste_momentum": ste_momentum,
                "ln_lr": ln_lr,
            }
        )
        return model, {"kind": "ste", "opt": opt, "ln_opt": ln_opt}, config, None

    if method == "swarm_v0_3":
        ModelCls, OptCls, stats_fn = _load_swarm_stack("v0_3")
        ms = 512 if max_step is None else max_step
        model = ModelCls(
            hidden_dim=hidden,
            n_bits=n_bits,
            ln_mode=ln_mode,
            activation=activation,
        ).to(device)
        opt = OptCls(
            model.swarm_layers(),
            recruit_rate=recruit_rate,
            max_step_prob=max_step_prob,
            grad_momentum=grad_momentum,
            max_step=ms,
            step_scale=step_scale,
            ln_params=model.ln_parameters(),
            ln_lr=ln_lr,
        )
        config.update(
            {
                "coding": "carry_safe_place_value_binary",
                "optimizer": "SwarmOptimizerV03",
                "n_bits": n_bits,
                "recruit_rate": recruit_rate,
                "max_step_prob": max_step_prob,
                "grad_momentum": grad_momentum,
                "max_step": ms,
                "step_scale": step_scale,
                "ln_lr": ln_lr,
            }
        )

        def _assert():
            model.assert_binary_invariants()

        return model, {"kind": "swarm", "opt": opt, "stats_fn": stats_fn}, config, _assert

    if method == "swarm_v0_4":
        ModelCls, OptCls, stats_fn = _load_swarm_stack("v0_4")
        ms = 64 if max_step is None else max_step
        model = ModelCls(
            hidden_dim=hidden,
            n_trits=n_trits,
            ln_mode=ln_mode,
            activation=activation,
        ).to(device)
        opt = OptCls(
            model.swarm_layers(),
            recruit_rate=recruit_rate,
            max_step_prob=max_step_prob,
            grad_momentum=grad_momentum,
            max_step=ms,
            step_scale=step_scale,
            ln_params=model.ln_parameters(),
            ln_lr=ln_lr,
        )
        config.update(
            {
                "coding": "balanced_ternary_place",
                "optimizer": "SwarmOptimizerV04",
                "n_trits": n_trits,
                "recruit_rate": recruit_rate,
                "max_step_prob": max_step_prob,
                "grad_momentum": grad_momentum,
                "max_step": ms,
                "step_scale": step_scale,
                "ln_lr": ln_lr,
            }
        )

        def _assert4():
            model.assert_ternary_invariants()

        return model, {"kind": "swarm", "opt": opt, "stats_fn": stats_fn}, config, _assert4

    raise ValueError(f"Unknown method: {method}")


def train_one(
    *,
    method: str,
    ln_mode: LNMode,
    epochs: int,
    patience: int,
    min_delta: float,
    seed: int,
    device: str,
    train_loader,
    test_loader,
    **build_kw,
) -> Dict[str, Any]:
    set_seed(seed)
    model, bundle, config, assert_fn = _build_method(
        method, ln_mode=ln_mode, device=device, **build_kw
    )
    config["device"] = device
    config["seed"] = seed
    config["epochs"] = epochs
    config["patience"] = patience
    config["min_delta"] = min_delta

    history: List[dict[str, Any]] = []
    best_test, best_epoch, stalled = -1.0, 0, 0
    best_state = None
    run_name = f"{method}_ln_{ln_mode}"
    print(f"\n===== {EXPERIMENT_ID} | {run_name} | seed={seed} =====", flush=True)
    t0_run = time.time()

    for epoch in range(1, epochs + 1):
        t0 = time.time()
        if bundle["kind"] == "ste":
            tr_acc, tr_loss = train_one_epoch_ste(
                model, bundle["opt"], train_loader, device, ln_opt=bundle.get("ln_opt")
            )
            flip = None
        else:
            tr_acc, tr_loss, flip = train_one_epoch_swarm(
                model, bundle["opt"], train_loader, device
            )
        te_acc, te_loss = evaluate(model, test_loader, device)
        if assert_fn is not None:
            assert_fn()
        row: dict[str, Any] = {
            "epoch": epoch,
            "train_acc": tr_acc,
            "train_loss": tr_loss,
            "test_acc": te_acc,
            "test_loss": te_loss,
            "epoch_sec": time.time() - t0,
        }
        if flip is not None:
            row["flip_frac"] = flip
            if hasattr(bundle["opt"], "last_step_frac"):
                row["step_frac"] = float(bundle["opt"].last_step_frac)
        history.append(row)

        if te_acc > best_test + min_delta:
            best_test, best_epoch, stalled = te_acc, epoch, 0
            best_state = {
                k: v.detach().cpu().clone() for k, v in model.state_dict().items()
            }
        else:
            stalled += 1

        flip_s = f" flip={flip:.4f}" if flip is not None else ""
        print(
            f"epoch {epoch:03d}/{epochs} | train={tr_acc:.4f} loss={tr_loss:.4f} | "
            f"test={te_acc:.4f} loss={te_loss:.4f}{flip_s} | "
            f"best={best_test:.4f}@{best_epoch} stall={stalled}/{patience} | "
            f"{row['epoch_sec']:.1f}s",
            flush=True,
        )
        if stalled >= patience:
            print("Early stop.", flush=True)
            break

    if best_state is not None:
        model.load_state_dict(best_state)
        model.to(device)
    final_acc, final_loss = evaluate(model, test_loader, device)
    wall_sec = time.time() - t0_run

    ck_dir = _REPO_ROOT / "checkpoints" / EXPERIMENT_ID
    ck_dir.mkdir(parents=True, exist_ok=True)
    ck_path = ck_dir / f"{run_name}_seed{seed}.pt"
    torch.save(
        {
            "model_state": model.state_dict(),
            "meta": {
                "experiment": EXPERIMENT_ID,
                "name": run_name,
                "config": config,
                "best_test_acc": best_test,
                "best_epoch": best_epoch,
            },
        },
        ck_path,
    )

    summary: dict[str, Any] = {
        "epochs_ran": len(history),
        "protocol": "ste_vs_swarm_v1",
    }
    if bundle["kind"] == "swarm" and bundle.get("stats_fn") is not None:
        try:
            summary["swarm_stats_final"] = bundle["stats_fn"](model)
        except Exception:  # noqa: BLE001
            pass

    run_id = soft_record_completed_run(
        experiment=EXPERIMENT_ID,
        name=run_name,
        config=config,
        history=history,
        seed=seed,
        wall_sec=wall_sec,
        best_test_acc=best_test,
        best_epoch=best_epoch,
        final_test_acc=final_acc,
        final_test_loss=final_loss,
        summary=summary,
        checkpoint_path=ck_path,
    )
    out = {
        "experiment": EXPERIMENT_ID,
        "name": run_name,
        "method": method,
        "ln_mode": ln_mode,
        "seed": seed,
        "best_test_acc": best_test,
        "best_epoch": best_epoch,
        "final_test_acc": final_acc,
        "final_test_loss": final_loss,
        "wall_sec": wall_sec,
        "epochs_ran": len(history),
        "run_id": run_id,
        "checkpoint_path": str(ck_path),
        "history": history,
        "config": config,
    }
    print(
        f"Done {run_name}: best_test={best_test:.4f} @{best_epoch} "
        f"wall={wall_sec:.1f}s run_id={run_id}",
        flush=True,
    )
    return out


def main(argv: Sequence[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="STE vs Swarm shared-protocol MNIST comparison")
    p.add_argument(
        "--methods",
        type=str,
        default=",".join(METHODS),
        help=f"Comma-separated methods from {METHODS}",
    )
    p.add_argument(
        "--ln-mode",
        choices=["all", *LN_MODES],
        default="all",
    )
    p.add_argument("--epochs", type=int, default=80)
    p.add_argument("--patience", type=int, default=10)
    p.add_argument("--min-delta", type=float, default=5e-4)
    p.add_argument("--hidden", type=int, default=128)
    p.add_argument("--activation", choices=("relu", "squared_relu"), default="relu")
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", type=str, default=None)
    p.add_argument("--data-root", type=str, default=None)
    # STE
    p.add_argument("--ste-lr", type=float, default=0.1)
    p.add_argument("--ste-momentum", type=float, default=0.9)
    p.add_argument("--ln-lr", type=float, default=1e-2)
    # Swarm shared
    p.add_argument("--n-bits", type=int, default=16)
    p.add_argument("--n-trits", type=int, default=10)
    p.add_argument("--recruit-rate", type=float, default=1e4)
    p.add_argument("--max-step-prob", type=float, default=0.5)
    p.add_argument(
        "--max-step",
        type=int,
        default=None,
        help="Swarm max integer/trit step (default: 512 for v0_3, 64 for v0_4)",
    )
    p.add_argument("--step-scale", type=float, default=1e6)
    p.add_argument("--grad-momentum", type=float, default=0.9)
    args = p.parse_args(list(argv) if argv is not None else None)

    methods = [m.strip() for m in args.methods.split(",") if m.strip()]
    for m in methods:
        if m not in METHODS:
            p.error(f"Unknown method {m!r}; choose from {METHODS}")
    ln_modes: list[LNMode] = (
        list(LN_MODES) if args.ln_mode == "all" else [args.ln_mode]  # type: ignore[list-item]
    )

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    data_root = args.data_root or str(_REPO_ROOT / "data")
    train_loader, test_loader = make_mnist_loaders(
        root=data_root,
        batch_size_train=args.batch_size,
        batch_size_test=1000,
        num_workers=0,
    )

    build_kw = dict(
        hidden=args.hidden,
        activation=args.activation,
        ste_lr=args.ste_lr,
        ste_momentum=args.ste_momentum,
        ln_lr=args.ln_lr,
        n_bits=args.n_bits,
        n_trits=args.n_trits,
        recruit_rate=args.recruit_rate,
        max_step_prob=args.max_step_prob,
        grad_momentum=args.grad_momentum,
        max_step=args.max_step,
        step_scale=args.step_scale,
    )

    results: list[dict[str, Any]] = []
    for method in methods:
        for ln_mode in ln_modes:
            results.append(
                train_one(
                    method=method,
                    ln_mode=ln_mode,
                    epochs=args.epochs,
                    patience=args.patience,
                    min_delta=args.min_delta,
                    seed=args.seed,
                    device=device,
                    train_loader=train_loader,
                    test_loader=test_loader,
                    **build_kw,
                )
            )

    print("\n===== ste_vs_swarm summary =====", flush=True)
    print(
        f"{'method':<14} {'ln_mode':<12} {'best_test':>10} {'best_ep':>8} {'wall_s':>8}",
        flush=True,
    )
    for r in results:
        print(
            f"{r['method']:<14} {r['ln_mode']:<12} {r['best_test_acc']:>10.4f} "
            f"{r['best_epoch']:>8} {r['wall_sec']:>8.1f}",
            flush=True,
        )
    print(
        "\nLogged to DuckDB. View with: ./scripts/report.sh  "
        "or  python -m binary_optimizers.store report",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
