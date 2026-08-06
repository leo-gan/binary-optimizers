#!/usr/bin/env python3
"""Experiment v0_8: Unary Swarm link-value + XOR writeback (WP-U1 existence)."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch
import torch.nn.functional as F

_THIS = Path(__file__).resolve().parent
_REPO = _THIS.parents[1]
sys.path.insert(0, str(_REPO))
sys.path.insert(0, str(_THIS))

from binary_optimizers.data.mnist import make_mnist_loaders  # noqa: E402
from binary_optimizers.store import db_notes, enrich_config, soft_record_completed_run  # noqa: E402
from binary_optimizers.training.budget import (  # noqa: E402
    EarlyStopTracker,
    add_budget_args,
    budget_from_args,
)
from binary_optimizers.training.loops import set_seed  # noqa: E402

from metrics import swarm_stats  # noqa: E402
from model import UnaryLinkMLP  # noqa: E402
from optimizer import UnaryLinkOptimizer  # noqa: E402

EXPERIMENT_ID = "v0_8_unary_link"


@torch.no_grad()
def evaluate(model, loader, device: str) -> tuple[float, float]:
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


def train_one_epoch(model, opt, loader, device: str) -> tuple[float, float, float]:
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


def train_run(
    *,
    experiment_id: str = EXPERIMENT_ID,
    budget,
    hidden: int,
    swarm_size: int,
    encoder: str,
    tanh_tau: float,
    opt_name: str,
    lr: float,
    momentum: float,
    decoder: str,
    alpha: float,
    p_min: float,
    p_max: float,
    p_noise: float,
    threshold: float,
    freeze_swarm: bool,
    ln_mode: str,
    activation: str,
    ln_lr: float,
    seed: int,
    device: str,
    train_loader,
    test_loader,
    results_dir: Path,
    run_tag: str = "default",
    in_dim: int = 28 * 28,
    n_classes: int = 10,
) -> Dict[str, Any]:
    set_seed(seed)
    model = UnaryLinkMLP(
        hidden_dim=hidden,
        swarm_size=swarm_size,
        encoder=encoder,  # type: ignore[arg-type]
        tanh_tau=tanh_tau,
        ln_mode=ln_mode,  # type: ignore[arg-type]
        activation=activation,  # type: ignore[arg-type]
        in_dim=in_dim,
        n_classes=n_classes,
    ).to(device)
    opt = UnaryLinkOptimizer(
        model.swarm_layers(),
        opt=opt_name,  # type: ignore[arg-type]
        lr=lr,
        momentum=momentum,
        decoder=decoder,  # type: ignore[arg-type]
        alpha=alpha,
        p_min=p_min,
        p_max=p_max,
        p_noise=p_noise,
        threshold=threshold,
        ln_params=model.ln_parameters(),
        ln_lr=ln_lr,
        freeze_swarm=freeze_swarm,
    )

    history: List[Dict[str, Any]] = []
    best_state: Optional[Dict[str, torch.Tensor]] = None
    tracker = EarlyStopTracker(budget)

    print(
        f"\n===== {experiment_id} | tag={run_tag} | S={swarm_size} | "
        f"enc={encoder} | opt={opt_name} | dec={decoder} | seed={seed} =====",
        flush=True,
    )

    for epoch in range(1, budget.max_epochs + 1):
        t0 = time.time()
        tr_acc, tr_loss, flip = train_one_epoch(model, opt, train_loader, device)
        te_acc, te_loss = evaluate(model, test_loader, device)
        model.assert_binary_invariants()
        stats = swarm_stats(model)
        dt = time.time() - t0
        row = {
            "epoch": epoch,
            "train_acc": tr_acc,
            "train_loss": tr_loss,
            "test_acc": te_acc,
            "test_loss": te_loss,
            "flip_frac": flip,
            "mean_abs_delta": opt.last_delta_abs_mean,
            "mean_abs_grad": opt.last_grad_abs_mean,
            "mean_abs_link_value": stats["mean_abs_link_value"],
            "frac_plus": stats["frac_plus"],
            "epoch_sec": dt,
        }
        history.append(row)
        decision = tracker.observe(epoch, te_acc)
        if decision.improved:
            best_state = {
                k: v.detach().cpu().clone() for k, v in model.state_dict().items()
            }
        print(
            f"epoch {epoch:03d} | train={tr_acc:.4f} test={te_acc:.4f} | "
            f"flip={flip:.4f} |Δ|={opt.last_delta_abs_mean:.3e} | "
            f"{tracker.status_str()} | {dt:.1f}s",
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

    out: Dict[str, Any] = {
        "experiment": experiment_id,
        "run_tag": run_tag,
        "seed": seed,
        "hidden": hidden,
        "swarm_size": swarm_size,
        "encoder": encoder,
        "tanh_tau": tanh_tau,
        "opt": opt_name,
        "lr": lr,
        "momentum": momentum,
        "decoder": decoder,
        "alpha": alpha,
        "p_min": p_min,
        "p_max": p_max,
        "p_noise": p_noise,
        "threshold": threshold,
        "freeze_swarm": freeze_swarm,
        "ln_mode": ln_mode,
        "activation": activation,
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
    }

    results_dir.mkdir(parents=True, exist_ok=True)
    safe_tag = run_tag.replace("/", "_")
    json_path = results_dir / f"{safe_tag}_seed{seed}.json"
    with open(json_path, "w") as f:
        json.dump(out, f, indent=2)

    cfg = enrich_config(
        experiment_id,
        {
            "run_tag": run_tag,
            "hidden": hidden,
            "swarm_size": swarm_size,
            "encoder": encoder,
            "opt": opt_name,
            "lr": lr,
            "decoder": decoder,
            "alpha": alpha,
            "p_max": p_max,
            "p_noise": p_noise,
            "ln_mode": ln_mode,
            "budget": budget.to_dict(),
        },
    )
    rid = soft_record_completed_run(
        experiment=experiment_id,
        name=safe_tag,
        config=cfg,
        history=history,
        seed=seed,
        wall_sec=out["wall_sec"],
        best_test_acc=out["best_test_acc"],
        best_epoch=out["best_epoch"],
        final_test_acc=final_test_acc,
        final_test_loss=final_test_loss,
        summary={"epochs_ran": len(history), "source_json": str(json_path)},
        notes=db_notes(experiment_id),
    )
    if rid:
        out["run_id"] = rid

    ckpt_dir = _REPO / "checkpoints" / experiment_id
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = ckpt_dir / f"{safe_tag}_seed{seed}.pt"
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
    return out


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="v0_8 Unary link Swarm training")
    add_budget_args(p)
    p.add_argument("--hidden", type=int, default=128)
    p.add_argument("--swarm-size", type=int, default=256)
    p.add_argument(
        "--encoder",
        choices=("fixed", "tanh", "signed_sqrt", "majority"),
        default="fixed",
    )
    p.add_argument("--tanh-tau", type=float, default=0.0, help="0 → S/2 for tanh")
    p.add_argument("--opt", choices=("sgd", "sgd_m", "adam"), default="sgd")
    p.add_argument("--lr", type=float, default=0.1)
    p.add_argument("--momentum", type=float, default=0.9)
    p.add_argument(
        "--decoder",
        choices=("density", "thresholded", "sign_noise"),
        default="density",
    )
    p.add_argument("--alpha", type=float, default=10.0)
    p.add_argument("--p-min", type=float, default=0.0)
    p.add_argument("--p-max", type=float, default=0.25)
    p.add_argument("--p-noise", type=float, default=0.001)
    p.add_argument("--threshold", type=float, default=1e-3)
    p.add_argument(
        "--freeze-swarm",
        action="store_true",
        help="Baseline: no XOR updates (frozen random swarm)",
    )
    p.add_argument("--ln-mode", choices=("none", "no_affine", "affine"), default="none")
    p.add_argument("--activation", choices=("relu", "squared_relu"), default="relu")
    p.add_argument("--ln-lr", type=float, default=1e-2)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--data-root", type=str, default=None)
    p.add_argument("--device", type=str, default=None)
    p.add_argument("--results-dir", type=str, default=None)
    p.add_argument("--run-tag", type=str, default="default")
    p.add_argument("--experiment-id", type=str, default=EXPERIMENT_ID)
    return p


def main(argv: Optional[List[str]] = None) -> None:
    args = build_arg_parser().parse_args(argv)
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    data_root = args.data_root or str(_REPO / "data")
    results_dir = Path(
        args.results_dir or (_REPO / "results" / args.experiment_id)
    )
    budget = budget_from_args(args)

    train_loader, test_loader = make_mnist_loaders(
        root=data_root,
        batch_size_train=args.batch_size,
        batch_size_test=1000,
        num_workers=0,
        pin_memory=device != "cpu",
    )

    train_run(
        experiment_id=args.experiment_id,
        budget=budget,
        hidden=args.hidden,
        swarm_size=args.swarm_size,
        encoder=args.encoder,
        tanh_tau=args.tanh_tau,
        opt_name=args.opt,
        lr=args.lr,
        momentum=args.momentum,
        decoder=args.decoder,
        alpha=args.alpha,
        p_min=args.p_min,
        p_max=args.p_max,
        p_noise=args.p_noise,
        threshold=args.threshold,
        freeze_swarm=args.freeze_swarm,
        ln_mode=args.ln_mode,
        activation=args.activation,
        ln_lr=args.ln_lr,
        seed=args.seed,
        device=device,
        train_loader=train_loader,
        test_loader=test_loader,
        results_dir=results_dir,
        run_tag=args.run_tag,
    )


if __name__ == "__main__":
    main()
