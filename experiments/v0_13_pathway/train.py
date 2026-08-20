#!/usr/bin/env python3
"""CLI + train loop for experiment v0_13 pathway optimizer."""

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
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))
if str(_THIS) not in sys.path:
    sys.path.insert(0, str(_THIS))

from binary_optimizers.assembly import mean_assembly_index  # noqa: E402
from binary_optimizers.data.mnist import make_mnist_loaders  # noqa: E402
from binary_optimizers.optimizers.pathway import PathwayOptimizer  # noqa: E402
from binary_optimizers.store import db_notes, enrich_config, soft_record_completed_run  # noqa: E402
from binary_optimizers.training.budget import (  # noqa: E402
    EarlyStopTracker,
    add_budget_args,
    budget_from_args,
)
from binary_optimizers.training.loops import set_seed  # noqa: E402

from model import PathwayMLP  # noqa: E402

EXPERIMENT_ID = "v0_13_pathway"
REPO_ROOT = _REPO


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


def train_one_epoch(model, opt: PathwayOptimizer, loader, device: str) -> tuple[float, float, float, float]:
    model.train()
    total = correct = 0
    loss_sum = 0.0
    accepted = proposed = 0
    n_steps = 0
    need_grad = opt.ranking == "ste_topk" or opt.mode == "hybrid"

    for x, y in loader:
        x, y = x.to(device), y.to(device)

        def closure():
            opt.zero_grad()
            logits = model(x)
            loss = F.cross_entropy(logits, y)
            if need_grad:
                loss.backward()
            return loss, logits

        loss = opt.step(closure)
        with torch.no_grad():
            logits = model(x)
            pred = logits.argmax(1)
        loss_sum += float(loss.detach()) * y.size(0)
        correct += pred.eq(y).sum().item()
        total += y.size(0)
        accepted += opt.last_accepted
        proposed += opt.last_proposed
        n_steps += 1

    accept_rate = accepted / max(1, proposed)
    return correct / max(1, total), loss_sum / max(1, total), accept_rate, proposed / max(1, n_steps)


def train_run(
    *,
    experiment_id: str = EXPERIMENT_ID,
    budget,
    hidden: int,
    mode: str,
    candidates: int,
    lambda_a: float,
    ranking: str,
    assembly_of: str,
    bn_lr: float,
    use_bn: bool,
    norm: str,
    seed: int,
    device: str,
    train_loader,
    test_loader,
    results_dir: Path,
    run_tag: str = "default",
) -> Dict[str, Any]:
    set_seed(seed)
    model = PathwayMLP(hidden_dim=hidden, norm=norm, use_bn=use_bn).to(device)  # type: ignore[arg-type]
    opt = PathwayOptimizer(
        model.parameters(),
        candidates=candidates,
        lambda_a=lambda_a,
        ranking=ranking,  # type: ignore[arg-type]
        assembly_of=assembly_of,  # type: ignore[arg-type]
        mode=mode,  # type: ignore[arg-type]
        bn_lr=bn_lr,
    )

    history: List[Dict[str, Any]] = []
    best_state: Optional[Dict[str, torch.Tensor]] = None
    tracker = EarlyStopTracker(budget)

    print(
        f"\n===== {experiment_id} | tag={run_tag} | mode={mode} | "
        f"norm={model.norm_name} | k={candidates} λ={lambda_a} | seed={seed} =====",
        flush=True,
    )

    for epoch in range(1, budget.max_epochs + 1):
        t0 = time.time()
        tr_acc, tr_loss, accept_rate, k_mean = train_one_epoch(
            model, opt, train_loader, device
        )
        te_acc, te_loss = evaluate(model, test_loader, device)
        model.assert_binary()
        a_w = mean_assembly_index(model.binary_parameters())
        nstats = model.norm_stats()
        dt = time.time() - t0
        row = {
            "epoch": epoch,
            "train_acc": tr_acc,
            "train_loss": tr_loss,
            "test_acc": te_acc,
            "test_loss": te_loss,
            "accept_rate": accept_rate,
            "candidates_mean": k_mean,
            "assembly_w": a_w,
            "delta_loss": opt.last_delta_loss,
            "delta_a": opt.last_delta_a,
            "epoch_sec": dt,
            **{f"norm_{k}": v for k, v in nstats.items()},
        }
        history.append(row)
        decision = tracker.observe(epoch, te_acc)
        if decision.improved:
            best_state = {
                k: v.detach().cpu().clone() for k, v in model.state_dict().items()
            }
        extra = ""
        if nstats:
            extra = (
                f" agree={nstats['agreement']:.3f} "
                f"nscale={nstats['scale']:.3f} |"
            )
        print(
            f"epoch {epoch:03d} | train={tr_acc:.4f} test={te_acc:.4f} | "
            f"accpt={accept_rate:.3f} ÂW={a_w:.3f} |{extra} "
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
    model.assert_binary()
    a_w_final = mean_assembly_index(model.binary_parameters())

    out: Dict[str, Any] = {
        "experiment": experiment_id,
        "run_tag": run_tag,
        "seed": seed,
        "hidden": hidden,
        "mode": mode,
        "candidates": candidates,
        "lambda_a": lambda_a,
        "ranking": ranking,
        "assembly_of": assembly_of,
        "use_bn": use_bn,
        "norm": model.norm_name,
        "bn_lr": bn_lr,
        "device": device,
        "epochs_ran": len(history),
        "budget": budget.to_dict(),
        "stop_meta": tracker.meta_dict(),
        "best_test_acc": tracker.best,
        "best_epoch": tracker.best_epoch,
        "final_test_acc": final_test_acc,
        "final_test_loss": final_test_loss,
        "final_assembly_w": a_w_final,
        "wall_sec": tracker.wall_sec,
        "history": history,
    }

    results_dir.mkdir(parents=True, exist_ok=True)
    safe_tag = run_tag.replace("/", "_")
    json_path = results_dir / f"{safe_tag}_seed{seed}.json"
    with open(json_path, "w") as f:
        json.dump(out, f, indent=2)

    ckpt_dir = REPO_ROOT / "checkpoints" / experiment_id
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
    print(f"Saved {ckpt_path}", flush=True)

    try:
        cfg = enrich_config(
            experiment_id,
            {
                "run_tag": run_tag,
                "hidden": hidden,
                "mode": mode,
                "candidates": candidates,
                "lambda_a": lambda_a,
                "ranking": ranking,
                "assembly_of": assembly_of,
                "use_bn": use_bn,
                "norm": model.norm_name,
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
            print(f"Stored run_id={rid} experiment={experiment_id}", flush=True)
    except Exception as exc:  # noqa: BLE001
        print(f"Warning: DuckDB record skipped: {exc}", flush=True)

    return out


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="v0_13 pathway optimizer training")
    add_budget_args(p)
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
    p.add_argument(
        "--norm",
        choices=("none", "pathway", "layernorm", "bn"),
        default="none",
        help="Hidden-activation norm after the first binary linear",
    )
    p.add_argument(
        "--bn",
        action="store_true",
        help="Deprecated: same as --norm bn",
    )
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
    results_dir = Path(args.results_dir or (_REPO / "results" / args.experiment_id))
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
        mode=args.mode,
        candidates=args.candidates,
        lambda_a=args.lambda_a,
        ranking=args.ranking,
        assembly_of=args.assembly_of,
        bn_lr=args.bn_lr,
        use_bn=args.bn,
        norm=args.norm,
        seed=args.seed,
        device=device,
        train_loader=train_loader,
        test_loader=test_loader,
        results_dir=results_dir,
        run_tag=args.run_tag,
    )


if __name__ == "__main__":
    main()
