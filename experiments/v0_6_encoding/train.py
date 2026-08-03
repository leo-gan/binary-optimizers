#!/usr/bin/env python3
"""Encoding atlas: fixed total n, vary structure (WP2)."""

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

from binary_optimizers.data.mnist import make_mnist_loaders  # noqa: E402
from binary_optimizers.training.loops import set_seed  # noqa: E402

from model import BitNetEncodingMLP  # noqa: E402
from optimizer import SwarmOptimizerV06  # noqa: E402

# --- step scaling (same spirit as v0.5 register) ---


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
    """One atlas cell."""

    kind: str  # encoding | unary
    n_bits: int | None = None
    encoding: str | None = None
    n_exp: int = 0
    swarm_size: int | None = None

    def tag(self) -> str:
        if self.kind == "unary":
            return f"unary_S{self.swarm_size}"
        if self.encoding == "fixed":
            return f"fixed_n{self.n_bits}"
        return f"{self.encoding}{self.n_exp}_n{self.n_bits}"

    def to_dict(self) -> dict[str, Any]:
        return {
            "kind": self.kind,
            "n_bits": self.n_bits,
            "encoding": self.encoding,
            "n_exp": self.n_exp,
            "swarm_size": self.swarm_size,
            "tag": self.tag(),
        }


def default_cells(*, include_rescue: bool) -> list[Cell]:
    cells: list[Cell] = []
    for n in (8, 16):
        cells.append(Cell(kind="encoding", n_bits=n, encoding="fixed", n_exp=0))
        for ne in (2, 3, 4):
            if ne < n:
                cells.append(
                    Cell(kind="encoding", n_bits=n, encoding="exp_mant", n_exp=ne)
                )
        cells.append(Cell(kind="encoding", n_bits=n, encoding="block_scale", n_exp=3))
    cells.append(Cell(kind="unary", swarm_size=256))
    if include_rescue:
        n = 32
        cells.append(Cell(kind="encoding", n_bits=n, encoding="fixed", n_exp=0))
        cells.append(Cell(kind="encoding", n_bits=n, encoding="exp_mant", n_exp=3))
        cells.append(Cell(kind="encoding", n_bits=n, encoding="block_scale", n_exp=3))
    return cells


def parse_cells(s: str) -> list[Cell]:
    """Parse 'fixed@8,exp_mant:2@8,unary:256'."""
    out: list[Cell] = []
    for part in s.split(","):
        part = part.strip()
        if not part:
            continue
        if part.startswith("unary:") or part.startswith("unary@"):
            S = int(part.split(":")[-1].split("@")[-1])
            out.append(Cell(kind="unary", swarm_size=S))
            continue
        # encoding@n or encoding:ne@n
        if "@" not in part:
            raise ValueError(f"cell needs @n_bits: {part}")
        left, n_s = part.rsplit("@", 1)
        n = int(n_s)
        if left == "fixed":
            out.append(Cell(kind="encoding", n_bits=n, encoding="fixed", n_exp=0))
        elif left.startswith("exp_mant:"):
            ne = int(left.split(":")[1])
            out.append(Cell(kind="encoding", n_bits=n, encoding="exp_mant", n_exp=ne))
        elif left.startswith("block_scale:"):
            ne = int(left.split(":")[1])
            out.append(
                Cell(kind="encoding", n_bits=n, encoding="block_scale", n_exp=ne)
            )
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


def _run_encoding_cell(
    cell: Cell,
    *,
    args,
    device: str,
    train_loader,
    test_loader,
    results_dir: Path,
) -> dict[str, Any]:
    assert cell.n_bits is not None and cell.encoding is not None
    set_seed(args.seed)
    n_mant = (
        cell.n_bits
        if cell.encoding != "exp_mant"
        else cell.n_bits - cell.n_exp
    )
    max_step_w, step_scale_w = scale_steps_for_mant(
        n_mant, args.max_step, args.step_scale
    )
    model = BitNetEncodingMLP(
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
    return _train_loop(
        model,
        opt,
        cell,
        args=args,
        device=device,
        train_loader=train_loader,
        test_loader=test_loader,
        results_dir=results_dir,
        extra={
            "max_step_used": max_step_w,
            "step_scale_used": step_scale_w,
            "n_mant": n_mant,
        },
        assert_fn=model.assert_binary_invariants,
        stats_fn=model.weight_stats,
    )


def _load_v01():
    """Load v0.1 model/optimizer under unique module names (avoid path clashes)."""
    import importlib.util

    v01 = _REPO / "experiments" / "v0_1"
    # v0.1 modules import bare "layers" etc.
    if str(v01) not in sys.path:
        sys.path.insert(0, str(v01))
    for bare in ("layers", "model", "optimizer", "metrics"):
        sys.modules.pop(bare, None)
    loaded = {}
    for bare in ("layers", "model", "optimizer"):
        path = v01 / f"{bare}.py"
        spec = importlib.util.spec_from_file_location(f"v01_{bare}", path)
        assert spec and spec.loader
        mod = importlib.util.module_from_spec(spec)
        sys.modules[f"v01_{bare}"] = mod
        sys.modules[bare] = mod  # for sibling imports during exec
        spec.loader.exec_module(mod)
        loaded[bare] = mod
    return loaded["model"].BitNetSwarmMLP, loaded["optimizer"].SwarmOptimizerV01


def _run_unary_cell(
    cell: Cell,
    *,
    args,
    device: str,
    train_loader,
    test_loader,
    results_dir: Path,
) -> dict[str, Any]:
    assert cell.swarm_size is not None
    BitNetSwarmMLP, SwarmOptimizerV01 = _load_v01()
    set_seed(args.seed)
    model = BitNetSwarmMLP(
        hidden_dim=args.hidden,
        swarm_size=cell.swarm_size,
        ln_mode=args.ln_mode,  # type: ignore[arg-type]
        activation=args.activation,  # type: ignore[arg-type]
    ).to(device)
    opt = SwarmOptimizerV01(
        model.swarm_layers(),
        recruit_rate=args.recruit_rate,
        max_flip_prob=args.max_flip_prob,
        grad_momentum=args.grad_momentum,
        ln_params=model.ln_parameters(),
        ln_lr=args.ln_lr,
    )

    def _stats():
        w = torch.cat([L.effective_weight().reshape(-1) for L in model.swarm_layers()])
        return {
            "mean_abs_w": float(w.abs().mean().item()),
            "std_w": float(w.std().item()),
            "frac_near_pm1": float((w.abs() > 0.9).float().mean().item()),
        }

    return _train_loop(
        model,
        opt,
        cell,
        args=args,
        device=device,
        train_loader=train_loader,
        test_loader=test_loader,
        results_dir=results_dir,
        extra={"swarm_size": cell.swarm_size},
        assert_fn=model.assert_binary_invariants,
        stats_fn=_stats,
    )


def _train_loop(
    model,
    opt,
    cell: Cell,
    *,
    args,
    device: str,
    train_loader,
    test_loader,
    results_dir: Path,
    extra: dict[str, Any],
    assert_fn,
    stats_fn,
) -> dict[str, Any]:
    history = []
    best_test, best_epoch, stalled = -1.0, 0, 0
    best_state = None
    print(f"\n===== v0_6 | {cell.tag()} | ln={args.ln_mode} | seed={args.seed} =====", flush=True)
    t0_run = time.time()
    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        tr_acc, tr_loss, flip = train_one_epoch(model, opt, train_loader, device)
        te_acc, te_loss = evaluate(model, test_loader, device)
        assert_fn()
        stats = stats_fn()
        row = {
            "epoch": epoch,
            "train_acc": tr_acc,
            "train_loss": tr_loss,
            "test_acc": te_acc,
            "test_loss": te_loss,
            "flip_frac": flip,
            "step_frac": getattr(opt, "last_step_frac", None),
            "epoch_sec": time.time() - t0,
            **{f"stat_{k}": v for k, v in stats.items()},
        }
        history.append(row)
        if te_acc > best_test + args.min_delta:
            best_test, best_epoch, stalled = te_acc, epoch, 0
            best_state = {
                k: v.detach().cpu().clone() for k, v in model.state_dict().items()
            }
        else:
            stalled += 1
        print(
            f"  ep {epoch:03d}/{args.epochs} train={tr_acc:.4f} test={te_acc:.4f} "
            f"best={best_test:.4f}@{best_epoch} stall={stalled}/{args.patience} "
            f"({row['epoch_sec']:.1f}s)",
            flush=True,
        )
        if stalled >= args.patience:
            print("  Early stop.", flush=True)
            break
    if best_state is not None:
        model.load_state_dict(best_state)
        model.to(device)
    final_acc, final_loss = evaluate(model, test_loader, device)
    final_stats = stats_fn()
    out = {
        "experiment": "v0_6_encoding",
        **cell.to_dict(),
        "ln_mode": args.ln_mode,
        "seed": args.seed,
        "hidden": args.hidden,
        "best_test_acc": best_test,
        "best_epoch": best_epoch,
        "epochs_ran": len(history),
        "final_test_acc": final_acc,
        "final_test_loss": final_loss,
        "wall_sec": time.time() - t0_run,
        "final_stats": final_stats,
        "history": history,
        "status": "completed",
        **extra,
    }
    results_dir.mkdir(parents=True, exist_ok=True)
    jp = results_dir / f"{cell.tag()}_ln_{args.ln_mode}_seed{args.seed}.json"
    with open(jp, "w") as f:
        json.dump(out, f, indent=2)
    print(f"  Saved {jp}", flush=True)
    return out


def main() -> None:
    p = argparse.ArgumentParser(description="v0.6 encoding atlas")
    p.add_argument(
        "--cells",
        type=str,
        default=None,
        help="Comma list e.g. fixed@8,exp_mant:2@8,unary:256 (default = plan grid)",
    )
    p.add_argument("--include-rescue", action="store_true", help="Add n=32 rescue cells")
    p.add_argument("--ln-mode", type=str, default="none")
    p.add_argument("--epochs", type=int, default=80)
    p.add_argument("--patience", type=int, default=5)
    p.add_argument("--min-delta", type=float, default=5e-4)
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
    p.add_argument("--max-flip-prob", type=float, default=0.15, help="unary only")
    p.add_argument("--grad-momentum", type=float, default=0.9)
    p.add_argument("--ln-lr", type=float, default=1e-2)
    p.add_argument("--activation", type=str, default="relu")
    args = p.parse_args()

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    data_root = args.data_root or str(_REPO / "data")
    results_dir = Path(args.results_dir or (_REPO / "results" / "v0_6_encoding"))
    cells = (
        parse_cells(args.cells)
        if args.cells
        else default_cells(include_rescue=args.include_rescue)
    )

    train_loader, test_loader = make_mnist_loaders(
        root=data_root,
        batch_size_train=args.batch_size,
        batch_size_test=1000,
        num_workers=0,
    )

    curve: list[dict[str, Any]] = []
    for cell in cells:
        try:
            if cell.kind == "unary":
                out = _run_unary_cell(
                    cell,
                    args=args,
                    device=device,
                    train_loader=train_loader,
                    test_loader=test_loader,
                    results_dir=results_dir,
                )
            else:
                out = _run_encoding_cell(
                    cell,
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
                    "final_stats": out.get("final_stats"),
                    "status": "completed",
                }
            )
        except Exception as e:
            print(f"FAILED {cell.tag()}: {e}", flush=True)
            traceback.print_exc()
            curve.append({**cell.to_dict(), "best_test_acc": None, "status": f"error:{type(e).__name__}"})

    summary = {
        "experiment": "v0_6_encoding",
        "seed": args.seed,
        "ln_mode": args.ln_mode,
        "hidden": args.hidden,
        "patience": args.patience,
        "note": (
            "WP2 encoding atlas: fixed n from WP1; vary structure; "
            "unary S=256 baseline; no per-cell hparam search"
        ),
        "curve": curve,
    }
    results_dir.mkdir(parents=True, exist_ok=True)
    sp = results_dir / f"summary_ln_{args.ln_mode}_seed{args.seed}.json"
    with open(sp, "w") as f:
        json.dump(summary, f, indent=2)
    # CSV
    csv_path = results_dir / f"curve_ln_{args.ln_mode}_seed{args.seed}.csv"
    keys = ["tag", "kind", "n_bits", "encoding", "n_exp", "swarm_size", "best_test_acc", "best_epoch", "epochs_ran", "wall_sec", "status"]
    with open(csv_path, "w") as f:
        f.write(",".join(keys) + "\n")
        for r in curve:
            f.write(",".join(str(r.get(k, "")) for k in keys) + "\n")
    print("\n===== v0_6_encoding curve =====", flush=True)
    for r in curve:
        print(
            f"  {r.get('tag', '?'):20s}  best={r.get('best_test_acc')}  status={r.get('status')}",
            flush=True,
        )
    print(f"Summary: {sp}", flush=True)


if __name__ == "__main__":
    main()
