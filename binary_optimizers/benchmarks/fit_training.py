"""
Fit-scale training with checkpoint cache.

Trains only when model/optimizer fingerprint is missing; otherwise loads
saved weights and reuses metrics. Benchmarks can load checkpoints without
retraining.
"""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Optional

import torch
import torch.nn as nn

from binary_optimizers.models.mnist import create_mnist_bit_mlp, create_mnist_bit_mlp_large
from binary_optimizers.optimizers.cosine_voting import CosineVotingOptimizer
from binary_optimizers.optimizers.ema_flip import EMAFlipOptimizer
from binary_optimizers.optimizers.hybrid_accumulator import HybridAccumulatorOptimizer
from binary_optimizers.optimizers.signum import MomentumVotingOptimizer
from binary_optimizers.optimizers.sparse_sign import SparseSignOptimizer
from binary_optimizers.optimizers.ste import STEOptimizer
from binary_optimizers.optimizers.threshold_if import ThresholdedIntegrateFireOptimizer
from binary_optimizers.optimizers.voting import VotingOptimizer
from binary_optimizers.profiling.memory import profile_model_memory
from binary_optimizers.training.checkpoints import (
    CHECKPOINT_SCHEMA_VERSION,
    TrainSpec,
    checkpoint_exists,
    load_checkpoint,
    load_meta,
    save_checkpoint,
)
from binary_optimizers.training.loops import (
    evaluate_accuracy,
    set_seed,
    train_one_epoch_classification,
)
from binary_optimizers.training.param_groups import split_binary_and_bn_params


FIT_OPTIMIZERS = (
    "adam",
    "ste",
    "voting",
    "signum",
    "threshold_if",
    "ema_flip",
    "cosine_voting",
    "sparse_sign",
    "hybrid_accumulator",
)

NEW_OPTIMIZERS = (
    "ema_flip",
    "cosine_voting",
    "sparse_sign",
    "hybrid_accumulator",
)

# Tag identifies this experiment series (bump via schema when optimizers change).
FIT_TAG = "fit_v2"


@dataclass
class FitTrial:
    trial: int
    seed: int
    per_epoch_train_acc: list[float] = field(default_factory=list)
    per_epoch_test_acc: list[float] = field(default_factory=list)
    per_epoch_time_s: list[float] = field(default_factory=list)
    total_time_s: float = 0.0
    best_test_acc: float = 0.0
    best_epoch: int = 0
    final_train_acc: float = 0.0
    final_test_acc: float = 0.0
    train_test_gap: float = 0.0
    memory: dict[str, Any] = field(default_factory=dict)
    from_cache: bool = False
    checkpoint_slug: str = ""
    packed_test_acc: float | None = None


@dataclass
class FitConfigResult:
    name: str
    optimizer: str
    model: str
    epochs: int
    trials: list[FitTrial] = field(default_factory=list)

    def summary(self) -> dict[str, Any]:
        finals = [t.final_test_acc for t in self.trials]
        bests = [t.best_test_acc for t in self.trials]
        times = [t.total_time_s for t in self.trials]
        gaps = [t.train_test_gap for t in self.trials]
        packed = [t.packed_test_acc for t in self.trials if t.packed_test_acc is not None]
        mean = sum(finals) / len(finals) if finals else 0.0
        best_mean = sum(bests) / len(bests) if bests else 0.0
        std = 0.0
        if len(finals) > 1:
            std = (sum((x - mean) ** 2 for x in finals) / (len(finals) - 1)) ** 0.5
        mem = self.trials[0].memory if self.trials else {}
        n_ep = max((len(t.per_epoch_test_acc) for t in self.trials), default=0)
        curve, train_curve = [], []
        for i in range(n_ep):
            vals = [t.per_epoch_test_acc[i] for t in self.trials if i < len(t.per_epoch_test_acc)]
            curve.append(sum(vals) / len(vals) if vals else 0.0)
            vals_tr = [t.per_epoch_train_acc[i] for t in self.trials if i < len(t.per_epoch_train_acc)]
            train_curve.append(sum(vals_tr) / len(vals_tr) if vals_tr else 0.0)
        return {
            "name": self.name,
            "optimizer": self.optimizer,
            "model": self.model,
            "epochs": self.epochs,
            "n_trials": len(self.trials),
            "final_test_acc_mean": mean,
            "final_test_acc_std": std,
            "best_test_acc_mean": best_mean,
            "total_time_mean_s": sum(times) / len(times) if times else 0.0,
            "train_test_gap_mean": sum(gaps) / len(gaps) if gaps else 0.0,
            "packed_test_acc_mean": sum(packed) / len(packed) if packed else None,
            "per_epoch_test_acc_mean": curve,
            "per_epoch_train_acc_mean": train_curve,
            "memory": mem,
            "is_new": self.optimizer in NEW_OPTIMIZERS,
            "from_cache": all(t.from_cache for t in self.trials) if self.trials else False,
            "checkpoint_slugs": [t.checkpoint_slug for t in self.trials],
        }


def _make_model(model_name: str) -> nn.Module:
    if model_name == "bit_mlp":
        return create_mnist_bit_mlp(hidden_dim=128)
    if model_name == "bit_mlp_large":
        return create_mnist_bit_mlp_large(hidden_dim=256)
    raise ValueError(f"Unknown model: {model_name}")


def default_optimizer_kwargs(
    name: str,
    *,
    steps_per_epoch: int,
    epochs: int,
) -> dict[str, Any]:
    """Canonical hyperparams (included in checkpoint fingerprint)."""
    total_steps = max(1, steps_per_epoch * epochs)
    if name == "adam":
        return {"lr": 1e-3}
    if name == "ste":
        return {"lr": 0.1, "momentum": 0.9}
    if name == "voting":
        return {"lr": 0.1, "momentum": 0.9, "push_rate": 0.3, "clip": 1.0}
    if name == "signum":
        return {"lr": 5e-3, "momentum": 0.9, "clip": 1.5}
    if name == "threshold_if":
        return {"lr": 0.1, "threshold": 0.02, "decay": 0.99}
    if name == "ema_flip":
        return {
            "lr": 0.05,
            "lr_min": 0.005,
            "momentum": 0.9,
            "threshold_scale": 0.35,
            "clip": 1.5,
            "flip_mode": False,
            "total_steps": total_steps,
        }
    if name == "cosine_voting":
        return {
            "lr_max": 0.1,
            "lr_min": 0.005,
            "momentum": 0.9,
            "total_steps": total_steps,
            "clip": 1.5,
            "confidence_threshold": 0.0,
            "restart_period": max(1, total_steps // 3),
        }
    if name == "sparse_sign":
        return {
            "lr": 0.05,
            "momentum": 0.9,
            "density": 0.8,
            "density_start": 1.0,
            "density_end": 0.4,
            "total_steps": total_steps,
            "clip": 1.5,
            "magnitude_biased": True,
        }
    if name == "hybrid_accumulator":
        return {
            "lr": 0.15,
            "momentum": 0.9,
            "init_threshold": 0.03,
            "target_fire_rate": 0.08,
            "decay": 0.95,
            "clip": 1.5,
            "soft_fire": True,
            "soft_alpha": 0.5,
        }
    raise ValueError(f"Unknown optimizer: {name}")


def _make_optimizer(
    name: str,
    model: nn.Module,
    kwargs: dict[str, Any],
) -> torch.optim.Optimizer:
    binary, continuous = split_binary_and_bn_params(model)
    # Most binary optimizers already special-case dim<2; still pass all params.
    # For adam, use full model; for others, single group is fine.
    params = model.parameters()
    if name == "adam":
        return torch.optim.Adam(params, lr=kwargs.get("lr", 1e-3))
    if name == "ste":
        return STEOptimizer(params, lr=kwargs["lr"], momentum=kwargs["momentum"])
    if name == "voting":
        return VotingOptimizer(
            params,
            lr=kwargs["lr"],
            momentum=kwargs["momentum"],
            push_rate=kwargs["push_rate"],
            clip=kwargs["clip"],
        )
    if name == "signum":
        return MomentumVotingOptimizer(
            params, lr=kwargs["lr"], momentum=kwargs["momentum"], clip=kwargs["clip"]
        )
    if name == "threshold_if":
        return ThresholdedIntegrateFireOptimizer(
            params,
            lr=kwargs["lr"],
            threshold=kwargs["threshold"],
            decay=kwargs["decay"],
        )
    if name == "ema_flip":
        return EMAFlipOptimizer(params, **kwargs)
    if name == "cosine_voting":
        return CosineVotingOptimizer(params, **kwargs)
    if name == "sparse_sign":
        return SparseSignOptimizer(params, **kwargs)
    if name == "hybrid_accumulator":
        return HybridAccumulatorOptimizer(params, **kwargs)
    raise ValueError(f"Unknown optimizer: {name}")


def make_train_spec(
    *,
    model_name: str,
    optimizer: str,
    epochs: int,
    seed: int,
    batch_size_train: int,
    steps_per_epoch: int,
    tag: str = FIT_TAG,
) -> TrainSpec:
    kwargs = default_optimizer_kwargs(
        optimizer, steps_per_epoch=steps_per_epoch, epochs=epochs
    )
    return TrainSpec(
        model=model_name,
        optimizer=optimizer,
        optimizer_kwargs=kwargs,
        dataset="mnist",
        epochs=epochs,
        seed=seed,
        batch_size_train=batch_size_train,
        schema_version=CHECKPOINT_SCHEMA_VERSION,
        tag=tag,
    )


@torch.no_grad()
def evaluate_packed_accuracy(model: nn.Module, loader, device: str) -> float:
    """
    Accuracy with 2D weights forced to ±1 (deployment / bitpacked proxy).

    Copies the model, signs all 2D parameters, evaluates STE/binary forward.
    BN and 1D params are left unchanged.
    """
    import copy

    m = copy.deepcopy(model).to(device)
    m.eval()
    for p in m.parameters():
        if p.dim() >= 2:
            p.data = torch.where(p.data >= 0, torch.ones_like(p.data), -torch.ones_like(p.data))
    return evaluate_accuracy(m, loader, device)


def train_or_load(
    spec: TrainSpec,
    *,
    train_loader,
    test_loader,
    device: str,
    checkpoint_root: str | Path = "checkpoints",
    force_retrain: bool = False,
    eval_packed: bool = True,
) -> tuple[nn.Module, FitTrial]:
    """
    Load checkpoint if fingerprint matches; otherwise train, save, return trial metrics.
    """
    model = _make_model(spec.model).to(device)
    steps_per_epoch = len(train_loader)

    if not force_retrain and checkpoint_exists(spec, checkpoint_root):
        print(f"  [cache hit] {spec.slug()}")
        meta = load_checkpoint(spec, model, root=checkpoint_root, map_location=device)
        m = meta.get("metrics") or {}
        trial = FitTrial(
            trial=0,
            seed=spec.seed,
            per_epoch_train_acc=list(m.get("per_epoch_train_acc") or []),
            per_epoch_test_acc=list(m.get("per_epoch_test_acc") or []),
            per_epoch_time_s=list(m.get("per_epoch_time_s") or []),
            total_time_s=float(m.get("total_time_s") or 0.0),
            best_test_acc=float(m.get("best_test_acc") or 0.0),
            best_epoch=int(m.get("best_epoch") or 0),
            final_train_acc=float(m.get("final_train_acc") or 0.0),
            final_test_acc=float(m.get("final_test_acc") or 0.0),
            train_test_gap=float(m.get("train_test_gap") or 0.0),
            memory=dict(m.get("memory") or {}),
            from_cache=True,
            checkpoint_slug=spec.slug(),
            packed_test_acc=m.get("packed_test_acc"),
        )
        # Optional re-benchmark packed acc without retraining
        if eval_packed and trial.packed_test_acc is None:
            trial.packed_test_acc = evaluate_packed_accuracy(model, test_loader, device)
            # refresh meta
            m["packed_test_acc"] = trial.packed_test_acc
            save_checkpoint(spec, model, m, root=checkpoint_root)
        return model, trial

    print(f"  [train] {spec.slug()}")
    set_seed(spec.seed)
    model = _make_model(spec.model).to(device)
    opt_kwargs = dict(spec.optimizer_kwargs)
    optimizer = _make_optimizer(spec.optimizer, model, opt_kwargs)

    # Warm-up for memory profile
    model.train()
    xb, yb = next(iter(train_loader))
    xb, yb = xb.to(device), yb.to(device)
    optimizer.zero_grad()
    model(xb).sum().backward()
    optimizer.step()
    optimizer.zero_grad()
    mem = profile_model_memory(model, optimizer).to_dict()

    trial = FitTrial(trial=0, seed=spec.seed, memory=mem, checkpoint_slug=spec.slug())
    best = 0.0
    best_ep = 0
    t0_all = time.perf_counter()

    for epoch in range(spec.epochs):
        t0 = time.perf_counter()
        tr = train_one_epoch_classification(model, optimizer, train_loader, device)
        te = evaluate_accuracy(model, test_loader, device)
        dt = time.perf_counter() - t0
        trial.per_epoch_train_acc.append(tr)
        trial.per_epoch_test_acc.append(te)
        trial.per_epoch_time_s.append(dt)
        if te > best:
            best = te
            best_ep = epoch + 1
        print(
            f"    ep {epoch+1}/{spec.epochs} train={tr:.4f} test={te:.4f} ({dt:.1f}s)"
        )

    trial.total_time_s = time.perf_counter() - t0_all
    trial.final_train_acc = trial.per_epoch_train_acc[-1]
    trial.final_test_acc = trial.per_epoch_test_acc[-1]
    trial.best_test_acc = best
    trial.best_epoch = best_ep
    trial.train_test_gap = trial.final_train_acc - trial.final_test_acc
    trial.from_cache = False

    if eval_packed:
        trial.packed_test_acc = evaluate_packed_accuracy(model, test_loader, device)
        print(f"    packed_test={trial.packed_test_acc:.4f}")

    metrics = {
        "per_epoch_train_acc": trial.per_epoch_train_acc,
        "per_epoch_test_acc": trial.per_epoch_test_acc,
        "per_epoch_time_s": trial.per_epoch_time_s,
        "total_time_s": trial.total_time_s,
        "best_test_acc": trial.best_test_acc,
        "best_epoch": trial.best_epoch,
        "final_train_acc": trial.final_train_acc,
        "final_test_acc": trial.final_test_acc,
        "train_test_gap": trial.train_test_gap,
        "memory": trial.memory,
        "packed_test_acc": trial.packed_test_acc,
        "steps_per_epoch": steps_per_epoch,
    }
    paths = save_checkpoint(spec, model, metrics, root=checkpoint_root)
    print(f"    saved {paths.weights}")
    return model, trial


def run_fit_training(
    *,
    optimizers: tuple[str, ...] | list[str] | None = None,
    model_name: str = "bit_mlp",
    epochs: int = 15,
    num_trials: int = 1,
    seed: int = 42,
    batch_size_train: int = 128,
    batch_size_test: int = 512,
    device: Optional[str] = None,
    data_root: str = "./data",
    checkpoint_root: str | Path = "checkpoints",
    force_retrain: bool = False,
    output_json: str | Path = "results/fit_training.json",
    eval_packed: bool = True,
) -> dict[str, Any]:
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    optimizers = tuple(optimizers or FIT_OPTIMIZERS)

    from binary_optimizers.data.mnist import make_mnist_loaders

    train_loader, test_loader = make_mnist_loaders(
        root=data_root,
        batch_size_train=batch_size_train,
        batch_size_test=batch_size_test,
        num_workers=0,
    )
    steps_per_epoch = len(train_loader)
    all_results: list[FitConfigResult] = []

    for opt_name in optimizers:
        cfg_name = f"{model_name}_{opt_name}"
        print(f"\n=== FIT {cfg_name} ===")
        result = FitConfigResult(
            name=cfg_name, optimizer=opt_name, model=model_name, epochs=epochs
        )
        for trial_i in range(num_trials):
            trial_seed = seed + trial_i
            spec = make_train_spec(
                model_name=model_name,
                optimizer=opt_name,
                epochs=epochs,
                seed=trial_seed,
                batch_size_train=batch_size_train,
                steps_per_epoch=steps_per_epoch,
            )
            _model, trial = train_or_load(
                spec,
                train_loader=train_loader,
                test_loader=test_loader,
                device=device,
                checkpoint_root=checkpoint_root,
                force_retrain=force_retrain,
                eval_packed=eval_packed,
            )
            trial.trial = trial_i
            result.trials.append(trial)
        all_results.append(result)

    payload = {
        "meta": {
            "dataset": "mnist",
            "model": model_name,
            "epochs": epochs,
            "num_trials": num_trials,
            "seed": seed,
            "device": device,
            "batch_size_train": batch_size_train,
            "batch_size_test": batch_size_test,
            "steps_per_epoch": steps_per_epoch,
            "checkpoint_root": str(checkpoint_root),
            "force_retrain": force_retrain,
            "tag": FIT_TAG,
            "schema_version": CHECKPOINT_SCHEMA_VERSION,
            "note": (
                "Fit-scale MNIST with checkpoint cache. Retrain only if model/optimizer "
                "fingerprint changes or --force-retrain."
            ),
        },
        "configs": [{**asdict(r), "summary": r.summary()} for r in all_results],
        "summaries": [r.summary() for r in all_results],
    }
    path = Path(output_json)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2))
    print(f"\nWrote {path}")
    return payload


def benchmark_checkpoints(
    *,
    optimizers: tuple[str, ...] | list[str] | None = None,
    model_name: str = "bit_mlp",
    epochs: int = 15,
    seed: int = 42,
    batch_size_train: int = 128,
    batch_size_test: int = 512,
    device: Optional[str] = None,
    data_root: str = "./data",
    checkpoint_root: str | Path = "checkpoints",
    output_json: str | Path = "results/checkpoint_benchmark.json",
) -> dict[str, Any]:
    """
    Load saved models only (no training) and re-evaluate STE + packed accuracy.
    Skips optimizers with no checkpoint.
    """
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    optimizers = tuple(optimizers or FIT_OPTIMIZERS)
    from binary_optimizers.data.mnist import make_mnist_loaders

    _train_loader, test_loader = make_mnist_loaders(
        root=data_root,
        batch_size_train=batch_size_train,
        batch_size_test=batch_size_test,
        num_workers=0,
    )
    # steps_per_epoch from train set size / batch
    steps_per_epoch = len(_train_loader)
    rows = []
    for opt_name in optimizers:
        spec = make_train_spec(
            model_name=model_name,
            optimizer=opt_name,
            epochs=epochs,
            seed=seed,
            batch_size_train=batch_size_train,
            steps_per_epoch=steps_per_epoch,
        )
        if not checkpoint_exists(spec, checkpoint_root):
            print(f"  [skip missing] {spec.slug()}")
            rows.append(
                {
                    "optimizer": opt_name,
                    "status": "missing",
                    "slug": spec.slug(),
                }
            )
            continue
        model = _make_model(model_name).to(device)
        meta = load_checkpoint(spec, model, root=checkpoint_root, map_location=device)
        t0 = time.perf_counter()
        ste_acc = evaluate_accuracy(model, test_loader, device)
        packed_acc = evaluate_packed_accuracy(model, test_loader, device)
        dt = time.perf_counter() - t0
        m = meta.get("metrics") or {}
        row = {
            "optimizer": opt_name,
            "status": "ok",
            "slug": spec.slug(),
            "fingerprint": spec.fingerprint(),
            "ste_test_acc": ste_acc,
            "packed_test_acc": packed_acc,
            "cached_best_test_acc": m.get("best_test_acc"),
            "cached_final_test_acc": m.get("final_test_acc"),
            "eval_time_s": dt,
            "is_new": opt_name in NEW_OPTIMIZERS,
        }
        rows.append(row)
        print(
            f"  {opt_name:20s} ste={ste_acc:.4f} packed={packed_acc:.4f} "
            f"(cached best={m.get('best_test_acc')})"
        )

    payload = {
        "meta": {
            "model": model_name,
            "epochs": epochs,
            "seed": seed,
            "device": device,
            "checkpoint_root": str(checkpoint_root),
            "note": "Benchmark from checkpoints only — no training.",
        },
        "rows": rows,
    }
    path = Path(output_json)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2))
    print(f"Wrote {path}")
    return payload


# Re-export analysis helpers from prior module body (kept for reports)
def analyze_fit_results(payload: dict[str, Any]) -> dict[str, Any]:
    summaries = list(payload.get("summaries") or [])
    summaries_sorted = sorted(
        summaries, key=lambda s: s.get("best_test_acc_mean", 0.0), reverse=True
    )
    baselines = [s for s in summaries if not s.get("is_new")]
    news = [s for s in summaries if s.get("is_new")]
    base_by = {s["optimizer"]: s for s in baselines}
    new_by = {s["optimizer"]: s for s in news}

    win_matrix: dict[str, dict[str, Any]] = {}
    for n_name, n_s in new_by.items():
        beats = {
            b_name: n_s["best_test_acc_mean"] > b_s["best_test_acc_mean"]
            for b_name, b_s in base_by.items()
        }
        win_matrix[n_name] = {
            "beats": beats,
            "n_wins": sum(1 for v in beats.values() if v),
            "n_baselines": len(beats),
            "best_test": n_s["best_test_acc_mean"],
            "final_test": n_s["final_test_acc_mean"],
            "gap": n_s["train_test_gap_mean"],
            "time_s": n_s["total_time_mean_s"],
            "packed": n_s.get("packed_test_acc_mean"),
        }

    diagnostics = []
    for s in summaries_sorted:
        curve = s.get("per_epoch_test_acc_mean") or []
        train_c = s.get("per_epoch_train_acc_mean") or []
        peak = max(curve) if curve else 0.0
        late = curve[-1] if curve else 0.0
        peak_ep = (curve.index(peak) + 1) if curve else 0
        decay = peak - late
        issues = []
        if late < 0.5:
            issues.append("underfit")
        if s.get("train_test_gap_mean", 0) > 0.08:
            issues.append("overfit_gap")
        if decay > 0.05 and peak_ep < max(1, len(curve) - 2):
            issues.append("late_decay")
        diagnostics.append(
            {
                "optimizer": s["optimizer"],
                "is_new": s.get("is_new", False),
                "best_test": s["best_test_acc_mean"],
                "final_test": s["final_test_acc_mean"],
                "final_train": train_c[-1] if train_c else 0.0,
                "gap": s["train_test_gap_mean"],
                "peak_epoch": peak_ep,
                "late_decay": decay,
                "time_s": s["total_time_mean_s"],
                "packed": s.get("packed_test_acc_mean"),
                "from_cache": s.get("from_cache"),
                "issues": issues,
            }
        )

    best_base = max(baselines, key=lambda s: s["best_test_acc_mean"]) if baselines else None
    best_new = max(news, key=lambda s: s["best_test_acc_mean"]) if news else None

    proposals = [
        {
            "optimizer": "cache",
            "title": "Checkpoint cache is source of truth",
            "detail": (
                "Weights live under checkpoints/<slug>/. Re-run fit only with "
                "--force-retrain or after changing model/optimizer kwargs/schema. "
                "Use experiments/run_benchmark_checkpoints.py to re-evaluate only."
            ),
            "priority": "high",
        }
    ]
    if best_new and best_base:
        if best_new["best_test_acc_mean"] + 1e-9 < best_base["best_test_acc_mean"]:
            proposals.append(
                {
                    "optimizer": "all_new",
                    "title": "Close gap to best baseline",
                    "detail": (
                        f"Best new `{best_new['optimizer']}` "
                        f"({best_new['best_test_acc_mean']:.4f}) vs baseline "
                        f"`{best_base['optimizer']}` "
                        f"({best_base['best_test_acc_mean']:.4f}). Continue Hybrid v2 ablations."
                    ),
                    "priority": "high",
                }
            )

    return {
        "ranking": [
            {
                "rank": i + 1,
                "optimizer": s["optimizer"],
                "is_new": s.get("is_new", False),
                "best_test_acc_mean": s["best_test_acc_mean"],
                "final_test_acc_mean": s["final_test_acc_mean"],
                "train_test_gap_mean": s["train_test_gap_mean"],
                "total_time_mean_s": s["total_time_mean_s"],
                "packed_test_acc_mean": s.get("packed_test_acc_mean"),
                "from_cache": s.get("from_cache"),
            }
            for i, s in enumerate(summaries_sorted)
        ],
        "win_matrix": win_matrix,
        "diagnostics": diagnostics,
        "best_baseline": best_base["optimizer"] if best_base else None,
        "best_new": best_new["optimizer"] if best_new else None,
        "proposals": proposals,
    }


def _fmt_bytes(n: float | int) -> str:
    n = float(n or 0)
    if n >= 1 << 20:
        return f"{n / (1 << 20):.2f} MB"
    if n >= 1 << 10:
        return f"{n / (1 << 10):.1f} KB"
    return f"{int(n)} B"


def render_fit_report(payload: dict[str, Any], analysis: dict[str, Any]) -> str:
    meta = payload.get("meta") or {}
    lines = [
        "# Fit-Scale Training Report (checkpointed)",
        "",
        f"- **Model:** {meta.get('model')}  ",
        f"- **Epochs / trials:** {meta.get('epochs')} / {meta.get('num_trials')}  ",
        f"- **Device:** {meta.get('device')}  ",
        f"- **Checkpoint root:** `{meta.get('checkpoint_root')}`  ",
        f"- **Tag / schema:** `{meta.get('tag')}` / v{meta.get('schema_version')}  ",
        f"- **Note:** {meta.get('note')}",
        "",
        "Checkpoints are reused unless model, optimizer, hyperparams, epochs, or seed change.",
        "",
        "## Ranking",
        "",
        "| Rank | Opt | New | Best | Final | Packed | Gap | Time | Cache |",
        "| :---: | :--- | :---: | ---: | ---: | ---: | ---: | ---: | :---: |",
    ]
    for r in analysis["ranking"]:
        pk = r.get("packed_test_acc_mean")
        pk_s = f"{pk:.4f}" if pk is not None else "—"
        lines.append(
            f"| {r['rank']} | `{r['optimizer']}` | {'★' if r['is_new'] else ''} | "
            f"{r['best_test_acc_mean']:.4f} | {r['final_test_acc_mean']:.4f} | {pk_s} | "
            f"{r['train_test_gap_mean']:+.4f} | {r['total_time_mean_s']:.1f} | "
            f"{'yes' if r.get('from_cache') else 'no'} |"
        )
    lines += [
        "",
        f"**Best baseline:** `{analysis.get('best_baseline')}`  ",
        f"**Best new:** `{analysis.get('best_new')}`",
        "",
        "## Win matrix (best test)",
        "",
    ]
    base_names = [s["optimizer"] for s in payload.get("summaries") or [] if not s.get("is_new")]
    if base_names:
        lines.append("| New | " + " | ".join(f"`{b}`" for b in base_names) + " | Wins |")
        lines.append("| :--- | " + " | ".join([":---:"] * len(base_names)) + " | :---: |")
        for n in NEW_OPTIMIZERS:
            w = analysis["win_matrix"].get(n)
            if not w:
                continue
            marks = ["✅" if w["beats"].get(b) else "❌" for b in base_names]
            lines.append(
                f"| `{n}` | " + " | ".join(marks) + f" | **{w['n_wins']}/{w['n_baselines']}** |"
            )

    lines += ["", "## Diagnostics", ""]
    lines.append("| Opt | Best | Final | Packed | Issues | Cache |")
    lines.append("| :--- | ---: | ---: | ---: | :--- | :---: |")
    for d in analysis["diagnostics"]:
        pk = d.get("packed")
        pk_s = f"{pk:.4f}" if pk is not None else "—"
        issues = ", ".join(d["issues"]) if d["issues"] else "—"
        lines.append(
            f"| `{d['optimizer']}` | {d['best_test']:.4f} | {d['final_test']:.4f} | "
            f"{pk_s} | {issues} | {'yes' if d.get('from_cache') else 'no'} |"
        )

    lines += ["", "## Proposals", ""]
    for i, p in enumerate(analysis.get("proposals") or [], 1):
        lines += [
            f"### {i}. [{p.get('priority','medium').upper()}] `{p['optimizer']}` — {p['title']}",
            "",
            p["detail"],
            "",
        ]

    lines += [
        "## Reproduce",
        "",
        "```bash",
        "# Train only missing fingerprints (or after optimizer/model changes)",
        "uv run python experiments/run_fit_training.py",
        "# Force retrain everything",
        "uv run python experiments/run_fit_training.py --force-retrain",
        "# Re-benchmark saved nets only (no training)",
        "uv run python experiments/run_benchmark_checkpoints.py",
        "```",
        "",
    ]
    return "\n".join(lines)


def write_fit_report(
    payload: dict[str, Any],
    *,
    output_md: str | Path = "results/fit_training_report.md",
    analysis_json: str | Path = "results/fit_training_analysis.json",
) -> dict[str, Any]:
    analysis = analyze_fit_results(payload)
    Path(output_md).parent.mkdir(parents=True, exist_ok=True)
    Path(output_md).write_text(render_fit_report(payload, analysis))
    Path(analysis_json).write_text(json.dumps(analysis, indent=2))
    return analysis
