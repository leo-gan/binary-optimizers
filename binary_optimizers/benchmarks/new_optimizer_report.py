"""Build improvement proofs for the four new optimizers from training-sweep results."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from binary_optimizers.benchmarks.training_sweep import NEW_OPTIMIZER_NAMES


# Design rationale written once (not regenerated from numbers).
OPTIMIZER_RATIONALE: dict[str, dict[str, str]] = {
    "ema_flip": {
        "title": "EMAFlipOptimizer",
        "vs": "MomentumRank / confidence-gated voting",
        "idea": (
            "EMA of the gradient with an adaptive threshold gate replaces expensive "
            "topk ranking; flip/update only where smoothed signal is strong."
        ),
        "pros": "O(n) steps; self-tuning threshold; less oscillation than raw sign votes",
        "cons": "Warm-up while the running mean calibrates",
    },
    "cosine_voting": {
        "title": "CosineVotingOptimizer",
        "vs": "VotingOptimizer / MomentumVotingOptimizer",
        "idea": (
            "Momentum sign voting with a built-in cosine LR schedule so early steps "
            "explore and late steps refine without an external scheduler."
        ),
        "pros": "No external LR scheduler; often lowest wall-clock among voting family",
        "cons": "Accuracy sensitive to total_steps vs actual training length",
    },
    "sparse_sign": {
        "title": "SparseSignOptimizer",
        "vs": "STE_SGD / MomentumVoting (dense updates)",
        "idea": (
            "Each step updates only a random density fraction of confident weights, "
            "acting as weight-space dropout with denser effective LR on active set."
        ),
        "pros": "Implicit regularization; same momentum memory as MomentumVoting",
        "cons": "CPU mask overhead; density and lr must be co-tuned",
    },
    "hybrid_accumulator": {
        "title": "HybridAccumulatorOptimizer",
        "vs": "MomentumVoting + ThresholdedIntegrateFire",
        "idea": (
            "EMA gradient tracking + accumulate-then-fire with per-group adaptive "
            "threshold targeting a fire rate — only apply updates with strong evidence."
        ),
        "pros": "Noise-filtered updates; self-regulating fire rate",
        "cons": "Two buffers per param (EMA + accumulator); can over-dampen late",
    },
}

BASELINE_OPTIMIZERS_SMALL = (
    "adam",
    "ste",
    "voting",
    "signum",
    "threshold_if",
)


def _fmt_bytes(n: float | int) -> str:
    n = float(n)
    if n >= 1 << 20:
        return f"{n / (1 << 20):.2f} MB"
    if n >= 1 << 10:
        return f"{n / (1 << 10):.1f} KB"
    return f"{int(n)} B"


def _summary_index(summaries: list[dict[str, Any]], model: str) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    for s in summaries:
        if s.get("model") != model:
            continue
        out[s["optimizer"]] = s
    return out


def _train_mem(s: dict[str, Any]) -> float:
    return float(((s.get("memory") or {}).get("training") or {}).get("total_bytes", 0))


def _inf_mem(s: dict[str, Any]) -> float:
    return float(((s.get("memory") or {}).get("inference") or {}).get("bitpacked_bytes", 0))


def build_new_optimizer_comparison(
    training: dict[str, Any],
    *,
    model: str = "bit_mlp_small",
) -> dict[str, Any]:
    """
    Compare each new optimizer to baselines on the same model.

    Returns structured tables + win matrix used for proof-of-improvement docs.
    """
    by_opt = _summary_index(training.get("summaries", []), model)
    baselines = {k: by_opt[k] for k in BASELINE_OPTIMIZERS_SMALL if k in by_opt}
    news = {k: by_opt[k] for k in NEW_OPTIMIZER_NAMES if k in by_opt}

    rows: list[dict[str, Any]] = []
    for name, s in {**baselines, **news}.items():
        rows.append(
            {
                "optimizer": name,
                "is_new": name in NEW_OPTIMIZER_NAMES,
                "test_acc_mean": float(s.get("test_acc_mean", 0.0)),
                "test_acc_std": float(s.get("test_acc_std", 0.0)),
                "total_time_mean_s": float(s.get("total_time_mean_s", 0.0)),
                "train_mem_bytes": _train_mem(s),
                "infer_mem_bytes": _inf_mem(s),
                "per_epoch_test_acc_mean": list(s.get("per_epoch_test_acc_mean") or []),
            }
        )
    rows.sort(key=lambda r: r["test_acc_mean"], reverse=True)

    # Win matrix: new opt beats baseline on accuracy
    wins: dict[str, dict[str, Any]] = {}
    for new_name, new_s in news.items():
        beat: dict[str, bool] = {}
        for base_name, base_s in baselines.items():
            beat[base_name] = float(new_s["test_acc_mean"]) > float(base_s["test_acc_mean"])
        wins[new_name] = {
            "beat": beat,
            "n_wins": sum(1 for v in beat.values() if v),
            "n_baselines": len(beat),
            "acc": float(new_s["test_acc_mean"]),
            "time_s": float(new_s["total_time_mean_s"]),
            "train_mem": _train_mem(new_s),
        }

    # Sequential story: each new opt vs closest related baseline(s)
    sequential = [
        {
            "step": 1,
            "optimizer": "ema_flip",
            "targets": ["signum", "voting", "ste"],
            "claim": "Adaptive EMA gate vs dense momentum / STE updates",
        },
        {
            "step": 2,
            "optimizer": "cosine_voting",
            "targets": ["voting", "signum"],
            "claim": "Cosine-annealed voting vs fixed-lr voting family",
        },
        {
            "step": 3,
            "optimizer": "sparse_sign",
            "targets": ["ste", "signum"],
            "claim": "Sparse confident sign updates vs dense STE/signum",
        },
        {
            "step": 4,
            "optimizer": "hybrid_accumulator",
            "targets": ["threshold_if", "signum", "voting"],
            "claim": "EMA + adaptive fire vs fixed-threshold IF / voting",
        },
    ]
    for item in sequential:
        opt = item["optimizer"]
        if opt not in news:
            item["status"] = "missing"
            item["proofs"] = []
            continue
        new_acc = float(news[opt]["test_acc_mean"])
        new_time = float(news[opt]["total_time_mean_s"])
        new_mem = _train_mem(news[opt])
        proofs = []
        for t in item["targets"]:
            if t not in baselines:
                continue
            b = baselines[t]
            proofs.append(
                {
                    "baseline": t,
                    "acc_delta": new_acc - float(b["test_acc_mean"]),
                    "time_delta_s": new_time - float(b["total_time_mean_s"]),
                    "mem_delta_bytes": new_mem - _train_mem(b),
                    "beats_acc": new_acc > float(b["test_acc_mean"]),
                    "beats_or_ties_time": new_time <= float(b["total_time_mean_s"]) * 1.05,
                    "beats_or_ties_mem": new_mem <= _train_mem(b) * 1.05,
                }
            )
        item["status"] = "ok"
        item["new_acc"] = new_acc
        item["new_time_s"] = new_time
        item["new_train_mem"] = new_mem
        item["proofs"] = proofs
        item["n_acc_wins"] = sum(1 for p in proofs if p["beats_acc"])

    return {
        "model": model,
        "meta": training.get("meta", {}),
        "rows": rows,
        "wins": wins,
        "sequential": sequential,
        "rationale": OPTIMIZER_RATIONALE,
    }


def render_new_optimizer_markdown(comparison: dict[str, Any]) -> str:
    """Render a full summary section for new optimizers."""
    meta = comparison.get("meta") or {}
    lines: list[str] = [
        "# New Binary Optimizers — Design, Proofs, and Pareto",
        "",
        "This document completes the experiment task for four optimizers designed "
        "to improve on the existing suite. Numbers come from the **scaffolding** "
        "training sweep (short epochs/batches) so they show relative structure, "
        "not publishable full-train accuracy.",
        "",
        f"- **Model (primary):** `{comparison['model']}`",
        f"- **Epochs / trials:** {meta.get('bit_mlp_small_epochs', meta.get('epochs', '?'))} / "
        f"{meta.get('num_trials', '?')}",
        f"- **Device:** {meta.get('device', '?')}",
        f"- **Note:** {meta.get('note', 'scaffold')}",
        "",
        "> **Honest scaffold caveat:** On real MNIST subsets with few batches, some "
        "> warm-up-heavy optimizers (EMAFlip, HybridAccumulator) can lag STE/voting. "
        "> Wins below are **relative proofs under the scaffold protocol** (accuracy, "
        "> time, memory vs named baselines), not a claim of state-of-the-art training.",
        "",
        "Dependencies remain **PyTorch** + **NumPy** (project defaults).",
        "",
        "---",
        "",
        "## 1. Existing landscape (baselines in sweep)",
        "",
        "| Family | Optimizer key | Mechanism |",
        "| :--- | :--- | :--- |",
        "| STE | `ste`, `adam` | Continuous weights + clamp / Adam moments |",
        "| Voting | `voting`, `signum` | Sign votes ± momentum |",
        "| Integrate-and-fire | `threshold_if` | Accumulate then fire updates |",
        "| Swarm | `swarm` | Population of binary weights |",
        "",
        "---",
        "",
        "## 2. Sequential design rationale",
        "",
    ]

    for name in NEW_OPTIMIZER_NAMES:
        r = comparison["rationale"][name]
        lines += [
            f"### 2.{list(NEW_OPTIMIZER_NAMES).index(name)+1} {r['title']} (`{name}`)",
            "",
            f"**Improves on:** {r['vs']}",
            "",
            f"**Core idea:** {r['idea']}",
            "",
            f"- *Pros:* {r['pros']}",
            f"- *Cons:* {r['cons']}",
            "",
        ]

    lines += [
        "---",
        "",
        "## 3. Scaffold results (same model)",
        "",
        "| Optimizer | New? | Test Acc (mean ± std) | Train time (s) | Train mem | Infer bitpacked |",
        "| :--- | :---: | ---: | ---: | ---: | ---: |",
    ]
    for row in comparison["rows"]:
        flag = "★" if row["is_new"] else ""
        lines.append(
            f"| `{row['optimizer']}` {flag} | {'yes' if row['is_new'] else 'no'} | "
            f"{row['test_acc_mean']:.4f} ± {row['test_acc_std']:.4f} | "
            f"{row['total_time_mean_s']:.2f} | {_fmt_bytes(row['train_mem_bytes'])} | "
            f"{_fmt_bytes(row['infer_mem_bytes'])} |"
        )

    lines += [
        "",
        "### Convergence (mean test acc per epoch)",
        "",
        "| Optimizer | " + " | ".join(
            f"Ep {i+1}" for i in range(max((len(r['per_epoch_test_acc_mean']) for r in comparison['rows']), default=0))
        ) + " |",
        "| :--- | " + " | ".join([":---:" ] * max((len(r['per_epoch_test_acc_mean']) for r in comparison['rows']), default=0)) + " |",
    ]
    n_ep = max((len(r["per_epoch_test_acc_mean"]) for r in comparison["rows"]), default=0)
    if n_ep:
        for row in comparison["rows"]:
            cells = []
            for i in range(n_ep):
                curve = row["per_epoch_test_acc_mean"]
                cells.append(f"{curve[i]:.3f}" if i < len(curve) else "—")
            lines.append(f"| `{row['optimizer']}` | " + " | ".join(cells) + " |")

    lines += [
        "",
        "---",
        "",
        "## 4. Proof of improvement (accuracy win matrix)",
        "",
        "A ✅ means the new optimizer's mean test accuracy **exceeds** that baseline "
        f"on `{comparison['model']}` under the scaffold protocol.",
        "",
    ]
    base_names = list(BASELINE_OPTIMIZERS_SMALL)
    header = "| New optimizer | " + " | ".join(f"`{b}`" for b in base_names) + " | Wins |"
    sep = "| :--- | " + " | ".join([":---:"] * len(base_names)) + " | :---: |"
    lines += [header, sep]
    for new_name in NEW_OPTIMIZER_NAMES:
        w = comparison["wins"].get(new_name)
        if not w:
            lines.append(f"| `{new_name}` | " + " | ".join(["—"] * len(base_names)) + " | 0 |")
            continue
        marks = []
        for b in base_names:
            if b not in w["beat"]:
                marks.append("—")
            else:
                marks.append("✅" if w["beat"][b] else "❌")
        lines.append(
            f"| `{new_name}` | " + " | ".join(marks)
            + f" | **{w['n_wins']}/{w['n_baselines']}** |"
        )

    lines += [
        "",
        "### Sequential proofs (each optimizer vs related baselines)",
        "",
    ]
    for item in comparison["sequential"]:
        lines.append(f"#### Step {item['step']}: `{item['optimizer']}`")
        lines.append("")
        lines.append(f"*{item['claim']}*")
        lines.append("")
        if item.get("status") != "ok":
            lines.append("_No sweep row for this optimizer._")
            lines.append("")
            continue
        lines.append(
            f"Scaffold metrics: acc={item['new_acc']:.4f}, "
            f"time={item['new_time_s']:.2f}s, train_mem={_fmt_bytes(item['new_train_mem'])}."
        )
        lines.append("")
        lines.append("| Baseline | Δ acc | Δ time (s) | Δ train mem | Beats acc? |")
        lines.append("| :--- | ---: | ---: | ---: | :---: |")
        for p in item["proofs"]:
            lines.append(
                f"| `{p['baseline']}` | {p['acc_delta']:+.4f} | {p['time_delta_s']:+.3f} | "
                f"{p['mem_delta_bytes']:+.0f} B | {'✅' if p['beats_acc'] else '❌'} |"
            )
        lines.append("")
        lines.append(
            f"**Accuracy wins on related baselines:** {item['n_acc_wins']}/{len(item['proofs'])}."
        )
        lines.append("")

    lines += [
        "---",
        "",
        "## 5. Trade-offs",
        "",
        "| Dimension | Typical strength among the four | Trade-off |",
        "| :--- | :--- | :--- |",
        "| Accuracy (scaffold) | Highest win-count new opt on matrix above | May need more epochs to warm up |",
        "| Wall-clock | Cosine / sparse variants aim for cheaper steps | Sparse masks cost CPU without sparse kernels |",
        "| Training memory | Voting-family single buffer | Hybrid keeps EMA + accumulator |",
        "| Inference memory | Same bitpacked footprint for STE MLP | Independent of optimizer after extract |",
        "",
        "---",
        "",
        "## 6. How to reproduce",
        "",
        "```bash",
        "uv run pytest tests/ -q",
        "uv run python experiments/run_full_pipeline.py",
        "# or",
        "uv run python experiments/run_training_sweep.py",
        "uv run python experiments/run_pareto_analysis.py",
        "```",
        "",
        "Artifacts: `results/training_sweep.json`, `results/pareto_analysis.md`, "
        "`results/new_optimizers_report.md`.",
        "",
    ]
    return "\n".join(lines)


def write_new_optimizer_report(
    training_json: str | Path = "results/training_sweep.json",
    output_md: str | Path = "results/new_optimizers_report.md",
    output_json: str | Path | None = "results/new_optimizers_report.json",
    model: str = "bit_mlp_small",
) -> dict[str, Any]:
    training = json.loads(Path(training_json).read_text())
    comparison = build_new_optimizer_comparison(training, model=model)
    md = render_new_optimizer_markdown(comparison)
    out = Path(output_md)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(md)
    if output_json is not None:
        # Drop non-serializable nothing; comparison is plain dicts
        Path(output_json).write_text(json.dumps(comparison, indent=2))
    return comparison
