"""
Pareto analysis combining training sweep results with inference benchmarks.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def _load_json(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text())


def _dominates(a: dict[str, float], b: dict[str, float], maximize: set[str], minimize: set[str]) -> bool:
    """True if a Pareto-dominates b on the given objectives."""
    better_or_equal = True
    strictly_better = False
    for k in maximize:
        if a[k] < b[k]:
            better_or_equal = False
        if a[k] > b[k]:
            strictly_better = True
    for k in minimize:
        if a[k] > b[k]:
            better_or_equal = False
        if a[k] < b[k]:
            strictly_better = True
    return better_or_equal and strictly_better


def compute_pareto_front(
    points: list[dict[str, Any]],
    *,
    maximize: list[str] | None = None,
    minimize: list[str] | None = None,
) -> list[dict[str, Any]]:
    maximize_s = set(maximize or ["test_acc_mean"])
    minimize_s = set(
        minimize
        or [
            "inference_memory_bitpacked",
            "inference_latency_ms",
            "training_memory_total",
            "total_time_mean_s",
        ]
    )
    front = []
    for p in points:
        if any(
            _dominates(
                {k: q[k] for k in maximize_s | minimize_s},
                {k: p[k] for k in maximize_s | minimize_s},
                maximize_s,
                minimize_s,
            )
            for q in points
            if q is not p
        ):
            continue
        front.append(p)
    return front


def build_combined_table(
    training: dict[str, Any],
    inference: dict[str, Any],
) -> list[dict[str, Any]]:
    """
    Merge training summaries with inference metrics.

    Inference latency/memory are attached from the MLP microbench when the
    model is STE-based, and from the swarm comparison when model is swarm.
    """
    # Reference inference stats
    mlp_mem = inference["mnist_bit_mlp"]["memory_majority"]
    mlp_lat = {
        r["batch_size"]: r["mean_ms"]
        for r in inference["mnist_bit_mlp"]["latencies"]
        if r["mode"] == "packed_binary"
    }
    mlp_lat_float = {
        r["batch_size"]: r["mean_ms"]
        for r in inference["mnist_bit_mlp"]["latencies"]
        if r["mode"] == "float"
    }
    ref_batch = sorted(mlp_lat.keys())[0] if mlp_lat else 1
    swarm = inference.get("swarm_comparison", {})

    rows = []
    for s in training.get("summaries", []):
        mem = s.get("memory") or {}
        train_mem = (mem.get("training") or {}).get("total_bytes", 0)
        inf_mem = (mem.get("inference") or {}).get("bitpacked_bytes", 0)

        if s["model"] == "swarm_mlp":
            latency = swarm.get("majority_cached_ms", {}).get("mean", 0.0)
            inf_mem = swarm.get("memory_majority_bitpacked", inf_mem)
        else:
            # Scale bitpacked memory from profile if available
            latency = mlp_lat.get(ref_batch, 0.0)
            if inf_mem == 0:
                inf_mem = mlp_mem.get("bitpacked_bytes", 0)

        # CIFAR models: use training profile inference memory; latency approx from linear bench
        if s["dataset"] == "cifar10":
            linear = inference.get("linear_layer", [])
            packed_lin = [
                r for r in linear if r["mode"] == "packed_binary" and r["batch_size"] == ref_batch
            ]
            latency = packed_lin[0]["mean_ms"] if packed_lin else latency

        row = {
            "name": s["name"],
            "dataset": s["dataset"],
            "model": s["model"],
            "optimizer": s["optimizer"],
            "test_acc_mean": s["test_acc_mean"],
            "test_acc_std": s["test_acc_std"],
            "total_time_mean_s": s["total_time_mean_s"],
            "training_memory_total": train_mem,
            "inference_memory_bitpacked": inf_mem,
            "inference_latency_ms": latency,
            "inference_latency_float_ms": mlp_lat_float.get(ref_batch, 0.0),
            "per_epoch_test_acc_mean": s.get("per_epoch_test_acc_mean", []),
        }
        rows.append(row)
    return rows


def run_pareto_analysis(
    *,
    training_json: str | Path = "results/training_sweep.json",
    inference_json: str | Path = "results/inference_benchmark.json",
    output_md: str | Path = "results/pareto_analysis.md",
    output_json: str | Path = "results/pareto_analysis.json",
) -> dict[str, Any]:
    training = _load_json(training_json)
    inference = _load_json(inference_json)
    rows = build_combined_table(training, inference)

    # Pareto on accuracy (max) vs inference memory (min) vs latency (min)
    front_mem = compute_pareto_front(
        rows,
        maximize=["test_acc_mean"],
        minimize=["inference_memory_bitpacked"],
    )
    front_speed = compute_pareto_front(
        rows,
        maximize=["test_acc_mean"],
        minimize=["inference_latency_ms"],
    )
    front_train = compute_pareto_front(
        rows,
        maximize=["test_acc_mean"],
        minimize=["training_memory_total", "total_time_mean_s"],
    )
    front_all = compute_pareto_front(rows)

    payload = {
        "rows": rows,
        "pareto_accuracy_vs_inference_memory": [r["name"] for r in front_mem],
        "pareto_accuracy_vs_inference_speed": [r["name"] for r in front_speed],
        "pareto_training_efficiency": [r["name"] for r in front_train],
        "pareto_all_objectives": [r["name"] for r in front_all],
        "meta": {
            "training_source": str(training_json),
            "inference_source": str(inference_json),
            "note": training.get("meta", {}).get("note", ""),
        },
    }

    md = _format_pareto_report(payload)

    Path(output_json).parent.mkdir(parents=True, exist_ok=True)
    Path(output_json).write_text(json.dumps(payload, indent=2))
    Path(output_md).write_text(md)
    print(f"Wrote {output_json}")
    print(f"Wrote {output_md}")
    return payload


def _fmt_bytes(n: int | float) -> str:
    n = float(n)
    if n >= 1e6:
        return f"{n / 1e6:.2f} MB"
    if n >= 1e3:
        return f"{n / 1e3:.1f} KB"
    return f"{int(n)} B"


def _format_pareto_report(payload: dict[str, Any]) -> str:
    rows = payload["rows"]
    lines = [
        "# Pareto Analysis Report",
        "",
        "Combines scaffolding training-sweep results with inference benchmarks.",
        "",
        f"**Note:** {payload['meta'].get('note', '')}",
        "",
        "## Full comparison",
        "",
        "| Config | Dataset | Test Acc (mean ± std) | Infer mem (bitpacked) | Infer latency (ms) | Train mem | Train time (s) |",
        "| :--- | :--- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for r in sorted(rows, key=lambda x: -x["test_acc_mean"]):
        lines.append(
            f"| {r['name']} | {r['dataset']} | "
            f"{r['test_acc_mean']:.4f} ± {r['test_acc_std']:.4f} | "
            f"{_fmt_bytes(r['inference_memory_bitpacked'])} | "
            f"{r['inference_latency_ms']:.4f} | "
            f"{_fmt_bytes(r['training_memory_total'])} | "
            f"{r['total_time_mean_s']:.2f} |"
        )

    lines += [
        "",
        "## Accuracy vs inference memory",
        "",
        "| Config | Acc | Bitpacked mem |",
        "| :--- | ---: | ---: |",
    ]
    for r in sorted(rows, key=lambda x: x["inference_memory_bitpacked"]):
        mark = " ★" if r["name"] in payload["pareto_accuracy_vs_inference_memory"] else ""
        lines.append(
            f"| {r['name']}{mark} | {r['test_acc_mean']:.4f} | "
            f"{_fmt_bytes(r['inference_memory_bitpacked'])} |"
        )

    lines += [
        "",
        "## Accuracy vs inference speed",
        "",
        "| Config | Acc | Latency (ms) |",
        "| :--- | ---: | ---: |",
    ]
    for r in sorted(rows, key=lambda x: x["inference_latency_ms"]):
        mark = " ★" if r["name"] in payload["pareto_accuracy_vs_inference_speed"] else ""
        lines.append(
            f"| {r['name']}{mark} | {r['test_acc_mean']:.4f} | {r['inference_latency_ms']:.4f} |"
        )

    lines += [
        "",
        "## Training efficiency (acc vs train mem & time)",
        "",
        "| Config | Acc | Train mem | Time (s) |",
        "| :--- | ---: | ---: | ---: |",
    ]
    for r in sorted(rows, key=lambda x: x["total_time_mean_s"]):
        mark = " ★" if r["name"] in payload["pareto_training_efficiency"] else ""
        lines.append(
            f"| {r['name']}{mark} | {r['test_acc_mean']:.4f} | "
            f"{_fmt_bytes(r['training_memory_total'])} | {r['total_time_mean_s']:.2f} |"
        )

    lines += [
        "",
        "## Pareto-optimal configurations",
        "",
        f"- **Accuracy vs inference memory:** {', '.join(payload['pareto_accuracy_vs_inference_memory']) or '—'}",
        f"- **Accuracy vs inference speed:** {', '.join(payload['pareto_accuracy_vs_inference_speed']) or '—'}",
        f"- **Training efficiency:** {', '.join(payload['pareto_training_efficiency']) or '—'}",
        f"- **All objectives:** {', '.join(payload['pareto_all_objectives']) or '—'}",
        "",
        "★ marks Pareto-optimal points on the corresponding frontier.",
        "",
        "## Interpretation",
        "",
        "- Higher accuracy with lower bitpacked memory is preferred for edge deployment.",
        "- Swarm configs store a population during training but can collapse to a majority",
        "  vote at inference, matching STE bitpacked memory while training memory stays higher.",
        "- These numbers come from **scaffolding** runs (few epochs/batches); absolute",
        "  accuracies are not final. The tables validate the analytics pipeline.",
        "",
    ]
    return "\n".join(lines)
