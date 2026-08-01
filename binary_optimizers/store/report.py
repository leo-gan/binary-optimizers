"""Human-readable experiment comparison report (DuckDB → stdout).

Usage::

    python -m binary_optimizers.store report
    python -m binary_optimizers.store report --experiment v0_4
"""

from __future__ import annotations

import re
from typing import Any, Mapping, Sequence

import duckdb

from .db import STATUS_COMPLETED, init_db, list_runs

# ---------------------------------------------------------------------------
# Static knowledge (from experiments/v0_*/PROTOCOL.md) — not inferred from DB
# ---------------------------------------------------------------------------

LN_MODE_EXPLAIN: dict[str, dict[str, str]] = {
    "none": {
        "id": "LN0",
        "cli": "none",
        "title": "No LayerNorm",
        "description": (
            "No LayerNorm before swarm linears. Purest latent-free path: "
            "only int8 agents + STE decode + FP activations. Often more "
            "stable long-run for v0.1; later place-value versions still "
            "use it as the pure baseline."
        ),
        "trainable": "None (no γ, β).",
    },
    "no_affine": {
        "id": "LN1",
        "cli": "no_affine",
        "title": "LayerNorm without affine",
        "description": (
            "LayerNorm with elementwise_affine=False: FP mean/variance "
            "normalization only. Stabilizes activations without introducing "
            "learned scale/bias on the norm. Primary 'almost pure' mode."
        ),
        "trainable": "None (no γ, β).",
    },
    "affine": {
        "id": "LN2",
        "cli": "affine",
        "title": "LayerNorm with affine (hybrid)",
        "description": (
            "LayerNorm with elementwise_affine=True: learned γ, β updated "
            "by a small FP SGD group (ln_lr). Hybrid: discrete swarm weights "
            "plus a continuous 'norm crutch'. Usually strongest accuracy; "
            "report as hybrid, not pure binary/ternary."
        ),
        "trainable": "γ, β via FP SGD (ln_lr).",
    },
}

# Ordered lineage for narrative + delta analysis
VERSION_LINEAGE: list[dict[str, Any]] = [
    {
        "id": "v0_1",
        "label": "v0.1",
        "title": "Latent-free unary binary Swarm",
        "coding": "Unary majority: int8 agents ∈ {-1,+1}; decode by majority/sign of sum",
        "changed": (
            "Baseline experiment: BitNet-style MLP on MNIST, no FP master "
            "weight, Swarm flips only, manual STE. Agents are redundant "
            "votes for each logical weight (swarm_size=32)."
        ),
        "vs_previous": None,
        "key_defaults": "hidden=128, swarm_size=32, ReLU (default)",
    },
    {
        "id": "v0_2",
        "label": "v0.2",
        "title": "Place-value (exponential) binary Swarm",
        "coding": "Place-value bits: agent i contributes ±2^i; multi-level s_norm ∈ [-1,1]",
        "changed": (
            "Same latent-free stack as v0.1, but exponential place-value "
            "coding instead of unary majority. LSB flips are small steps; "
            "MSB flips are large. LSB-easier flip schedule (prob ∝ 2^{-i})."
        ),
        "vs_previous": "v0.1 unary majority Swarm",
        "key_defaults": "hidden=128, n_bits=16",
    },
    {
        "id": "v0_3",
        "label": "v0.3",
        "title": "Carry-safe place-value Swarm",
        "coding": "Binary place-value via integer register v; bits re-encoded after ±1 steps",
        "changed": (
            "Keeps binary place-value weights, but updates a carry-safe "
            "integer v ← clip(v±1) then re-encodes bits. No independent "
            "random MSB flips — smoother late training. Large accuracy jump."
        ),
        "vs_previous": "v0.2 independent bit flips",
        "key_defaults": "n_bits=16, adaptive Δv ≤ 512",
    },
    {
        "id": "v0_4",
        "label": "v0.4",
        "title": "Balanced ternary place-value Swarm",
        "coding": "Digits d_i ∈ {-1,0,+1}, places 3^i; carry-safe base-3 steps",
        "changed": (
            "Alphabet becomes balanced ternary (BitNet-adjacent). "
            "n_trits=10 ≈ 16 binary bits of range. Natural sparsity via "
            "zero digits (zero_frac). Competitive with v0.3."
        ),
        "vs_previous": "v0.2/v0.3 binary place-value",
        "key_defaults": "n_trits=10, adaptive Δs ≤ 64",
    },
]

_LN_ORDER = ("none", "no_affine", "affine")
_EXP_ORDER = ("v0_1", "v0_2", "v0_3", "v0_4")


def _ln_mode_from_run(run: Mapping[str, Any]) -> str | None:
    cfg = run.get("config") or {}
    if isinstance(cfg, dict) and cfg.get("ln_mode") is not None:
        return str(cfg["ln_mode"])
    name = run.get("name") or ""
    m = re.match(r"^ln_(none|no_affine|affine)$", str(name))
    if m:
        return m.group(1)
    if str(name).startswith("ln_"):
        return str(name)[3:]
    return None


def _fmt_acc(x: float | None, width: int = 6) -> str:
    if x is None:
        return " " * (width - 1) + "—"
    return f"{x:.4f}"


def _pp(d: float | None) -> str:
    """Percentage points (×100) for readability."""
    if d is None:
        return "—"
    sign = "+" if d >= 0 else ""
    return f"{sign}{100.0 * d:.2f} pp"


def _sort_experiments(ids: Sequence[str]) -> list[str]:
    rank = {e: i for i, e in enumerate(_EXP_ORDER)}

    def key(e: str) -> tuple[int, str]:
        return (rank.get(e, 1000), e)

    return sorted(set(ids), key=key)


def collect_run_grid(
    runs: Sequence[Mapping[str, Any]],
) -> dict[str, dict[str, dict[str, Any]]]:
    """experiment -> ln_mode -> best run dict (by best_test_acc)."""
    grid: dict[str, dict[str, dict[str, Any]]] = {}
    for run in runs:
        if run.get("status") and run["status"] != STATUS_COMPLETED:
            # still allow if status missing
            if run.get("best_test_acc") is None:
                continue
        exp = str(run.get("experiment") or "unknown")
        ln = _ln_mode_from_run(run)
        if ln is None:
            ln = str(run.get("name") or "unknown")
        acc = run.get("best_test_acc")
        bucket = grid.setdefault(exp, {})
        prev = bucket.get(ln)
        if prev is None or (acc is not None and (prev.get("best_test_acc") is None or acc > prev["best_test_acc"])):
            bucket[ln] = dict(run)
            bucket[ln]["_ln_mode"] = ln
    return grid


def format_ln_modes_section() -> str:
    lines = [
        "## LayerNorm modes (`ln_mode`)",
        "",
        "Shared ablation across v0.1–v0.4. Applied **before each swarm linear**.",
        "",
    ]
    for key in _LN_ORDER:
        info = LN_MODE_EXPLAIN[key]
        lines.append(f"### `{info['cli']}` ({info['id']}) — {info['title']}")
        lines.append("")
        lines.append(info["description"])
        lines.append("")
        lines.append(f"- **Trainable:** {info['trainable']}")
        lines.append("")
    lines.extend(
        [
            "**How to read results:** `none` / `no_affine` are the pure(r) "
            "latent-free settings; `affine` adds FP γ,β and is a **hybrid** upper bound.",
            "",
        ]
    )
    return "\n".join(lines)


def format_version_section(
    experiments_present: Sequence[str] | None = None,
) -> str:
    present = set(experiments_present) if experiments_present else None
    lines = [
        "## Version lineage (what changed)",
        "",
        "Each version keeps MNIST + BitNet-style MLP + Swarm-style discrete "
        "updates; the **weight coding / update rule** is what evolves.",
        "",
    ]
    for i, ver in enumerate(VERSION_LINEAGE):
        if present is not None and ver["id"] not in present:
            # still show full lineage for context if any later version present
            if not any(v["id"] in present for v in VERSION_LINEAGE[i:]):
                continue
        lines.append(f"### {ver['label']} — {ver['title']}")
        lines.append("")
        lines.append(f"- **Coding:** {ver['coding']}")
        if ver["vs_previous"]:
            lines.append(f"- **Baseline:** {ver['vs_previous']}")
        lines.append(f"- **What changed:** {ver['changed']}")
        lines.append(f"- **Defaults:** {ver['key_defaults']}")
        lines.append("")
    return "\n".join(lines)


def format_pivot_table(grid: dict[str, dict[str, dict[str, Any]]]) -> str:
    exps = _sort_experiments(list(grid.keys()))
    # discover ln columns: preferred order first, then extras
    ln_cols: list[str] = []
    seen: set[str] = set()
    for pref in _LN_ORDER:
        if any(pref in grid.get(e, {}) for e in exps):
            ln_cols.append(pref)
            seen.add(pref)
    for e in exps:
        for ln in grid[e]:
            if ln not in seen:
                ln_cols.append(ln)
                seen.add(ln)

    headers = ["experiment", *[f"ln_{c}" if not c.startswith("ln_") else c for c in ln_cols], "best", "best_mode", "wall_best(s)"]
    # shorter headers
    headers = ["experiment"] + [c for c in ln_cols] + ["best", "best_mode", "wall_s"]

    lines = [
        "## Results comparison (best test accuracy)",
        "",
        "Values are **best test accuracy** over the run (early-stopped checkpoint). "
        "One seed per cell when only seed=42 is stored.",
        "",
    ]
    # markdown table
    lines.append("| " + " | ".join(headers) + " |")
    aligns = [" :--- "] + [" ---: "] * (len(headers) - 1)
    lines.append("|" + "|".join(aligns) + "|")

    for exp in exps:
        row_accs: dict[str, float | None] = {}
        row_runs: dict[str, dict[str, Any]] = {}
        for ln in ln_cols:
            r = grid.get(exp, {}).get(ln)
            if r is None:
                row_accs[ln] = None
            else:
                row_accs[ln] = r.get("best_test_acc")
                row_runs[ln] = r
        best_ln = None
        best_acc: float | None = None
        for ln, acc in row_accs.items():
            if acc is None:
                continue
            if best_acc is None or acc > best_acc:
                best_acc = acc
                best_ln = ln
        wall = None
        if best_ln and best_ln in row_runs:
            wall = row_runs[best_ln].get("wall_sec")
        cells = [exp]
        for ln in ln_cols:
            cells.append(_fmt_acc(row_accs[ln]))
        cells.append(_fmt_acc(best_acc))
        cells.append(best_ln or "—")
        cells.append(f"{wall:.0f}" if wall is not None else "—")
        lines.append("| " + " | ".join(cells) + " |")
    lines.append("")
    return "\n".join(lines)


def format_detail_table(grid: dict[str, dict[str, dict[str, Any]]]) -> str:
    lines = [
        "## Per-run detail",
        "",
        "| experiment | ln_mode | seed | best_test | best_epoch | wall_s | epochs_ran |",
        "| :--- | :--- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for exp in _sort_experiments(list(grid.keys())):
        for ln in list(_LN_ORDER) + sorted(
            k for k in grid[exp] if k not in _LN_ORDER
        ):
            r = grid[exp].get(ln)
            if r is None:
                continue
            summary = r.get("summary") or {}
            epochs = summary.get("epochs_ran") if isinstance(summary, dict) else None
            seed = r.get("seed")
            wall = r.get("wall_sec")
            be = r.get("best_epoch")
            lines.append(
                "| {exp} | {ln} | {seed} | {acc} | {be} | {wall} | {ep} |".format(
                    exp=exp,
                    ln=ln,
                    seed="" if seed is None else str(seed),
                    acc=_fmt_acc(r.get("best_test_acc")),
                    be="" if be is None else str(be),
                    wall=f"{wall:.0f}" if wall is not None else "—",
                    ep="" if epochs is None else str(epochs),
                )
            )
    lines.append("")
    return "\n".join(lines)


def format_analysis(grid: dict[str, dict[str, dict[str, Any]]]) -> str:
    lines = [
        "## Simple analysis",
        "",
    ]
    exps = _sort_experiments(list(grid.keys()))
    if not exps:
        lines.append("No completed runs in the database.")
        lines.append("")
        return "\n".join(lines)

    # overall best
    overall: tuple[str, str, float] | None = None
    for exp in exps:
        for ln, r in grid[exp].items():
            acc = r.get("best_test_acc")
            if acc is None:
                continue
            if overall is None or acc > overall[2]:
                overall = (exp, ln, float(acc))
    if overall:
        lines.append(
            f"- **Best overall:** `{overall[0]}` / `ln_mode={overall[1]}` "
            f"→ **{overall[2]:.4f}** test acc."
        )

    # version-to-version at each ln_mode
    lines.append("- **Version-to-version deltas** (same `ln_mode`, best test acc):")
    for i in range(1, len(exps)):
        prev_e, cur_e = exps[i - 1], exps[i]
        parts = []
        for ln in _LN_ORDER:
            a = grid.get(prev_e, {}).get(ln, {}).get("best_test_acc")
            b = grid.get(cur_e, {}).get(ln, {}).get("best_test_acc")
            if a is None or b is None:
                continue
            d = float(b) - float(a)
            parts.append(f"{ln}: {_pp(d)}")
        if parts:
            lines.append(f"  - `{prev_e}` → `{cur_e}`: " + "; ".join(parts))
        else:
            lines.append(f"  - `{prev_e}` → `{cur_e}`: (no overlapping ln_mode rows)")

    # LN effect within each experiment: affine - none, no_affine - none
    lines.append("- **LayerNorm effect within each experiment** (vs `none`):")
    for exp in exps:
        base = grid.get(exp, {}).get("none", {}).get("best_test_acc")
        if base is None:
            lines.append(f"  - `{exp}`: no `none` run")
            continue
        bits = []
        for ln in ("no_affine", "affine"):
            acc = grid.get(exp, {}).get(ln, {}).get("best_test_acc")
            if acc is None:
                continue
            bits.append(f"{ln}: {_pp(float(acc) - float(base))}")
        lines.append(
            f"  - `{exp}` (none={float(base):.4f}): "
            + ("; ".join(bits) if bits else "no other LN modes")
        )

    # narrative takeaways from known lineage + numbers
    lines.append("- **Takeaways:**")
    if "v0_1" in grid and "v0_2" in grid:
        lines.append(
            "  - **v0.1 → v0.2:** place-value coding (multi-level weights + LSB-biased "
            "flips) lifts accuracy over unary majority, especially with LN."
        )
    if "v0_2" in grid and "v0_3" in grid:
        lines.append(
            "  - **v0.2 → v0.3:** carry-safe ±1 integer steps are the large jump "
            "(~+3–4 pp); smoother than independent bit flips."
        )
    if "v0_3" in grid and "v0_4" in grid:
        a3 = grid["v0_3"].get("affine", {}).get("best_test_acc")
        a4 = grid["v0_4"].get("affine", {}).get("best_test_acc")
        if a3 is not None and a4 is not None:
            d = float(a4) - float(a3)
            lines.append(
                f"  - **v0.3 → v0.4:** ternary place-value is competitive with "
                f"carry-safe binary (affine Δ {_pp(d)}); adds natural zero-digit sparsity."
            )
    lines.append(
        "  - **`affine` vs pure modes:** when `affine` wins, treat it as a hybrid "
        "ceiling (FP γ,β). Compare pure progress on `none` / `no_affine`."
    )
    lines.append("")
    return "\n".join(lines)


def build_report(
    conn: duckdb.DuckDBPyConnection,
    *,
    experiment: str | None = None,
    include_detail: bool = True,
) -> str:
    """Build a full markdown report from the experiment store."""
    init_db(conn)
    runs = list_runs(conn, experiment=experiment, order_by="experiment", descending=False)
    # list_runs orders by experiment string; re-filter status
    runs = [r for r in runs if r.get("status") in (None, STATUS_COMPLETED) or r.get("best_test_acc") is not None]
    grid = collect_run_grid(runs)
    present = _sort_experiments(list(grid.keys()))

    parts = [
        "# Experiment analysis report",
        "",
        "Source: DuckDB experiment store"
        + (f" (filter: experiment={experiment})" if experiment else " (all experiments)")
        + f". Runs used: **{sum(len(v) for v in grid.values())}**.",
        "",
        format_ln_modes_section(),
        format_version_section(present if experiment is None else present),
        format_pivot_table(grid),
        format_analysis(grid),
    ]
    if include_detail:
        parts.append(format_detail_table(grid))
    parts.append(
        "---\n\n"
        "*Regenerate: `python -m binary_optimizers.store report`. "
        "Refresh data: `python -m binary_optimizers.store import`.*"
    )
    return "\n".join(parts)


def print_report(
    conn: duckdb.DuckDBPyConnection,
    *,
    experiment: str | None = None,
    include_detail: bool = True,
) -> str:
    text = build_report(conn, experiment=experiment, include_detail=include_detail)
    print(text)
    return text
