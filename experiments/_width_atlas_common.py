"""Shared helpers for v0.5 width-atlas experiments."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from typing import Any


_REPO = Path(__file__).resolve().parents[1]


def load_exp_module(exp_dir_name: str, module_file: str, unique_name: str):
    """Load experiments/<exp_dir_name>/<module_file> under a unique module name."""
    path = _REPO / "experiments" / exp_dir_name / module_file
    if not path.is_file():
        raise FileNotFoundError(path)
    # Sibling imports inside v0_1/v0_3 use bare "layers", "model", etc.
    exp_dir = str(path.parent)
    if exp_dir not in sys.path:
        sys.path.insert(0, exp_dir)
    if str(_REPO) not in sys.path:
        sys.path.insert(0, str(_REPO))

    spec = importlib.util.spec_from_file_location(unique_name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load {path}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[unique_name] = mod
    # Ensure bare name resolves for cross-imports within the experiment package.
    bare = module_file.replace(".py", "")
    sys.modules[bare] = mod
    spec.loader.exec_module(mod)
    return mod


def parse_int_list(s: str) -> list[int]:
    parts = [p.strip() for p in s.split(",") if p.strip()]
    return [int(p) for p in parts]


def approx_register_state_bytes(hidden: int, n_bits: int) -> int:
    """int8 population only (not activations/grads): two layers 784→H and H→10."""
    return n_bits * (hidden * 784 + 10 * hidden)


def approx_unary_state_bytes(hidden: int, swarm_size: int) -> int:
    return swarm_size * (hidden * 784 + 10 * hidden)


def estimate_ok(state_bytes: int, limit_gb: float = 4.0) -> bool:
    """Rough gate: int8 state * ~5 (float view + grad + overhead) < limit."""
    return (state_bytes * 5) < limit_gb * (1024**3)


def write_curve_csv(path: Path, rows: list[dict[str, Any]], width_key: str) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    keys = [
        width_key,
        "best_test_acc",
        "best_epoch",
        "epochs_ran",
        "wall_sec",
        "approx_state_bytes",
        "ln_mode",
        "status",
    ]
    with open(path, "w") as f:
        f.write(",".join(keys) + "\n")
        for r in rows:
            f.write(
                ",".join(str(r.get(k, "")) for k in keys) + "\n"
            )


def load_result_rows(
    results_dir: Path,
    *,
    width_key: str,
    prefix: str,
    ln_mode: str,
    seed: int,
) -> list[dict[str, Any]]:
    """Load curve rows from per-width result JSONs (merge partial runs).

    Files look like ``nbits16_ln_none_seed42.json`` or ``S64_ln_none_seed42.json``.
    """
    import json
    import re

    if not results_dir.is_dir():
        return []
    pat = re.compile(
        rf"^{re.escape(prefix)}(\d+)_ln_{re.escape(ln_mode)}_seed{seed}\.json$"
    )
    rows: list[dict[str, Any]] = []
    for path in sorted(results_dir.glob("*.json")):
        m = pat.match(path.name)
        if not m:
            continue
        with open(path) as f:
            data = json.load(f)
        w = int(m.group(1))
        rows.append(
            {
                width_key: data.get(width_key, w),
                "ln_mode": data.get("ln_mode", ln_mode),
                "approx_state_bytes": data.get("approx_state_bytes"),
                "best_test_acc": data.get("best_test_acc"),
                "best_epoch": data.get("best_epoch"),
                "epochs_ran": data.get("epochs_ran"),
                "wall_sec": data.get("wall_sec"),
                "status": data.get("status", "completed"),
            }
        )
    rows.sort(key=lambda r: int(r[width_key]))
    return rows


def write_summary(
    results_dir: Path,
    *,
    experiment: str,
    seed: int,
    ln_mode: str,
    hidden: int,
    widths_requested: list[int],
    note: str,
    width_key: str,
    this_run_curve: list[dict[str, Any]],
    disk_prefix: str,
) -> Path:
    """Write summary + CSV. Prefer full on-disk merge over this-run-only curve."""
    import json

    results_dir.mkdir(parents=True, exist_ok=True)
    disk = load_result_rows(
        results_dir,
        width_key=width_key,
        prefix=disk_prefix,
        ln_mode=ln_mode,
        seed=seed,
    )
    # Prefer disk rows (all completed widths); fall back to this run if empty.
    curve = disk if disk else this_run_curve
    summary = {
        "experiment": experiment,
        "seed": seed,
        "ln_mode": ln_mode,
        "hidden": hidden,
        "widths_requested": widths_requested,
        "note": note,
        "curve": curve,
        "this_run_curve": this_run_curve,
    }
    sp = results_dir / f"summary_ln_{ln_mode}_seed{seed}.json"
    with open(sp, "w") as f:
        json.dump(summary, f, indent=2)
    write_curve_csv(
        results_dir / f"curve_ln_{ln_mode}_seed{seed}.csv",
        curve,
        width_key,
    )
    return sp
