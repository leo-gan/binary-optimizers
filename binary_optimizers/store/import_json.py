"""Import legacy results/**/*.json experiment files into the DuckDB store."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path
from typing import Any

from .db import connect, init_db, record_completed_run, run_exists
from .paths import default_db_path, repo_root

# Keys that belong on the run row / summary rather than config.
_SCALAR_KEYS = {
    "best_test_acc",
    "best_epoch",
    "final_test_acc",
    "final_test_loss",
    "wall_sec",
    "epochs_ran",
}

_SUMMARY_KEYS = {
    "swarm_stats_final",
    "baseline_v0_2_best",
    "baseline_v0_3_note",
    "baseline_v0_1_best",
    "notes",
}

_SKIP_TOP_KEYS = {"history", "json_path"}

_SUMMARY_NAME_RE = re.compile(r"^summary_seed\d+\.json$")


def stable_run_id(
    experiment: str,
    name: str,
    seed: int | None,
    config: dict[str, Any],
) -> str:
    payload = {
        "experiment": experiment,
        "name": name,
        "seed": seed,
        "config": config,
    }
    blob = json.dumps(payload, sort_keys=True, default=str)
    return hashlib.sha256(blob.encode()).hexdigest()[:32]


def _run_name_from_payload(data: dict[str, Any], path: Path) -> str:
    if "ln_mode" in data:
        return f"ln_{data['ln_mode']}"
    stem = path.stem
    # e.g. ln_affine_seed42 -> ln_affine
    m = re.match(r"^(.*)_seed\d+$", stem)
    if m:
        return m.group(1)
    return stem


def parse_result_json(path: Path) -> dict[str, Any]:
    """Parse a result JSON file into record_completed_run kwargs."""
    with open(path) as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise ValueError(f"Expected object in {path}")

    experiment = str(data.get("experiment") or path.parent.name)
    seed = data.get("seed")
    if seed is not None:
        seed = int(seed)
    name = _run_name_from_payload(data, path)
    history = data.get("history") or []
    if not isinstance(history, list):
        raise ValueError(f"history must be a list in {path}")

    config: dict[str, Any] = {}
    summary: dict[str, Any] = {}
    for key, val in data.items():
        if key in _SKIP_TOP_KEYS or key in _SCALAR_KEYS:
            continue
        if key in ("experiment", "seed", "ln_mode"):
            # ln_mode kept in config for filtering; experiment/seed are columns
            if key == "ln_mode":
                config[key] = val
            continue
        if key in _SUMMARY_KEYS:
            summary[key] = val
        else:
            config[key] = val

    if "epochs_ran" in data:
        summary["epochs_ran"] = data["epochs_ran"]
    summary["source_json"] = str(path)

    rid = stable_run_id(experiment, name, seed, config)
    return {
        "run_id": rid,
        "experiment": experiment,
        "name": name,
        "seed": seed,
        "config": config,
        "history": history,
        "wall_sec": data.get("wall_sec"),
        "best_test_acc": data.get("best_test_acc"),
        "best_epoch": data.get("best_epoch"),
        "final_test_acc": data.get("final_test_acc"),
        "final_test_loss": data.get("final_test_loss"),
        "summary": summary or None,
    }


def iter_result_json_files(results_dir: Path) -> list[Path]:
    files: list[Path] = []
    if not results_dir.is_dir():
        return files
    for path in sorted(results_dir.rglob("*.json")):
        if _SUMMARY_NAME_RE.match(path.name):
            continue
        files.append(path)
    return files


def import_results(
    results_dir: Path | str | None = None,
    *,
    db_path: Path | str | None = None,
    replace: bool = False,
) -> dict[str, int]:
    """
    Import all run JSON files under ``results_dir``.

    Returns counts: ``{"imported": n, "skipped": n, "failed": n}``.
    """
    root = Path(results_dir) if results_dir is not None else repo_root() / "results"
    conn = connect(db_path)
    init_db(conn)
    stats = {"imported": 0, "skipped": 0, "failed": 0}
    try:
        for path in iter_result_json_files(root):
            try:
                kwargs = parse_result_json(path)
                rid = kwargs["run_id"]
                if run_exists(conn, rid) and not replace:
                    stats["skipped"] += 1
                    continue
                record_completed_run(conn, replace=replace, **kwargs)
                stats["imported"] += 1
            except Exception as exc:  # noqa: BLE001 — collect per-file failures
                stats["failed"] += 1
                print(f"FAIL {path}: {exc}")
    finally:
        conn.close()
    return stats


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        description="Import results/**/*.json into the experiment DuckDB store."
    )
    p.add_argument(
        "--results-dir",
        type=Path,
        default=None,
        help="Root results directory (default: <repo>/results)",
    )
    p.add_argument(
        "--db",
        type=Path,
        default=None,
        help=f"DuckDB path (default: {default_db_path()})",
    )
    p.add_argument(
        "--replace",
        action="store_true",
        help="Replace existing runs with the same stable run_id",
    )
    args = p.parse_args(argv)
    stats = import_results(args.results_dir, db_path=args.db, replace=args.replace)
    print(
        f"Import done: imported={stats['imported']} "
        f"skipped={stats['skipped']} failed={stats['failed']}"
    )
    return 1 if stats["failed"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
