"""DuckDB-backed experiment store: runs + long-form metrics."""

from __future__ import annotations

import json
import logging
import subprocess
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import duckdb

from .paths import default_db_path

logger = logging.getLogger(__name__)

STATUS_RUNNING = "running"
STATUS_COMPLETED = "completed"
STATUS_FAILED = "failed"

_SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS runs (
    run_id VARCHAR PRIMARY KEY,
    experiment VARCHAR NOT NULL,
    name VARCHAR,
    status VARCHAR NOT NULL,
    seed INTEGER,
    created_at TIMESTAMP NOT NULL,
    finished_at TIMESTAMP,
    wall_sec DOUBLE,
    best_test_acc DOUBLE,
    best_epoch INTEGER,
    final_test_acc DOUBLE,
    final_test_loss DOUBLE,
    config JSON NOT NULL,
    summary JSON,
    checkpoint_path VARCHAR,
    git_commit VARCHAR,
    notes VARCHAR
);

CREATE TABLE IF NOT EXISTS metrics (
    run_id VARCHAR NOT NULL,
    step INTEGER NOT NULL,
    metric VARCHAR NOT NULL,
    value DOUBLE NOT NULL,
    PRIMARY KEY (run_id, step, metric)
);
"""


def connect(path: Path | str | None = None) -> duckdb.DuckDBPyConnection:
    """Open (or create) the experiment database."""
    db_path = Path(path) if path is not None else default_db_path()
    db_path.parent.mkdir(parents=True, exist_ok=True)
    return duckdb.connect(str(db_path))


def init_db(conn: duckdb.DuckDBPyConnection) -> None:
    """Create tables if they do not exist."""
    conn.execute(_SCHEMA_SQL)


def _now() -> datetime:
    return datetime.now(timezone.utc).replace(tzinfo=None)


def _json_dumps(obj: Any) -> str:
    return json.dumps(obj, default=str, sort_keys=True)


def try_git_commit() -> str | None:
    """Best-effort ``git rev-parse HEAD``; returns None if unavailable."""
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            timeout=2,
            check=False,
        )
        if out.returncode == 0:
            return out.stdout.strip() or None
    except (OSError, subprocess.SubprocessError):
        pass
    return None


def new_run_id() -> str:
    return uuid.uuid4().hex


def start_run(
    conn: duckdb.DuckDBPyConnection,
    experiment: str,
    name: str,
    config: Mapping[str, Any],
    *,
    seed: int | None = None,
    run_id: str | None = None,
    git_commit: str | None = ...,  # type: ignore[assignment]
    notes: str | None = None,
) -> str:
    """Insert a run with status=running. Returns run_id."""
    rid = run_id or new_run_id()
    if git_commit is ...:
        git_commit = try_git_commit()
    conn.execute(
        """
        INSERT INTO runs (
            run_id, experiment, name, status, seed, created_at,
            config, summary, git_commit, notes
        ) VALUES (?, ?, ?, ?, ?, ?, ?::JSON, NULL, ?, ?)
        """,
        [
            rid,
            experiment,
            name,
            STATUS_RUNNING,
            seed,
            _now(),
            _json_dumps(dict(config)),
            git_commit,
            notes,
        ],
    )
    return rid


def log_metrics(
    conn: duckdb.DuckDBPyConnection,
    run_id: str,
    step: int,
    metrics: Mapping[str, Any],
) -> int:
    """Insert numeric metrics for one step. Non-numeric values are skipped."""
    rows: list[tuple[str, int, str, float]] = []
    for key, raw in metrics.items():
        if key == "epoch":
            continue
        try:
            val = float(raw)
        except (TypeError, ValueError):
            continue
        if val != val:  # NaN
            continue
        rows.append((run_id, int(step), str(key), val))
    if not rows:
        return 0
    conn.executemany(
        """
        INSERT OR REPLACE INTO metrics (run_id, step, metric, value)
        VALUES (?, ?, ?, ?)
        """,
        rows,
    )
    return len(rows)


def log_history(
    conn: duckdb.DuckDBPyConnection,
    run_id: str,
    history: Sequence[Mapping[str, Any]],
) -> int:
    """Bulk-log a list of per-epoch dicts (each may include ``epoch``)."""
    n = 0
    for i, row in enumerate(history):
        step = int(row.get("epoch", i + 1))
        n += log_metrics(conn, run_id, step, row)
    return n


def finish_run(
    conn: duckdb.DuckDBPyConnection,
    run_id: str,
    *,
    status: str = STATUS_COMPLETED,
    wall_sec: float | None = None,
    best_test_acc: float | None = None,
    best_epoch: int | None = None,
    final_test_acc: float | None = None,
    final_test_loss: float | None = None,
    summary: Mapping[str, Any] | None = None,
    checkpoint_path: str | Path | None = None,
    notes: str | None = None,
) -> None:
    """Mark a run finished and store scalar summary fields."""
    ck = str(checkpoint_path) if checkpoint_path is not None else None
    summary_json = _json_dumps(dict(summary)) if summary is not None else None
    conn.execute(
        """
        UPDATE runs SET
            status = ?,
            finished_at = ?,
            wall_sec = COALESCE(?, wall_sec),
            best_test_acc = COALESCE(?, best_test_acc),
            best_epoch = COALESCE(?, best_epoch),
            final_test_acc = COALESCE(?, final_test_acc),
            final_test_loss = COALESCE(?, final_test_loss),
            summary = COALESCE(?::JSON, summary),
            checkpoint_path = COALESCE(?, checkpoint_path),
            notes = COALESCE(?, notes)
        WHERE run_id = ?
        """,
        [
            status,
            _now(),
            wall_sec,
            best_test_acc,
            best_epoch,
            final_test_acc,
            final_test_loss,
            summary_json,
            ck,
            notes,
            run_id,
        ],
    )


def fail_run(
    conn: duckdb.DuckDBPyConnection,
    run_id: str,
    notes: str | None = None,
) -> None:
    finish_run(conn, run_id, status=STATUS_FAILED, notes=notes)


def delete_run(conn: duckdb.DuckDBPyConnection, run_id: str) -> None:
    conn.execute("DELETE FROM metrics WHERE run_id = ?", [run_id])
    conn.execute("DELETE FROM runs WHERE run_id = ?", [run_id])


def run_exists(conn: duckdb.DuckDBPyConnection, run_id: str) -> bool:
    row = conn.execute(
        "SELECT 1 FROM runs WHERE run_id = ? LIMIT 1", [run_id]
    ).fetchone()
    return row is not None


def _row_to_run(row: tuple[Any, ...], columns: Sequence[str]) -> dict[str, Any]:
    d = dict(zip(columns, row))
    for key in ("config", "summary"):
        val = d.get(key)
        if isinstance(val, str):
            try:
                d[key] = json.loads(val)
            except json.JSONDecodeError:
                pass
        elif val is not None and not isinstance(val, (dict, list)):
            # duckdb may return dict-like already
            try:
                d[key] = json.loads(str(val))
            except (json.JSONDecodeError, TypeError):
                pass
    return d


_RUN_COLUMNS = (
    "run_id",
    "experiment",
    "name",
    "status",
    "seed",
    "created_at",
    "finished_at",
    "wall_sec",
    "best_test_acc",
    "best_epoch",
    "final_test_acc",
    "final_test_loss",
    "config",
    "summary",
    "checkpoint_path",
    "git_commit",
    "notes",
)


def get_run(conn: duckdb.DuckDBPyConnection, run_id: str) -> dict[str, Any] | None:
    cols = ", ".join(_RUN_COLUMNS)
    row = conn.execute(
        f"SELECT {cols} FROM runs WHERE run_id = ?", [run_id]
    ).fetchone()
    if row is None:
        return None
    return _row_to_run(row, _RUN_COLUMNS)


def list_runs(
    conn: duckdb.DuckDBPyConnection,
    *,
    experiment: str | None = None,
    status: str | None = None,
    order_by: str = "best_test_acc",
    descending: bool = True,
    limit: int | None = None,
) -> list[dict[str, Any]]:
    """List runs with optional filters."""
    allowed = {
        "best_test_acc",
        "created_at",
        "finished_at",
        "wall_sec",
        "experiment",
        "name",
        "seed",
    }
    if order_by not in allowed:
        raise ValueError(f"order_by must be one of {sorted(allowed)}")
    direction = "DESC" if descending else "ASC"
    clauses: list[str] = []
    params: list[Any] = []
    if experiment is not None:
        clauses.append("experiment = ?")
        params.append(experiment)
    if status is not None:
        clauses.append("status = ?")
        params.append(status)
    where = f"WHERE {' AND '.join(clauses)}" if clauses else ""
    lim = f"LIMIT {int(limit)}" if limit is not None else ""
    cols = ", ".join(_RUN_COLUMNS)
    # NULLS LAST for descending best_test_acc
    nulls = "NULLS LAST" if descending else "NULLS FIRST"
    sql = (
        f"SELECT {cols} FROM runs {where} "
        f"ORDER BY {order_by} {direction} {nulls} {lim}"
    )
    rows = conn.execute(sql, params).fetchall()
    return [_row_to_run(r, _RUN_COLUMNS) for r in rows]


def load_history(
    conn: duckdb.DuckDBPyConnection,
    run_id: str,
    metric: str = "test_acc",
) -> list[tuple[int, float]]:
    """Return (step, value) pairs ordered by step."""
    rows = conn.execute(
        """
        SELECT step, value FROM metrics
        WHERE run_id = ? AND metric = ?
        ORDER BY step
        """,
        [run_id, metric],
    ).fetchall()
    return [(int(s), float(v)) for s, v in rows]


def compare_best(
    conn: duckdb.DuckDBPyConnection,
    experiment: str | None = None,
) -> list[dict[str, Any]]:
    """Best completed run per (experiment, name), ordered by accuracy."""
    clauses = ["status = ?"]
    params: list[Any] = [STATUS_COMPLETED]
    if experiment is not None:
        clauses.append("experiment = ?")
        params.append(experiment)
    where = " AND ".join(clauses)
    rows = conn.execute(
        f"""
        SELECT experiment, name, seed, best_test_acc, best_epoch, run_id, wall_sec
        FROM (
            SELECT *,
                ROW_NUMBER() OVER (
                    PARTITION BY experiment, name
                    ORDER BY best_test_acc DESC NULLS LAST
                ) AS rn
            FROM runs
            WHERE {where}
        ) t
        WHERE rn = 1
        ORDER BY best_test_acc DESC NULLS LAST
        """,
        params,
    ).fetchall()
    cols = (
        "experiment",
        "name",
        "seed",
        "best_test_acc",
        "best_epoch",
        "run_id",
        "wall_sec",
    )
    return [dict(zip(cols, r)) for r in rows]


def format_compare_markdown(
    rows: Iterable[Mapping[str, Any]],
    title: str = "Best runs",
) -> str:
    lines = [f"# {title}", ""]
    lines.append("| experiment | name | seed | best_test_acc | best_epoch | wall_sec |")
    lines.append("| :--- | :--- | ---: | ---: | ---: | ---: |")
    for r in rows:
        acc = r.get("best_test_acc")
        acc_s = f"{acc:.4f}" if acc is not None else ""
        wall = r.get("wall_sec")
        wall_s = f"{wall:.1f}" if wall is not None else ""
        be = r.get("best_epoch")
        be_s = "" if be is None else str(be)
        seed = r.get("seed")
        seed_s = "" if seed is None else str(seed)
        lines.append(
            f"| {r.get('experiment', '')} | {r.get('name', '')} | {seed_s} | "
            f"{acc_s} | {be_s} | {wall_s} |"
        )
    return "\n".join(lines)


def record_completed_run(
    conn: duckdb.DuckDBPyConnection,
    *,
    experiment: str,
    name: str,
    config: Mapping[str, Any],
    history: Sequence[Mapping[str, Any]],
    seed: int | None = None,
    run_id: str | None = None,
    wall_sec: float | None = None,
    best_test_acc: float | None = None,
    best_epoch: int | None = None,
    final_test_acc: float | None = None,
    final_test_loss: float | None = None,
    summary: Mapping[str, Any] | None = None,
    checkpoint_path: str | Path | None = None,
    notes: str | None = None,
    replace: bool = False,
) -> str:
    """
    Insert a finished run + history in one shot.

    If ``run_id`` already exists and ``replace`` is False, returns existing id
    without changes. If ``replace`` is True, deletes and re-inserts.
    """
    rid = run_id or new_run_id()
    if run_exists(conn, rid):
        if not replace:
            return rid
        delete_run(conn, rid)
    start_run(
        conn,
        experiment,
        name,
        config,
        seed=seed,
        run_id=rid,
    )
    log_history(conn, rid, history)
    finish_run(
        conn,
        rid,
        wall_sec=wall_sec,
        best_test_acc=best_test_acc,
        best_epoch=best_epoch,
        final_test_acc=final_test_acc,
        final_test_loss=final_test_loss,
        summary=summary,
        checkpoint_path=checkpoint_path,
        notes=notes,
    )
    return rid


def soft_record_completed_run(**kwargs: Any) -> str | None:
    """Like ``record_completed_run`` but logs and returns None on failure."""
    try:
        conn = kwargs.pop("conn", None)
        path = kwargs.pop("path", None)
        own_conn = conn is None
        if own_conn:
            conn = connect(path)
            init_db(conn)
        try:
            return record_completed_run(conn, **kwargs)
        finally:
            if own_conn:
                conn.close()
    except Exception:
        logger.exception("Failed to record experiment run to database")
        return None
