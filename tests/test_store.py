"""Tests for binary_optimizers.store (DuckDB experiment store)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from binary_optimizers.store import (
    build_report,
    compare_best,
    connect,
    finish_run,
    format_compare_markdown,
    get_run,
    import_results,
    init_db,
    list_runs,
    load_history,
    log_history,
    log_metrics,
    parse_result_json,
    record_completed_run,
    soft_record_completed_run,
    stable_run_id,
    start_run,
)


@pytest.fixture
def conn(tmp_path: Path):
    db = tmp_path / "test.duckdb"
    c = connect(db)
    init_db(c)
    yield c
    c.close()


def test_init_idempotent(conn):
    init_db(conn)
    init_db(conn)
    n = conn.execute("SELECT COUNT(*) FROM runs").fetchone()[0]
    assert n == 0


def test_lifecycle_and_history(conn):
    rid = start_run(
        conn,
        "v0_4",
        "ln_affine",
        {"hidden": 128, "ln_mode": "affine"},
        seed=42,
    )
    assert get_run(conn, rid)["status"] == "running"
    n = log_metrics(
        conn,
        rid,
        1,
        {"test_acc": 0.9, "train_loss": 0.5, "epoch": 1, "note": "skip"},
    )
    assert n == 2
    log_history(
        conn,
        rid,
        [
            {"epoch": 2, "test_acc": 0.91, "flip_frac": 0.1},
            {"epoch": 3, "test_acc": 0.92},
        ],
    )
    finish_run(
        conn,
        rid,
        best_test_acc=0.92,
        best_epoch=3,
        final_test_acc=0.92,
        final_test_loss=0.1,
        wall_sec=12.5,
        summary={"epochs_ran": 3},
        checkpoint_path="/tmp/x.pt",
    )
    run = get_run(conn, rid)
    assert run is not None
    assert run["status"] == "completed"
    assert run["best_test_acc"] == pytest.approx(0.92)
    assert run["config"]["hidden"] == 128
    assert run["summary"]["epochs_ran"] == 3
    assert run["checkpoint_path"] == "/tmp/x.pt"
    hist = load_history(conn, rid, "test_acc")
    assert hist == [(1, 0.9), (2, 0.91), (3, 0.92)]


def test_list_runs_filter(conn):
    record_completed_run(
        conn,
        experiment="v0_3",
        name="ln_none",
        config={"a": 1},
        history=[{"epoch": 1, "test_acc": 0.8}],
        seed=1,
        best_test_acc=0.8,
    )
    record_completed_run(
        conn,
        experiment="v0_4",
        name="ln_affine",
        config={"a": 2},
        history=[{"epoch": 1, "test_acc": 0.95}],
        seed=2,
        best_test_acc=0.95,
    )
    only = list_runs(conn, experiment="v0_4")
    assert len(only) == 1
    assert only[0]["name"] == "ln_affine"
    top = list_runs(conn, order_by="best_test_acc", descending=True)
    assert top[0]["best_test_acc"] == pytest.approx(0.95)


def test_record_idempotent_skip(conn):
    rid = stable_run_id("v0_4", "ln_affine", 42, {"hidden": 128})
    record_completed_run(
        conn,
        run_id=rid,
        experiment="v0_4",
        name="ln_affine",
        config={"hidden": 128},
        history=[{"epoch": 1, "test_acc": 0.9}],
        seed=42,
        best_test_acc=0.9,
    )
    record_completed_run(
        conn,
        run_id=rid,
        experiment="v0_4",
        name="ln_affine",
        config={"hidden": 128},
        history=[{"epoch": 1, "test_acc": 0.99}],
        seed=42,
        best_test_acc=0.99,
        replace=False,
    )
    assert get_run(conn, rid)["best_test_acc"] == pytest.approx(0.9)
    record_completed_run(
        conn,
        run_id=rid,
        experiment="v0_4",
        name="ln_affine",
        config={"hidden": 128},
        history=[{"epoch": 1, "test_acc": 0.99}],
        seed=42,
        best_test_acc=0.99,
        replace=True,
    )
    assert get_run(conn, rid)["best_test_acc"] == pytest.approx(0.99)


def test_compare_best_and_markdown(conn):
    for name, acc in [("ln_none", 0.95), ("ln_affine", 0.97), ("ln_none", 0.94)]:
        record_completed_run(
            conn,
            experiment="v0_4",
            name=name,
            config={},
            history=[],
            seed=42,
            best_test_acc=acc,
        )
    rows = compare_best(conn, experiment="v0_4")
    by_name = {r["name"]: r["best_test_acc"] for r in rows}
    assert by_name["ln_none"] == pytest.approx(0.95)
    assert by_name["ln_affine"] == pytest.approx(0.97)
    md = format_compare_markdown(rows)
    assert "ln_affine" in md
    assert "0.9700" in md


def test_parse_and_import_json(tmp_path: Path):
    results = tmp_path / "results" / "v0_4"
    results.mkdir(parents=True)
    payload = {
        "experiment": "v0_4",
        "coding": "balanced_ternary_place",
        "ln_mode": "affine",
        "seed": 42,
        "hidden": 128,
        "n_trits": 10,
        "best_test_acc": 0.9737,
        "best_epoch": 16,
        "final_test_acc": 0.9737,
        "final_test_loss": 0.11,
        "wall_sec": 100.0,
        "epochs_ran": 2,
        "swarm_stats_final": {"zero_frac": 0.3},
        "history": [
            {
                "epoch": 1,
                "train_acc": 0.8,
                "test_acc": 0.9,
                "flip_frac": 0.1,
            },
            {
                "epoch": 2,
                "train_acc": 0.85,
                "test_acc": 0.95,
                "flip_frac": 0.09,
            },
        ],
    }
    jp = results / "ln_affine_seed42.json"
    jp.write_text(json.dumps(payload))
    (results / "summary_seed42.json").write_text(
        json.dumps({"experiment": "v0_4", "runs": []})
    )

    parsed = parse_result_json(jp)
    assert parsed["experiment"] == "v0_4"
    assert parsed["name"] == "ln_affine"
    assert parsed["seed"] == 42
    assert parsed["config"]["n_trits"] == 10
    assert parsed["config"]["ln_mode"] == "affine"
    assert len(parsed["history"]) == 2

    db = tmp_path / "exp.duckdb"
    stats = import_results(tmp_path / "results", db_path=db)
    assert stats == {"imported": 1, "skipped": 0, "failed": 0}
    stats2 = import_results(tmp_path / "results", db_path=db)
    assert stats2["skipped"] == 1
    assert stats2["imported"] == 0

    conn = connect(db)
    runs = list_runs(conn, experiment="v0_4")
    assert len(runs) == 1
    assert runs[0]["best_test_acc"] == pytest.approx(0.9737)
    hist = load_history(conn, runs[0]["run_id"], "test_acc")
    assert hist == [(1, 0.9), (2, 0.95)]
    conn.close()


def test_soft_record_does_not_raise(tmp_path: Path):
    # Invalid path parent that we can still create — use good path
    rid = soft_record_completed_run(
        path=tmp_path / "soft.duckdb",
        experiment="x",
        name="y",
        config={},
        history=[{"epoch": 1, "test_acc": 0.5}],
        best_test_acc=0.5,
    )
    assert rid is not None


def test_import_real_results_if_present():
    """Optional smoke: import repo results/ when present (not required in CI)."""
    root = Path(__file__).resolve().parents[1]
    results = root / "results"
    if not results.is_dir():
        pytest.skip("no results/ directory")
    jsons = list(results.rglob("ln_*.json"))
    if not jsons:
        pytest.skip("no ln_*.json result files")
    # Use in-memory-like temp via separate path under results is gitignored;
    # use a tmp file next to package via pytest tmp — re-open as unit style.
    # This test only checks parse on one real file.
    parsed = parse_result_json(jsons[0])
    assert "experiment" in parsed
    assert parsed["history"]


def test_build_report_contains_sections(conn):
    for exp, ln, acc in [
        ("v0_1", "none", 0.92),
        ("v0_1", "affine", 0.91),
        ("v0_2", "none", 0.93),
        ("v0_2", "affine", 0.94),
    ]:
        record_completed_run(
            conn,
            experiment=exp,
            name=f"ln_{ln}",
            config={"ln_mode": ln, "hidden": 128},
            history=[{"epoch": 1, "test_acc": acc}],
            seed=42,
            best_test_acc=acc,
            best_epoch=1,
            wall_sec=100.0,
            summary={"epochs_ran": 5},
        )
    text = build_report(conn)
    assert "# Experiment analysis report" in text
    assert "LayerNorm modes" in text
    assert "no_affine" in text
    assert "affine" in text
    assert "`none`" in text or "none" in text
    assert "Version lineage" in text
    assert "v0.1" in text and "v0.2" in text
    assert "Results comparison" in text
    assert "Simple analysis" in text
    assert "0.9400" in text or "0.94" in text
    assert "Per-run detail" in text
    text_nodetail = build_report(conn, include_detail=False)
    assert "Per-run detail" not in text_nodetail