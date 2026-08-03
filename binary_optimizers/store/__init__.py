"""Experiment result store (DuckDB).

Typical usage::

    from binary_optimizers.store import connect, init_db, record_completed_run

    conn = connect()
    init_db(conn)
    record_completed_run(
        conn,
        experiment="v0_4",
        name="ln_affine",
        config={"hidden": 128},
        history=[{"epoch": 1, "test_acc": 0.9}],
        best_test_acc=0.9,
    )

Import legacy JSON::

    python -m binary_optimizers.store.import_json
    python -m binary_optimizers.store.import_json --replace
"""

from .db import (
    STATUS_COMPLETED,
    STATUS_FAILED,
    STATUS_RUNNING,
    compare_best,
    connect,
    fail_run,
    finish_run,
    format_compare_markdown,
    get_run,
    init_db,
    list_runs,
    load_history,
    log_history,
    log_metrics,
    new_run_id,
    record_completed_run,
    soft_record_completed_run,
    start_run,
    try_git_commit,
)
from .import_json import import_results, parse_result_json, stable_run_id
from .paths import ENV_DB, default_db_path, repo_root
from .report import build_report, print_report
from .versions import (  # noqa: F401
    REGISTRY,
    TRAIN_BUDGET_PROTOCOL,
    config_version_fields,
    db_notes,
    enrich_config,
    get_meta,
)

__all__ = [
    "ENV_DB",
    "REGISTRY",
    "STATUS_COMPLETED",
    "STATUS_FAILED",
    "STATUS_RUNNING",
    "TRAIN_BUDGET_PROTOCOL",
    "build_report",
    "compare_best",
    "config_version_fields",
    "connect",
    "db_notes",
    "default_db_path",
    "enrich_config",
    "fail_run",
    "finish_run",
    "format_compare_markdown",
    "get_meta",
    "get_run",
    "import_results",
    "init_db",
    "list_runs",
    "load_history",
    "log_history",
    "log_metrics",
    "new_run_id",
    "parse_result_json",
    "print_report",
    "record_completed_run",
    "repo_root",
    "soft_record_completed_run",
    "stable_run_id",
    "start_run",
    "try_git_commit",
]
