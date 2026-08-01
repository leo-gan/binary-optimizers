"""Allow ``python -m binary_optimizers.store`` for quick queries."""

from __future__ import annotations

import argparse
import sys

from .db import (
    compare_best,
    connect,
    format_compare_markdown,
    init_db,
    list_runs,
)
from .import_json import main as import_main
from .paths import default_db_path
from .report import print_report


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    p = argparse.ArgumentParser(description="Experiment store utilities")
    p.add_argument(
        "--db",
        default=None,
        help=f"DuckDB path (default: {default_db_path()})",
    )
    sub = p.add_subparsers(dest="cmd", required=True)

    p_import = sub.add_parser("import", help="Import results/**/*.json")
    p_import.add_argument("--results-dir", default=None)
    p_import.add_argument("--replace", action="store_true")

    p_list = sub.add_parser("list", help="List runs")
    p_list.add_argument("--experiment", default=None)
    p_list.add_argument("--limit", type=int, default=20)

    p_best = sub.add_parser("best", help="Best run per experiment/name")
    p_best.add_argument("--experiment", default=None)

    p_report = sub.add_parser(
        "report",
        help="Print comparison tables, version diffs, and LN-mode notes",
    )
    p_report.add_argument("--experiment", default=None, help="Filter to one experiment id")
    p_report.add_argument(
        "--no-detail",
        action="store_true",
        help="Omit per-run detail table",
    )

    args = p.parse_args(argv)

    if args.cmd == "import":
        import_argv = []
        if args.db:
            import_argv.extend(["--db", str(args.db)])
        if args.results_dir:
            import_argv.extend(["--results-dir", str(args.results_dir)])
        if args.replace:
            import_argv.append("--replace")
        return import_main(import_argv)

    conn = connect(args.db)
    init_db(conn)
    try:
        if args.cmd == "list":
            rows = list_runs(
                conn, experiment=args.experiment, limit=args.limit
            )
            for r in rows:
                acc = r.get("best_test_acc")
                acc_s = f"{acc:.4f}" if acc is not None else "  n/a "
                print(
                    f"{acc_s}  {r.get('experiment')}  {r.get('name')}  "
                    f"seed={r.get('seed')}  {r.get('run_id')[:12]}…"
                )
            return 0
        if args.cmd == "best":
            rows = compare_best(conn, experiment=args.experiment)
            print(format_compare_markdown(rows))
            return 0
        if args.cmd == "report":
            print_report(
                conn,
                experiment=args.experiment,
                include_detail=not args.no_detail,
            )
            return 0
    finally:
        conn.close()
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
