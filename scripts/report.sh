#!/usr/bin/env bash
# Generate the experiment analysis report from the DuckDB store.
#
# Usage (from repo root):
#   ./scripts/report.sh
#   ./scripts/report.sh --no-detail
#   ./scripts/report.sh --experiment v0_4
#   ./scripts/report.sh --db /path/to/experiments.duckdb
#
# Optional: write markdown to a file
#   ./scripts/report.sh > results/report.md

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

exec uv run python -m binary_optimizers.store report "$@"
