#!/usr/bin/env bash
# Print the STE vs Swarm comparison report from the DuckDB store.
#
# Covers shared-protocol runs (experiment=ste_vs_swarm): method × LN tables,
# Swarm−STE deltas, and per-run detail. See experiments/ste_vs_swarm/PROTOCOL.md.
#
# Usage (from repo root):
#   ./scripts/report_ste_vs_swarm.sh
#   ./scripts/report_ste_vs_swarm.sh --db /path/to/experiments.duckdb
#   ./scripts/report_ste_vs_swarm.sh > results/ste_vs_swarm_report.md
#
# Populate data first (if empty):
#   ./scripts/run_ste_vs_swarm.sh
#   ./scripts/run_ste_vs_swarm.sh --methods ste_sgd,swarm_v0_3 --ln-mode affine

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

exec uv run python -m binary_optimizers.store report --experiment ste_vs_swarm "$@"
