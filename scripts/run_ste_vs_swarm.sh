#!/usr/bin/env bash
# Run shared-protocol STE vs Swarm comparison (logs to DuckDB).
#
# Usage (from repo root):
#   ./scripts/run_ste_vs_swarm.sh
#   ./scripts/run_ste_vs_swarm.sh --methods ste_sgd,swarm_v0_3 --ln-mode affine
#   ./scripts/run_ste_vs_swarm.sh --epochs 5 --patience 2

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

exec uv run python experiments/ste_vs_swarm/train.py "$@"
