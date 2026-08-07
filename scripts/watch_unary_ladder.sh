#!/usr/bin/env bash
# Emit only terminal status for Grok monitor (DONE/FAILED at end).
# Usage: bash scripts/watch_unary_ladder.sh
# Pair with: bash scripts/run_unary_ladder.sh (or nohup that script).
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

while true; do
  if [[ -f logs/unary_ladder_done.txt ]]; then
    echo DONE
    exit 0
  fi
  # Ladder runner gone without done marker → failed/killed
  if ! pgrep -f 'run_unary_ladder\.sh' >/dev/null 2>&1; then
    sleep 2
    if [[ -f logs/unary_ladder_done.txt ]]; then
      echo DONE
      exit 0
    fi
    echo FAILED
    exit 1
  fi
  sleep 30
done
