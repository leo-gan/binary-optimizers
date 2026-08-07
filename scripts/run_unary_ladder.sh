#!/usr/bin/env bash
# Sequential Unary link Swarm ladder (WP-U2 → U5).
# Usage (from repo root): bash scripts/run_unary_ladder.sh
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
mkdir -p logs
export PYTHONUNBUFFERED=1

echo "===== $(date -Is) WP-U2 v0_9 decoder ablations ====="
uv run python experiments/v0_9_unary_decoder/train.py --seed 42 --device "${DEVICE:-cpu}" \
  2>&1 | tee logs/v0_9_decoder.log

echo "===== $(date -Is) WP-U3 v0_10 width atlas ====="
uv run python experiments/v0_10_unary_width/train.py --seed 42 --device "${DEVICE:-cpu}" \
  2>&1 | tee logs/v0_10_width.log

echo "===== $(date -Is) WP-U4 v0_11 encoder atlas ====="
uv run python experiments/v0_11_unary_encoder/train.py --seed 42 --device "${DEVICE:-cpu}" \
  2>&1 | tee logs/v0_11_encoder.log

echo "===== $(date -Is) WP-U5 v0_12 CIFAR probe ====="
uv run python experiments/v0_12_unary_cifar/train.py --seed 42 --device "${DEVICE:-cpu}" \
  2>&1 | tee logs/v0_12_cifar.log

echo "===== $(date -Is) ALL DONE =====" | tee logs/unary_ladder_done.txt
