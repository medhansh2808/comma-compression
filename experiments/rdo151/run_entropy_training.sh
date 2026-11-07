#!/usr/bin/env bash
# Explicit GPU job: train existing probability model, then measure a verified
# full archive. No package installs, driver changes, or automatic process kills.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"
source .venv/bin/activate
OUT="$ROOT/experiments/rdo151/work/entropy_rate_v1"
STEPS="${1:-6000}"
mkdir -p "$ROOT/experiments/rdo151/logs"
RESUME=()
if [ -f "$OUT/checkpoint.pt" ]; then RESUME=(--resume); fi
python -u experiments/rdo151/train_entropy.py train \
  --device cuda --steps "$STEPS" --output "$OUT" "${RESUME[@]}" \
  2>&1 | tee -a experiments/rdo151/logs/entropy_rate_train.log
python -u experiments/rdo151/train_entropy.py evaluate \
  --device cuda --output "$OUT" \
  2>&1 | tee -a experiments/rdo151/logs/entropy_rate_evaluate.log
