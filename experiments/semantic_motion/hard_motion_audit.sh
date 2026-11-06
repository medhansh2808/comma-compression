#!/usr/bin/env bash
# CPU only. Preserve original fits and benchmark; no training or GPU calls.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"
source .venv/bin/activate
export CUDA_VISIBLE_DEVICES=""
BASE=experiments/semantic_motion
TOKENS=experiments/rdo151/cache/tokens151.npy
for NAME in global rank4; do
  if [ ! -f "$BASE/runs/${NAME}_gated/audit.json" ]; then
    python -u "$BASE/motion_codec.py" refine \
      --tokens "$TOKENS" --model "$BASE/runs/$NAME/motion.bin" \
      --output "$BASE/runs/${NAME}_gated" \
      2>&1 | tee "$BASE/runs/${NAME}_gated.log"
  fi
done
if [ ! -f "$BASE/runs/benchmark_v2/summary.json" ]; then
  python -u "$BASE/motion_codec.py" benchmark \
    --tokens "$TOKENS" \
    --models "$BASE/runs/global/motion.bin" "$BASE/runs/rank4/motion.bin" \
             "$BASE/runs/global_gated/motion.bin" "$BASE/runs/rank4_gated/motion.bin" \
    --output "$BASE/runs/benchmark_v2" \
    2>&1 | tee "$BASE/runs/benchmark_v2.log"
fi
cat "$BASE/runs/benchmark_v2/summary.md"
