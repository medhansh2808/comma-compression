#!/usr/bin/env bash
# Explicit GPU launch. Fits both controls sequentially and resumes checkpoints.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"
source .venv/bin/activate
TOKENS=experiments/rdo151/cache/tokens151.npy
BASE=experiments/semantic_motion
STEPS="${1:-3000}"
test -s "$TOKENS"
mkdir -p "$BASE/runs"
python "$BASE/motion_codec.py" smoke
for NAME in global rank4; do
  RANK=0
  if [ "$NAME" = rank4 ]; then RANK=4; fi
  RESUME=()
  if [ -f "$BASE/runs/$NAME/checkpoint.pt" ]; then RESUME=(--resume); fi
  python -u "$BASE/motion_codec.py" train \
    --tokens "$TOKENS" --output "$BASE/runs/$NAME" \
    --rank "$RANK" --steps "$STEPS" --batch 4 \
    --device cuda "${RESUME[@]}" \
    2>&1 | tee -a "$BASE/runs/$NAME.log"
done
echo "Both motion fits finished. Run the CPU benchmark separately."
