#!/usr/bin/env bash
set -euo pipefail
task_repo_root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)
cd "$task_repo_root"
source .venv/bin/activate
task_run_root=${1:-"$task_repo_root/experiments/rdo151/work/score_queue_$(date +%Y%m%d_%H%M%S)"}
mkdir "$task_run_root"
task_run_root=$(cd "$task_run_root" && pwd)
export CUDA_VISIBLE_DEVICES=0
export PYTHONUNBUFFERED=1
printf 'Run directory: %s\n' "$task_run_root"
python -m unittest discover -s experiments/score_search -p 'test_*.py'
python -u experiments/score_search/hpac.py \
  --epochs 12 --lr 0.003 --accum-frames 8 --native-candidates 3 \
  --output "$task_run_root/hpac" \
  2>&1 | tee "$task_run_root/hpac.log" &&
python -u experiments/score_search/token_rdo.py \
  --parent-run "$task_run_root/hpac" \
  --shortlist-per-frame 12 --budgets 128 512 2048 4096 \
  --output "$task_run_root/token_rdo" \
  2>&1 | tee "$task_run_root/token_rdo.log"
