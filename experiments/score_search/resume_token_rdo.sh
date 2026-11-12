#!/usr/bin/env bash
set -euo pipefail
task_repo_root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)
cd "$task_repo_root"
source .venv/bin/activate
if [ "$#" -ne 1 ]; then
  printf 'Usage: bash experiments/score_search/resume_token_rdo.sh QUEUE_DIRECTORY\n' >&2
  exit 2
fi
task_queue=$(cd -- "$1" && pwd)
export CUDA_VISIBLE_DEVICES=0
export PYTHONUNBUFFERED=1
python -m unittest discover -s experiments/score_search -p 'test_*.py'
python -u experiments/score_search/token_rdo.py \
  --resume --check-only --parent-run "$task_queue/hpac" \
  --output "$task_queue/token_rdo" \
  2>&1 | tee -a "$task_queue/token_rdo-resume.log" &&
python -u experiments/score_search/token_rdo.py \
  --resume --parent-run "$task_queue/hpac" \
  --shortlist-per-frame 12 --budgets 128 512 2048 4096 \
  --output "$task_queue/token_rdo" \
  2>&1 | tee -a "$task_queue/token_rdo-resume.log"
