# Fix and resume token editing

This addresses the observed `Proposal renderer differs from actual master at
frame 0` failure. The original proposal renderer used batch 1 while the actual
decoder used batch 8. Floating point kernels can produce different rounded
pixels at different batch sizes. This is a supported explanation for the
failure, not a locally GPU-confirmed diagnosis.

Exact candidate rendering now uses the decoder's original eight-frame batch,
and exact candidate metrics use the evaluator's eight-pair batch. Baseline
RGB parity remains mandatory and produces a diagnostic JSON if it fails.
The gradient model uses singleton rendering to keep activation memory low,
with its forward pixels anchored to the verified decoder output. Its derivative
is explicitly approximate: it proposes edits, while batched measured metrics
and real archive re-encoding decide acceptance. Evaluator preprocessing inputs
are checked directly, instead of requiring different gradient/inference kernels
to produce bit-identical predictions.

## Resume the interrupted run

After committing the updated source files on the laptop and pulling them on
the server, run inside tmux:

```bash
cd ~/medhansh/comma-compression-rdo151
bash experiments/score_search/resume_token_rdo.sh \
  experiments/rdo151/work/score_queue_20261112_134613
```

The launcher checks baseline rendering/metrics at frames 0,7,8,599, checks
the gradient/preprocessing path at the first frame with proposals, and then
continues token editing. It does not repeat HPAC training or native HPAC
candidate evaluation. It reuses the interrupted token run's runtime snapshot,
baseline RGB and completed probability trace if present and valid. If baseline
RGB or a trace is missing, the baseline decode is repeated as needed.

If any parity check still fails, stop and inspect renderer_parity_batch_*.json
or metric_parity_frame_*.json. The check has not been relaxed to accept different
decoder pixels. GPU execution and exact parity must still be checked on the
server; local validation covered syntax and five CPU regression checks.

## New checkpoint behavior

Proposal progress is written atomically to proposal_state.json after every
completed frame. A partial frame is recomputed. Completed native encodes and
evaluations are reused when archive, input and target-token hashes match.
An interrupted encode/decode restarts that candidate. Saved baseline inputs,
runtime sources, ground truth and PyTorch version are validated on resume.

The older proposal_checks.json file alone is not a complete frame checkpoint;
the original frame-0 failure has no completed proposal frames to recover.
Future resumes use proposal_state.json. Use the same launcher command again
after an interruption. Run only one process against an output directory.

Actual compression candidate archive.zip files are still generated. No ZIP
bundle needs to be copied to or extracted on the server; source changes travel
through GitHub.
