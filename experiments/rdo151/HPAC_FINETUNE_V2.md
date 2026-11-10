# HPAC fine-tuning v2

This is an extension of the existing experiments/rdo151/train_entropy.py.
It requires that module, profile_reference.py, the extracted reference runtime,
and the verified leader151.zip and tokens151.npy cache. No new packages.

Train on the server:

```bash
cd ~/medhansh/comma-compression-rdo151
source .venv/bin/activate
set -o pipefail
mkdir -p experiments/rdo151/logs
CUDA_VISIBLE_DEVICES=0 python -u experiments/rdo151/finetune_hpac_v2.py \
  --device cuda --epochs 8 --accum-frames 8 --lr 0.001 \
  2>&1 | tee experiments/rdo151/logs/hpac_finetune_v2.log
```

Use tmux. Detach with Ctrl+B, release, then D. Do not press Ctrl+C to detach.
The output directory must not already exist. Existing runs are never overwritten.
There is no resume command; last.pt saves model and optimizer state for recovery
or subsequent analysis. Use a different --output for a separate new run.

## What changes

- Train existing HPAC parameters, with frame_embed.weight, m2/b2/e2 and any
  parameter whose name ends in exponent frozen.
- Shuffle and visit all 600 frames every epoch. Each frame contributes ordinary
  token cross entropy, averaged across accum-frames before an optimizer update.
- Reuse the existing quantized forward, logit rounding, weight projection and
  reference row widths. No additional model is introduced.
- Serialize, reload, verify representative deployed logits, then measure all
  600 frames before selecting a checkpoint. Selection charges stored HPAC plus
  shared temporal model bytes. Every epoch is retained regardless of selection.

This remains a fixed-correction-table proxy: it does not run the decoder's
complete adaptive corrector or native arithmetic coder. In particular it does
not measure the reference's 42,801-byte stream-shortening advantage. A smaller
combined_proxy_bytes is a candidate for a real encode, not proof of savings.
The 11888-byte baseline model and ~153111-byte proxy are sanity references,
not substitutes for reference archive parity.

## Outputs

Default directory: experiments/rdo151/work/hpac_finetune_v2.

- config.json: selected parameter names, seed and settings.
- training.jsonl: training progress and gradient norm.
- measurements.jsonl: full-sequence serialized checkpoint proxy measurements.
- epochs.jsonl: comparisons to the baseline and current best epoch.
- best.json, best_hpac.bin, best_prev2.bin: selected deployed parameter state.
- epoch_001/, etc.: every serialized epoch and its measurements.
- last.pt: final floating training state and optimizer.

changed_hpac_bytes exposes whether quantized parameters actually changed.
If best.json still reports epoch 0, no measured checkpoint beat the original.
Do not submit a model archive assembled with the old token stream: changed
probabilities require a new stream encoded with that same deployed model, an
exact token round trip, and full video evaluation.

## Verification status

Syntax compilation checked locally. The complete runtime and GPU execution
were not available locally. At startup the existing trainer's check() validates
reference model serialization, full reference archive reconstruction and sampled
dense/sparse causal inference parity. Every candidate additionally checks
load/save equality and quantized logits on frames 0,199,399,599. These sampled
checks do not replace native full-sequence token round-trip verification.

## Reading results

```bash
cat experiments/rdo151/work/hpac_finetune_v2/best.json
cat experiments/rdo151/work/hpac_finetune_v2/epochs.jsonl
```

All 600 frames are training data for this sequence-specific compressor;
these measurements make no claim about performance on unseen videos.
