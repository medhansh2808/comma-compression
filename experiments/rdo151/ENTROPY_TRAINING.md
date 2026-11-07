# One targeted training run: fine-tune the existing entropy model

This experiment directly targets token coding rate in the leader's current
representation. It does not train motion, change token values, approximate
the renderer, or alter the pose carrier. Identical decoded tokens and RGB
state mean distortion is held fixed; the objective is lower total rate.

## What is trained

The existing quantized HPAC spatial/temporal model and its shared t-2 branch:
34,250 trainable parameters. The 600x8 frame embeddings stay frozen. Every
base-model weight row stays within its original serialized signed precision.
Biases and exponents remain in the deployed ranges. No additional network or
decoder architecture is introduced. All model values are carried in the
existing HPAC state and temporal extension formats.

The forward pass uses the inherited straight-through integer operations,
causal spatial masks and deployed logit quantization. It sees full cached
maps under teacher forcing, but the causal masks prevent access to symbols
not available to the decoder. Sampled scan groups are compared against the
decoder's sparse evaluator with the current/future groups zeroed.

## Objective and limitation

Optimize conditional negative log likelihood of the actual tokens, after the
existing fixed correction table. The logit quantization is applied with a
straight-through gradient, and the minimum coded frequency is approximated
with a 31-bit cost cap, also using a straight-through gradient.

The native adaptive corrector and complete arithmetic-stream termination are
not differentiable parts of this training loss. Thus **this is a rate proxy,
not a claim to optimize exact final archive bytes by gradient descent**.

Every 500 steps, measure a fixed, evenly spaced 16-frame probe using the
deployed integer-frequency conversion, and serialize the quantized model.
Checkpoint selection combines the estimated sequence-rate proxy and actual
compressed entropy-model bytes. Those probe estimates must not be compared
as absolute byte counts with the published 108,891-byte stream: they precede
the adaptive corrector and do not include its finalization behavior.

This is per-video compression fitting. The probe is a deterministic selection
sample from that known video, not a generalization benchmark or hidden test.

## Actual success criterion

After training, use the selected quantized checkpoint in the existing native
encoder on all 600 unchanged maps. Count the complete archive, including the
new serialized HPAC and temporal state. Try both the ordinary finalization
and the existing shortening rule; accept either only after exact decoding.
Verification stops on the first mismatching causal symbol group.

Verify that renderer, pose carrier, correction table, warp signal and visual
extensions are unchanged. Mark success only if a verified archive is smaller
than 164,361 bytes. A lower training loss alone is not success. Loss of the
reference's special shortening advantage is charged in full.

The reference is retained as the initial best checkpoint. If no trained
checkpoint improves the combined proxy, the expensive re-encode is skipped.
Full official RGB evaluation/T4 testing remains separate if a candidate wins.

## Install nothing; use the existing checkout

Prerequisites are the existing challenge `.venv`, prepared #151 source tree,
reference archive, token cache, C compiler and `profile_reference.py` from the
previous update. This package adds source files only. Existing environments,
drivers, training artifacts and reference files are not replaced.

CPU preparation check (no GPU allocation):

```bash
cd ~/medhansh/comma-compression-rdo151
source .venv/bin/activate
CUDA_VISIBLE_DEVICES="" python experiments/rdo151/train_entropy.py check --device cpu
```

This checks exact original model serialization, byte-identical original archive
rebuild, sampled dense/causal-sparse forward parity, and finite nonzero gradients.
It does not assert that the published CUDA stream decodes on every CPU backend.

## Explicit GPU launch

When GPU use is intended, start/attach tmux `medh` and launch:

```bash
cd ~/medhansh/comma-compression-rdo151
unset CUDA_VISIBLE_DEVICES
bash experiments/rdo151/run_entropy_training.sh 6000
```

This launches one 6,000-step fit, then the full native encode/verification of
the selected checkpoint. Fits and native coding run sequentially. Training
prints every 50 steps, probes/checkpoints every 500. Estimate runtime from the
actual step timing; CUDA training was not timed in development. The full
causal coding/verification can take longer than the dense training itself.

Detach with Ctrl+B then D. Reattach with `tmux attach -t medh`. Rerunning the
same shell command resumes the checkpoint. At most 499 training steps are lost
after an interruption. Completed encoded streams are reused after provenance
checks, so interrupted verification does not require another completed encode.

To use the GPU elsewhere after training, the individual commands are available:

```bash
python -u experiments/rdo151/train_entropy.py train --device cuda --steps 6000
python -u experiments/rdo151/train_entropy.py evaluate --device cuda
```

The shell launcher is the simpler default. Setup never launches it automatically.

## Results

```bash
cat experiments/rdo151/work/entropy_rate_v1/best.json
cat experiments/rdo151/work/entropy_rate_v1/evaluation.json
```

Outputs are under the already ignored `experiments/rdo151/work/entropy_rate_v1`.
Logs are `experiments/rdo151/logs/entropy_rate_train.log` and
`experiments/rdo151/logs/entropy_rate_evaluate.log`.

Per-model immutable evaluation directories are keyed by quantized-state hash.
A file named `encoder_input.zip` is an internal model-loading scaffold with
the old token stream: it is **not a valid final candidate**. Only archives
explicitly marked `roundtrip_verified: true` in evaluation results are valid.
`best_verified_archive.zip` is saved inside that evaluation directory only if
the candidate is strictly smaller than the reference.

The selected state may change after extending training; completed earlier
evaluations remain in their hash-specific directories. Resuming requires the
same learning rate, seed, input cache and probe selection.

## Validation before delivery

- Original HPAC and temporal-state serialization reproduced exact input bytes.
- Rebuilding the original complete archive reproduced its SHA-256.
- Dense teacher-forced outputs matched sampled sparse causal decoder groups.
- Quantized-model gradients were finite and nonzero.
- A short CPU training loop and checkpoint resume were exercised.
- A deliberately changed temporal state was packed and native-encoded for one
  frame; independent decoding recovered the exact reference tokens.
- Full 600-frame candidate encoding and CUDA training remain untested.

Architecture, fitted starting state, formats and native coder are inherited
from the public #151/#150 lineage. Preserve their attribution and licenses.
This is a bounded encoder-side experiment, not a public submission description.
