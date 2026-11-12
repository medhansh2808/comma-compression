# Corrected HPAC plus distortion-aware token editing

Run `bash experiments/score_search/run_both.sh` inside tmux with the existing
repository venv. GPU 0 is selected explicitly. The two stages are joined with
`&&` and pipefail. A runtime error stops the queue; a successful experiment
that finds no improvement keeps the baseline and allows the next experiment.
No packages are installed and no training or evaluation starts on the laptop.

## Inputs

Existing files on the server:

- experiments/rdo151/work/rd/xp1.zip: the current 163872-byte candidate.
- experiments/rdo151/cache/tokens151_o2.npy: corrected token cache, SHA256
  d64c2a094dbdcdf6b381c94810f9d672ac33c3c960eb6ae35ba09df0af51c3a1.
- experiments/rdo151/work/reference: current working XP1-compatible decoder
  sources, including the added rc64 encoder and native corrector C sources.
- experiments/rdo151/train_entropy.py and profile_reference.py: existing
  training-format and integer-frequency helpers.
- experiments/rdo151/rd/common.py, vendor/frame_utils.py and vendor/modules.py,
  plus the evaluation model weights at the paths common.py already uses.
- experiments/rdo151/work/rd/gt_dali.npz: existing ground truth cache. A missing
  cache is an error; this code never creates AV-decoded replacement ground truth.

This extends the experiment code already present on the server. It is not a
standalone submission or a complete replacement for the repository.

## Isolation and validation

Each stage snapshots executable runtime sources into its new output directory,
then compiles private RC64 and corrector shared libraries at -O2 with floating
point contraction disabled and no fast math. Existing runtime sources,
libraries, archives and completed experiment results are not edited.

Startup checks corrected token provenance, exact model serialization, archive
rebuild and sampled dense/sparse entropy-model agreement. Baseline scoring uses
actual archive decoding and both distortion networks. The first run requires
the XP1 score to reproduce 0.1253025222561036 within 0.000002. This is a local
numerical consistency gate, not official evaluation or proof of cross-device
portability. Snapshot source, compiler, archive and ground-truth hashes are saved.

Training/evaluation requires temporary disk space for rendered RGB: about
3.67 GB per full video. During token proposals this baseline video stays on disk;
later candidate videos are evaluated one at a time and deleted after scoring.
A crash may leave frames.raw for diagnosis. No source-transfer ZIP is created
on the server. archive.zip files are the actual compression candidates required
by the experiment. Model_input.zip is an intermediate archive with updated HPAC
state; its token stream is deliberately replaced before any decode/evaluation.

## Stage 1: HPAC

Fine-tune the existing weights and biases on all 600 frames, with 8-frame
gradient accumulation, 12 epochs and a cosine learning rate from 0.003 to
0.0003. Integer model forward and row widths use the existing reference format.
Frame embeddings, exponents and shared temporal parameters are frozen.

After every epoch, serialize and reload, verify representative quantized
logits, and measure all 600 frames using the fixed-corrector frequency proxy.
Save every epoch. Rank distinct deployed checkpoints by the combined token
proxy plus actual serialized HPAC/temporal model size, and encode the top 3
with the real adaptive corrector and native arithmetic coder. This proxy
shortlist may miss a checkpoint that would win with the native coder.

Every candidate token stream must decode exactly to the corrected cached
tokens. A smaller archive gets full RGB rendering and score measurement.
best.json points to the baseline unless an actually measured candidate wins.
The 42801-byte shortening anomaly is not assumed or reproduced; full native
streams are used, with no unverified truncation.

## Stage 2: token edits

Read the first stage's selected archive and exact runtime snapshot. Actually
decode it, capture rate-saving alternative labels at their native coding
probabilities, and render the baseline. Only alternatives saving more than
0.25 estimated bits are considered; retain up to 512 per frame.

The exact edit renderer and metrics use the decoder's batch of eight.
Differentiate through the token embedding, singleton renderer and RGB rounding using STE,
SegNet and PoseNet to rank proposals. The segmentation surrogate is a sigmoid
of the wrong-vs-target logit margin. The pose term uses the derivative of the
full-video square-root score at the baseline distortion. The singleton gradient is an approximate proposal derivative with forward
pixels anchored to verified batched decoder output. Check batched renderer
pixels and differentiable metric preprocessing against the actual baseline path.

For up to 12 proposals per frame, render each individual edited token map and
evaluate exact label disagreements and pose error. Rank using that measured
single-edit distortion change minus estimated rate savings. Enforce 8-pixel
Chebyshev spacing within each frame to limit interactions. This is a heuristic,
not a guarantee: rendering has broader spatial effects, coding contexts change
after edits, and temporal probability feedback changes actual stream size.

Build candidates with up to 128,512,2048,4096 edits. Re-encode each full token
sequence, decode to verify the edited tokens, then perform full-video rendering
and score measurement with actual archive bytes. Only a measured lower score
replaces best.json. If no promising individual edits exist, finish with the
baseline and report that result.

## Progress and results

The launcher prints the new run directory. Logs are hpac.log and token_rdo.log.
Each stage writes best.json, summary.json, per-candidate results and a runtime
snapshot. HPAC keeps epoch models and last.pt; HPAC has no automatic resume.
Token editing now supports --resume: see FIX_AND_RESUME.md.
Use a new run directory for a new attempt rather than deleting an existing run.

Detach tmux using Ctrl+B, release, D. Ctrl+C interrupts the job.

## Verification performed before delivery

Python syntax and shell syntax checked; CPU score-accounting and proposal-
selection unit checks run. The supplied native RC64 source was compiled at -O2
and tested with streaming encode/decode examples locally. Full model execution,
GPU memory demand and end-to-end integration remain unverified locally: the
reference archives, evaluation weights and GPU are on the server. Startup
invariants are intended to stop rather than silently continue on a mismatch.

Training on all 600 frames is sequence-specific compression, not an unseen-
video generalization experiment. Evaluation is local; official evaluation and
a portable submission decoder remain separate steps.
