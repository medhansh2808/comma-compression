# Causal context probability adapter, first fit

This is a training/probability-measurement experiment, not an integrated codec
or valid new submission. Frozen starting architecture/state derives from #151,
#150 and their credited lineage. Preserve their licenses and attribution.

25 input features: five relative corrected HPAC logits clipped to [-16,0],
one-hot warped previous token, warped t-2 token, unwarped previous-frame
boundary bucket, and current frozen HPAC-plus-fixed-table argmax. No current
true token or current complete boundary mask is an input. Upstream HPAC uses
its inherited causal masks. No current-neighbor features are added in v1.

8 hidden units, ReLU, five output adjustments: 253 scalar parameters. Integer
weights/biases bounded to [-127,127]; first layer rescales by 128, nearest-even
rounds and clamps hidden codes to [0,127]. Last layer rescales by 128 and clamps
to [-32,32], yielding at most +/-4 logit corrections on a 1/8 grid. Corrections
apply only where previous boundary distance <=1 or the frozen top-two-logit
margin <=4. Both conditions are available to a sequential decoder.

Zero last-layer weights/biases give exactly zero corrections. First-layer
weights have fixed seeded random initialization. This initialization yields
zero initial gradient to the first layer until the last layer changes.
Training uses straight-through gradients through integer quantization and
unweighted sampled token NLL (31-bit forward cap), with HPAC completely frozen.
One random frame and 16384 uniformly sampled positions per update. Adam .05
in integer-code units, 2000 steps. No claim this optimizer is proven stable.
Checkpoints/probes every 250 steps; logs every 50. Seed 1234. Same-config resume
supported by --resume; otherwise refuses an existing run directory.

Probe selection uses 16 fixed frames from the known compression target. It is
not a held-out generalization benchmark. Probe evaluation serializes/reloads
integer parameters and charges every model byte: 518 bytes including CAD1
header. A reference fallback with no adapter overhead stays initial best.
Training checkpoints are not charged as transmitted data; only packed inference
parameters are. A real container would also need framing/integration overhead.

The separate measure command checks all 600 frames using unweighted integer
frequency cost. It measures BEFORE the adaptive native corrector, excluding
stream termination. Thus estimated_net_saving_bytes is a PROXY, not an archive
saving. The 42.8KB reference shortening may disappear. Native coding, exact
roundtrip, full archive size, RGB evaluation and T4 timing remain future work.
No automatic full encode, decoder edit, official submission, or model swap.

Prerequisites: existing .venv, train_entropy.py, profile_reference.py, reference
runtime/archive and verified cached tokens. Install nothing on the server.

CPU check:
CUDA_VISIBLE_DEVICES="" python experiments/context_adapter/run.py check --device cpu

GPU training (separate explicit launch):
CUDA_VISIBLE_DEVICES=0 python -u experiments/context_adapter/run.py train --device cuda

GPU full-sequence proxy measurement (separate explicit launch):
CUDA_VISIBLE_DEVICES=0 python -u experiments/context_adapter/run.py measure --device cuda

Outputs: experiments/context_adapter/runs/v1/{best.cad,best.json,checkpoint.pt,
training.jsonl,probes.jsonl,full_measurement.json}. Defaults can be changed with
--output. On interruption rerun training with --resume. At most 249 updates lost.
GPU usage competes with other jobs. CUDA timing and peak memory unmeasured.

Local validation: NumPy integer serialization, zero-output parity, signed
nearest-even rounding and position locality; Python syntax checks passed.
Workspace has no PyTorch or original runtime after maintenance, so actual
reference/PyTorch integration is explicitly deferred to check on the server.
It must PASS before training. That check verifies upstream dense/sparse causal
logit parity, original archive/model reconstruction, zero corrections on the
reference, and nonzero PyTorch forward versus NumPy integer forward parity.
These are bounded tests, not a proof of every context/device combination.
