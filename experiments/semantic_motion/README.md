# Learn motion, encode the corrections

Question: can a small learned spatial motion representation explain enough of
the optimized driving-video token sequence to pay for its own transmitted
parameters? This is a new experimental token codec built on the cached #151
maps. It is not a complete challenge submission or a claim of a new research
algorithm. Motion compensation and low-rank representations are established
ideas; the experiment is how they behave on this evaluator-specific sequence.

## What changes

Instead of changing the leader's fixed t-2 shift, fit a displacement field
between consecutive semantic token maps. A model represents flow using a
per-frame global field plus four learned spatial deformation bases mixed by
per-frame coefficients. It can express different movement in different image
regions. It does not identify objects or learn a physical world model.

Train differentiably using bilinear sampling, a categorical prediction-error
surrogate, smoothness and magnitude penalties. Parameters are quantized with
straight-through gradients. The deployed CPU codec reconstructs the quantized
field, rounds it to integer displacements and copies previous-frame labels.
It records exact corrections wherever the prediction is wrong. An independent
decoder reconstructs each previous map before predicting the next one.

No trained parameter, first frame or residual is free: all are inside each
`.smc` archive and counted in its byte size. The initial residual coder is a
shared Brotli backend for all ablations. The training objective is not the
exact compressed byte count. Better prediction may still mean worse rate.

## Three comparisons

| Case | Prediction | Learned state |
|---|---|---|
| identity | Copy the preceding map | None |
| global | Fit a bounded global affine field per transition | 6 parameters per transition |
| rank4 | Global field plus shared spatial motion patterns | Global parameters, 4 bases and per-frame mixtures |

The affine field is passed through a smooth bound, so the deployed global
field is only approximately affine at large displacement magnitudes. A rank
of zero is the global-only model. All three must recover identical tokens.
The rank4 comparison tests whether local deformation helps enough to pay for
the extra model. This is per-video fitting, not an unseen-video test.

## Preparation and checks (CPU only)

Use the existing challenge `.venv`; it already supplies Torch, NumPy, Pillow
and Brotli. No new GPU packages, drivers or original-video downloads.

```bash
cd ~/medhansh/comma-compression-rdo151
source .venv/bin/activate
CUDA_VISIBLE_DEVICES="" python experiments/semantic_motion/motion_codec.py smoke
```

The smoke check covers gradients, quantized Torch/NumPy flow agreement, warp
direction, lossless state packing, independent token decoding, and preview
generation. It uses synthetic maps and CPU tensors. It does not test CUDA.

## Explicit GPU launch

Launch only when you intend to use the GPU. Setup does not start training.
The two fits run sequentially for 3,000 steps each. This schedule is a first
experiment, not a claim of convergence. Logs report elapsed time every 50
steps. Checkpoints save every 250 steps; rerunning the shell script resumes
from the latest checkpoint. On an interrupted run at most 249 steps are lost.

```bash
tmux new-session -s semantic-motion
```

Inside tmux:

```bash
cd ~/medhansh/comma-compression-rdo151
unset CUDA_VISIBLE_DEVICES
bash experiments/semantic_motion/train_pair.sh 3000
```

Detach using Ctrl+B then D. Reattach using `tmux attach -t semantic-motion`.
To extend both existing fits to 6,000 total steps, rerun the script with 6000.
Do not change rank/grid/data when resuming an existing output directory.

## Benchmark and produce visualizations (CPU only)

This separate command encodes all three cases, decodes each archive from its
own bytes without the original token cache, compares the recovered maps, and
generates the byte table plus GIFs. The original cache is used only to verify
equality, not to reconstruct the output. It does not import Torch.

```bash
cd ~/medhansh/comma-compression-rdo151
source .venv/bin/activate
CUDA_VISIBLE_DEVICES="" python -u experiments/semantic_motion/motion_codec.py benchmark \
  --tokens experiments/rdo151/cache/tokens151.npy \
  --models experiments/semantic_motion/runs/global/motion.bin \
           experiments/semantic_motion/runs/rank4/motion.bin \
  --output experiments/semantic_motion/runs/benchmark_v1
```

Use a new benchmark directory after extending training. Results include
`summary.md`, `summary.json`, `.smc` token archives, and a GIF for each model.
The GIF shows original token colours, prediction, flow and correction pixels
at selected transitions across the video. It is not real-time playback.

Standalone decode, with no original cache or Torch required:

```bash
CUDA_VISIBLE_DEVICES="" python experiments/semantic_motion/motion_codec.py decode \
  --archive experiments/semantic_motion/runs/benchmark_v1/rank4.smc \
  --output experiments/semantic_motion/runs/rank4_recovered.npy
```

## What would make the result interesting

- Lower prediction error and fewer residual bytes, but a larger total file:
  motion representation is useful; its parameter rate is the bottleneck.
- Global motion helps, but rank4 does not: shared deformation bases may be
  too restrictive or insufficiently trained.
- Rank4 reduces total rate versus global: local motion buys real compression
  under this residual coder. It still must beat the strong HPAC baseline.
- Large errors around newly visible regions or moving boundaries: inspect
  keyframes, occlusion handling, or a richer motion-conditioned residual coder.

The reference token stream is 108,891 bytes. This is a useful reference line,
not an apples-to-apples full submission score: the reference also carries HPAC
model bytes, while this prototype charges its motion state inside `.smc`.
No renderer, pose-carrier integration or official evaluation is provided in
this prototype. Those are next steps only if the rate results justify them.

## Validation performed before delivery

- CPU smoke checks passed under Torch 2.9.0+cpu.
- A 200-step synthetic translation fit reduced actual integer-warp prediction
  error from 12.5% to approximately 0.2083%.
- Its residual stream shrank from 559 to 221 bytes, but motion overhead made
  the complete toy token archive larger: 759 to 874 bytes. This is an example
  of why prediction quality is not the compression objective by itself.
- Synthetic exact round trips passed for identity, global and local models.
- Full driving-video training and CUDA behavior remain untested.

## Sources and credit

The optimized token maps, inherited semantic renderer and pose representation
come from the public comma challenge lineage, especially #150 and #151. This
experiment does not claim those as new work. Preserve the existing repository
licenses and `LINEAGE.md`.

- https://github.com/commaai/comma_video_compression_challenge/pull/151
- https://github.com/commaai/comma_video_compression_challenge/pull/150
- https://github.com/tensorsofthewall/comma_semantic_folded

Read and understand experiment code before incorporating any part into a
challenge submission; follow the challenge's AI-use policy. The deliverable
here is an experimental implementation and benchmark, not a public PR draft.
