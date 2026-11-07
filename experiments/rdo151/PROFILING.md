# Frozen-reference coding-cost measurement

No model training, parameter updates, token edits, RGB reconstruction or
SegNet/PoseNet evaluation. The profiler derives a bounded prefix loop from
the existing decoder and installs observational hooks. It does not rewrite
the decoder source, probability rows or archive. All decoded symbols must
match the verified reference cache; a mismatch stops the run immediately.

## CPU support and numerical parity

The decoder supports CPU inference, but that does not guarantee reproducing
the CUDA-scored arithmetic stream. Tiny probability differences can change
an arithmetic decoder's subsequent symbols. In a local Torch 2.9.0+cpu test,
the uninstrumented decoder already diverged from the GPU reference in the
first frame. The profiler's parity gate rejected that run. This is not a
training requirement or evidence that every CPU backend will fail.

Test one frame on the server's CPU first. If it fails the cache comparison,
do not run 600 CPU frames and do not bypass verification. Try the CUDA backend
when the GPU is available. That uses the frozen model for inference only.
Exact backend reproduction is required to attribute costs to the reference.
No CUDA integration test could be run in the development environment.

## Existing environment only

Use the challenge `.venv` already installed. No fresh environment or downloads
are required. The profiler compiles the existing native coder/corrector in a
temporary directory using the existing C compiler. It uses the token cache,
reference archive, and optional learned motion state already present.

CPU-only native counter self-test:

```bash
cd ~/medhansh/comma-compression-rdo151
source .venv/bin/activate
CUDA_VISIBLE_DEVICES="" python experiments/rdo151/profile_reference.py --selftest
```

One-frame CPU parity check:

```bash
CUDA_VISIBLE_DEVICES="" python -u experiments/rdo151/profile_reference.py \
  --device cpu --frames 1 \
  --motion-state experiments/semantic_motion/runs/rank4_gated/motion.bin \
  --output experiments/rdo151/work/profile_cpu_parity1
```

If CPU parity succeeds, run a 20-frame timing sample in tmux:

```bash
CUDA_VISIBLE_DEVICES="" python -u experiments/rdo151/profile_reference.py \
  --device cpu --frames 20 \
  --motion-state experiments/semantic_motion/runs/rank4_gated/motion.bin \
  --output experiments/rdo151/work/profile_cpu20
```

Check `summary.json` for estimated full-sequence time. This estimate excludes
some initialization effects and does not imply the prefix represents the full
cost distribution. If time is acceptable, use `--frames 600` and a new output
directory such as `experiments/rdo151/work/profile_cpu600`.

If CPU parity fails, a separate CUDA check (only when GPU use is intended):

```bash
unset CUDA_VISIBLE_DEVICES
python -u experiments/rdo151/profile_reference.py \
  --device cuda --frames 1 \
  --motion-state experiments/semantic_motion/runs/rank4_gated/motion.bin \
  --output experiments/rdo151/work/profile_cuda_parity1
```

After that passes, change frames to 600 and output to
`experiments/rdo151/work/profile_cuda600`. Do not start multiple profiling
processes simultaneously. Existing output directories are preserved. Prefix
runs are separate runs; full runs start from frame zero because the decoder
and adaptive corrector have causal state.

## What is measured

- Integer-frequency symbol cost `-log2(f_actual / 2^31)`, using the native
  coder's float32 conversion, flooring, winner tie handling and sum balancing.
- Cost aggregated by frame, true class, prediction correctness, token-boundary
  distance, and reference probability-cost bins.
- With a motion state: reference cost at positions where motion fixes or
  harms copying, plus positions where both succeed or both fail.
- Native reader bit positions, real stream bits read, and implicit zero bits
  supplied after the stored stream ends.
- A shadow encoder fed the same probabilities and decoded symbols. Read-only
  native counters report emitted bits and pending underflow bits. Its raw
  finalization and shortened candidate are compared with the published stream
  only for a complete 600-frame run.
- Cache parity for every decoded symbol, not merely a final hash.

The shadow encoder is not a new candidate optimization. It records how the
existing encoding path accounts for its bits. The shortened candidate is not
declared valid unless it is byte-identical to the already verified published
token stream; prefix shortening is not treated as validated.

## Outputs

- `config.json`: input fingerprints and device.
- `frames.csv`: flushed after each completed frame; costs, read positions,
  pending bits, timings and parity status.
- `summary.json`: aggregate measurements and timing estimate after success.
- `expensive_tokens.jsonl`: highest-cost token locations per frame.
- `block_frequency_bits.npz`: costs summed into 16x16 blocks for later plots.
- `shadow_raw.bin`: the shadow encoder's final output.
- `failure.json`: failure details when decoding or verification fails.

Optional `--save-full-maps` writes float32 costs for every token: about 472 MB
for all 600 frames. It is off by default. Profiling outputs are under the
already ignored `work/` directory. Return results to the laptop with SCP.

## Interpretation limits

Frequency NLL is not literal stored bytes per token. Reader counters include
lookahead, and their clipped frame increments are not independent frame code
lengths. Finalization, rounding, implicit zero extension and emitted pending
bits require separate accounting. Do not add all bucket families together:
class, boundary and prediction buckets describe overlapping partitions.

The pre-corrector cost is a proxy computed with a minimum integer frequency;
it is not the actual byte result of removing the adaptive corrector. Likewise,
reference cost at motion-corrected positions is not a prediction of the saving
from a motion-conditioned model. Motion can also affect confidence at pixels
whose top prediction does not change.

## Validation

- Native encoder/decoder round-trip passed on 10,000 probability rows.
- Integer-frequency totals and expected native code lengths passed.
- The CPU parity gate caught a first-frame reference mismatch.
- A decoder control without instrumentation also mismatched the cached CUDA
  output, confirming that CPU output must not silently be accepted here.
- Complete reference profiling and CUDA parity remain to be tested on the
  user's matching environment. No full-sequence bit-cost claim is made yet.
