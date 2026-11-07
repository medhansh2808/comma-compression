# Full-video ordinary versus cost-weighted bias gradients

Run from existing #151 checkout and environment. No package installation.
Only the five head biases are allowed to change. Original tokens, weights,
exponents, renderer and pose carrier remain fixed. Every candidate starts
from the reference, not a previous update or failed checkpoint.

For all 600 frames, compute two gradient directions:
- Ordinary: equally weighted token negative log probability in bits.
- Cost weighted: multiply each token loss by its detached original coding-loss
  estimate, clipped to 31 bits, then normalize by the global sum of weights.
  Weights are computed at the reference. No gradient flows through weights.

Normalize each direction by its maximum absolute component. Independently
try offsets of 1,2,4 bias units and a reverse-direction -1 control. Serialize,
reload, check first-frame logit parity, and score each unique quantized state
on all 600 frames using UNWEIGHTED integer-frequency probability cost. Duplicate
states are skipped. Fixed-frame sample selection is no longer used.

Report original model size and unchanged model recompressed with exactly the
same compression search as candidates. Report both same-compressor and actual
storage differences. The real original stored model remains the baseline to
beat; recompression overhead is not excused when selecting a candidate.

This proxy is before the native adaptive corrector and stream termination.
No automatic encoding/evaluation follows. A lower proxy is not a byte saving
or leaderboard result. Special stream-shortening benefits may be lost. A full
native encode and exact decode would be a separate next step.

GPU command (explicit; competes with other GPU jobs):

```bash
CUDA_VISIBLE_DEVICES=0 python -u experiments/rdo151/weighted_bias.py --device cuda
```

Defaults: all 600 frames; output work/weighted_bias_v1; refuses existing output
directory. Choose --output with a new path for another run. Prefix --frames
values below 600 are smoke tests and cannot save a changed best candidate.
CPU mode: CUDA_VISIBLE_DEVICES="" and --device cpu (considerably slower).
Prints progress every 50 frames per pass. Timing depends on device sharing;
use the baseline pass timing to judge subsequent passes (up to eight unique
candidate passes plus the gradient pass). No resume is implemented.

Outputs summary.json, best.json, best_hpac.bin, best_prev2.bin. Unchanged
reference is the initial best. No valid new archive is produced by this script.

Prerequisites: train_entropy.py, diagnose_entropy.py, profile_reference.py,
prepared original archive/runtime and verified token cache from prior setup.
