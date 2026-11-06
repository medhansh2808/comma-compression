# Hard-warp audit of the learned motion fields

The v1 full-sequence benchmark was worse than copying the previous token map.
Improved minibatch soft losses do not establish improved integer prediction.

This diagnostic reuses the fitted models, with no additional GPU training.
For each transition, it evaluates motion strengths 0, 0.25, 0.5, 0.75, 1,
1.5 and 2 on the actual hard, integer-warp prediction. Zero means copy the
previous frame. It selects a nonzero strength only if that strictly reduces
the token mismatch count. The model and strength choices are transmitted and
charged inside the token archive. The decoder uses only archive bytes.

The selector therefore guarantees no worse pixel mismatch than identity on
these same fitted maps, but it does not guarantee better compression. It is
an encoder-side per-video selection, not a generalization or learned-gating
result. No result should be described as improved forecasting ability.

The audit answers whether some transitions benefit from the learned field,
and whether integer motion magnitude or harmful transitions explain the v1
failure. If it rarely finds useful motion, stronger optimization or a changed
representation is needed. If prediction improves but bytes do not, the model
and residual coding costs remain the bottleneck. This alone does not isolate
soft-loss mismatch from insufficient fitting.

```bash
cd ~/medhansh/comma-compression-rdo151
bash experiments/semantic_motion/hard_motion_audit.sh
```

This writes new `global_gated`, `rank4_gated` and `benchmark_v2` directories.
It preserves all v1 states, reports and archives. Outputs include per-frame
choices, aggregate hard errors, actual model/residual/archive byte counts,
five GIFs, and independent exact decoding checks. Runtime is CPU-only and
depends on token dimensions and Brotli compression. It can run in tmux.

The updated `motion_codec.py` reads both v1 and new files. New archives contain
a `strengths` state array; decode them with the updated implementation.
