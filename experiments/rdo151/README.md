# Temporal context experiment on submission #151

This is an unmeasured, bounded experiment on the public #151 decoder, not a leaderboard result. The inherited architecture, weights and tokens belong to the credited public submission lineage. The first GPU control must reproduce the reference token stream byte for byte. No encoder-side training code for #151 was published in that PR.

The pinned decoder applies the same warp to t-1 and t-2. Compare its reference rule against applying that warp twice to t-2, and against leaving t-2 unwarped. These rules require no fitted side information. Every candidate must losslessly reconstruct the same token maps. The renderer, pose carrier, corrector and trailer stay unchanged. The trained context model may prefer the original rule; improvement is not assumed.

CPU: `python experiments/rdo151/prepare.py`, then `python experiments/rdo151/check_cpu.py`.

GPU, in order: `python experiments/rdo151/run.py cache`, then `reference`, `double`, `zero`. Run each case in a separate process. Keep the smallest verified archive only if it improves on the reference. Inspect `work/*/result.json` for actual bytes and the rate contribution to score. A verified 1,000-byte saving reduces score by about 0.000666 with unchanged distortion. Still run the official evaluator and check the 30-minute T4 runtime before submission.

Preparation clones the exact public commit, checks the release SHA-256, creates isolated variants, derives an encoder from the pinned decoder symbol loop, and copies the existing RC64 encoder companion from this repository. Existing work directories are preserved. Models, archives, caches and generated copies are excluded from git; rerun preparation on each machine. About 4 GB per inflated raw video is needed if you run full evaluations.

Sources and attribution:

- https://github.com/commaai/comma_video_compression_challenge/pull/151
- https://github.com/darskkaa/comma_video_compression_challenge/tree/56ba57f073cf8cc97713e4c0c517830142f81063/submissions/darskkaa_rdo
- https://github.com/tensorsofthewall/comma_semantic_folded
- Existing `LINEAGE.md` and source licenses, including the public #130–#150 lineage.

The challenge allows AI assistance on parts of code, but asks entrants to write and read most submitted code themselves. Read the generated encoder/decoder differences and understand the probability updates before using this experiment in a submission. This directory does not draft a public PR or claim ownership of inherited code.
