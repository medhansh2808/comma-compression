# Lossless boundary representation benchmark

CPU-only NumPy/Brotli prototype, using existing verified #151 token cache.
No PyTorch, GPU, fitting, or changes to existing decoder/artifacts.

Compares four independently decodable formats:
1. Whole raw map compressed directly (control).
2. Horizontal transition bitmask and labels at each row start/transition.
3. Same mask XORed against previous frame's mask, plus absolute transition labels.
4. Row run lengths (uint16) and run labels.

Every row starts with a transmitted label. For mask formats, decoded transition
locations determine run lengths. Repeating the transmitted labels by these
lengths reconstructs all pixels. For temporal mask format, cumulative XOR
recovers each mask from previous reconstructed mask. First mask is absolute.
This encodes horizontal runs, not general polygon tracing/chain codes.
All location, label, initial-frame and container bytes count. No target-derived
boundary information is available for free to the decoder.

Each file is written and independently decoded from disk, then compared with
the full token cache. Also checks uniform, random, alternating-label toy inputs.
Default Brotli quality 6. Outputs immutable runs/v1; refuses existing path.
Requires only the existing numpy and brotli packages. Source cache is read-only.

```bash
CUDA_VISIBLE_DEVICES="" python -u experiments/boundary_codec/benchmark.py
cat experiments/boundary_codec/runs/v1/summary.json
```

These are standalone token representation sizes, not full official archives.
The reference token stream is 108,891 bytes; with main entropy state plus temporal
extension it is 120,779 bytes. Complete integration/accounting would be needed
for a leaderboard claim; do not assume all HPAC state can simply be removed
without examining shared dependencies. This benchmark does not use the unusual
reference arithmetic-stream shortening. A loss shows these simple encodings
are insufficient, not that all contour coding is ineffective.
