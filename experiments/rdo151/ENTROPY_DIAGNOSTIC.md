# Output-bias diagnostic

Starts from untouched #151, never from the failed training checkpoint. Changes
only the five existing head biases; weights, exponents, embeddings, temporal
branch, renderer and tokens are unchanged. No new stored parameter channel.

Computes the average gradient on four fixed frames (0,199,399,599), normalizes
its largest absolute component to one, then tests independent offsets of
0.25,0.5,1,2,4 integer-bias units. Scale -1 tests the reverse direction.
Each candidate resets to reference; these are not cumulative training steps.

After serialization, reloads the actual model, verifies identical serialized
bytes and checks logits against the floating shadow model on the first training
frame. Measures soft NLL, deployed integer-frequency rate proxy and compressed
model bytes on training frames and twelve separate fixed probe frames.
Reports how many rounded biases and canonical serialized bytes changed.
Only saves a changed candidate if combined proxy improves on both frame sets.
The frame sets come from the same known video; this is not a generalization test.

The loss and frequency proxy still exclude the native adaptive corrector and
final stream termination. No full encoding, video evaluation or leaderboard
claim is made by this script. Even a promising bias update may lose the original
stream's shortening benefit. Quantization can leave small offsets unchanged.

Use existing environment and prepared reference/cache; install nothing.

```bash
python -u experiments/rdo151/diagnose_entropy.py --device cuda
```

Default output: experiments/rdo151/work/entropy_diagnostic_v1. It refuses to
reuse an existing output directory; use --output with a new directory if needed.
Reads original cache; preserves all previous experiment outputs. It does not
launch training or native coding. GPU mode competes with other GPU processes.
CPU mode is available with CUDA_VISIBLE_DEVICES="" and --device cpu, but will
be slower. Report: diagnostic.json. A saved best.json is only a proxy result.

For transfer back to laptop, use SCP. For source updates to server, use GitHub.
