# Reference arithmetic-stream termination trace

Read-only reference replay using profile_reference.py. Builds a temporary
instrumented native encoder, without modifying inherited source files. Existing
decoder, probabilities, reference artifact and earlier outputs stay unchanged.

Records:
- termination_events.csv: symbols with at least one renormalization in frames
  430,431,432; target/frequency/cumulative below, five float probabilities,
  before/after interval, emitted/pending counters and E1/E2/E3 branch counts.
- termination_groups.csv: coder state after every batch from frame 430 onward.
- termination_summary.json: last emission in traced window and reference parity.
- Standard profiler outputs, including final exact shortened-stream comparison.

Frame numbers are zero-based. group is native batch order within the frame;
symbol_index is local to that batch, not a raster pixel index. Rows exclude
symbols with no renormalization. E1 emits zero; E2 emits one; both flush pending
opposite bits. E3 defers output while rescaling an interval in the middle half.
These are descriptive events, not a proof that a modification preserves ending.

Requires existing verified token cache, pinned reference/runtime, C compiler,
numpy and existing torch environment. No installation or training.

CPU selftest:
CUDA_VISIBLE_DEVICES="" python experiments/rdo151/trace_termination.py --selftest

Full CPU trace:
CUDA_VISIBLE_DEVICES="" python -u experiments/rdo151/trace_termination.py --device cpu --frames 600 --output experiments/rdo151/work/termination_cpu600

This replays all frames because arithmetic state is sequential; an existing
frames.csv cannot reconstruct missing symbol-level states. Budget roughly the
previous full profile's 27 minutes plus tracing overhead on that machine.
No GPU allocation in CPU mode. Standard output prints progress each frame.
Output directory must be new. No automatic candidate encoding follows.

Validation: native random-probability round trip and coding-cost selftest;
active event tracing produced byte-identical output to uninstrumented encoder
on 3000 random symbols. Full server reference trace is still to run. Final
results are trustworthy only if all tokens match and shortened stream matches
published bytes.
