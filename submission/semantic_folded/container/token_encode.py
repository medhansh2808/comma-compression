"""Reference token encoder derived from token_decode.py; same prediction order.
Retains the reference warp choices, model weights and corrector updates.
"""
from __future__ import annotations

import hashlib
import os
import sys
import tempfile
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))


from container.context_warp import apply_warp  # noqa: E402
from container.payload_format import split_inner_payload, read_archive, write_archive  # noqa: E402
from container.warp_signal import decode_warp_signal  # noqa: E402
from container.archive_writer import EXTERNAL_ATTR_PR141  # noqa: E402


POST_PURGE_HOOKS = []

def encode_tokens(archive_path: Path, pr140_runtime_dir: Path, source_tokens: np.ndarray, device_name: str = "cuda"):
    """Decode a candidate archive end-to-end, using only archive data.

    `pr140_runtime_dir` points at a directory containing the inherited,
    UNMODIFIED `runtime/` package (e.g. external/pr140) -- used only for its
    container-parsing (`read_residual_archive`) and model-loading code, which
    this function does not alter.

    Returns (decoded_tokens: torch.Tensor[N,H,W] uint8, info: dict).
    """
    import torch

    sys.path.insert(0, str(pr140_runtime_dir))
    # Force a fresh import bound to THIS runtime dir (PR140 vs PR141 trees
    # both define a top-level `runtime` package; avoid stale sys.modules
    # from a previous call in the same process).
    for name in list(sys.modules):
        if name == "runtime" or name.startswith("runtime."):
            del sys.modules[name]
    for _hook in POST_PURGE_HOOKS:  # optional decoder shims (empty by default => behaviour unchanged)
        _hook()

    from runtime.residual_archive import (  # noqa: E402
        read_residual_archive,
        _boundary_buckets,
        _sparse_class,
        _probability_table,
        NUM_CLASSES,
    )
    from runtime.ihs2 import materialize_ihs1  # noqa: E402
    from runtime.residual_archive import _rr8_select_corrector  # noqa: E402
    from runtime.f26_inflate import _load_renderer, _configure_device  # noqa: E402
    from runtime.hpac_inference import optimize_sparse_evaluator  # noqa: E402
    from runtime.entropy.rc64_encoder import NativeEncoder  # noqa: E402

    p_bytes = read_archive(archive_path)
    core, warp_signal = split_inner_payload(p_bytes)

    renderer_dir = pr140_runtime_dir / "cpr1"
    renderer = _load_renderer(renderer_dir)
    device, device_report = _configure_device(torch, device_name, 4)

    warp_ids = decode_warp_signal(warp_signal, count=renderer.N)
    assert len(warp_ids) == renderer.N, f"decoded {len(warp_ids)} warp ids, expected {renderer.N}"

    with tempfile.TemporaryDirectory() as tmp:
        tmp_archive = Path(tmp) / "core_only.zip"
        write_archive(tmp_archive, core, EXTERNAL_ATTR_PR141)
        parts = read_residual_archive(tmp_archive)  # inherited, UNMODIFIED parsing logic

    base_hpac = materialize_ihs1(parts.hpac_blob, renderer)
    model = renderer.load_hpac(base_hpac, device)
    masks = renderer.group_masks(device)
    sparse = _sparse_class(renderer_dir)(model, renderer.EVAL_H, renderer.EVAL_W)
    optimize_sparse_evaluator(sparse)

    group_plans = []
    for mask in masks:
        mask_array = mask.detach().cpu().numpy()
        flat_positions = np.flatnonzero(mask_array.reshape(-1))
        group_plans.append((torch.from_numpy(flat_positions).to(device), flat_positions))

    corrector = _rr8_select_corrector(renderer.EVAL_H * renderer.EVAL_W)
    stream_encoder = NativeEncoder(Path(os.environ["CPR1_RC64_LIBRARY"]))
    if source_tokens.shape != (renderer.N, renderer.EVAL_H, renderer.EVAL_W):
        raise ValueError("cached token shape does not match renderer")
    if source_tokens.dtype != np.uint8 or np.any(source_tokens >= NUM_CLASSES):
        raise ValueError("expected uint8 token IDs in [0, NUM_CLASSES)")
    previous = torch.zeros((1, renderer.EVAL_H, renderer.EVAL_W), dtype=torch.long, device=device)
    decoded_tokens = torch.empty((renderer.N, renderer.EVAL_H, renderer.EVAL_W), dtype=torch.uint8)

    t_start = time.time()
    with torch.inference_mode():
        for frame in range(renderer.N):
            index = torch.tensor([frame], dtype=torch.long, device=device)
            current = torch.zeros_like(previous)
            warp_id = warp_ids[frame]

            if frame:
                previous_cpu = previous[0].to(device="cpu", dtype=torch.uint8).numpy()
                boundary = _boundary_buckets(previous_cpu).reshape(-1)
            else:
                boundary = np.full(renderer.EVAL_H * renderer.EVAL_W, 4, dtype=np.uint8)

            # THE ONLY DECODER-LOOP CHANGE from the inherited decode_production_tokens:
            # warp the tensor fed into prepare_frame_context. `boundary` above and
            # `corrector.end_frame` below continue to use the literal, unwarped
            # `previous` -- unchanged from the inherited loop.
            warped_previous = apply_warp(previous, warp_id)
            context = model.prepare_frame_context(index, warped_previous)

            corrector.begin_frame(boundary)
            for group, (device_positions, flat_positions) in enumerate(group_plans):
                selected = sparse.selected_logits(current, context, group)
                base_logits = selected.cpu().numpy()
                predicted = base_logits.argmax(axis=1).astype(np.int64)
                feature = boundary[flat_positions].astype(np.int64) * NUM_CLASSES + predicted
                corrected = base_logits + parts.table.values[feature]
                probability = _probability_table(corrected, renderer.HPAC_LOGIT_PRECISION)
                state = corrector.group_state(probability, predicted, flat_positions)
                row64 = corrector.coding_row(state)
                symbols = np.asarray(source_tokens[frame]).reshape(-1)[flat_positions].astype(np.int64)
                stream_encoder.encode(row64, symbols)
                corrector.observe(state, symbols)
                current.reshape(-1)[device_positions] = torch.from_numpy(symbols).to(device)
            decoded_tokens[frame] = current[0].to(device="cpu", dtype=torch.uint8)
            corrector.end_frame(decoded_tokens[frame].numpy().reshape(-1))
            previous = current

    elapsed = time.time() - t_start
    del model
    info = {
        "decoded_token_sha256": hashlib.sha256(decoded_tokens.numpy().tobytes()).hexdigest(),
        "decode_runtime_seconds": elapsed,
        "warp_ids_decoded": len(warp_ids),
        "warp_choice_counts": {w: warp_ids.count(w) for w in sorted(set(warp_ids))},
        "parts_semantic_blob_sha256": hashlib.sha256(parts.semantic_blob).hexdigest(),
        "parts_carrier_blob_sha256": hashlib.sha256(parts.carrier_blob).hexdigest(),
        "parts_hpac_blob_sha256": hashlib.sha256(parts.hpac_blob).hexdigest(),
        "parts_residual_payload_sha256": hashlib.sha256(parts.residual_payload).hexdigest(),
        "parts_token_stream_bytes": len(parts.token_stream),
        "parts_token_stream_sha256": hashlib.sha256(parts.token_stream).hexdigest(),
    }
    payload = stream_encoder.finish()
    stream_encoder.close()
    info["encoded_stream_bytes"] = len(payload)
    info["encoded_stream_sha256"] = hashlib.sha256(payload).hexdigest()
    return payload, info
