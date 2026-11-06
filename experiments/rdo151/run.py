"""Bounded temporal-context experiment; reference parity must pass first."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import struct
import sys
import time
import numpy as np

HERE = Path(__file__).resolve().parent

def sha(data):
    return hashlib.sha256(data).hexdigest()

def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("case", choices=("cache", "reference", "double", "zero"))
    args = ap.parse_args()
    case = "reference" if args.case == "cache" else args.case
    dest = HERE / "work" / case
    reference = HERE / "cache/leader151.zip"
    sys.path[:0] = [str(dest), str(dest / "external/pr141")]
    os.chdir(dest)
    runtime = dest / "external/pr141/runtime"
    os.environ["CPR1_RC64_LIBRARY"] = str(runtime / "entropy/rc64_encode_backend.so")
    os.environ["F26_CORRECTOR_NATIVE_LIBRARY"] = str(runtime / "f26_corrector_native.so")
    import torch
    if not torch.cuda.is_available():
        ap.error("CUDA is required; preparation alone runs on CPU")
    from canonical.inflate_canonical import install_rp1_shim
    install_rp1_shim()
    from decoder import numerics
    from container import token_decode, token_encode
    # Rowpack installs its runtime-purge hooks on the decoder; share them with the encoder.
    token_encode.POST_PURGE_HOOKS = token_decode.POST_PURGE_HOOKS
    token_path = HERE / "cache/tokens151.npy"
    if args.case == "cache":
        if token_path.exists():
            ap.error("cache exists; keep it or use a new experiment checkout")
        tokens, info = token_decode.decode_tokens(reference, dest / "external/pr141", "cuda")
        np.save(token_path, tokens.numpy(), allow_pickle=False)
        info["archive_sha256"] = sha(reference.read_bytes())
        token_path.with_suffix(".json").write_text(json.dumps(info, indent=2) + "\n")
        print(json.dumps(info, indent=2), flush=True)
        return
    metadata = json.loads(token_path.with_suffix(".json").read_text())
    tokens = np.load(token_path, allow_pickle=False)
    if metadata["archive_sha256"] != sha(reference.read_bytes()) or metadata["decoded_token_sha256"] != sha(tokens.tobytes()):
        raise ValueError("token provenance/hash mismatch")
    gate = HERE / "work/reference/result.json"
    if case != "reference":
        control = json.loads(gate.read_text())
        if not control["reference_stream_identical"] or not control["roundtrip_verified"]:
            raise ValueError("reference parity must pass before experiments")
    result_file = dest / "result.json"
    if result_file.exists():
        ap.error("result exists; keep completed experiments immutable")
    start = time.monotonic()
    from decoder.inflate_core import _load
    from container.payload_format import read_archive, split_inner_payload, write_archive
    from container.archive_writer import EXTERNAL_ATTR_PR141
    parts, _, _ = _load(reference)
    core, signal = split_inner_payload(read_archive(reference))
    if not parts.token_stream or not core.endswith(parts.token_stream):
        raise ValueError("unexpected token stream layout")
    print(f"Encoding {case}: full 600-frame sequence", flush=True)
    stream, info = token_encode.encode_tokens(reference, dest / "external/pr141", tokens, "cuda")
    identical = stream == parts.token_stream
    if case == "reference" and not identical:
        raise ValueError("reference encoder did not reproduce the published token stream; stop")
    candidate = dest / "archive.zip"
    write_archive(candidate, core[:-len(parts.token_stream)] + stream + signal + struct.pack("<H", len(signal)), EXTERNAL_ATTR_PR141)
    print("Verifying candidate with its decoder", flush=True)
    decoded, _ = token_decode.decode_tokens(candidate, dest / "external/pr141", "cuda")
    if not np.array_equal(decoded.numpy(), tokens):
        raise ValueError("token round-trip failed")
    saved = reference.stat().st_size - candidate.stat().st_size
    result = dict(case=case, archive_bytes=candidate.stat().st_size, saved_bytes=saved,
                  score_change_from_rate=-25 * saved / 37545489,
                  reference_stream_identical=identical, roundtrip_verified=True,
                  token_sha256=sha(tokens.tobytes()), archive_sha256=sha(candidate.read_bytes()),
                  elapsed_seconds=time.monotonic()-start)
    result_file.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2), flush=True)

if __name__ == "__main__":
    main()
