"""Re-encode cached reference tokens, preserve other sections, optionally verify."""
from __future__ import annotations
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SUB = ROOT / "submission/semantic_folded"

def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--archive", type=Path, default=SUB / "archive.zip")
    ap.add_argument("--tokens", type=Path, default=ROOT / "cache/tokens.npy")
    ap.add_argument("--output", type=Path, default=ROOT / "runs/reference_reencoded/archive.zip")
    ap.add_argument("--verify", action="store_true")
    a = ap.parse_args()
    archive, tokens_path, output = a.archive.resolve(), a.tokens.resolve(), a.output.resolve()
    if output == archive or output.exists():
        ap.error("choose a new output path; existing archives will not be overwritten")
    tokens = np.load(tokens_path, mmap_mode="r", allow_pickle=False)
    metadata = json.loads(tokens_path.with_suffix(".json").read_text())
    archive_sha = hashlib.sha256(archive.read_bytes()).hexdigest()
    source_sha = hashlib.sha256(tokens.tobytes()).hexdigest()
    if metadata["archive_sha256"] != archive_sha:
        ap.error("cached tokens belong to another archive")
    if metadata["decoded_token_sha256"] != source_sha:
        ap.error("cached tokens do not match their recorded checksum")
    sys.path.insert(0, str(SUB))
    sys.path.insert(0, str(SUB / "external/pr141"))
    os.chdir(SUB)
    from canonical.inflate_canonical import install_rp1_shim
    install_rp1_shim()
    from decoder import numerics
    from decoder.inflate_core import _load
    from container.payload_format import read_archive, split_inner_payload, write_archive
    from container.archive_writer import EXTERNAL_ATTR_PR141
    from container.token_encode import encode_tokens
    from runtime.entropy.rc64_encoder import build_encoder_backend

    with tempfile.TemporaryDirectory(prefix="comma-encode-") as temporary:
        temporary = Path(temporary)
        os.environ["CPR1_RC64_LIBRARY"] = str(build_encoder_backend(temporary / "rc64.so"))
        native = temporary / "corrector.so"
        subprocess.run([
            os.environ.get("CC", "cc"), "-O3", "-std=c11", "-shared", "-fPIC",
            "-ffp-contract=off", "-fno-fast-math",
            str(SUB / "external/pr141/runtime/f26_corrector_native.c"),
            "-o", str(native), "-lm",
        ], check=True)
        os.environ["F26_CORRECTOR_NATIVE_LIBRARY"] = str(native)

        parts, _, _ = _load(archive)
        core, signal = split_inner_payload(read_archive(archive))
        if not parts.token_stream or not core.endswith(parts.token_stream):
            raise ValueError("token stream is not the expected final core section")
        prefix = core[:-len(parts.token_stream)]
        print("Encoding reference tokens with unchanged probabilities...", flush=True)
        stream, info = encode_tokens(archive, SUB / "external/pr141", tokens, "cuda")
        if info["decoded_token_sha256"] != source_sha:
            raise ValueError("encoder did not consume the expected token maps")
        output.parent.mkdir(parents=True, exist_ok=True)
        # Preserve the original signal, including any renderer/pose extensions.
        import struct
        write_archive(output, prefix + stream + signal + struct.pack("<H", len(signal)),
                      EXTERNAL_ATTR_PR141)
        info.update(
            original_archive_bytes=archive.stat().st_size,
            candidate_archive_bytes=output.stat().st_size,
            source_archive_sha256=archive_sha,
            token_roundtrip_verified=False,
        )
        if a.verify:
            from container.token_decode import decode_tokens
            print("Checking candidate with the existing token decoder...", flush=True)
            decoded, decoded_info = decode_tokens(output, SUB / "external/pr141", "cuda")
            if decoded_info["decoded_token_sha256"] != source_sha:
                raise ValueError("candidate token round-trip FAILED")
            info["token_roundtrip_verified"] = True
        output.with_suffix(".json").write_text(json.dumps(info, indent=2) + "\n")
        print(json.dumps(info, indent=2))

if __name__ == "__main__":
    main()
