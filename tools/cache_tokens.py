"""Extract the reference semantic tokens for encoder experiments."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]
SUBMISSION = ROOT / "submission/semantic_folded"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--archive", type=Path,
        default=SUBMISSION / "archive.zip",
    )
    parser.add_argument("--output", type=Path, default=ROOT / "cache/tokens.npy")
    args = parser.parse_args()

    archive = args.archive.resolve()
    output = args.output.resolve()
    metadata = output.with_suffix(".json")

    if not archive.is_file():
        parser.error(f"archive not found: {archive}")
    if output.exists() or metadata.exists():
        parser.error(f"cache already exists: {output}; choose another --output")

    compiler = shutil.which(os.environ.get("CC", "cc"))
    if compiler is None:
        parser.error("a C compiler is required")

    import numpy as np
    import torch

    if not torch.cuda.is_available():
        parser.error("this extraction requires a CUDA GPU")

    sys.path.insert(0, str(SUBMISSION))
    runtime = SUBMISSION / "external/pr141/runtime"

    with tempfile.TemporaryDirectory(prefix="comma-native-") as directory:
        build = Path(directory)
        libraries = (
            (
                runtime / "entropy/rc64_backend.c",
                build / "rc64_backend.so",
                "CPR1_RC64_LIBRARY",
                [],
            ),
            (
                runtime / "f26_corrector_native.c",
                build / "f26_corrector_native.so",
                "F26_CORRECTOR_NATIVE_LIBRARY",
                ["-ffp-contract=off", "-fno-fast-math", "-lm"],
            ),
        )
        for source, destination, variable, flags in libraries:
            subprocess.run(
                [
                    compiler, "-O3", "-std=c11", "-shared", "-fPIC",
                    str(source), "-o", str(destination), *flags,
                ],
                check=True,
            )
            os.environ[variable] = str(destination)

        # Some inherited modules resolve paths relative to the submission.
        os.chdir(SUBMISSION)
        from canonical.inflate_canonical import install_rp1_shim
        install_rp1_shim()
        from decoder import numerics  # applies reference numerical settings
        from container.token_decode import decode_tokens

        print("Decoding reference tokens...", flush=True)
        tokens, info = decode_tokens(
            archive, SUBMISSION / "external/pr141", "cuda"
        )
        array = tokens.cpu().numpy()
        output.parent.mkdir(parents=True, exist_ok=True)
        with output.open("wb") as handle:
            np.save(handle, array, allow_pickle=False)

        info.update(
            archive_sha256=hashlib.sha256(archive.read_bytes()).hexdigest(),
            shape=list(array.shape),
            dtype=str(array.dtype),
        )
        metadata.write_text(json.dumps(info, indent=2) + "\n")
        print(f"Saved {output}")
        print(f"Token SHA-256: {info['decoded_token_sha256']}")


if __name__ == "__main__":
    main()
