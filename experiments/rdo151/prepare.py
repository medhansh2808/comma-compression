"""Prepare three isolated, pinned #151 entropy-coding experiments (CPU)."""
from pathlib import Path
import hashlib
import shutil
import subprocess
import urllib.request

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
PIN = "56ba57f073cf8cc97713e4c0c517830142f81063"
SHA = "9e2de76d931c5a7aca29f7e9008c4d77b34ec037d042113e2fd082607391f315"

def replace_once(s, old, new):
    if s.count(old) != 1:
        raise ValueError(f"expected exactly one occurrence: {old!r}")
    return s.replace(old, new)

def main():
    vendor = HERE / "vendor"
    if not vendor.exists():
        subprocess.run(["git", "clone", "--no-checkout", "https://github.com/darskkaa/comma_video_compression_challenge.git", str(vendor)], check=True)
        subprocess.run(["git", "-C", str(vendor), "checkout", "--detach", PIN], check=True)
    actual = subprocess.check_output(["git", "-C", str(vendor), "rev-parse", "HEAD"], text=True).strip()
    if actual != PIN or subprocess.check_output(["git", "-C", str(vendor), "status", "--porcelain"], text=True).strip():
        raise ValueError("vendor must be clean and at the pinned commit")
    cache = HERE / "cache"
    cache.mkdir(exist_ok=True)
    archive = cache / "leader151.zip"
    if not archive.exists():
        temporary = archive.with_suffix(".part")
        urllib.request.urlretrieve("https://github.com/darskkaa/comma_video_compression_challenge/releases/download/v1.0/archive.zip", temporary)
        if hashlib.sha256(temporary.read_bytes()).hexdigest() != SHA:
            raise ValueError("download checksum mismatch")
        temporary.rename(archive)
    if hashlib.sha256(archive.read_bytes()).hexdigest() != SHA:
        raise ValueError("reference archive checksum mismatch")
    for case in ("reference", "double", "zero"):
        dest = HERE / "work" / case
        if dest.exists():
            print(f"Keeping existing {dest}; use a clean experiment checkout to regenerate.")
            continue
        shutil.copytree(vendor / "submissions/darskkaa_rdo", dest)
        shutil.copyfile(archive, dest / "archive.zip")
        decoder = dest / "container/token_decode.py"
        source = decoder.read_text()
        old = "model.prepare_frame_context(index, warped_previous, apply_warp(previous2, warp_id))"
        replacement = {
            "reference": old,
            "double": "model.prepare_frame_context(index, warped_previous, apply_warp(apply_warp(previous2, warp_id), warp_id))",
            "zero": "model.prepare_frame_context(index, warped_previous, previous2)",
        }[case]
        source = replace_once(source, old, replacement)
        decoder.write_text(source)
        # Derive the encoder from the pinned decoder so model setup and symbol order agree.
        encoder = replace_once(source, 'def decode_tokens(archive_path: Path, pr140_runtime_dir: Path, device_name: str = "cuda"):', 'def encode_tokens(archive_path: Path, pr140_runtime_dir: Path, source_tokens, device_name: str = "cuda"):')
        encoder = replace_once(encoder, "from runtime.entropy.rc64 import NativeDecoder", "from runtime.entropy.rc64_encoder import NativeEncoder")
        encoder = replace_once(encoder, 'decoder = NativeDecoder(Path(os.environ["CPR1_RC64_LIBRARY"]), parts.token_stream)', 'encoder = NativeEncoder(Path(os.environ["CPR1_RC64_LIBRARY"]))\n    if source_tokens.shape != (renderer.N, renderer.EVAL_H, renderer.EVAL_W) or source_tokens.dtype != np.uint8 or np.any(source_tokens >= NUM_CLASSES):\n        raise ValueError("invalid source tokens")')
        encoder = replace_once(encoder, "symbols = decoder.decode(row64).astype(np.int64)", "symbols = source_tokens[frame].reshape(-1)[flat_positions].astype(np.int64)\n                encoder.encode(row64, symbols)")
        encoder = replace_once(encoder, "return decoded_tokens, info", "payload = encoder.finish()\n    encoder.close()\n    return payload, info")
        (dest / "container/token_encode.py").write_text(encoder)
        for name in ("rc64_encoder.py", "rc64_encode_backend.c"):
            shutil.copyfile(ROOT / "submission/semantic_folded/external/pr141/runtime/entropy" / name, dest / "external/pr141/runtime/entropy" / name)
        # CPU-only syntax and native compilation checks.
        compile(source, str(decoder), "exec")
        compile(encoder, str(dest / "container/token_encode.py"), "exec")
        runtime = dest / "external/pr141/runtime"
        for cfile in (runtime / "entropy/rc64_encode_backend.c", runtime / "f26_corrector_native.c"):
            subprocess.run(["cc", "-O3", "-std=c11", "-shared", "-fPIC", "-ffp-contract=off", "-fno-fast-math", str(cfile), "-o", str(cfile.with_suffix(".so")), "-lm"], check=True)
        print(f"Prepared {case}")
    print("CPU preparation complete. Run run.py on CUDA next.")

if __name__ == "__main__":
    main()
