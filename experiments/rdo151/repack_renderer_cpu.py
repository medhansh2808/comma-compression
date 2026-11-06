"""Lossless CPU-only packing search on #151. Never changes the video token stream.

Use from the comma-compression-rdo151 checkout. Requires numpy and brotli.
Reuses the shipped pure integer renderer coder, without importing Torch.
Searches existing per-array adaptation increments (1..255) by ideal code
length; accepts the combined candidate only after actual coding and exact
RID2-byte recovery. This is a search heuristic, not exhaustive archive RDO.
Also sweeps Brotli settings on the unchanged HPAC bytes and RID9 metadata.
"""
import argparse
import ast
import copy
import hashlib
import io
import json
import math
from pathlib import Path
import shutil
import struct
import time
import zipfile

import brotli
import numpy as np

SHA151 = "9e2de76d931c5a7aca29f7e9008c4d77b34ec037d042113e2fd082607391f315"
HEADER = struct.Struct("<4sBBBBHHH")

def sha(b):
    return hashlib.sha256(b).hexdigest()

def load_coder(folder):
    # Extract only the exact pure integer coder definitions from the credited
    # source. Avoid module imports that initialize Torch or load GPU state.
    ns = dict(np=np, struct=struct, brotli=brotli)
    selected = []
    for n in ast.parse((folder / "folding/rid2.py").read_text()).body:
        if isinstance(n, ast.ClassDef) and n.name in {"RID2Error", "Enc", "Dec", "Model"}:
            selected.append(n)
        elif isinstance(n, ast.Assign) and any(isinstance(t, ast.Tuple) and [getattr(v, "id", "") for v in t.elts] == ["TOP", "HALF", "QUART"] for t in n.targets):
            selected.append(n)
    if len(selected) != 5:
        raise ValueError("unexpected integer-coder source")
    exec(compile(ast.Module(body=selected, type_ignores=[]), "rid2_cpu", "exec"), ns)
    selected = [n for n in ast.parse((folder / "folding/rid9.py").read_text()).body if isinstance(n, (ast.Assign, ast.FunctionDef))]
    exec(compile(ast.Module(body=selected, type_ignores=[]), "rid9_cpu", "exec"), ns)
    return ns

def best_brotli(original):
    raw = brotli.decompress(original)
    best = original
    for mode in (brotli.MODE_GENERIC, brotli.MODE_TEXT, brotli.MODE_FONT):
        for window in (16, 18, 20, 22, 24):
            trial = brotli.compress(raw, quality=11, lgwin=window, mode=mode)
            if len(trial) < len(best):
                if brotli.decompress(trial) != raw:
                    raise ValueError("Brotli recovery failed")
                best = trial
    return best

def cost(symbols, alphabet, inc):
    counts = np.ones(alphabet, dtype=np.int64)
    total = alphabet
    bits = 0.0
    for symbol in symbols:
        bits += math.log2(total / int(counts[symbol]))
        counts[symbol] += inc
        total += inc
        if total > 32768:
            counts = (counts + 1) // 2
            total = int(counts.sum())
    return bits

def search_renderer(body, ns):
    if body[:6] != b"SM3R\x01\x09":
        raise ValueError("expected RID9 renderer")
    original8 = ns["decode_to_blob8"](body)
    specs = []
    offset = 7
    for _ in range(body[6]):
        gap, offset = ns["_unvarint"](body, offset)
        count, offset = ns["_unvarint"](body, offset)
        bits, inc = body[offset:offset+2]
        offset += 2
        specs.append((gap, count, bits, inc))
    tail, offset = ns["_unvarint"](body, offset)
    size = struct.unpack_from("<I", body, offset)[0]
    offset += 4
    gaps = body[offset:offset+size]
    decoder = ns["Dec"](body[offset+size:])
    arrays = [ns["_code_array"](decoder, count, bits, inc, decode=True) for gap, count, bits, inc in specs]
    best_incs = []
    for i, ((gap, count, bits, inc), values) in enumerate(zip(specs, arrays)):
        symbols = (values + (1 << (bits - 1))).tolist()
        current_cost = cost(symbols, 1 << bits, inc)
        best_cost, best_inc = current_cost, inc
        for candidate_inc in range(1, 256):
            c = cost(symbols, 1 << bits, candidate_inc)
            if c < best_cost:
                best_cost, best_inc = c, candidate_inc
        best_incs.append(best_inc)
        print(f"array {i+1}/{len(arrays)}: inc {inc}->{best_inc}; estimated saving {current_cost-best_cost:.2f} bits", flush=True)
    new_gaps = best_brotli(gaps)

    def encode(increments, gap_stream):
        header = bytearray(body[:7])
        enc = ns["Enc"]()
        for (gap, count, bits, old_inc), values, inc in zip(specs, arrays, increments):
            header += ns["_varint"](gap) + ns["_varint"](count) + bytes([bits, inc])
            ns["_code_array"](enc, values, bits, inc)
        header += ns["_varint"](tail)
        result = bytes(header) + struct.pack("<I", len(gap_stream)) + gap_stream + enc.finish()
        if ns["decode_to_blob8"](result) != original8:
            raise ValueError("renderer bytes changed")
        return result

    old_incs = [s[3] for s in specs]
    best = body
    # Retain original arithmetic stream if its termination is more compact.
    gap_only = body[:offset-4] + struct.pack("<I", len(new_gaps)) + new_gaps + body[offset+size:]
    if ns["decode_to_blob8"](gap_only) != original8:
        raise ValueError("metadata-only recovery failed")
    candidates = [gap_only, encode(old_incs, new_gaps), encode(best_incs, new_gaps)]
    for trial in candidates:
        print(f"RID9 candidate: {len(trial)} bytes (original {len(body)})", flush=True)
        if len(trial) < len(best):
            best = trial
    return best, sha(original8)

def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root", type=Path, default=Path.cwd())
    args = ap.parse_args()
    start = time.monotonic()
    root = args.root.resolve() / "experiments/rdo151"
    folder = root / "work/reference"
    original = folder / "archive.zip"
    original_bytes = original.read_bytes()
    if sha(original_bytes) != SHA151:
        raise ValueError("expected exact verified #151 reference archive")
    with zipfile.ZipFile(io.BytesIO(original_bytes)) as z:
        if z.namelist() != ["p"]:
            raise ValueError("unexpected ZIP layout")
        zipinfo = copy.copy(z.getinfo("p"))
        payload = z.read("p")
    if payload[:4] != b"BLK2":
        raise ValueError("expected BLK2")
    expected, count = struct.unpack_from("<IB", payload, 4)
    size = int.from_bytes(payload[10:13], "little")
    if count != 1 or payload[9] != 0 or size != expected:
        raise ValueError("expected one raw model block")
    models, preserved_tail = payload[13:13+size], payload[13+size:]
    fields = list(HEADER.unpack_from(models))
    if fields[:3] != [b"RX1M", 1, 2] or fields[4] != 24 or HEADER.size + sum(fields[-3:]) != len(models):
        raise ValueError("unexpected model header")
    h, s, c = fields[-3:]
    pos = HEADER.size
    hpac = models[pos:pos+h]
    semantic = models[pos+h:pos+h+s]
    carrier = models[pos+h+s:]
    if semantic[:6] != b"LAY1\x01\x00" or int.from_bytes(semantic[6:8], "little") != len(semantic)-8:
        raise ValueError("unexpected semantic segment layout")
    ns = load_coder(folder)
    new_body, semantic_sha = search_renderer(semantic[8:], ns)
    new_semantic = semantic[:6] + struct.pack("<H", len(new_body)) + new_body
    print("Searching HPAC Brotli settings...", flush=True)
    new_hpac = best_brotli(hpac)
    if brotli.decompress(new_hpac) != brotli.decompress(hpac):
        raise ValueError("HPAC bytes changed")
    fields[-3:] = [len(new_hpac), len(new_semantic), len(carrier)]
    new_models = HEADER.pack(*fields) + new_hpac + new_semantic + carrier
    new_payload = b"BLK2" + struct.pack("<IB", len(new_models), 1) + b"\0" + len(new_models).to_bytes(3, "little") + new_models + preserved_tail
    if new_payload[13+len(new_models):] != preserved_tail:
        raise ValueError("token/residual/extension bytes changed")
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as z:
        z.writestr(zipinfo, new_payload)
    candidate = buffer.getvalue()
    result = dict(original_bytes=len(original_bytes), candidate_bytes=len(candidate), saved_bytes=len(original_bytes)-len(candidate),
        hpac_saved=h-len(new_hpac), renderer_saved=s-len(new_semantic), decoded_renderer_sha256=semantic_sha,
        preserved_tail_sha256=sha(preserved_tail), archive_sha256=sha(candidate), elapsed_seconds=time.monotonic()-start,
        cpu_structural_checks=True, official_evaluation_complete=False)
    print(json.dumps(result, indent=2), flush=True)
    if len(candidate) >= len(original_bytes):
        print("No improvement; reference preserved.")
        return
    destination = root / "work/renderer_repack"
    if destination.exists():
        raise ValueError("destination exists; preserving previous work")
    shutil.copytree(folder, destination)
    (destination / "result.json").unlink(missing_ok=True)
    (destination / "archive.zip").write_bytes(candidate)
    (destination / "repack_result.json").write_text(json.dumps(result, indent=2) + "\n")
    print(f"Saved candidate: {destination / 'archive.zip'}")

if __name__ == "__main__":
    main()
