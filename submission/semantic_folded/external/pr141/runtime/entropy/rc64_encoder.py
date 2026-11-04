"""Native streaming encoder for the inherited five-class RC64 decoder."""
from __future__ import annotations
import ctypes
import os
from pathlib import Path
import subprocess
import numpy as np
from .rc64 import Rc64Error, _load_library

def build_encoder_backend(destination: Path) -> Path:
    destination = Path(destination).resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    source = Path(__file__).with_name("rc64_encode_backend.c")
    subprocess.run([
        os.environ.get("CC", "cc"), "-O3", "-std=c11", "-shared", "-fPIC",
        "-ffp-contract=off", "-fno-fast-math", str(source),
        "-o", str(destination), "-lm",
    ], check=True)
    return destination

class NativeEncoder:
    def __init__(self, library_path: Path):
        self.context = None
        self.finished = False
        self.library = lib = _load_library(Path(library_path))
        lib.rc64_encoder_create.argtypes = []
        lib.rc64_encoder_create.restype = ctypes.c_void_p
        lib.rc64_encoder_destroy.argtypes = [ctypes.c_void_p]
        lib.rc64_encoder_destroy.restype = None
        lib.rc64_encoder_encode_probabilities.argtypes = [
            ctypes.c_void_p, ctypes.POINTER(ctypes.c_float),
            ctypes.c_size_t, ctypes.POINTER(ctypes.c_int32),
        ]
        lib.rc64_encoder_encode_probabilities.restype = ctypes.c_int
        lib.rc64_encoder_finish.argtypes = [ctypes.c_void_p]
        lib.rc64_encoder_finish.restype = ctypes.c_int
        lib.rc64_encoder_data.argtypes = [ctypes.c_void_p]
        lib.rc64_encoder_data.restype = ctypes.c_void_p
        lib.rc64_encoder_size.argtypes = [ctypes.c_void_p]
        lib.rc64_encoder_size.restype = ctypes.c_size_t
        self.context = lib.rc64_encoder_create()
        if not self.context:
            raise Rc64Error("encoder allocation failed")

    def encode(self, probabilities, symbols):
        if not self.context or self.finished:
            raise Rc64Error("encoder closed or finished")
        p = np.ascontiguousarray(probabilities, dtype=np.float32)
        raw = np.asarray(symbols)
        if p.ndim != 2 or p.shape[1] != 5 or not p.size:
            raise Rc64Error("probabilities must have shape [N, 5]")
        if raw.shape != (p.shape[0],) or not np.issubdtype(raw.dtype, np.integer):
            raise Rc64Error("symbols must be an integer array with shape [N]")
        if np.any(raw < 0) or np.any(raw >= 5):
            raise Rc64Error("symbols must be in [0, 4]")
        s = np.ascontiguousarray(raw, dtype=np.int32)
        status = self.library.rc64_encoder_encode_probabilities(
            self.context, p.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            len(s), s.ctypes.data_as(ctypes.POINTER(ctypes.c_int32)),
        )
        if status:
            self.close()
            raise Rc64Error(f"native RC64 encode failed: {status}")

    def finish(self):
        if not self.context:
            raise Rc64Error("encoder closed")
        if self.library.rc64_encoder_finish(self.context):
            self.close()
            raise Rc64Error("encoder finalization failed")
        self.finished = True
        size = self.library.rc64_encoder_size(self.context)
        return ctypes.string_at(self.library.rc64_encoder_data(self.context), size)

    def close(self):
        if self.context:
            self.library.rc64_encoder_destroy(self.context)
            self.context = None

    def __del__(self):
        self.close()
