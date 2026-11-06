"""Check streaming encode/decode compatibility using the actual native coder."""
from pathlib import Path
import sys
import numpy as np

HERE = Path(__file__).resolve().parent
runtime = HERE / "work/reference/external/pr141"
sys.path.insert(0, str(runtime))
from runtime.entropy.rc64 import NativeDecoder
from runtime.entropy.rc64_encoder import NativeEncoder

rng = np.random.default_rng(1234)
p = rng.dirichlet(np.ones(5), size=10000).astype(np.float32)
p[:100] = np.float32(0.2)  # equal-probability ties
p[100:200] = np.array([0.999996, 1e-6, 1e-6, 1e-6, 1e-6], np.float32)
symbols = np.array([rng.choice(5, p=row.astype(float)/row.sum(dtype=float)) for row in p], dtype=np.int32)
lib = runtime / "runtime/entropy/rc64_encode_backend.so"
encoder = NativeEncoder(lib)
for start in range(0, len(p), 137):
    encoder.encode(p[start:start+137], symbols[start:start+137])
payload = encoder.finish()
encoder.close()
decoder = NativeDecoder(lib, payload)
actual = np.concatenate([decoder.decode(p[start:start+83]) for start in range(0, len(p), 83)])
decoder.close()
np.testing.assert_array_equal(actual, symbols)
print(f"PASS: 10,000 symbols, different encoder/decoder chunk sizes; {len(payload)} bytes")
