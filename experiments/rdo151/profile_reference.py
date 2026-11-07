"""Measure the pinned #151 decoder without training or modifying its probabilities.

Default device is CPU. Prefix runs preserve the full model and causal order.
Integer-frequency NLL is a diagnostic, not a per-token allocation of stored
archive bytes. Reader positions include lookahead and implicit zero padding.
"""
import argparse
import csv
import ctypes
import hashlib
import importlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time
import types

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
SUB = ROOT / "experiments/rdo151/work/reference"
RUNTIME = SUB / "external/pr141"
ARCHIVE_SHA = "9e2de76d931c5a7aca29f7e9008c4d77b34ec037d042113e2fd082607391f315"
TOKEN_SHA = "092ff87541fe0f4bdbff07759ae84cafd14e7e7e6f1385334d4177ef8c941d77"
TOTAL = 1 << 31

def sha(data):
    return hashlib.sha256(data).hexdigest()

def frequencies(rows, *, allow_zero=False):
    """Mirror rc64_backend.c float32 -> integer frequencies and winner balance."""
    p = np.asarray(rows,dtype=np.float32)
    if p.ndim != 2 or p.shape[1] != 5 or not np.isfinite(p).all():
        raise ValueError("invalid probability rows")
    if np.any(p < 0) or (not allow_zero and np.any(p <= 0)):
        raise ValueError("nonpositive coded probability")
    wide = p.astype(np.float64)
    if np.any(wide.sum(1)<0.99998) or np.any(wide.sum(1)>1.00002) or np.any(wide>1.00002):
        raise ValueError("probability sum outside native tolerance")
    f = np.maximum((wide*TOTAL).astype(np.int64),1)
    winner = p.argmax(1)
    f[np.arange(len(f)),winner] += TOTAL-f.sum(1)
    if np.any(f <= 0) or np.any(f >= TOTAL) or np.any(f.sum(1) != TOTAL):
        raise ValueError("invalid quantized frequencies")
    return f

def build_native(directory):
    source = RUNTIME / "runtime/entropy/rc64_encode_backend.c"
    wrapper = directory / "measured.c"
    # Read-only inspection functions; no arithmetic-coder operations change.
    wrapper.write_text('#include '+json.dumps(str(source))+'\n'+'''
uint64_t profile_pending(void *v) { return ((rc64_encoder *)v)->pending; }
uint64_t profile_emitted(void *v) { return ((rc64_encoder *)v)->bits; }
''')
    library = directory / "rc64.so"
    compiler = os.environ.get("CC","cc")
    subprocess.run([compiler,"-O3","-std=c11","-shared","-fPIC","-ffp-contract=off","-fno-fast-math",str(wrapper),"-o",str(library),"-lm"],check=True)
    corrector = directory / "corrector.so"
    subprocess.run([compiler,"-O3","-std=c11","-shared","-fPIC","-ffp-contract=off","-fno-fast-math",str(RUNTIME/"runtime/f26_corrector_native.c"),"-o",str(corrector),"-lm"],check=True)
    return library,corrector

def shorten_candidate(raw):
    b = bytearray(raw[:-1])
    while b and b[-1] == 255:
        b.pop()
    if not b:
        return raw
    b[-1] += 1
    return bytes(b)

class Profile:
    def __init__(self,args,tokens):
        self.args,self.tokens = args,tokens
        self.n,self.h,self.w = tokens.shape
        self.limit = min(args.frames,self.n)
        self.frame = -1
        self.buckets = {}
        self.rows = []
        self.heatmaps = []
        self.start = time.monotonic()
        self.decoder = self.shadow = None
        self.fullmap = None
        if args.save_full_maps:
            self.fullmap = np.lib.format.open_memmap(args.output/"frequency_bits.npy",mode="w+",dtype=np.float32,shape=(self.limit,self.h,self.w))
        self.csv_file = (args.output/"frames.csv").open("w",newline="")
        self.writer = None
        self.top_file = (args.output/"expensive_tokens.jsonl").open("w")
        self.motion = self.motion_state = None
        if args.motion_state:
            source = ROOT/"experiments/semantic_motion/motion_codec.py"
            spec = importlib.util.spec_from_file_location("measured_motion",source)
            self.motion = importlib.util.module_from_spec(spec);spec.loader.exec_module(self.motion)
            self.motion_state = self.motion.unpack_state(args.motion_state.read_bytes())
            if tuple(self.motion_state["shape"]) != tokens.shape:
                raise ValueError("motion-state shape differs from reference cache")

    def frame_start(self,frame):
        self.frame = frame
        self.frame_time = time.monotonic()
        self.bits = np.zeros(self.h*self.w,np.float64)
        self.proxy = np.zeros_like(self.bits)
        self.base_argmax = np.zeros(self.h*self.w,np.uint8)
        self.final_argmax = np.zeros(self.h*self.w,np.uint8)
        self.seen = np.zeros(self.h*self.w,bool)
        self.reader_before = self.decoder.bit_position

    def install(self):
        # Installed only after the pinned decoder purges runtime.* imports.
        profile = self
        from runtime.entropy import rc64
        from runtime import native_free_corrector as nc
        original_decoder = rc64.NativeDecoder
        original_corrector = nc.NativeFreeCorrector
        from runtime.entropy.rc64_encoder import NativeEncoder

        class MeasuredDecoder(original_decoder):
            def __init__(self,library,payload):
                super().__init__(library,payload)
                profile.decoder = self
                profile.stream_bytes = len(payload)
                profile.initial_lookahead_bits = self.bit_position
                profile.shadow = NativeEncoder(library)
                lib = profile.shadow.library
                for name in ("profile_pending","profile_emitted"):
                    fn = getattr(lib,name)
                    fn.argtypes = [ctypes.c_void_p];fn.restype = ctypes.c_uint64

            def decode(self,rows):
                symbols = super().decode(rows)
                profile.group(rows,symbols)
                profile.shadow.encode(rows,symbols)
                return symbols

        class MeasuredCorrector(original_corrector):
            def group_state(self,probability,predicted,positions):
                result = super().group_state(probability,predicted,positions)
                profile.prob_before = probability
                profile.predicted = predicted
                profile.positions = positions
                return result

        rc64.NativeDecoder = MeasuredDecoder
        nc.NativeFreeCorrector = MeasuredCorrector

    def group(self,rows,symbols):
        pos = np.asarray(self.positions,dtype=np.int64)
        expected = self.tokens[self.frame].reshape(-1)[pos]
        if not np.array_equal(symbols,expected):
            bad = int(np.flatnonzero(symbols!=expected)[0])
            y,x = divmod(int(pos[bad]),self.w)
            raise ValueError(f"reference/cache mismatch at frame {self.frame}, y={y}, x={x}; stop")
        if np.any(self.seen[pos]):
            raise ValueError("duplicate symbol position in causal scan")
        self.seen[pos] = True
        f = frequencies(rows)
        before = frequencies(self.prob_before,allow_zero=True)
        ar = np.arange(len(symbols))
        self.bits[pos] = -np.log2(f[ar,symbols].astype(np.float64)/TOTAL)
        self.proxy[pos] = -np.log2(before[ar,symbols].astype(np.float64)/TOTAL)
        self.base_argmax[pos] = self.predicted
        self.final_argmax[pos] = f.argmax(1)

    def add_bucket(self,key,mask):
        mask = np.asarray(mask).reshape(-1)
        v = self.buckets.setdefault(key,dict(symbols=0,frequency_bits=0.,pre_corrector_proxy_bits=0.))
        v["symbols"] += int(mask.sum())
        v["frequency_bits"] += float(self.bits[mask].sum())
        v["pre_corrector_proxy_bits"] += float(self.proxy[mask].sum())

    def frame_done(self,frame,decoded):
        if frame != self.frame or not self.seen.all() or not np.array_equal(decoded,self.tokens[frame]):
            raise ValueError("frame coverage/parity failure")
        from runtime.residual_archive import _boundary_buckets
        actual = np.asarray(decoded).reshape(-1)
        self.add_bucket("all",np.ones_like(actual,bool))
        for label in range(5):
            self.add_bucket(f"class_{label}",actual==label)
        base_wrong = self.base_argmax != actual
        final_wrong = self.final_argmax != actual
        self.add_bucket("hpac_argmax_wrong",base_wrong)
        self.add_bucket("hpac_argmax_right",~base_wrong)
        self.add_bucket("final_argmax_wrong",final_wrong)
        self.add_bucket("final_argmax_right",~final_wrong)
        boundary = _boundary_buckets(np.asarray(decoded)).reshape(-1)
        for distance in range(5):
            self.add_bucket(f"current_boundary_distance_{distance}",boundary==distance)
        for low,high in ((0,.01),(.01,.1),(.1,1),(1,4),(4,8),(8,16),(16,32)):
            self.add_bucket(f"frequency_cost_{low}_to_{high}",(self.bits>=low)&(self.bits<high))
        if frame:
            previous = np.asarray(self.tokens[frame-1])
            copy_wrong = previous.reshape(-1) != actual
            self.add_bucket("copy_wrong",copy_wrong)
            self.add_bucket("copy_right",~copy_wrong)
            if self.motion:
                prediction = self.motion.predict(previous,self.motion_state,frame-1)
                motion_wrong = prediction.reshape(-1) != actual
                self.add_bucket("motion_fixes_copy",copy_wrong & ~motion_wrong)
                self.add_bucket("motion_harms_copy",~copy_wrong & motion_wrong)
                self.add_bucket("both_copy_and_motion_wrong",copy_wrong & motion_wrong)
                self.add_bucket("both_copy_and_motion_right",~copy_wrong & ~motion_wrong)
        else:
            self.add_bucket("first_frame",np.ones_like(actual,bool))
        after = self.decoder.bit_position
        stored_bits = self.stream_bytes*8
        paid = min(after,stored_bits)-min(self.reader_before,stored_bits)
        virtual = max(after-stored_bits,0)-max(self.reader_before-stored_bits,0)
        lib,handle = self.shadow.library,self.shadow.context
        record = dict(frame=frame,symbols=self.h*self.w,frequency_bits=float(self.bits.sum()),
            pre_corrector_proxy_bits=float(self.proxy.sum()),hpac_argmax_errors=int(base_wrong.sum()),
            final_argmax_errors=int(final_wrong.sum()),reader_bit_position=after,
            stored_bits_read_this_frame=paid,implicit_zero_bits_read_this_frame=virtual,
            shadow_emitted_bits=int(lib.profile_emitted(handle)),shadow_pending_underflow_bits=int(lib.profile_pending(handle)),
            frame_seconds=time.monotonic()-self.frame_time,token_parity=True)
        self.rows.append(record)
        if self.writer is None:
            self.writer = csv.DictWriter(self.csv_file,fieldnames=list(record));self.writer.writeheader()
        self.writer.writerow(record);self.csv_file.flush()
        k=min(self.args.top_k,self.bits.size)
        indices=np.argpartition(self.bits,-k)[-k:]
        indices=indices[np.argsort(self.bits[indices])[::-1]]
        for i in indices:
            y,x=divmod(int(i),self.w)
            self.top_file.write(json.dumps(dict(frame=frame,y=y,x=x,token=int(actual[i]),frequency_bits=float(self.bits[i]),
                current_boundary_distance=int(boundary[i]),hpac_argmax=int(self.base_argmax[i]),final_argmax=int(self.final_argmax[i])))+"\n")
        self.top_file.flush()
        self.heatmaps.append(self.bits.reshape(self.h//16,16,self.w//16,16).sum((1,3)).astype(np.float32))
        if self.fullmap is not None:
            self.fullmap[frame]=self.bits.reshape(self.h,self.w).astype(np.float32);self.fullmap.flush()
        completed=frame+1
        mean_seconds=sum(r["frame_seconds"] for r in self.rows)/completed
        print(json.dumps(dict(frame=completed,of=self.limit,frequency_bytes_equivalent=record["frequency_bits"]/8,
            frame_seconds=record["frame_seconds"],estimated_remaining_seconds=mean_seconds*(self.limit-completed),
            token_parity=True,virtual_zero_bits=virtual)),flush=True)

    def finish(self,info):
        emitted=self.shadow.library.profile_emitted(self.shadow.context)
        pending=self.shadow.library.profile_pending(self.shadow.context)
        raw=self.shadow.finish();self.shadow.close()
        (self.args.output/"shadow_raw.bin").write_bytes(raw)
        shortened=shorten_candidate(raw)
        from decoder.inflate_core import _load
        published=_load(self.args.archive)[0].token_stream
        read_position=self.decoder.bit_position
        self.decoder.close()
        np.savez_compressed(self.args.output/"block_frequency_bits.npz",bits=np.stack(self.heatmaps),block_size=np.array(16))
        totals=self.buckets["all"]
        for row in self.buckets.values():
            row["byte_equivalent"]=row["frequency_bits"]/8
            row["fraction_of_frequency_bits"]=row["frequency_bits"]/totals["frequency_bits"] if totals["frequency_bits"] else 0.
        summary=dict(device=self.args.device,frames_profiled=self.limit,total_reference_frames=self.n,
            full_reference=self.limit==self.n,all_profiled_tokens_match_cache=True,
            source_archive_sha256=sha(self.args.archive.read_bytes()),reference_token_stream_bytes=self.stream_bytes,
            decoded_prefix_sha256=info["decoded_token_sha256"],
            probability_cost_bits=totals["frequency_bits"],probability_cost_byte_equivalent=totals["frequency_bits"]/8,
            pre_corrector_proxy_bits=totals["pre_corrector_proxy_bits"],
            initial_reader_lookahead_bits=self.initial_lookahead_bits,
            final_reader_position_bits=read_position,
            stored_stream_bits_read=min(read_position,self.stream_bytes*8),
            implicit_zero_bits_read=max(read_position-self.stream_bytes*8,0),
            unused_stored_stream_bits=max(self.stream_bytes*8-read_position,0),
            shadow_emitted_bits_before_finish=int(emitted),shadow_pending_bits_before_finish=int(pending),
            shadow_raw_bytes=len(raw),shadow_shortened_candidate_bytes=len(shortened),
            shadow_shortened_matches_published=(shortened==published) if self.limit==self.n else None,
            shadow_shortened_candidate_verified=(shortened==published) if self.limit==self.n else False,
            frame_seconds_total=sum(r["frame_seconds"] for r in self.rows),
            estimated_full_600_frame_seconds=sum(r["frame_seconds"] for r in self.rows)/self.limit*self.n,
            wall_seconds=time.monotonic()-self.start,buckets=self.buckets,
            notes=["Bucket families overlap; do not sum across families.",
                   "Frequency bits are -log2(integer_frequency/2^31), not literal archive bits per pixel.",
                   "Reader positions include lookahead; clamped reader increments are not independent frame code lengths.",
                   "Pre-corrector proxy floors zero frequencies to one; it is not an actual corrector-removal re-encode.",
                   "Motion buckets measure reference costs at changed locations, not the achievable saving of a motion-conditioned coder.",
                   "Prefix timing cannot establish full-sequence cost or termination behavior."])
        (self.args.output/"summary.json").write_text(json.dumps(summary,indent=2)+"\n")
        self.csv_file.close();self.top_file.close()
        return summary

def selftest():
    sys.path.insert(0,str(RUNTIME))
    from runtime.entropy.rc64 import NativeDecoder
    from runtime.entropy.rc64_encoder import NativeEncoder
    rng=np.random.default_rng(2026)
    rows=rng.dirichlet(np.ones(5),size=10000).astype(np.float32)
    rows[:50]=0.2
    f=frequencies(rows)
    assert np.all(f.sum(1)==TOTAL)
    symbols=rng.integers(0,5,len(rows),dtype=np.int32)
    with tempfile.TemporaryDirectory() as tmp:
        lib,_=build_native(Path(tmp))
        encoder=NativeEncoder(lib);encoder.encode(rows,symbols);payload=encoder.finish();encoder.close()
        decoder=NativeDecoder(lib,payload)
        actual=decoder.decode(rows)
        np.testing.assert_array_equal(actual,symbols)
        # Integer-frequency self-information tracks native arithmetic size,
        # allowing a few bits of finalization/padding on an ordinary test.
        bits=-np.log2(f[np.arange(len(f)),symbols]/TOTAL).sum()
        assert abs(len(payload)*8-bits)<16
        decoder.close()
    print("PASS: native round trip, quantized frequencies, and coding-cost accounting (CPU only)")

def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--device",choices=("cpu","cuda"),default="cpu")
    ap.add_argument("--frames",type=int,default=20)
    ap.add_argument("--archive",type=Path,default=ROOT/"experiments/rdo151/cache/leader151.zip")
    ap.add_argument("--tokens",type=Path,default=ROOT/"experiments/rdo151/cache/tokens151.npy")
    ap.add_argument("--motion-state",type=Path)
    ap.add_argument("--output",type=Path,default=ROOT/"experiments/rdo151/work/profile_cpu20")
    ap.add_argument("--top-k",type=int,default=64)
    ap.add_argument("--save-full-maps",action="store_true")
    ap.add_argument("--selftest",action="store_true")
    args=ap.parse_args()
    if args.device=="cpu":
        os.environ["CUDA_VISIBLE_DEVICES"]=""
    if args.selftest:
        selftest();return
    if not 1<=args.frames<=600 or args.top_k<1:
        ap.error("frames must be 1..600 and top-k positive")
    args.archive=args.archive.resolve();args.tokens=args.tokens.resolve();args.output=args.output.resolve()
    if args.motion_state:
        args.motion_state=args.motion_state.resolve()
    if args.output.exists():
        ap.error("output exists; choose a new output directory")
    if sha(args.archive.read_bytes())!=ARCHIVE_SHA:
        ap.error("archive is not the pinned #151 release")
    tokens=np.load(args.tokens,mmap_mode="r",allow_pickle=False)
    if tokens.shape!=(600,384,512) or tokens.dtype!=np.uint8 or sha(tokens.tobytes())!=TOKEN_SHA:
        ap.error("cache is not the verified #151 token sequence")
    source_path=SUB/"container/token_decode.py"
    source=source_path.read_text()
    replacements={
      "for frame in range(renderer.N):":"for frame in range(PROFILE.limit):\n            PROFILE.frame_start(frame)",
      "decoded_tokens = torch.empty((renderer.N, renderer.EVAL_H, renderer.EVAL_W), dtype=torch.uint8)":"decoded_tokens = torch.empty((PROFILE.limit, renderer.EVAL_H, renderer.EVAL_W), dtype=torch.uint8)",
      "            previous = current":"            previous = current\n            PROFILE.frame_done(frame, decoded_tokens[frame].numpy())",
    }
    for old,new in replacements.items():
        if source.count(old)!=1:
            raise ValueError(f"decoder source differs; cannot safely instrument {old!r}")
        source=source.replace(old,new)
    compile(source,str(source_path),"exec")
    args.output.mkdir(parents=True)
    (args.output/"config.json").write_text(json.dumps(dict(device=args.device,frames=args.frames,
        decoder_source_sha256=sha(source_path.read_bytes()),archive_sha256=ARCHIVE_SHA,tokens_sha256=TOKEN_SHA,
        motion_state=str(args.motion_state) if args.motion_state else None),indent=2)+"\n")
    profile=Profile(args,tokens)
    sys.path[:0]=[str(SUB),str(RUNTIME)]
    os.chdir(SUB)
    from canonical.inflate_canonical import install_rp1_shim
    install_rp1_shim()
    from decoder import numerics
    from container import token_decode
    module=types.ModuleType("container._measured_decoder")
    module.__file__=str(source_path)
    module.PROFILE=profile
    exec(compile(source,str(source_path),"exec"),module.__dict__)
    module.POST_PURGE_HOOKS=token_decode.POST_PURGE_HOOKS+[profile.install]
    with tempfile.TemporaryDirectory(prefix="comma-profile-") as tmp:
        lib,corrector=build_native(Path(tmp))
        os.environ["CPR1_RC64_LIBRARY"]=str(lib)
        os.environ["F26_CORRECTOR_NATIVE_LIBRARY"]=str(corrector)
        print(f"Profiling {profile.limit} causal frames on {args.device}; no training",flush=True)
        try:
            decoded,info=module.decode_tokens(args.archive,RUNTIME,args.device)
            np.testing.assert_array_equal(decoded.numpy(),tokens[:profile.limit])
            summary=profile.finish(info)
        except Exception as error:
            (args.output/"failure.json").write_text(json.dumps(dict(device=args.device,
                completed_frames=len(profile.rows),token_parity=False,error_type=type(error).__name__,
                error=str(error),note="Do not interpret partial measurements as a valid reference profile."),indent=2)+"\n")
            raise
    print(json.dumps({k:v for k,v in summary.items() if k not in {"buckets","notes"}},indent=2),flush=True)
    print(f"Saved {args.output/'summary.json'}",flush=True)

if __name__=="__main__":
    main()
