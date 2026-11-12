"""Private runtime setup, actual archive encoding and full-video measurement."""
import hashlib
import json
import os
from pathlib import Path
import shutil
import struct
import subprocess
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
EXP = ROOT / 'experiments/rdo151'
TOKEN_SHA = 'd64c2a094dbdcdf6b381c94810f9d672ac33c3c960eb6ae35ba09df0af51c3a1'
ORIGINAL_BYTES = 37545489
RAW_SHAPE = (1200, 874, 1164, 3)
RATE_PER_BIT = 25 / (8 * ORIGINAL_BYTES)


def sha(path):
    with Path(path).open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()


def write_json(path, value):
    path = Path(path)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    tmp.replace(path)


def log(**values):
    print(json.dumps(values, allow_nan=False), flush=True)


def score(seg, pose, nbytes):
    return 100 * seg + (10 * pose) ** .5 + 25 * nbytes / ORIGINAL_BYTES


def resolve_archive(args):
    if args.parent_run is not None:
        p = args.parent_run.expanduser().resolve()
        result = json.loads((p / 'best.json').read_text())
        archive = Path(result['archive'])
        if sha(archive) != result['archive_sha256']:
            raise ValueError('Parent selected archive changed')
        # Carry the exact decoder associated with the selected archive.
        args.runtime_source = p / 'runtime'
        return archive
    return args.archive.expanduser().resolve()


def session(args):
    archive = resolve_archive(args)
    out = args.output.expanduser().resolve()
    if out.exists():
        raise FileExistsError(f'Preserving existing run: {out}; choose another --output')
    for p in (archive, EXP/'train_entropy.py', EXP/'profile_reference.py',
              EXP/'cache/tokens151_o2.npy', EXP/'work/rd/gt_dali.npz'):
        if not p.is_file():
            raise FileNotFoundError(p)
    source = (args.runtime_source or EXP/'work/reference').expanduser().resolve()
    if not source.is_dir():
        raise FileNotFoundError(source)
    out.mkdir(parents=True)
    private = out / 'runtime'
    # Snapshot executable sources only, not old experiment data or shared objects.
    allowed = {'.py', '.c', '.cpp', '.h', '.sh'}
    for p in source.rglob('*'):
        if not p.is_file() or p.suffix not in allowed or '__pycache__' in p.parts:
            continue
        target = private / p.relative_to(source)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(p, target)
    rt = private/'external/pr141'
    libs = out/'native'
    libs.mkdir()
    compiler = os.environ.get('CC', 'cc')
    for name, src in [('rc64',rt/'runtime/entropy/rc64_encode_backend.c'),
                      ('corrector',rt/'runtime/f26_corrector_native.c')]:
        subprocess.run([compiler, '-O2', '-std=c11', '-shared', '-fPIC',
                        '-ffp-contract=off', '-fno-fast-math', str(src),
                        '-o', str(libs/f'{name}.so'), '-lm'], check=True)
    os.environ['CPR1_RC64_LIBRARY'] = str(libs/'rc64.so')
    os.environ['F26_CORRECTOR_NATIVE_LIBRARY'] = str(libs/'corrector.so')
    sys.path.insert(0, str(EXP))
    import train_entropy as base
    base.SUB, base.RUNTIME = private, rt
    base.ARCHIVE, base.ARCHIVE_SHA = archive, sha(archive)
    base.TOKENS, base.TOKEN_SHA = EXP/'cache/tokens151_o2.npy', TOKEN_SHA
    ref = base.setup('cuda', base.TOKENS)
    base.check(ref)
    torch = ref['torch']
    # Preserve official renderer kernels even if the entropy runtime configures
    # different convolution flags while loading its sparse evaluator.
    from decoder.numerics import official_flags
    official_flags()
    gt_path = EXP/'work/rd/gt_dali.npz'
    with np.load(gt_path) as gt:
        gp, gs = gt['pose'].copy(), gt['seg'].copy()
    if gp.shape != (600,6) or gs.shape != (600,384,512):
        raise ValueError(f'Unexpected DALI GT shapes: {gp.shape}, {gs.shape}')
    # Reuse the already-working evaluation helper but prohibit AV fallback.
    sys.path.insert(0, str(EXP/'rd'))
    import common
    if Path(common.GT).resolve() != gt_path.resolve():
        raise ValueError('Evaluation helper is not using gt_dali.npz')
    metrics_net = common.load_nets()
    manifest = dict(archive=str(archive), archive_sha256=sha(archive),
                    tokens_sha256=TOKEN_SHA, gt_sha256=sha(gt_path),
                    compiler=subprocess.check_output([compiler,'--version'], text=True).splitlines()[0],
                    source_sha256={str(p.relative_to(private)):sha(p)
                                   for p in private.rglob('*') if p.is_file()},
                    torch_version=torch.__version__, cuda_device=torch.cuda.get_device_name(0))
    write_json(out/'manifest.json', manifest)
    return dict(out=out, archive=archive, base=base, ref=ref, private=private,
                rt=rt, torch=torch, common=common, net=metrics_net, gp=gp, gs=gs)


def evaluate(s, archive, folder, *, expected_tokens=None, keep_raw=False):
    """Actual archive -> real token decode -> rendered RGB -> both metrics."""
    folder.mkdir(parents=True, exist_ok=True)
    raw = folder/'frames.raw'
    if shutil.disk_usage(folder).free < np.prod(RAW_SHAPE)+1024**3:
        raise RuntimeError('Need space for one 3.67 GB RGB evaluation file')
    from decoder import inflate_core
    from decoder.numerics import official_flags
    official_flags()
    started = time.monotonic()
    info = inflate_core.inflate(archive, raw)
    if expected_tokens is not None and info['token_sha'] != expected_tokens:
        raise ValueError('Actual archive decoded to unexpected tokens')
    sd, pd = s['common'].eval_raw(raw, net=s['net'])
    result = dict(archive=str(archive.resolve()), archive_sha256=sha(archive),
                  bytes=archive.stat().st_size, seg=float(sd.mean()), pose=float(pd.mean()),
                  token_sha256=info['token_sha'], local_full_evaluation=True,
                  official_evaluation=False, seconds=time.monotonic()-started)
    result['score'] = score(result['seg'],result['pose'],result['bytes'])
    np.savez(folder/'perpair.npz', seg=sd, pose=pd)
    write_json(folder/'evaluation.json',result)
    log(phase='actual_evaluation', **result)
    if not keep_raw:
        raw.unlink()
    return result, raw


def select(s, result):
    write_json(s['out']/'best.json', result)


def encode(s, input_archive, tokens, folder):
    """Replace only the token stream; accept full native stream after decode."""
    folder.mkdir(parents=True,exist_ok=True)
    from container import token_encode, token_decode
    from container.payload_format import read_archive,split_inner_payload,write_archive
    from container.archive_writer import EXTERNAL_ATTR_PR141
    from decoder.inflate_core import _load
    token_encode.POST_PURGE_HOOKS = token_decode.POST_PURGE_HOOKS
    parts,_,_ = _load(input_archive)
    core,signal = split_inner_payload(read_archive(input_archive))
    if not parts.token_stream or not core.endswith(parts.token_stream):
        raise ValueError('Unexpected token stream framing')
    started = time.monotonic()
    stream,_ = token_encode.encode_tokens(input_archive,s['rt'],tokens,'cuda')
    dest = folder/'archive.zip'
    write_archive(dest,core[:-len(parts.token_stream)]+stream+signal+struct.pack('<H',len(signal)),EXTERNAL_ATTR_PR141)
    decoded,_ = token_decode.decode_tokens(dest,s['rt'],'cuda')
    if not np.array_equal(decoded.numpy(),tokens):
        raise ValueError('Native token round trip failed')
    result = dict(archive=str(dest),archive_sha256=sha(dest),bytes=dest.stat().st_size,
                  token_stream_bytes=len(stream),roundtrip_verified=True,
                  seconds=time.monotonic()-started)
    write_json(folder/'encoding.json',result)
    log(phase='native_encoding',**result)
    return dest


def common_arguments(ap):
    ap.add_argument('--archive',type=Path,default=EXP/'work/rd/xp1.zip')
    ap.add_argument('--parent-run',type=Path)
    ap.add_argument('--runtime-source',type=Path)
    ap.add_argument('--output',type=Path,required=True)
