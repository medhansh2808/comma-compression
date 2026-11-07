"""Bounded output-bias line search; no training resume or automatic full encode."""
import argparse
import json
import math
from pathlib import Path
import time
import numpy as np
import train_entropy as te


def frame_list(value):
    frames = [int(x) for x in value.split(',')]
    if not frames or len(set(frames)) != len(frames) or any(x < 0 or x >= 600 for x in frames):
        raise argparse.ArgumentTypeError('unique frame indices in [0, 599] required')
    return frames


def measure(ref, frames):
    torch = ref['torch']
    from torch.nn import functional as F
    with torch.no_grad():
        nll = sum(float(F.cross_entropy(*te.logits_for(ref, f))) / math.log(2) for f in frames) / len(frames)
    rate = te.probe(ref, frames)
    canonical, record = te.serialize(ref, ref['model'])
    _, size = te.storage(ref, canonical, record)
    return dict(bits_per_token=nll, rate_proxy_bytes=rate,
                stored_entropy_model_bytes=size, combined_proxy_bytes=rate + size)


def reload_serialized(ref, canonical, record):
    from container.hpac2 import unpack_record_share
    base = ref['renderer'].load_hpac(canonical, ref['device'])
    model = ref['model_class'](**ref['kwargs']).to(ref['device']).eval()
    missing = model.load_state_dict(base.state_dict(), strict=False)
    if set(missing.missing_keys) != {'m2', 'b2', 'e2'} or missing.unexpected_keys:
        raise ValueError('serialized model layout mismatch')
    unpack_record_share(record, model)
    return model


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--device', choices=('cpu', 'cuda'), default='cpu')
    ap.add_argument('--tokens', type=Path, default=te.TOKENS)
    ap.add_argument('--output', type=Path, default=te.ROOT/'experiments/rdo151/work/entropy_diagnostic_v1')
    ap.add_argument('--train-frames', type=frame_list, default=frame_list('0,199,399,599'))
    ap.add_argument('--probe-frames', type=frame_list, default=frame_list('39,79,119,159,239,279,319,359,439,479,519,559'))
    ap.add_argument('--scales', default='0.25,0.5,1,2,4,-1')
    args = ap.parse_args()
    args.tokens = args.tokens.resolve(); args.output = args.output.resolve()
    scales = [float(x) for x in args.scales.split(',')]
    if not scales or not all(math.isfinite(x) for x in scales):
        ap.error('finite scales required')
    if set(args.train_frames) & set(args.probe_frames):
        ap.error('training and probe frames must be disjoint')
    args.output.mkdir(parents=True, exist_ok=False)
    ref = te.setup(args.device, args.tokens)
    te.check(ref)
    torch = ref['torch']; model = ref['model']
    for p in model.parameters(): p.requires_grad_(False)
    bias = model.head.bias
    bias.requires_grad_(True)
    original = bias.detach().clone()
    started = time.monotonic()
    baseline_train = measure(ref, args.train_frames)
    baseline_probe = measure(ref, args.probe_frames)
    canonical0, record0 = te.serialize(ref, model)
    best = dict(step=0, is_reference=True, **baseline_probe)
    te.save_best(ref, args.output, canonical0, record0, best)
    from torch.nn import functional as F
    for frame in args.train_frames:
        loss = F.cross_entropy(*te.logits_for(ref, frame)) / (len(args.train_frames)*math.log(2))
        loss.backward()
    gradient = bias.grad.detach().clone()
    if not torch.isfinite(gradient).all() or gradient.abs().max() == 0:
        raise ValueError('invalid output bias gradient')
    direction = -gradient / gradient.abs().max()
    report = dict(device=args.device, train_frames=args.train_frames, probe_frames=args.probe_frames,
                  original_bias=original.cpu().tolist(), gradient=gradient.cpu().tolist(),
                  baseline_train=baseline_train, baseline_probe=baseline_probe, candidates=[])
    print(json.dumps({'baseline': report}), flush=True)
    try:
        for index, scale in enumerate(scales, 1):
            with torch.no_grad(): bias.copy_((original + scale*direction).clamp(-32768,32767))
            canonical, record = te.serialize(ref, model)
            reloaded = reload_serialized(ref, canonical, record)
            # Evaluate deployed serialized parameters, never the floating shadow weights.
            ref['model'] = reloaded
            if te.serialize(ref, reloaded) != (canonical, record):
                raise ValueError('serialization round trip failed')
            for frame in args.train_frames[:1]:
                with torch.no_grad():
                    ref['model'] = model; before, _ = te.logits_for(ref, frame)
                    ref['model'] = reloaded; after, _ = te.logits_for(ref, frame)
                if not torch.equal(before, after):
                    raise ValueError('serialized logits differ')
            train = measure(ref, args.train_frames)
            probe = measure(ref, args.probe_frames)
            row = dict(scale=scale, changed_bias_values=int((bias.round()!=original.round()).sum()),
                       changed_serialized_bytes=sum(a!=b for a,b in zip(canonical0,canonical)),
                       train=train, probe=probe, serialized_logits_verified=True)
            report['candidates'].append(row)
            if train['combined_proxy_bytes'] < baseline_train['combined_proxy_bytes'] and probe['combined_proxy_bytes'] < best['combined_proxy_bytes']:
                best = dict(step=index, is_reference=False, **probe, diagnostic_scale=scale)
                te.save_best(ref, args.output, canonical, record, best)
            ref['model'] = model
            print(json.dumps(row), flush=True)
            (args.output/'diagnostic.json').write_text(json.dumps(report, indent=2)+'\n')
    finally:
        ref['model'] = model
        with torch.no_grad(): bias.copy_(original)
    report['best'] = best; report['elapsed_seconds'] = time.monotonic()-started
    report['actual_archive_evaluated'] = False
    (args.output/'diagnostic.json').write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps({'best':best,'elapsed_seconds':report['elapsed_seconds']}), flush=True)

if __name__ == '__main__': main()
