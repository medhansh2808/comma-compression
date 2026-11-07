"""Compare full-sequence ordinary and detached token-cost-weighted bias gradients."""
import argparse
import json
import math
from pathlib import Path
import time
import numpy as np
import train_entropy as te
from diagnose_entropy import reload_serialized


def recompressed_size(ref, canonical, record):
    # Force the same repacking/compression search even for unchanged reference.
    control = dict(ref, canonical=b'')
    return te.storage(control, canonical, record)[1]


def rate(ref, frames, label):
    from runtime.residual_archive import _probability_table
    from profile_reference import frequencies
    total = 0.
    started = time.monotonic()
    with ref['torch'].no_grad():
        for i, frame in enumerate(frames, 1):
            logits, target = te.logits_for(ref, frame)
            flat = logits[0].permute(1,2,0).reshape(-1,5).cpu().numpy()
            freq = frequencies(_probability_table(flat, ref['renderer'].HPAC_LOGIT_PRECISION), allow_zero=True)
            symbols = target.cpu().numpy().reshape(-1)
            total += float(-np.log2(freq[np.arange(len(freq)),symbols]/float(1<<31)).sum())
            if i % 50 == 0 or i == len(frames):
                print(json.dumps(dict(phase=label,frames=i,of=len(frames),elapsed_seconds=time.monotonic()-started)),flush=True)
    return total/8


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--device',choices=('cpu','cuda'),default='cuda')
    ap.add_argument('--tokens',type=Path,default=te.TOKENS)
    ap.add_argument('--output',type=Path,default=te.ROOT/'experiments/rdo151/work/weighted_bias_v1')
    ap.add_argument('--frames',type=int,default=600,help='prefix length; values below 600 are smoke tests only')
    ap.add_argument('--scales',default='1,2,4,-1')
    args=ap.parse_args()
    args.tokens=args.tokens.resolve();args.output=args.output.resolve()
    scales=[float(x) for x in args.scales.split(',')]
    if not 1<=args.frames<=600 or not scales or not all(math.isfinite(x) for x in scales):ap.error('invalid frame count or scales')
    args.output.mkdir(parents=True,exist_ok=False)
    ref=te.setup(args.device,args.tokens);te.check(ref)
    torch=ref['torch'];model=ref['model']
    for p in model.parameters():p.requires_grad_(False)
    bias=model.head.bias;bias.requires_grad_(True)
    original=bias.detach().clone();canonical0,record0=te.serialize(ref,model)
    frames=list(range(args.frames));started=time.monotonic()
    original_size=te.storage(ref,canonical0,record0)[1]
    control_size=recompressed_size(ref,canonical0,record0)
    baseline=rate(ref,frames,'baseline')
    report=dict(full_sequence=args.frames==600,frames=args.frames,device=args.device,
        original_bias=original.cpu().tolist(),baseline_rate_proxy_bytes=baseline,
        original_model_bytes=original_size,recompressed_unchanged_model_bytes=control_size,
        compression_control_overhead_bytes=control_size-original_size,
        weighting='detached min(token NLL bits,31); globally normalized over all sampled tokens',candidates=[])
    best=dict(step=0,is_reference=True,rate_proxy_bytes=baseline,stored_entropy_model_bytes=original_size,
              combined_proxy_bytes=baseline+original_size)
    te.save_best(ref,args.output,canonical0,record0,best)
    ordinary=torch.zeros_like(bias);weighted=torch.zeros_like(bias);weight_sum=0.;token_count=0
    from torch.nn import functional as F
    for i,frame in enumerate(frames,1):
        logits,target=te.logits_for(ref,frame)
        nll=-F.log_softmax(logits,dim=1).gather(1,target[:,None]).squeeze(1)/math.log(2)
        weights=nll.detach().clamp(0,31)
        # Mean losses keep gradients numerically modest; restore token counts for global normalization.
        count=nll.numel()
        ordinary+=torch.autograd.grad(nll.mean(),bias,retain_graph=True)[0].detach()*count
        weighted+=torch.autograd.grad((nll*weights).mean(),bias)[0].detach()*count
        weight_sum+=float(weights.sum());token_count+=count
        if i%50==0 or i==len(frames):print(json.dumps(dict(phase='gradients',frames=i,of=len(frames))),flush=True)
    ordinary/=token_count
    weighted/=max(weight_sum,1e-30)
    report['gradients']={name:g.cpu().tolist() for name,g in [('ordinary',ordinary),('cost_weighted',weighted)]}
    seen={te.sha(canonical0+record0):'reference'}
    try:
        for name,g in [('ordinary',ordinary),('cost_weighted',weighted)]:
            if not torch.isfinite(g).all() or g.abs().max()==0:raise ValueError('invalid gradient')
            direction=-g/g.abs().max()
            for scale in scales:
                with torch.no_grad():bias.copy_((original+scale*direction).clamp(-32768,32767))
                canonical,record=te.serialize(ref,model);key=te.sha(canonical+record)
                row=dict(direction=name,scale=scale,bias=bias.detach().round().cpu().tolist(),state_sha256=key)
                if key in seen:
                    row['duplicate_of']=seen[key]
                else:
                    seen[key]=f'{name}:{scale}'
                    reloaded=reload_serialized(ref,canonical,record);ref['model']=reloaded
                    if te.serialize(ref,reloaded)!=(canonical,record):raise ValueError('serialization differs')
                    with torch.no_grad():
                        after,_=te.logits_for(ref,0);ref['model']=model;before,_=te.logits_for(ref,0)
                    if not torch.equal(before,after):raise ValueError('serialized logits differ')
                    ref['model']=reloaded
                    size=recompressed_size(ref,canonical,record)
                    token_bytes=rate(ref,frames,f'{name}:{scale}')
                    row.update(rate_proxy_bytes=token_bytes,stored_entropy_model_bytes=size,
                        combined_proxy_bytes=token_bytes+size,
                        token_proxy_change_bytes=token_bytes-baseline,
                        same_compressor_model_change_bytes=size-control_size,
                        actual_model_change_bytes=size-original_size,
                        serialized_logits_verified=True)
                    if args.frames==600 and row['combined_proxy_bytes']<best['combined_proxy_bytes']:
                        best=dict(step=len(report['candidates'])+1,is_reference=False,**row)
                        te.save_best(ref,args.output,canonical,record,best)
                report['candidates'].append(row);ref['model']=model
                print(json.dumps(row),flush=True)
                (args.output/'summary.json').write_text(json.dumps(report,indent=2)+'\n')
    finally:
        ref['model']=model
        with torch.no_grad():bias.copy_(original)
    report.update(best=best,elapsed_seconds=time.monotonic()-started,actual_archive_evaluated=False)
    (args.output/'summary.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(dict(best=best,elapsed_seconds=report['elapsed_seconds'])),flush=True)

if __name__=='__main__':main()
