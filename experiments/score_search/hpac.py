"""Fine-tune corrected HPAC, then select by native archive size and full score."""
import argparse
import json
import math
import time

import numpy as np
import shared as sh


def deployed(s, canonical, record):
    ref=s['ref']
    from container import hpac2
    loaded=ref['renderer'].load_hpac(canonical,ref['device'])
    model=ref['model_class'](**ref['kwargs']).to(ref['device']).eval()
    missing=model.load_state_dict(loaded.state_dict(),strict=False)
    if set(missing.missing_keys)!={'m2','b2','e2'} or missing.unexpected_keys:
        raise ValueError('Deployed HPAC schema differs')
    hpac2.unpack_record_share(record,model)
    other=dict(ref,model=model)
    if s['base'].serialize(other,model)!=(canonical,record):
        raise ValueError('Deployment serialization is not stable')
    with s['torch'].no_grad():
        for frame in (0,199,399,599):
            a,_=s['base'].logits_for(ref,frame)
            b,_=s['base'].logits_for(other,frame)
            if not s['torch'].equal(a,b):
                raise ValueError(f'Quantized training/deployed logits differ: {frame}')
    return other


def measure(s,epoch):
    base,ref=s['base'],s['ref']
    canonical,record=base.serialize(ref,ref['model'])
    other=deployed(s,canonical,record)
    total=0.
    started=time.monotonic()
    for frame in range(600):
        total+=base.probe(other,[frame])/600
        if (frame+1)%100==0:
            sh.log(phase='hpac_proxy',epoch=epoch,frames=frame+1,of=600,
                   seconds=time.monotonic()-started)
    _,nb=base.storage(ref,canonical,record)
    row=dict(epoch=epoch,rate_proxy_bytes=total,stored_entropy_model_bytes=nb,
             combined_proxy_bytes=total+nb,full_sequence=True,
             hpac_sha256=base.sha(canonical),prev2_sha256=base.sha(record),
             changed_hpac_bytes=sum(a!=b for a,b in zip(canonical,ref['canonical'])),
             proxy='fixed-corrector integer-frequency cost; not native coder cost')
    folder=s['out']/f'epoch_{epoch:03d}'
    folder.mkdir()
    base.save_best(ref,folder,canonical,record,row)
    sh.log(phase='hpac_checkpoint',**row)
    return row


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    sh.common_arguments(ap)
    ap.add_argument('--epochs',type=int,default=12)
    ap.add_argument('--lr',type=float,default=.003)
    ap.add_argument('--accum-frames',type=int,default=8)
    ap.add_argument('--native-candidates',type=int,default=3)
    ap.add_argument('--seed',type=int,default=1234)
    args=ap.parse_args()
    if min(args.epochs,args.accum_frames,args.native_candidates)<1 or args.lr<=0:
        ap.error('Positive epochs, accumulation, native-candidates and lr required')
    s=sh.session(args)
    torch,base,ref=s['torch'],s['base'],s['ref']
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    baseline,_=sh.evaluate(s,s['archive'],s['out']/'baseline',expected_tokens=sh.TOKEN_SHA)
    if args.parent_run is None and abs(baseline['score']-.1253025222561036)>2e-6:
        raise ValueError('XP1 baseline score differs from supplied report; inspect baseline/evaluation.json')
    best=baseline
    sh.select(s,best)
    model=ref['model']
    params=[]
    names={}
    for name,p in model.named_parameters():
        trainable=name not in {'frame_embed.weight','m2','b2','e2'} and not name.endswith('exponent')
        p.requires_grad_(trainable)
        if trainable:
            params.append(p)
            names[name]=p.numel()
    if not params:
        raise ValueError('No trainable HPAC weights')
    sh.write_json(s['out']/'training_config.json',dict(epochs=args.epochs,lr=args.lr,
        accum_frames=args.accum_frames,seed=args.seed,trainable=names,
        native_candidates=args.native_candidates,tokens_sha256=sh.TOKEN_SHA))
    optimizer=torch.optim.Adam(params,lr=args.lr)
    updates_per_epoch=math.ceil(600/args.accum_frames)
    scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,
        T_max=updates_per_epoch*args.epochs,eta_min=args.lr/10)
    rng=np.random.default_rng(args.seed)
    rows=[measure(s,0)]
    updates=0
    from torch.nn import functional as F
    for epoch in range(1,args.epochs+1):
        order=rng.permutation(600)
        for start in range(0,600,args.accum_frames):
            frames=order[start:start+args.accum_frames]
            optimizer.zero_grad(set_to_none=True)
            loss_sum=0.
            with torch.enable_grad():
                for frame in frames:
                    logits,target=base.logits_for(ref,int(frame))
                    loss=F.cross_entropy(logits,target)/math.log(2)
                    if not torch.isfinite(loss):
                        raise ValueError('Nonfinite HPAC loss')
                    (loss/len(frames)).backward()
                    loss_sum+=float(loss.detach())
            norm=torch.nn.utils.clip_grad_norm_(params,1.,error_if_nonfinite=True)
            if updates==0 and float(norm)==0:
                raise ValueError('HPAC quantized backward has zero gradient')
            optimizer.step()
            base.project(ref)
            scheduler.step()
            updates+=1
            if updates==1 or updates%10==0:
                row=dict(epoch=epoch,updates=updates,frames_this_epoch=start+len(frames),
                         bits_per_token=loss_sum/len(frames),gradient_norm=float(norm),
                         lr=optimizer.param_groups[0]['lr'])
                with (s['out']/'training.jsonl').open('a') as f:
                    f.write(json.dumps(row)+'\n')
                sh.log(phase='hpac_training',**row)
        rows.append(measure(s,epoch))
        torch.save(dict(epoch=epoch,model=model.state_dict(),optimizer=optimizer.state_dict(),
                        scheduler=scheduler.state_dict()),s['out']/'last.pt')
    # Rank by proxy, deduplicate deployed models, but allow proxy losers into
    # native measurement: adaptive-corrector costs may rank them differently.
    seen={(rows[0]['hpac_sha256'],rows[0]['prev2_sha256'])}
    candidates=[]
    for row in sorted(rows[1:],key=lambda r:r['combined_proxy_bytes']):
        key=(row['hpac_sha256'],row['prev2_sha256'])
        if key not in seen:
            seen.add(key)
            candidates.append(row)
        if len(candidates)>=args.native_candidates:
            break
    for row in candidates:
        folder=s['out']/f"epoch_{row['epoch']:03d}"
        model_archive=folder/'model_input.zip'
        base.pack_input(ref,(folder/'best_hpac.bin').read_bytes(),
                        (folder/'best_prev2.bin').read_bytes(),model_archive)
        candidate=sh.encode(s,model_archive,np.asarray(ref['tokens']),folder/'native')
        if candidate.stat().st_size<best['bytes']:
            # Same target tokens. A full render checks that the packed model
            # change also preserves the real decoder's reconstruction/metrics.
            actual,_=sh.evaluate(s,candidate,folder/'actual',expected_tokens=sh.TOKEN_SHA)
            if actual['score']<best['score']:
                best=actual
                sh.select(s,best)
    sh.write_json(s['out']/'summary.json',dict(baseline=baseline,best=best,
        score_change=best['score']-baseline['score'],native_candidates=len(candidates),
        completed=True))
    sh.log(phase='hpac_finished',best_score=best['score'],best_archive=best['archive'])


if __name__=='__main__':
    main()
