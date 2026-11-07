"""Train/evaluate a bounded context adapter. Probability proxy only; no archive claim."""
import argparse
import json
import math
from pathlib import Path
import sys
import time
import numpy as np
from model import D,H,SHAPES,initial,pack,unpack,features,apply,predict,selftest
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'experiments/rdo151'))


def frame_data(ref,frame):
    import train_entropy as te
    with ref['torch'].no_grad():
        logits,target=te.logits_for(ref,frame)
        _,_,p1,p2,boundary=te.sample(ref,frame)
    logits=logits[0].permute(1,2,0).reshape(-1,5).cpu().numpy()
    x,gate=features(logits,p1.cpu().numpy(),p2.cpu().numpy(),boundary.cpu().numpy())
    return logits,x,gate,target.cpu().numpy().reshape(-1)


def table_bits(ref,logits,target):
    from runtime.residual_archive import _probability_table
    from profile_reference import frequencies
    f=frequencies(_probability_table(logits,ref['renderer'].HPAC_LOGIT_PRECISION),allow_zero=True)
    return float(-np.log2(f[np.arange(len(f)),target]/float(1<<31)).sum())


def measure(ref,frames,arrays,label):
    base=adapted=0.;started=time.monotonic()
    for i,frame in enumerate(frames,1):
        logits,x,gate,target=frame_data(ref,frame)
        base+=table_bits(ref,logits,target)
        adapted+=table_bits(ref,apply(logits,x,gate,arrays),target)
        if i%50==0 or i==len(frames):print(json.dumps(dict(phase=label,frame=i,of=len(frames),elapsed_seconds=time.monotonic()-started)),flush=True)
    factor=600/len(frames)/8
    return dict(frames_measured=len(frames),reference_proxy_bytes=base*factor,
        adapter_proxy_bytes=adapted*factor,adapter_model_bytes=len(pack(arrays)),
        estimated_net_saving_bytes=(base-adapted)*factor-len(pack(arrays)),
        full_sequence=len(frames)==600,actual_archive_evaluated=False)


def torch_model(torch,device):
    class Adapter(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.values=torch.nn.ParameterList([torch.nn.Parameter(torch.tensor(a,dtype=torch.float32,device=device)) for a in initial()])
        def q(self,x):
            rounded=x.clamp(-127,127).round()
            return x+(rounded-x).detach()
        def round(self,x):return x+(x.round()-x).detach()
        def forward(self,x):
            w1,b1,w2,b2=[self.q(a) for a in self.values]
            h=(self.round(x@w1.T/128)+b1).clamp(0,127)
            return (self.round(h@w2.T/128)+b2).clamp(-32,32)
        def arrays(self):return [p.detach().clamp(-127,127).round().cpu().numpy().astype(np.int16) for p in self.values]
    return Adapter()


def check(ref):
    import train_entropy as te
    te.check(ref)
    if ref['renderer'].HPAC_LOGIT_PRECISION!=8:raise ValueError('expected 1/8 logit grid')
    torch=ref['torch'];m=torch_model(torch,ref['device'])
    logits,x,g,target=frame_data(ref,0)
    if not np.array_equal(apply(logits,x,g,m.arrays()),logits):raise ValueError('zero parity')
    # Exercise nonzero parameters; compare training forward against integer deployment.
    with torch.no_grad():
        m.values[2].copy_(torch.arange(40,device=ref['device']).reshape(5,8)-20)
        m.values[3].copy_(torch.arange(5,device=ref['device'])-2)
    testx=x[:2048]
    with torch.no_grad():out=m(torch.tensor(testx,dtype=torch.float32,device=ref['device'])).cpu().numpy()
    if not np.array_equal(out,predict(testx,m.arrays())):raise ValueError('torch/integer forward mismatch')
    m0=torch_model(torch,ref['device'])
    xt=torch.tensor(testx,dtype=torch.float32,device=ref['device'])
    bt=torch.tensor(logits[:len(testx)],device=ref['device'])
    gt=torch.tensor(g[:len(testx),None],device=ref['device'],dtype=torch.float32)
    yt=torch.tensor(target[:len(testx)].astype(np.int64),device=ref['device'])
    loss=torch.nn.functional.cross_entropy(bt+m0(xt)*gt/8,yt)
    loss.backward()
    gradients=[p.grad for p in m0.parameters() if p.grad is not None]
    if not gradients or not all(torch.isfinite(v).all() for v in gradients) or not any(v.abs().max()>0 for v in gradients):
        raise ValueError('adapter gradients are nonfinite or all zero')
    # Upstream causal masking was checked by te.check; adapter has no target-map argument.
    print('PASS: reference zero parity; torch/integer deployment parity; finite nonzero gradients; upstream causal sparse checks',flush=True)


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('command',choices=('check','train','measure'))
    ap.add_argument('--device',choices=('cpu','cuda'),default='cuda')
    ap.add_argument('--tokens',type=Path,default=ROOT/'experiments/rdo151/cache/tokens151.npy')
    ap.add_argument('--output',type=Path,default=ROOT/'experiments/context_adapter/runs/v1')
    ap.add_argument('--steps',type=int,default=2000)
    ap.add_argument('--lr',type=float,default=.05,help='Adam step in integer code units')
    ap.add_argument('--batch',type=int,default=16384)
    ap.add_argument('--resume',action='store_true')
    args=ap.parse_args();args.tokens=args.tokens.resolve();args.output=args.output.resolve()
    if args.steps<1 or args.batch<1 or not math.isfinite(args.lr) or args.lr<=0:ap.error('invalid training settings')
    selftest()
    import train_entropy as te
    ref=te.setup(args.device,args.tokens)
    for p in ref['model'].parameters():p.requires_grad_(False)
    check(ref)
    if args.command=='check':return
    if args.command=='measure':
        arrays=unpack((args.output/'best.cad').read_bytes())
        result=measure(ref,list(range(600)),arrays,'full_sequence_measurement')
        (args.output/'full_measurement.json').write_text(json.dumps(result,indent=2)+'\n')
        print(json.dumps(result,indent=2));return
    torch=ref['torch'];m=torch_model(torch,ref['device'])
    optimizer=torch.optim.Adam(m.parameters(),lr=args.lr)
    rng=np.random.default_rng(1234);start=0
    probe=np.linspace(0,599,16,dtype=int).tolist()
    cfg=dict(archive_sha256=te.ARCHIVE_SHA,tokens_sha256=te.TOKEN_SHA,lr=args.lr,batch=args.batch,probe=probe,model='CAD1-25x8x5')
    if args.resume:
        saved=torch.load(args.output/'checkpoint.pt',map_location=ref['device'],weights_only=True)
        if saved['config']!=cfg:raise ValueError('resume config differs')
        m.load_state_dict(saved['model']);optimizer.load_state_dict(saved['optimizer'])
        rng.bit_generator.state=json.loads(saved['rng']);start=saved['step']
        best=json.loads((args.output/'best.json').read_text())
    else:
        args.output.mkdir(parents=True,exist_ok=False)
        (args.output/'config.json').write_text(json.dumps(cfg,indent=2)+'\n')
        (args.output/'best.cad').write_bytes(pack(m.arrays()))
        best=dict(step=0,is_reference=True,estimated_net_saving_bytes=0.,actual_archive_evaluated=False)
        (args.output/'best.json').write_text(json.dumps(best,indent=2)+'\n')
    started=time.monotonic()
    for step in range(start,args.steps):
        frame=int(rng.integers(600));logits,x,gate,target=frame_data(ref,frame)
        ids=rng.integers(len(x),size=args.batch)
        xt=torch.tensor(x[ids],device=ref['device'],dtype=torch.float32)
        base=torch.tensor(logits[ids],device=ref['device'])
        gt=torch.tensor(gate[ids,None],device=ref['device'],dtype=torch.float32)
        yt=torch.tensor(target[ids].astype(np.int64),device=ref['device'])
        adapted=(base+m(xt)*gt/8).clamp(-4096,32767/8)
        logp=torch.nn.functional.log_softmax(adapted,dim=1).gather(1,yt[:,None]).squeeze(1)
        capped=logp+(logp.clamp(min=-31*math.log(2))-logp).detach()
        loss=-capped.mean()/math.log(2)
        optimizer.zero_grad(set_to_none=True);loss.backward()
        if not all(p.grad is None or torch.isfinite(p.grad).all() for p in m.parameters()):raise ValueError('nonfinite gradient')
        torch.nn.utils.clip_grad_norm_(m.parameters(),1.)
        optimizer.step()
        with torch.no_grad():
            for p in m.parameters():p.clamp_(-127,127)
        if (step+1)%50==0 or step==start:
            row=dict(step=step+1,frame=frame,bits_per_token=float(loss.detach()),elapsed_seconds=time.monotonic()-started)
            with (args.output/'training.jsonl').open('a') as f:f.write(json.dumps(row)+'\n')
            print(json.dumps(row),flush=True)
        if (step+1)%250==0 or step+1==args.steps:
            blob=pack(m.arrays());arrays=unpack(blob)
            result=measure(ref,probe,arrays,'serialized_probe')
            result['step']=step+1
            with (args.output/'probes.jsonl').open('a') as f:f.write(json.dumps(result)+'\n')
            if result['estimated_net_saving_bytes']>best['estimated_net_saving_bytes']:
                best=dict(result,is_reference=False)
                (args.output/'best.cad').write_bytes(blob)
                (args.output/'best.json').write_text(json.dumps(best,indent=2)+'\n')
            tmp=args.output/'checkpoint.tmp'
            torch.save(dict(step=step+1,model=m.state_dict(),optimizer=optimizer.state_dict(),rng=json.dumps(rng.bit_generator.state),config=cfg),tmp)
            tmp.replace(args.output/'checkpoint.pt')
            print(json.dumps(dict(probe=result,best_step=best['step'])),flush=True)
    print('Training finished. Run measure separately for all 600 frames. No new archive has been produced.',flush=True)

if __name__=='__main__':main()
