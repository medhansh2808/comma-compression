"""Gradient-shortlisted token edits; exact single-edit metrics; real archive selection."""
import argparse
import hashlib
import math
import time

import numpy as np
import shared as sh


def collect_hook(s, limit):
    """Record rate-saving alternatives from the real baseline arithmetic decode."""
    from container import token_decode
    from profile_reference import frequencies
    masks=[np.flatnonzero(m.cpu().numpy().reshape(-1))
           for m in s['ref']['renderer'].group_masks(s['ref']['device'])]
    rows=[[] for _ in range(600)]
    calls=[0]
    def install():
        from runtime.entropy import rc64
        parent=rc64.NativeDecoder
        class Recorded(parent):
            def decode(self,probabilities):
                symbols=super().decode(probabilities)
                frame,group=divmod(calls[0],len(masks))
                calls[0]+=1
                if frame>=600:
                    raise ValueError('Unexpected number of decoder groups')
                pos=masks[group]
                expected=np.asarray(s['ref']['tokens'][frame]).reshape(-1)[pos]
                if not np.array_equal(symbols,expected):
                    raise ValueError('Baseline decode differs from corrected cache')
                f=frequencies(probabilities,allow_zero=True)
                alt=f.argmax(1)
                gain=np.log2(f[np.arange(len(f)),alt]/f[np.arange(len(f)),symbols])
                keep=np.flatnonzero((alt!=symbols)&(gain>.25))
                for k in keep:
                    rows[frame].append((int(pos[k]),int(alt[k]),float(gain[k])))
                return symbols
        rc64.NativeDecoder=Recorded
    token_decode.POST_PURGE_HOOKS.append(install)
    def finish():
        token_decode.POST_PURGE_HOOKS.remove(install)
        if calls[0]!=600*len(masks):
            raise ValueError('Incomplete probability trace')
        return [sorted(r,key=lambda x:x[2],reverse=True)[:limit] for r in rows]
    return finish


class FrameModel:
    def __init__(self,s,raw,baseline):
        self.s=s
        self.t=s['torch']
        self.raw=np.memmap(raw,dtype=np.uint8,mode='r',shape=sh.RAW_SHAPE)
        self.baseline=baseline
        from runtime.f26_inflate import _load_renderer
        from decoder import extfmt
        r=_load_renderer(s['rt']/'cpr1')
        self.renderer=r
        extensions=extfmt.parse_payload(s['ref']['extension'])
        if extensions['ext2'] is not None:
            raise ValueError('This proposal model does not support EXT2 masters')
        self.sem=r.SemanticTokenRenderer(96)
        state=r.unpack_variant_semantic_or_none(s['ref']['parts'].semantic_blob,self.sem.state_dict())
        if state is None:
            raise ValueError('Expected the reference semantic renderer format')
        self.sem.load_state_dict(state,strict=True)
        self.sem=self.sem.eval().to('cuda')
        for p in self.sem.parameters():
            p.requires_grad_(False)
        from frame_utils import rgb_to_yuv6
        if not hasattr(rgb_to_yuv6,'__wrapped__'):
            raise ValueError('Expected differentiable yuv6 implementation under no_grad wrapper')
        self.yuv6=rgb_to_yuv6.__wrapped__

    def render(self,labels,frame,embedding=None,ste=False):
        t=self.t
        from torch.nn import functional as F
        handle=None
        if embedding is not None:
            handle=self.sem.token_embed.register_forward_hook(lambda module,inputs,result:embedding)
        try:
            rgb=self.sem(labels,t.tensor([frame],device='cuda'))
        finally:
            if handle is not None:
                handle.remove()
        master=F.interpolate(rgb,size=(874,1164),mode='bilinear',align_corners=False).clamp(0,255)
        return master+(master.round()-master).detach() if ste else master.round()

    def slave(self,frame):
        return self.t.from_numpy(np.array(self.raw[2*frame])).permute(2,0,1)[None].to('cuda').float()

    def differentiable_metrics(self,master,slave):
        from torch.nn import functional as F
        seg=self.s['net'].segnet(F.interpolate(master,size=(384,512),mode='bilinear'))
        pair=self.t.stack([slave,master],1).reshape(2,3,874,1164)
        x=self.yuv6(F.interpolate(pair,size=(384,512),mode='bilinear'))
        pose=self.s['net'].posenet(x.reshape(1,2*x.shape[1],*x.shape[-2:]))['pose'][...,:6]
        return seg,pose

    def gradient_rank(self,frame,proposals):
        t=self.t
        labels=t.from_numpy(np.array(self.s['ref']['tokens'][frame],dtype=np.int64))[None].to('cuda')
        with t.enable_grad():
            embedding=self.sem.token_embed(labels).detach().requires_grad_(True)
            master=self.render(labels,frame,embedding=embedding,ste=True)
            saved=t.from_numpy(np.array(self.raw[2*frame+1])).permute(2,0,1)[None].to('cuda')
            if not t.equal(master.detach().to(t.uint8),saved):
                raise ValueError(f'Proposal renderer differs from actual master at frame {frame}')
            slave=self.slave(frame)
            seg,pose=self.differentiable_metrics(master,slave)
            # Verify the differentiable preprocessing against the actual metric.
            pair=t.stack([slave,master.detach()],1).to(t.uint8).permute(0,1,3,4,2)
            actual_pose,actual_seg=self.s['common'].outputs(self.s['net'],pair)
            if not t.equal(seg.detach().argmax(1).to(t.uint8),actual_seg):
                raise ValueError('Differentiable SegNet preprocessing does not match evaluation')
            if not t.allclose(pose.detach(),actual_pose,rtol=1e-5,atol=1e-6):
                raise ValueError('Differentiable PoseNet preprocessing does not match evaluation')
            target=t.from_numpy(self.s['gs'][frame].astype(np.int64))[None].to('cuda')
            correct=seg.gather(1,target[:,None]).squeeze(1)
            other=seg.masked_fill(t.nn.functional.one_hot(target,5).permute(0,3,1,2).bool(),-float('inf')).amax(1)
            # Smooth approximation to a label disagreement; exact mismatches
            # will be checked for every shortlisted individual edit.
            seg_surrogate=t.sigmoid(other-correct).mean()
            gp=t.from_numpy(self.s['gp'][frame])[None].to('cuda')
            pose_mse=(pose.float()-gp).square().mean()
            pose_slope=5/math.sqrt(max(10*self.baseline['pose'],1e-20))
            loss=100/600*seg_surrogate+pose_slope/600*pose_mse
            grad=t.autograd.grad(loss,embedding)[0][0]
            if not t.isfinite(grad).all():
                raise ValueError('Nonfinite token gradient')
            pos=np.array([p[0] for p in proposals])
            alt=np.array([p[1] for p in proposals])
            yy=t.as_tensor(pos//512,device='cuda')
            xx=t.as_tensor(pos%512,device='cuda')
            aa=t.as_tensor(alt,device='cuda')
            old=labels[0,yy,xx]
            delta=self.sem.token_embed.weight[aa]-self.sem.token_embed.weight[old]
            estimate=(grad[yy,xx]*delta).sum(1).detach().cpu().numpy()
        return sorted([(float(d)-sh.RATE_PER_BIT*p[2],p,float(d))
                       for d,p in zip(estimate,proposals)],key=lambda x:x[0])

    def exact_pair(self,frame,labels):
        t=self.t
        with t.no_grad():
            master=self.render(t.from_numpy(labels.astype(np.int64))[None].to('cuda'),frame)
            pair=t.stack([self.slave(frame),master],1).to(t.uint8).permute(0,1,3,4,2)
            pose,seg=self.s['common'].outputs(self.s['net'],pair)
            target=t.from_numpy(self.s['gs'][frame])[None].to('cuda')
            mismatch=float((seg!=target).double().mean())
            gp=t.from_numpy(self.s['gp'][frame])[None].to('cuda')
            mse=float((pose.float()-gp).square().mean())
        return mismatch,mse


def independent_change(baseline,old_seg,old_pose,new_seg,new_pose):
    pose=max(0.,baseline['pose']+(new_pose-old_pose)/600)
    return 100*(new_seg-old_seg)/600+math.sqrt(10*pose)-math.sqrt(10*baseline['pose'])


def choose_edits(rows,budget,radius=8):
    """Greedy ranking with spatial spacing to reduce same-frame interactions."""
    selected=[]
    positions={}
    for row in sorted(rows,key=lambda r:r['estimated_score_change']):
        if row['estimated_score_change']>=0:
            continue
        frame,pos=row['frame'],row['position']
        y,x=divmod(pos,512)
        previous=positions.setdefault(frame,[])
        if any(max(abs(y-a),abs(x-b))<=radius for a,b in previous):
            continue
        selected.append(row)
        previous.append((y,x))
        if len(selected)>=budget:
            break
    return selected


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    sh.common_arguments(ap)
    ap.add_argument('--candidates-per-frame',type=int,default=512)
    ap.add_argument('--shortlist-per-frame',type=int,default=12)
    ap.add_argument('--budgets',type=int,nargs='+',default=[128,512,2048,4096])
    args=ap.parse_args()
    if min(args.candidates_per_frame,args.shortlist_per_frame,*args.budgets)<1:
        ap.error('Candidate counts and budgets must be positive')
    s=sh.session(args)
    finish=collect_hook(s,args.candidates_per_frame)
    baseline,raw=sh.evaluate(s,s['archive'],s['out']/'baseline',expected_tokens=sh.TOKEN_SHA,keep_raw=True)
    proposals=finish()
    sh.select(s,baseline)
    np.savez_compressed(s['out']/'rate_proposals.npz',
        frame=np.array([f for f,p in enumerate(proposals) for _ in p],dtype=np.int16),
        position=np.array([x[0] for p in proposals for x in p],dtype=np.int32),
        alternative=np.array([x[1] for p in proposals for x in p],dtype=np.uint8),
        estimated_bits_saved=np.array([x[2] for p in proposals for x in p],dtype=np.float32))
    model=FrameModel(s,raw,baseline)
    per=np.load(s['out']/'baseline/perpair.npz')
    old_seg,old_pose=per['seg'].copy(),per['pose'].copy()
    per.close()
    candidates=[]
    started=time.monotonic()
    for frame in range(600):
        if proposals[frame]:
            ranked=model.gradient_rank(frame,proposals[frame])[:args.shortlist_per_frame]
            labels=np.array(s['ref']['tokens'][frame])
            for _,(pos,alt,gain),surrogate in ranked:
                edited=labels.copy()
                edited.reshape(-1)[pos]=alt
                sd,pd=model.exact_pair(frame,edited)
                dd=independent_change(baseline,float(old_seg[frame]),float(old_pose[frame]),sd,pd)
                candidates.append(dict(frame=frame,position=pos,old=int(labels.reshape(-1)[pos]),
                    alternative=alt,estimated_bits_saved=gain,surrogate_distortion_change=surrogate,
                    exact_single_edit_distortion_change=dd,
                    estimated_score_change=dd-sh.RATE_PER_BIT*gain))
        if (frame+1)%10==0:
            sh.log(phase='token_proposals',frames=frame+1,of=600,tested=len(candidates),
                   promising=sum(r['estimated_score_change']<0 for r in candidates),
                   seconds=time.monotonic()-started)
        if (frame+1)%50==0:
            sh.write_json(s['out']/'proposal_checks.json',candidates)
    sh.write_json(s['out']/'proposal_checks.json',candidates)
    del model.raw
    raw.unlink()
    best=baseline
    tried=set()
    results=[]
    for budget in sorted(set(args.budgets)):
        edits=choose_edits(candidates,budget)
        if not edits or len(edits) in tried:
            continue
        tried.add(len(edits))
        folder=s['out']/f'edits_{len(edits):05d}'
        folder.mkdir()
        sh.write_json(folder/'edits.json',edits)
        tokens=np.array(s['ref']['tokens'])
        for e in edits:
            tokens[e['frame']].reshape(-1)[e['position']]=e['alternative']
        expected=hashlib.sha256(tokens.tobytes()).hexdigest()
        archive=sh.encode(s,s['archive'],tokens,folder/'native')
        actual,_=sh.evaluate(s,archive,folder/'actual',expected_tokens=expected)
        actual['edited_tokens']=len(edits)
        results.append(actual)
        if actual['score']<best['score']:
            best=actual
            sh.select(s,best)
    sh.write_json(s['out']/'summary.json',dict(baseline=baseline,best=best,candidates=results,
        score_change=best['score']-baseline['score'],completed=True))
    sh.log(phase='token_rdo_finished',best_score=best['score'],best_archive=best['archive'])


if __name__=='__main__':
    main()
