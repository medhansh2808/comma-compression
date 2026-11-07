"""Quantization-aware rate fine-tuning of #151's existing HPAC model.

Tokens and RGB-generating state stay fixed. Training uses the pre-corrector
conditional-rate proxy. Actual total archive size and exact decoding decide
success. No new decoder architecture or free parameter side-channel.
"""
import argparse
import hashlib
import importlib
import json
import math
import os
from pathlib import Path
import struct
import subprocess
import sys
import tempfile
import time
import types

import brotli
import numpy as np

ROOT=Path(__file__).resolve().parents[2]
SUB=ROOT/"experiments/rdo151/work/reference"
RUNTIME=SUB/"external/pr141"
ARCHIVE=ROOT/"experiments/rdo151/cache/leader151.zip"
TOKENS=ROOT/"experiments/rdo151/cache/tokens151.npy"
ARCHIVE_SHA="9e2de76d931c5a7aca29f7e9008c4d77b34ec037d042113e2fd082607391f315"
TOKEN_SHA="092ff87541fe0f4bdbff07759ae84cafd14e7e7e6f1385334d4177ef8c941d77"
HEADER=struct.Struct("<4sBBBBHHH")

def sha(b):return hashlib.sha256(b).hexdigest()

def setup(device,tokens_path):
    if device=="cpu":os.environ["CUDA_VISIBLE_DEVICES"]=""
    import torch
    torch.set_num_threads(4)
    if device=="cuda" and not torch.cuda.is_available():raise ValueError("CUDA unavailable")
    if sha(ARCHIVE.read_bytes())!=ARCHIVE_SHA:raise ValueError("reference archive differs")
    tokens=np.load(tokens_path,mmap_mode="r",allow_pickle=False)
    if tokens.shape!=(600,384,512) or tokens.dtype!=np.uint8 or sha(tokens.tobytes())!=TOKEN_SHA:
        raise ValueError("reference token cache differs")
    sys.path[:0]=[str(SUB),str(RUNTIME)]
    os.chdir(SUB)
    from canonical.inflate_canonical import install_rp1_shim
    install_rp1_shim()
    from decoder import numerics,extfmt
    from decoder.inflate_core import _load
    from runtime.f26_inflate import _load_renderer,_configure_device
    from runtime import ihs2
    from hpac import rowpack
    from container import hpac2
    from container.warp_signal import decode_warp_signal
    from container.payload_format import read_archive,split_inner_payload
    parts,warp,payload=_load(ARCHIVE)
    renderer=_load_renderer(RUNTIME/"cpr1")
    dev,_=_configure_device(torch,device,4)
    layout=ihs2.layout_from_runtime(renderer)
    canonical=rowpack.inverse(parts.hpac_blob,layout) if rowpack.is_rowpack(parts.hpac_blob,layout) else parts.hpac_blob
    if not canonical.startswith(b"IHS1"):raise ValueError("expected IHS1 state")
    record=extfmt.parse_payload(payload)["prev2"]
    if record is None or record[0]!=hpac2.EXT_TAG_SHARE:raise ValueError("expected shared temporal branch")
    base=renderer.load_hpac(canonical,dev)
    integer=importlib.import_module("hpac_integer")
    kwargs=dict(num_pairs=renderer.N,num_classes=renderer.NUM_CLASSES,patch=renderer.HPAC_PATCH,
        delta=renderer.HPAC_DELTA,channels=renderer.HPAC_CHANNELS,frame_dim=renderer.HPAC_FILM_DIM,
        norm_mode="none",activation="relu",use_frame_scale=True,weight_bound=127,activation_bound=127,
        use_weight_scales=True,weight_exponent_min=-6,use_spm=True,use_norm_gates=False)
    cls=hpac2.make_hpacs_class(integer)
    model=cls(**kwargs).to(dev).eval()
    missing=model.load_state_dict(base.state_dict(),strict=False)
    if set(missing.missing_keys)!={"m2","b2","e2"} or missing.unexpected_keys:
        raise ValueError("model state layout differs")
    hpac2.unpack_record_share(record,model)
    depths=ihs2._unpack_nibbles(canonical[4:4+layout.depth_bytes],layout.depth_count)
    core,w2=split_inner_payload(read_archive(ARCHIVE))
    return dict(torch=torch,device=dev,tokens=tokens,model=model,model_class=cls,kwargs=kwargs,
        parts=parts,renderer=renderer,layout=layout,canonical=canonical,depths=depths,
        record=record,warp_ids=decode_warp_signal(warp,600),core=core,signal=w2,
        extension=payload,warp=warp)

def serialize(ref,model):
    """Same IHS1 row widths and schema as reference; exact integer values."""
    from runtime import ihs2
    from integer_model_io import _weight_rows
    from container.hpac2 import pack_record_share
    modules=dict(model.named_modules());params=dict(model.named_parameters())
    rows=[]
    for name,start,end in ref["layout"].module_ranges:
        module=modules[name]
        w,_,_=module.codes()
        rows.extend(r.detach().cpu().numpy().astype(np.int16) for r in _weight_rows(module,w))
    head=ref["canonical"][:4+ref["layout"].depth_bytes]
    body=ihs2._write_signed_rows(tuple(rows),ref["depths"],ref["layout"])
    tail=[]
    for field in ref["layout"].tail_fields:
        values=params[field.name].detach().round().cpu().numpy()
        dtype=np.dtype(field.dtype)
        if np.any(values<np.iinfo(dtype).min) or np.any(values>np.iinfo(dtype).max):raise ValueError("tail overflow")
        tail.append(values.astype(dtype).tobytes())
    return head+body+b"".join(tail),pack_record_share(model)

def storage(ref,canonical,record):
    from hpac import rowpack
    from container.payload_format import read_archive
    from runtime.block_container import decode_model_blocks
    models,_=decode_model_blocks(ref["core"])
    fields=HEADER.unpack_from(models)
    old_stream=models[HEADER.size:HEADER.size+fields[-3]]
    if canonical==ref["canonical"]:
        stream=old_stream
    else:
        packed=rowpack.forward(canonical,ref["layout"])
        if rowpack.inverse(packed,ref["layout"])!=canonical:raise ValueError("HPAC rowpack failed")
        stream=min((brotli.compress(packed,quality=11,mode=mode,lgwin=win)
                    for mode in (brotli.MODE_GENERIC,brotli.MODE_TEXT,brotli.MODE_FONT)
                    for win in (16,24)),key=len)
    return stream,len(stream)+len(record)

def pack_input(ref,canonical,record,path):
    from runtime.block_container import decode_model_blocks
    from container.payload_format import write_archive
    from container.archive_writer import EXTERNAL_ATTR_PR141
    from decoder import extfmt
    models,tail=decode_model_blocks(ref["core"])
    fields=list(HEADER.unpack_from(models))
    if fields[:3]!=[b"RX1M",1,2] or fields[4]!=24:raise ValueError("model header differs")
    stream,_=storage(ref,canonical,record)
    old_hpac_len=fields[-3]
    unchanged=models[HEADER.size+old_hpac_len:]
    fields[-3]=len(stream)
    models=HEADER.pack(*fields)+stream+unchanged
    prefix=b"BLK2"+struct.pack("<IB",len(models),1)+bytes([0])+len(models).to_bytes(3,"little")+models
    if ref["extension"].count(ref["record"])!=1:raise ValueError("temporal extension not unique")
    payload=ref["extension"].replace(ref["record"],record)
    signal=extfmt.join_trailer(ref["warp"],payload)
    write_archive(path,prefix+tail+signal+struct.pack("<H",len(signal)),EXTERNAL_ATTR_PR141)

def project(ref):
    torch=ref["torch"];model=ref["model"];modules=dict(model.named_modules())
    with torch.no_grad():
        for name,start,end in ref["layout"].module_ranges:
            module=modules[name]
            d=torch.as_tensor(ref["depths"][start:end].astype(np.int64),device=ref["device"])
            lo=torch.where(d>0,-torch.pow(2.,d-1),0.)
            hi=torch.where(d>0,torch.pow(2.,d-1)-1,0.)
            shape=(len(d),)+(1,)*(module.weight.ndim-1)
            module.weight.clamp_(lo.reshape(shape),hi.reshape(shape))
            module.bias.clamp_(-32768,32767)
            if hasattr(module,"exponent"):module.exponent.clamp_(module.exponent_min,0)
        model.m2.clamp_(-127,127);model.b2.clamp_(-32768,32767);model.e2.clamp_(-8,0)

def sample(ref,frame):
    torch=ref["torch"]
    from container.context_warp import apply_warp
    from runtime.residual_archive import _boundary_buckets
    tokens=ref["tokens"];dev=ref["device"]
    current=torch.from_numpy(np.array(tokens[frame],dtype=np.int64))[None].to(dev)
    zero=torch.zeros_like(current)
    p1=torch.from_numpy(np.array(tokens[frame-1],dtype=np.int64))[None].to(dev) if frame else zero
    p2=torch.from_numpy(np.array(tokens[frame-2],dtype=np.int64))[None].to(dev) if frame>=2 else zero
    boundary=_boundary_buckets(np.asarray(tokens[frame-1])) if frame else np.full(tokens.shape[1:],4,np.uint8)
    warp=ref["warp_ids"][frame]
    return current,torch.tensor([frame],device=dev),apply_warp(p1,warp),apply_warp(p2,warp),torch.from_numpy(boundary.astype(np.int64)).to(dev)

def logits_for(ref,frame):
    current,idx,p1,p2,boundary=sample(ref,frame)
    context=ref["model"].prepare_frame_context(idx,p1,p2)
    logits=ref["model"].cached_context_logits(current,context)
    predicted=logits.detach().argmax(1)[0]
    table=ref["torch"].as_tensor(ref["parts"].table.values,device=ref["device"])
    corrections=table[boundary*5+predicted].permute(2,0,1)[None]
    corrected=logits+corrections
    precision=ref["renderer"].HPAC_LOGIT_PRECISION
    quantized=(corrected*precision).round().clamp(-32768,32767)/precision
    corrected=corrected+(quantized-corrected).detach()
    return corrected,current

def probe(ref,frames):
    from runtime.residual_archive import _probability_table
    from profile_reference import frequencies
    bits=0
    with ref["torch"].no_grad():
        for frame in frames:
            logits,target=logits_for(ref,int(frame))
            flat=logits[0].permute(1,2,0).reshape(-1,5).cpu().numpy()
            q=_probability_table(flat,ref["renderer"].HPAC_LOGIT_PRECISION)
            f=frequencies(q,allow_zero=True)
            symbols=target.cpu().numpy().reshape(-1)
            bits+=float(-np.log2(f[np.arange(len(f)),symbols]/float(1<<31)).sum())
    return bits/len(frames)*len(ref["tokens"])/8

def dense_sparse_check(ref):
    torch=ref["torch"]
    from runtime.residual_archive import _sparse_class
    from runtime.hpac_inference import optimize_sparse_evaluator
    other=ref["model_class"](**ref["kwargs"]).to(ref["device"]).eval()
    other.load_state_dict(ref["model"].state_dict())
    sparse=_sparse_class(RUNTIME/"cpr1")(other,384,512)
    optimize_sparse_evaluator(sparse)
    current,idx,p1,p2,_=sample(ref,0)
    with torch.no_grad():
        context=ref["model"].prepare_frame_context(idx,p1,p2)
        dense=ref["model"].cached_context_logits(current,context)[0].permute(1,2,0).reshape(-1,5)
        ctx2=other.prepare_frame_context(idx,p1,p2)
        masks=ref["renderer"].group_masks(ref["device"])
        order=torch.empty((384,512),device=ref["device"],dtype=torch.long)
        for group,mask in enumerate(masks):order[mask]=group
        for group in (0,len(masks)//3,2*len(masks)//3,len(masks)-1):
            positions=masks[group].reshape(-1).nonzero().reshape(-1)
            partial=current.masked_fill((order>=group)[None],0)
            actual=sparse.selected_logits(partial,ctx2,group)
            if not torch.equal(actual,dense[positions]):raise ValueError(f"dense/sparse quantized logits differ in group {group}")
    del sparse,other

def check(ref):
    canonical,record=serialize(ref,ref["model"])
    if canonical!=ref["canonical"] or record!=ref["record"]:raise ValueError("initial serialization not exact")
    with tempfile.TemporaryDirectory() as d:
        p=Path(d)/"archive.zip";pack_input(ref,canonical,record,p)
        if p.read_bytes()!=ARCHIVE.read_bytes():raise ValueError("initial container rebuild not exact")
    dense_sparse_check(ref)
    print("PASS: exact HPAC serialization, temporal serialization, complete archive rebuild, dense/sparse causal logits",flush=True)

def save_best(ref,out,canonical,record,metrics):
    (out/"best_hpac.bin").write_bytes(canonical)
    (out/"best_prev2.bin").write_bytes(record)
    (out/"best.json").write_text(json.dumps(metrics,indent=2)+"\n")

def train(args,ref):
    torch=ref["torch"]
    from torch.nn import functional as F
    out=args.output
    checkpoint=out/"checkpoint.pt"
    if checkpoint.exists() and not args.resume:raise ValueError("checkpoint exists; use --resume")
    out.mkdir(parents=True,exist_ok=True)
    model=ref["model"]
    for name,p in model.named_parameters():p.requires_grad_(name!="frame_embed.weight")
    optimizer=torch.optim.Adam([p for p in model.parameters() if p.requires_grad],lr=args.lr)
    rng=torch.Generator().manual_seed(args.seed)
    frames=np.unique(np.linspace(0,599,args.probe_frames).astype(int)).tolist()
    start_step=0
    config=dict(archive_sha256=ARCHIVE_SHA,tokens_sha256=TOKEN_SHA,probe_frames=frames,
                lr=args.lr,seed=args.seed,weight_precision="fixed reference row widths",frame_embeddings_frozen=True)
    if args.resume:
        saved=torch.load(checkpoint,map_location=ref["device"],weights_only=True)
        if saved["config"]!=config:raise ValueError("resume settings differ")
        model.load_state_dict(saved["model"]);optimizer.load_state_dict(saved["optimizer"])
        rng.set_state(saved["rng"].cpu());start_step=saved["step"]
        best=json.loads((out/"best.json").read_text())
    else:
        print("Measuring deterministic initial rate probe...",flush=True)
        rate=probe(ref,frames)
        canonical,record=serialize(ref,model)
        _,model_bytes=storage(ref,canonical,record)
        best=dict(step=0,rate_proxy_bytes=rate,stored_entropy_model_bytes=model_bytes,
                  combined_proxy_bytes=rate+model_bytes,is_reference=True,config=config)
        save_best(ref,out,canonical,record,best)
    initial_parameters=sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(json.dumps(dict(trainable_parameters=initial_parameters,baseline_probe=best,
          objective="quantized conditional rate proxy; final native archive bytes decide success")),flush=True)
    started=time.monotonic()
    for step in range(start_step,args.steps):
        frame=int(torch.randint(600,(1,),generator=rng).item())
        logits,target=logits_for(ref,frame)
        logp=F.log_softmax(logits,dim=1).gather(1,target[:,None]).squeeze(1)
        capped=logp+(logp.clamp(min=-31*math.log(2))-logp).detach()
        loss=-capped.mean()/math.log(2)
        optimizer.zero_grad(set_to_none=True);loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(),1.)
        optimizer.step();project(ref)
        if (step+1)%args.log_every==0 or step==start_step:
            entry=dict(step=step+1,frame=frame,conditional_bits_per_token=float(loss.detach()),elapsed_seconds=time.monotonic()-started)
            with (out/"training.jsonl").open("a") as f:f.write(json.dumps(entry)+"\n")
            print(json.dumps(entry),flush=True)
        if (step+1)%args.probe_every==0 or step+1==args.steps:
            rate=probe(ref,frames)
            canonical,record=serialize(ref,model)
            _,model_bytes=storage(ref,canonical,record)
            entry=dict(step=step+1,rate_proxy_bytes=rate,stored_entropy_model_bytes=model_bytes,
                       combined_proxy_bytes=rate+model_bytes,is_reference=canonical==ref["canonical"] and record==ref["record"],config=config)
            with (out/"probes.jsonl").open("a") as f:f.write(json.dumps(entry)+"\n")
            if entry["combined_proxy_bytes"]<best["combined_proxy_bytes"]:
                best=entry;save_best(ref,out,canonical,record,best)
                print("Saved new best quantized-rate candidate",flush=True)
            tmp=checkpoint.with_suffix(".tmp")
            torch.save(dict(step=step+1,config=config,model=model.state_dict(),optimizer=optimizer.state_dict(),rng=rng.get_state()),tmp)
            tmp.replace(checkpoint)
            print(json.dumps(dict(probe=entry,best_step=best["step"])),flush=True)
    print(f"Training complete. Best candidate metadata: {out/'best.json'}",flush=True)

def verify_decoder(archive,tokens,device):
    from container import token_decode
    source_path=SUB/"container/token_decode.py"
    source=source_path.read_text()
    old="                symbols = decoder.decode(row64).astype(np.int64)"
    new=old+'\n                if not np.array_equal(symbols, EXPECTED[frame].reshape(-1)[flat_positions]):\n                    raise ValueError(f"token round-trip failed at frame {frame}, group {group}")'
    if source.count(old)!=1:raise ValueError("decoder source differs")
    module=types.ModuleType("container._training_verifier");module.__file__=str(source_path);module.EXPECTED=tokens
    exec(compile(source.replace(old,new),str(source_path),"exec"),module.__dict__)
    module.POST_PURGE_HOOKS=token_decode.POST_PURGE_HOOKS
    decoded,info=module.decode_tokens(archive,RUNTIME,device)
    np.testing.assert_array_equal(decoded.numpy(),tokens)
    return info

def evaluate(args,ref):
    from profile_reference import build_native,shorten_candidate
    from canonical.inflate_canonical import install_rp1_shim
    from container import token_decode,token_encode
    from container.payload_format import read_archive,split_inner_payload,write_archive
    from container.archive_writer import EXTERNAL_ATTR_PR141
    from decoder.inflate_core import _load
    run=args.output
    best=json.loads((run/"best.json").read_text())
    canonical=(run/"best_hpac.bin").read_bytes();record=(run/"best_prev2.bin").read_bytes()
    model_id=sha(canonical+record)
    out=run/"evaluations"/model_id[:16]
    out.mkdir(parents=True,exist_ok=True)
    if (out/"evaluation.json").exists():
        result=json.loads((out/"evaluation.json").read_text())
        (run/"evaluation.json").write_text(json.dumps(result,indent=2)+"\n")
        print("Keeping completed evaluation for this quantized state",flush=True)
        print(json.dumps(result,indent=2));return
    if best["is_reference"]:
        result=dict(improved=False,reason="training did not beat the original combined proxy; no re-encode needed",best=best)
        (out/"evaluation.json").write_text(json.dumps(result,indent=2)+"\n")
        (run/"evaluation.json").write_text(json.dumps(result,indent=2)+"\n")
        print(json.dumps(result,indent=2));return
    encoder_input=out/"encoder_input.zip"
    pack_input(ref,canonical,record,encoder_input)
    install_rp1_shim();token_encode.POST_PURGE_HOOKS=token_decode.POST_PURGE_HOOKS
    with tempfile.TemporaryDirectory() as d:
        library,corrector=build_native(Path(d))
        os.environ["CPR1_RC64_LIBRARY"]=str(library);os.environ["F26_CORRECTOR_NATIVE_LIBRARY"]=str(corrector)
        raw_path=out/"raw_stream.bin"
        encode_meta=out/"encode_info.json"
        input_sha=sha(encoder_input.read_bytes())
        if raw_path.exists() and encode_meta.exists():
            raw=raw_path.read_bytes();cached=json.loads(encode_meta.read_text())
            if cached["raw_sha256"]!=sha(raw) or cached["encoder_input_sha256"]!=input_sha or cached["tokens_sha256"]!=TOKEN_SHA:
                raise ValueError("cached encoded stream provenance differs")
            print("Reusing completed encoded stream; continuing verification",flush=True)
        else:
            print("Re-encoding 600 unchanged token maps with the trained entropy model",flush=True)
            raw,info=token_encode.encode_tokens(encoder_input,RUNTIME,ref["tokens"],args.device)
            raw_path.write_bytes(raw)
            encode_meta.write_text(json.dumps(dict(raw_sha256=sha(raw),encoder_input_sha256=input_sha,tokens_sha256=TOKEN_SHA,info=info),indent=2)+"\n")
        parts,_,_=_load(encoder_input)
        core,signal=split_inner_payload(read_archive(encoder_input))
        if not core.endswith(parts.token_stream):raise ValueError("unexpected token layout")
        prefix=core[:-len(parts.token_stream)]
        choices=[]
        shorter=shorten_candidate(raw)
        if shorter!=raw:choices.append(("shortened",shorter))
        choices.append(("raw",raw))
        attempts=[]
        verified=None
        for name,stream in choices:
            candidate=out/f"candidate_{name}.zip"
            write_archive(candidate,prefix+stream+signal+struct.pack("<H",len(signal)),EXTERNAL_ATTR_PR141)
            print(f"Verifying {name}: {candidate.stat().st_size} archive bytes",flush=True)
            try:
                decoded_info=verify_decoder(candidate,ref["tokens"],args.device)
            except ValueError as error:
                attempts.append(dict(name=name,verified=False,error=str(error)));continue
            new_parts,warp,payload=_load(candidate)
            from decoder import extfmt
            old_ext=extfmt.parse_payload(ref["extension"]);new_ext=extfmt.parse_payload(payload)
            if new_parts.semantic_blob!=ref["parts"].semantic_blob or new_parts.carrier_blob!=ref["parts"].carrier_blob or new_parts.residual_payload!=ref["parts"].residual_payload or warp!=ref["warp"] or old_ext["ext1"]!=new_ext["ext1"] or old_ext["ext2"]!=new_ext["ext2"]:
                raise ValueError("RGB-generating state or token decoding table changed")
            attempts.append(dict(name=name,verified=True,archive_bytes=candidate.stat().st_size))
            verified=candidate
            break
        if verified is None:raise ValueError("neither stream representation verified")
    saved=ARCHIVE.stat().st_size-verified.stat().st_size
    result=dict(improved=saved>0,original_bytes=ARCHIVE.stat().st_size,candidate_bytes=verified.stat().st_size,
        saved_bytes=saved,score_change_from_rate=-25*saved/37545489,roundtrip_verified=True,
        rgb_state_unchanged=True,archive_sha256=sha(verified.read_bytes()),candidate=str(verified),
        raw_token_bytes=len(raw),attempts=attempts,best=best,official_evaluation_complete=False)
    (out/"evaluation.json").write_text(json.dumps(result,indent=2)+"\n")
    (run/"evaluation.json").write_text(json.dumps(result,indent=2)+"\n")
    if saved>0:(out/"best_verified_archive.zip").write_bytes(verified.read_bytes())
    print(json.dumps(result,indent=2),flush=True)

def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument("command",choices=("check","train","evaluate"))
    ap.add_argument("--device",choices=("cpu","cuda"),default="cuda")
    ap.add_argument("--tokens",type=Path,default=TOKENS)
    ap.add_argument("--output",type=Path,default=ROOT/"experiments/rdo151/work/entropy_rate_v1")
    ap.add_argument("--steps",type=int,default=6000)
    ap.add_argument("--lr",type=float,default=.01)
    ap.add_argument("--seed",type=int,default=1234)
    ap.add_argument("--probe-frames",type=int,default=16)
    ap.add_argument("--probe-every",type=int,default=500)
    ap.add_argument("--log-every",type=int,default=50)
    ap.add_argument("--resume",action="store_true")
    args=ap.parse_args()
    args.tokens=args.tokens.resolve();args.output=args.output.resolve()
    if min(args.steps,args.probe_frames,args.probe_every,args.log_every)<1 or args.probe_frames>600 or args.lr<=0:ap.error("invalid schedule")
    ref=setup(args.device,args.tokens)
    if args.command=="check":
        check(ref)
        from torch.nn import functional as F
        logits,target=logits_for(ref,0)
        loss=F.cross_entropy(logits,target)
        loss.backward()
        gradients=[p.grad for n,p in ref["model"].named_parameters() if p.grad is not None]
        if not gradients or not all(ref["torch"].isfinite(g).all() for g in gradients) or not any(g.abs().max()>0 for g in gradients):raise ValueError("invalid gradients")
        print("PASS: finite nonzero gradients through quantized probability model",flush=True)
    elif args.command=="train":
        check(ref);train(args,ref)
    else:evaluate(args,ref)

if __name__=="__main__":main()
