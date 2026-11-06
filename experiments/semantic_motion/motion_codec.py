"""Research prototype: learned motion + exact semantic-token residuals.

Training uses Torch. The standalone token decoder uses NumPy and Brotli only.
All motion parameters, the first map, and residuals are charged to the file.
This is a token-codec experiment, not an official RGB-video submission.
"""
import argparse
import hashlib
import json
from pathlib import Path
import struct
import time

import brotli
import numpy as np

MAGIC = b"SMC1"
STEPS = dict(affine=1/16, coefficients=1/256, basis=1/128)
PALETTE = np.array([[38,45,60], [91,171,236], [82,185,126], [239,184,70], [218,98,130]], np.uint8)

def sha(data):
    return hashlib.sha256(data).hexdigest()

def read_tokens(path):
    tokens = np.load(path, mmap_mode="r", allow_pickle=False)
    if tokens.ndim != 3 or tokens.dtype != np.uint8 or min(tokens.shape) < 2 or tokens.max() >= 5:
        raise ValueError("expected uint8 [frames,height,width] tokens with IDs 0..4")
    return tokens

def identity_state(shape):
    return dict(shape=list(shape), rank=0, grid=[2,2], max_flow=0.0, arrays={})

def pack_state(state):
    specs, raw = [], bytearray()
    for name, a in state["arrays"].items():
        a = np.ascontiguousarray(a, dtype="<i2")
        specs.append(dict(name=name, shape=list(a.shape), offset=len(raw), bytes=a.nbytes))
        raw.extend(a.tobytes())
    header = {k:v for k,v in state.items() if k != "arrays"}
    header["specs"] = specs
    meta = json.dumps(header, separators=(",", ":"), sort_keys=True).encode()
    return struct.pack("<I", len(meta)) + meta + brotli.compress(bytes(raw), quality=9)

def unpack_state(data):
    size = struct.unpack_from("<I", data)[0]
    header = json.loads(data[4:4+size])
    raw = brotli.decompress(data[4+size:])
    arrays = {}
    end = 0
    for spec in header.pop("specs"):
        if spec["offset"] != end or spec["bytes"] != int(np.prod(spec["shape"]))*2:
            raise ValueError("invalid motion state layout")
        end += spec["bytes"]
        arrays[spec["name"]] = np.frombuffer(raw[spec["offset"]:end], "<i2").reshape(spec["shape"]).copy()
    if end != len(raw):
        raise ValueError("motion state has trailing bytes")
    header["arrays"] = arrays
    return header

def resize_bilinear(a, height, width):
    # Fixed align_corners=True convention, shared by CPU encoder and decoder.
    ys = np.linspace(0, a.shape[-2]-1, height)
    xs = np.linspace(0, a.shape[-1]-1, width)
    y0, x0 = np.floor(ys).astype(int), np.floor(xs).astype(int)
    y1, x1 = np.minimum(y0+1, a.shape[-2]-1), np.minimum(x0+1, a.shape[-1]-1)
    fy, fx = (ys-y0)[None,:,None], (xs-x0)[None,None,:]
    top = a[:,y0][:,:,x0]*(1-fx) + a[:,y0][:,:,x1]*fx
    bot = a[:,y1][:,:,x0]*(1-fx) + a[:,y1][:,:,x1]*fx
    return top*(1-fy) + bot*fy

def coarse_flow(state, index):
    gh, gw = state["grid"]
    arrays = state["arrays"]
    a = arrays["affine"][index].astype(np.float64)*STEPS["affine"]
    yy, xx = np.meshgrid(np.linspace(-1,1,gh), np.linspace(-1,1,gw), indexing="ij")
    features = np.stack([np.ones_like(xx), xx, yy])
    flow = np.zeros((2,gh,gw), np.float64)
    for d in range(2):
        for k in range(3):
            flow[d] += a[d,k]*features[k]
    if state["rank"]:
        c = arrays["coefficients"][index].astype(np.float64)*STEPS["coefficients"]
        b = arrays["basis"].astype(np.float64)*STEPS["basis"]
        for r in range(state["rank"]):
            flow += c[r]*b[r]
    limit = state["max_flow"]
    return limit*np.tanh(flow/limit)

def predict(previous, state, index):
    if not state["arrays"]:
        return previous.copy()
    height, width = previous.shape
    flow = resize_bilinear(coarse_flow(state, index), height, width)
    # Actual deployed codec uses deterministic integer nearest-neighbor warps.
    dx, dy = np.rint(flow).astype(np.int32)
    yy, xx = np.indices((height,width))
    return previous[np.clip(yy+dy,0,height-1), np.clip(xx+dx,0,width-1)]

def encode(tokens, state, destination, quality=6):
    state_bytes = pack_state(state)
    # Decode transmitted state before encoding; original floating weights
    # never influence deployed prediction or residual construction.
    deployed = unpack_state(state_bytes)
    compressor = brotli.Compressor(quality=quality, lgwin=24)
    chunks = [compressor.process(np.ascontiguousarray(tokens[0]).tobytes())]
    misses, identity_misses = 0, 0
    for t in range(1,len(tokens)):
        current = np.asarray(tokens[t])
        prediction = predict(np.asarray(tokens[t-1]), deployed, t-1)
        mismatch = prediction != current
        residual = np.where(mismatch, current+1, 0).astype(np.uint8)
        chunks.append(compressor.process(residual.tobytes()))
        misses += int(mismatch.sum())
        identity_misses += int(np.count_nonzero(tokens[t-1] != current))
        if t % 100 == 0:
            print(f"Encoded {t}/{len(tokens)-1} frame transitions", flush=True)
    chunks.append(compressor.finish())
    residual_bytes = b"".join(chunks)
    meta = json.dumps(dict(shape=list(tokens.shape), quality=quality, tokens_sha256=sha(tokens.tobytes())), separators=(",", ":")).encode()
    payload = MAGIC + struct.pack("<II",len(meta),len(state_bytes)) + meta + state_bytes + residual_bytes
    destination.write_bytes(payload)
    denominator = (len(tokens)-1)*tokens.shape[1]*tokens.shape[2]
    return dict(token_archive_bytes=len(payload), model_bytes=len(state_bytes), residual_bytes=len(residual_bytes),
        header_bytes=12+len(meta), prediction_error=misses/denominator,
        identity_prediction_error=identity_misses/denominator, archive_sha256=sha(payload))

def decode(source):
    data = source.read_bytes()
    if data[:4] != MAGIC:
        raise ValueError("invalid token archive")
    meta_size, state_size = struct.unpack_from("<II",data,4)
    pos = 12
    meta = json.loads(data[pos:pos+meta_size]); pos += meta_size
    state = unpack_state(data[pos:pos+state_size]); pos += state_size
    shape = tuple(meta["shape"])
    if tuple(state["shape"]) != shape:
        raise ValueError("state/token shape mismatch")
    raw = brotli.decompress(data[pos:])
    if len(raw) != int(np.prod(shape)):
        raise ValueError("invalid residual size")
    residuals = np.frombuffer(raw,np.uint8).reshape(shape)
    tokens = np.empty(shape,np.uint8)
    tokens[0] = residuals[0]
    for t in range(1,shape[0]):
        prediction = predict(tokens[t-1],state,t-1)
        r = residuals[t]
        if r.max() > 5:
            raise ValueError("invalid residual symbol")
        tokens[t] = np.where(r == 0,prediction,r-1)
    if sha(tokens.tobytes()) != meta["tokens_sha256"]:
        raise ValueError("token checksum FAILED")
    return tokens, state

def make_model(shape, rank, grid, limit, device):
    import torch
    from torch import nn
    from torch.nn import functional as F
    n,h,w = shape
    gh,gw = grid

    class Motion(nn.Module):
        def __init__(self):
            super().__init__()
            self.affine = nn.Parameter(torch.zeros(n-1,2,3))
            if rank:
                self.coefficients = nn.Parameter(torch.randn(n-1,rank)*0.1)
                self.basis = nn.Parameter(torch.randn(rank,2,gh,gw)*0.2)
            yy,xx = torch.meshgrid(torch.linspace(-1,1,gh),torch.linspace(-1,1,gw),indexing="ij")
            self.register_buffer("features",torch.stack([torch.ones_like(xx),xx,yy]))

        def forward(self, ids):
            def quant(name):
                p = getattr(self,name)
                step = STEPS[name]
                return p+(torch.round(p/step)*step-p).detach()
            flow = torch.einsum("bdk,khw->bdhw",quant("affine")[ids],self.features)
            if rank:
                flow = flow+torch.einsum("br,rdhw->bdhw",quant("coefficients")[ids],quant("basis"))
            return F.interpolate(limit*torch.tanh(flow/limit),size=(h,w),mode="bilinear",align_corners=True)

        def export(self):
            arrays = {}
            for name in ("affine","coefficients","basis"):
                if hasattr(self,name):
                    values = np.rint(getattr(self,name).detach().cpu().numpy()/STEPS[name])
                    if np.max(np.abs(values)) > 32767:
                        raise ValueError("quantized motion state overflow")
                    arrays[name] = values.astype("<i2")
            return dict(shape=list(shape),rank=rank,grid=list(grid),max_flow=limit,arrays=arrays)

    return Motion().to(device)

def train(args):
    import torch
    from torch.nn import functional as F
    torch.manual_seed(args.seed)
    torch.set_num_threads(args.threads)
    tokens = read_tokens(args.tokens)
    if args.device == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA unavailable")
    output = args.output.resolve()
    checkpoint = output / "checkpoint.pt"
    if checkpoint.exists() and not args.resume:
        raise ValueError("checkpoint exists; use --resume or a new output")
    output.mkdir(parents=True,exist_ok=True)
    device = torch.device(args.device)
    labels = torch.from_numpy(np.array(tokens)).to(device)
    model = make_model(tokens.shape,args.rank,(args.grid_h,args.grid_w),args.max_flow,device)
    optimizer = torch.optim.Adam(model.parameters(),lr=args.lr)
    n,h,w = tokens.shape
    yy,xx = torch.meshgrid(torch.linspace(-1,1,h,device=device),torch.linspace(-1,1,w,device=device),indexing="ij")
    base = torch.stack([xx,yy],-1)[None]
    scale = torch.tensor([2/(w-1),2/(h-1)],device=device)
    start_step = 0
    if args.resume:
        saved = torch.load(checkpoint,map_location=device,weights_only=True)
        if saved["config"] != dict(shape=list(tokens.shape),rank=args.rank,grid=[args.grid_h,args.grid_w],max_flow=args.max_flow):
            raise ValueError("resume configuration differs")
        if saved["tokens_sha256"] != sha(tokens.tobytes()):
            raise ValueError("resume tokens differ")
        model.load_state_dict(saved["model"])
        optimizer.load_state_dict(saved["optimizer"])
        torch.set_rng_state(saved["rng"].cpu())
        if device.type == "cuda":
            torch.cuda.set_rng_state(saved["cuda_rng"].cpu(),device)
        start_step = saved["step"]
    config = dict(shape=list(tokens.shape),rank=args.rank,grid=[args.grid_h,args.grid_w],max_flow=args.max_flow)
    history = output / "training.jsonl"
    start = time.monotonic()
    for step in range(start_step,args.steps):
        ids = torch.randint(n-1,(args.batch,),device=device)
        previous = F.one_hot(labels[ids].long(),5).permute(0,3,1,2).float()
        target = labels[ids+1].long()
        flow = model(ids)
        grid = base+flow.permute(0,2,3,1)*scale
        probabilities = F.grid_sample(previous,grid,mode="bilinear",padding_mode="border",align_corners=True)
        # Blur the categorical source slightly during training to give motion
        # gradients near boundaries; the deployed codec remains integer/exact.
        soft_source = F.avg_pool2d(previous,5,stride=1,padding=2,count_include_pad=False)
        soft_p = F.grid_sample(soft_source,grid,mode="bilinear",padding_mode="border",align_corners=True)
        p = 0.8*probabilities+0.2*soft_p
        correct = p.gather(1,target[:,None]).squeeze(1)
        # Linear mismatch surrogate stays informative for mostly-correct maps.
        data_loss = (1-correct).mean()
        smooth = (flow[:,:,1:]-flow[:,:,:-1]).square().mean()+(flow[:,:,:,1:]-flow[:,:,:,:-1]).square().mean()
        magnitude = flow.square().mean()
        loss = data_loss+args.smooth*smooth+args.magnitude*magnitude
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        if (step+1)%args.log_every == 0 or step == start_step:
            record = dict(step=step+1,loss=float(loss.detach()),soft_error=float(data_loss.detach()),elapsed_seconds=time.monotonic()-start)
            with history.open("a") as f:
                f.write(json.dumps(record)+"\n")
            print(json.dumps(record),flush=True)
        if (step+1)%args.save_every == 0 or step+1 == args.steps:
            saved = dict(step=step+1,config=config,tokens_sha256=sha(tokens.tobytes()),model=model.state_dict(),optimizer=optimizer.state_dict(),rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state(device) if device.type == "cuda" else None)
            temporary = checkpoint.with_suffix(".tmp")
            torch.save(saved,temporary)
            temporary.replace(checkpoint)
            state_path = output / "motion.bin"
            state_path.write_bytes(pack_state(model.export()))
    print(f"Training finished. Motion state: {output / 'motion.bin'}",flush=True)

def preview(tokens,state,path):
    from PIL import Image,ImageDraw
    frames = []
    n,h,w = tokens.shape
    target_w = min(w,320)
    target_h = max(1,round(h*target_w/w))
    for t in np.unique(np.linspace(1,n-1,min(48,n-1)).astype(int)):
        prediction = predict(np.asarray(tokens[t-1]),state,t-1)
        error = prediction != tokens[t]
        coarse = coarse_flow(state,t-1) if state["arrays"] else np.zeros((2,2,2))
        flow = resize_bilinear(coarse,h,w)
        magnitude = np.sqrt((flow*flow).sum(0))/max(state["max_flow"],1)
        angle = (np.arctan2(flow[1],flow[0])+np.pi)/(2*np.pi)
        hsv = np.stack([angle*255,np.full_like(angle,220),np.clip(magnitude*4,0,1)*255],-1).astype(np.uint8)
        images = [Image.fromarray(PALETTE[tokens[t]]),Image.fromarray(PALETTE[prediction]),Image.fromarray(hsv,mode="HSV").convert("RGB"),Image.fromarray(np.repeat((error*255).astype(np.uint8)[...,None],3,axis=2))]
        canvas = Image.new("RGB",(target_w*2,(target_h+24)*2),(16,20,28))
        draw = ImageDraw.Draw(canvas)
        for i,(im,label) in enumerate(zip(images,[f"Original tokens: frame {t}","Predicted from previous frame","Learned flow (colour=direction)",f"Exact corrections: {error.mean():.2%}"])):
            x,y = (i%2)*target_w,(i//2)*(target_h+24)
            canvas.paste(im.resize((target_w,target_h),Image.Resampling.NEAREST),(x,y+24))
            draw.text((x+5,y+5),label,fill="white")
        frames.append(canvas)
    frames[0].save(path,save_all=True,append_images=frames[1:],duration=180,loop=0)

def benchmark(args):
    tokens = read_tokens(args.tokens)
    args.output.mkdir(parents=True,exist_ok=True)
    states = [("identity",identity_state(tokens.shape))]
    for path in args.models:
        state = unpack_state(path.read_bytes())
        if tuple(state["shape"]) != tokens.shape:
            raise ValueError("model trained on a different token shape")
        states.append((path.parent.name,state))
    if len({name for name,_ in states}) != len(states):
        raise ValueError("model directories need distinct names")
    summary = []
    for name,state in states:
        archive = args.output / f"{name}.smc"
        if archive.exists():
            raise ValueError(f"preserving existing result: {archive}")
        print(f"Benchmarking {name}",flush=True)
        result = encode(tokens,state,archive,args.quality)
        recovered,_ = decode(archive)
        if not np.array_equal(recovered,tokens):
            raise ValueError("exact token verification failed")
        result.update(name=name,roundtrip_verified=True,reference_token_bytes=args.reference_token_bytes,
                      token_bytes_vs_reference=result["token_archive_bytes"]-args.reference_token_bytes,
                      official_submission=False)
        (args.output / f"{name}.json").write_text(json.dumps(result,indent=2)+"\n")
        preview(tokens,state,args.output / f"{name}.gif")
        summary.append(result)
        print(json.dumps(result,indent=2),flush=True)
    (args.output / "summary.json").write_text(json.dumps(summary,indent=2)+"\n")
    lines = ["# Learned motion token-codec experiment","", "All candidates reconstruct the exact cached token maps. These are token payload sizes, not official RGB-video submission scores.","", "| Model | Motion state bytes | Residual bytes | Total token bytes | Prediction error | Exact decode |", "|---|---:|---:|---:|---:|---|"]
    for r in summary:
        lines.append(f"| {r['name']} | {r['model_bytes']} | {r['residual_bytes']} | {r['token_archive_bytes']} | {r['prediction_error']:.3%} | yes |")
    lines.extend(["",f"Published #151 token stream: {args.reference_token_bytes:,} bytes. Its HPAC model is shared with its renderer/container and is not included in that stream-only count; this prototype charges its own motion model. Integration and full archive accounting are required for a leaderboard comparison.","", "GIFs show selected frame transitions across the sequence, not continuous real-time playback."])
    (args.output / "summary.md").write_text("\n".join(lines)+"\n")

def smoke():
    import tempfile
    import torch
    torch.set_num_threads(2)
    rng = np.random.default_rng(7)
    tokens = rng.integers(0,5,(7,18,24),dtype=np.uint8)
    with tempfile.TemporaryDirectory() as directory:
        directory = Path(directory)
        for rank in (0,2):
            model = make_model(tokens.shape,rank,(3,4),8,"cpu")
            optimizer = torch.optim.Adam(model.parameters(),lr=0.03)
            flow = model(torch.tensor([0,1]))
            loss = (flow-1.25).square().mean()
            loss.backward()
            assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())
            optimizer.step()
            state = model.export()
            packed = pack_state(state)
            restored = unpack_state(packed)
            for key in state["arrays"]:
                np.testing.assert_array_equal(state["arrays"][key],restored["arrays"][key])
            # Quantized training and CPU deployed coarse flows should agree.
            with torch.no_grad():
                trained_flow = model(torch.tensor([0])).numpy()[0]
            cpu_flow = resize_bilinear(coarse_flow(restored,0),18,24)
            np.testing.assert_allclose(trained_flow,cpu_flow,rtol=1e-5,atol=1e-5)
            path = directory / f"rank{rank}.smc"
            encode(tokens,state,path,quality=1)
            result,_ = decode(path)
            np.testing.assert_array_equal(result,tokens)
        path = directory / "identity.smc"
        encode(tokens,identity_state(tokens.shape),path,quality=1)
        result,_ = decode(path)
        np.testing.assert_array_equal(result,tokens)
        state = identity_state((3,18,24))
        shifted = np.stack([tokens[0],np.roll(tokens[0],1,axis=1),np.roll(tokens[0],2,axis=1)])
        arrays = np.zeros((2,2,3),dtype="<i2")
        arrays[:,0,0] = -16
        state.update(max_flow=32.,arrays=dict(affine=arrays))
        assert np.mean(predict(shifted[0],state,0)!=shifted[1]) < np.mean(shifted[0]!=shifted[1])
        preview(tokens,identity_state(tokens.shape),directory / "smoke.gif")
    print("PASS: gradients, quantized flow parity, motion direction, state packing, exact round trips, and preview")

def main():
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest="command",required=True)
    sub.add_parser("smoke")
    de = sub.add_parser("decode")
    de.add_argument("--archive",type=Path,required=True)
    de.add_argument("--output",type=Path,required=True)
    tr = sub.add_parser("train")
    tr.add_argument("--tokens",type=Path,required=True)
    tr.add_argument("--output",type=Path,required=True)
    tr.add_argument("--rank",type=int,default=4)
    tr.add_argument("--grid-h",type=int,default=8)
    tr.add_argument("--grid-w",type=int,default=12)
    tr.add_argument("--max-flow",type=float,default=24.)
    tr.add_argument("--steps",type=int,default=3000)
    tr.add_argument("--batch",type=int,default=4)
    tr.add_argument("--lr",type=float,default=0.03)
    tr.add_argument("--smooth",type=float,default=0.02)
    tr.add_argument("--magnitude",type=float,default=0.00002)
    tr.add_argument("--seed",type=int,default=1234)
    tr.add_argument("--threads",type=int,default=4)
    tr.add_argument("--log-every",type=int,default=50)
    tr.add_argument("--save-every",type=int,default=250)
    tr.add_argument("--device",choices=("cpu","cuda"),default="cuda")
    tr.add_argument("--resume",action="store_true")
    be = sub.add_parser("benchmark")
    be.add_argument("--tokens",type=Path,required=True)
    be.add_argument("--models",type=Path,nargs="*",default=[])
    be.add_argument("--output",type=Path,required=True)
    be.add_argument("--quality",type=int,default=6)
    be.add_argument("--reference-token-bytes",type=int,default=108891)
    args = ap.parse_args()
    if args.command == "smoke":
        smoke()
    elif args.command == "decode":
        if args.output.exists():
            ap.error("output exists; choose a new destination")
        recovered,_ = decode(args.archive)
        args.output.parent.mkdir(parents=True,exist_ok=True)
        np.save(args.output,recovered,allow_pickle=False)
        print(json.dumps(dict(shape=list(recovered.shape),tokens_sha256=sha(recovered.tobytes()),verified=True)))
    elif args.command == "train":
        if args.rank < 0 or min(args.grid_h,args.grid_w) < 2 or args.max_flow <= 0 or min(args.steps,args.batch,args.save_every,args.log_every,args.threads) < 1:
            ap.error("invalid training dimensions or schedule")
        train(args)
    else:
        benchmark(args)

if __name__ == "__main__":
    main()
