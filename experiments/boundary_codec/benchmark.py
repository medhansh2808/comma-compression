"""Lossless boundary representation benchmark. CPU only; no submission integration."""
import argparse
import hashlib
import json
from pathlib import Path
import struct
import time
import brotli
import numpy as np

HEADER=struct.Struct('<4sBIIIQQ')
SHA='092ff87541fe0f4bdbff07759ae84cafd14e7e7e6f1385334d4177ef8c941d77'
ROOT=Path(__file__).resolve().parents[2]


def encode(tokens,mode,quality):
    shape=tokens.shape
    if mode==0:
        a=brotli.compress(tokens.tobytes(),quality=quality);b=b''
    else:
        mask=np.ones(shape,dtype=bool)
        mask[:,:,1:]=tokens[:,:,1:]!=tokens[:,:,:-1]
        labels=tokens[mask]
        if mode in (1,2):
            if mode==2:
                delta=mask.copy();delta[1:]=mask[1:]^mask[:-1];mask=delta
            locations=np.packbits(mask.reshape(-1),bitorder='little').tobytes()
        elif mode==3:
            indices=np.flatnonzero(mask.reshape(-1))
            lengths=np.diff(np.append(indices,tokens.size)).astype('<u2')
            locations=lengths.tobytes()
        else:raise ValueError('mode')
        a=brotli.compress(locations,quality=quality)
        b=brotli.compress(labels.tobytes(),quality=quality)
    return HEADER.pack(b'BND1',mode,*shape,len(a),len(b))+a+b


def decode(data):
    magic,mode,t,h,w,na,nb=HEADER.unpack_from(data)
    if magic!=b'BND1' or len(data)!=HEADER.size+na+nb:raise ValueError('bad container')
    a=brotli.decompress(data[HEADER.size:HEADER.size+na])
    if mode==0:
        if nb or len(a)!=t*h*w:raise ValueError('raw length')
        return np.frombuffer(a,np.uint8).reshape(t,h,w).copy()
    labels=np.frombuffer(brotli.decompress(data[HEADER.size+na:]),np.uint8)
    if mode in (1,2):
        if len(a)!=(t*h*w+7)//8:raise ValueError('mask length')
        mask=np.unpackbits(np.frombuffer(a,np.uint8),bitorder='little',count=t*h*w).reshape(t,h,w).astype(bool)
        if mode==2:np.logical_xor.accumulate(mask,axis=0,out=mask)
        if not mask[:,:,0].all():raise ValueError('missing row seed')
        indices=np.flatnonzero(mask.reshape(-1))
        lengths=np.diff(np.append(indices,t*h*w))
    elif mode==3:
        lengths=np.frombuffer(a,'<u2')
        if not len(lengths) or not lengths.min() or int(lengths.sum())!=t*h*w:raise ValueError('run lengths')
    else:raise ValueError('mode')
    if len(labels)!=len(lengths) or (labels>4).any():raise ValueError('labels')
    return np.repeat(labels,lengths).reshape(t,h,w)


def selftest():
    rng=np.random.default_rng(123)
    examples=[np.zeros((3,4,7),np.uint8),rng.integers(0,5,(3,4,7),dtype=np.uint8),np.tile(np.arange(5,dtype=np.uint8),(2,3,1))]
    for x in examples:
        for mode in range(4):
            if not np.array_equal(decode(encode(x,mode,4)),x):raise ValueError('selftest failed')
    print('PASS: uniform, random, alternating labels; all four formats decode exactly',flush=True)


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--tokens',type=Path,default=ROOT/'experiments/rdo151/cache/tokens151.npy')
    ap.add_argument('--output',type=Path,default=ROOT/'experiments/boundary_codec/runs/v1')
    ap.add_argument('--quality',type=int,default=6)
    ap.add_argument('--selftest',action='store_true')
    args=ap.parse_args();selftest()
    if args.selftest:return
    if not 0<=args.quality<=11:ap.error('quality must be 0..11')
    tokens=np.load(args.tokens,allow_pickle=False,mmap_mode='r')
    if tokens.shape!=(600,384,512) or tokens.dtype!=np.uint8 or hashlib.sha256(tokens.tobytes()).hexdigest()!=SHA:raise ValueError('expected verified #151 cache')
    args.output.mkdir(parents=True,exist_ok=False)
    rows=[]
    for mode,name in enumerate(('raw_control','boundary_mask','temporal_boundary_mask','row_run_lengths')):
        started=time.monotonic();data=encode(tokens,mode,args.quality)
        path=args.output/(name+'.bnd');path.write_bytes(data)
        # Decode bytes read from disk; encoder inputs are not supplied to decoder.
        restored=decode(path.read_bytes())
        if not np.array_equal(restored,tokens):raise ValueError('roundtrip mismatch')
        _,_,_,_,_,a,b=HEADER.unpack_from(data)
        row=dict(name=name,total_bytes=len(data),location_or_raw_bytes=a,label_bytes=b,
            header_bytes=HEADER.size,exact_decode=True,elapsed_seconds=time.monotonic()-started)
        rows.append(row);print(json.dumps(row),flush=True)
        del restored,data
    report=dict(token_sha256=SHA,quality=args.quality,rows=rows,
        reference_token_stream_bytes=108891,reference_token_plus_main_hpac_and_temporal_state_bytes=120779,
        official_archive_evaluated=False,
        notes=['Standalone token formats only; renderer and pose carrier excluded.',
               'Reference includes its verified stream-shortening advantage.',
               'Boundary locations, labels and format headers are all charged.',
               'This prototype uses generic Brotli, not a learned contour entropy model.'])
    (args.output/'summary.json').write_text(json.dumps(report,indent=2)+'\n')

if __name__=='__main__':main()
