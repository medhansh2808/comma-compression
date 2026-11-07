"""Small integer context adapter; inputs contain no current target token map."""
import struct
import numpy as np

D,H,C=25,8,5
HEADER=struct.Struct('<4sHHHH')
SHAPES=((H,D),(H,),(C,H),(C,))


def initial(seed=1234):
    rng=np.random.default_rng(seed)
    return [rng.integers(-16,17,SHAPES[0]).astype(np.int16),np.zeros(H,np.int16),
            np.zeros((C,H),np.int16),np.zeros(C,np.int16)]


def pack(arrays):
    if len(arrays)!=4:raise ValueError('parameter count')
    chunks=[]
    for a,shape in zip(arrays,SHAPES):
        a=np.asarray(a)
        if a.shape!=shape or not np.isfinite(a).all() or (a!=np.rint(a)).any() or (np.abs(a)>127).any():raise ValueError('invalid integer parameters')
        chunks.append(a.astype('<i2').tobytes())
    return HEADER.pack(b'CAD1',1,D,H,C)+b''.join(chunks)


def unpack(blob):
    if len(blob)!=HEADER.size+2*sum(np.prod(x) for x in SHAPES):raise ValueError('model size')
    if HEADER.unpack_from(blob)!=(b'CAD1',1,D,H,C):raise ValueError('model format')
    flat=np.frombuffer(blob, dtype='<i2',offset=HEADER.size);result=[];i=0
    for shape in SHAPES:
        n=int(np.prod(shape));result.append(flat[i:i+n].reshape(shape).copy());i+=n
    if (np.abs(flat.astype(np.int32))>127).any():raise ValueError('model range')
    return result


def divide_even(x):
    # Exact nearest-even division by 128, including negative ties.
    q=x//128;r=x%128
    return q+((r>64)|((r==64)&((q&1)!=0)))


def predict(features,arrays):
    x=np.asarray(features,dtype=np.int64)
    w1,b1,w2,b2=[a.astype(np.int64) for a in arrays]
    h=np.clip(divide_even(x@w1.T)+b1,0,127)
    delta=np.clip(divide_even(h@w2.T)+b2,-32,32)
    return delta.astype(np.int16) # logit units of 1/8


def features(logits,previous,previous2,boundary):
    logits=np.asarray(logits,np.float32).reshape(-1,5)
    q=np.rint(logits*8).astype(np.int64)
    relative=np.clip(q-q.max(1,keepdims=True),-128,0).astype(np.int16)
    one=np.eye(5,dtype=np.int16)*128
    p=np.asarray(previous).reshape(-1).astype(int)
    p2=np.asarray(previous2).reshape(-1).astype(int)
    b=np.asarray(boundary).reshape(-1).astype(int)
    if not (len(p)==len(p2)==len(b)==len(q)) or min(p.min(),p2.min(),b.min())<0 or max(p.max(),p2.max(),b.max())>4:raise ValueError('context')
    x=np.concatenate([relative,one[p],one[p2],one[b],one[q.argmax(1)]],axis=1)
    sortedq=np.sort(q,axis=1)
    gate=(b<=1)|((sortedq[:,-1]-sortedq[:,-2])<=32)
    return x,gate


def apply(logits,x,gate,arrays):
    q=np.clip(np.rint(np.asarray(logits,np.float32).reshape(-1,5)*8),-32768,32767).astype(np.int32)
    delta=predict(x,arrays).astype(np.int32)*np.asarray(gate)[:,None]
    return np.clip(q+delta,-32768,32767).astype(np.float32)/8


def selftest():
    rng=np.random.default_rng(7);a=initial()
    logits=rng.integers(-80,80,(64,5)).astype(np.float32)/8
    previous=rng.integers(0,5,64);boundary=rng.integers(0,5,64)
    x,g=features(logits,previous,previous,boundary)
    assert np.array_equal(apply(logits,x,g,a),logits)
    a[2][:]=rng.integers(-127,128,a[2].shape);a[3][:]=rng.integers(-8,9,5)
    blob=pack(a);assert pack(unpack(blob))==blob
    assert np.array_equal(predict(x,a),predict(x,unpack(blob)))
    values=np.arange(-100000,100000,dtype=np.int64)
    assert np.array_equal(divide_even(values),np.rint(values/128).astype(np.int64))
    # Selected output only depends on selected causal context, never other current positions.
    changed=logits.copy();changed[1:]+=3
    xx,gg=features(changed,previous,previous,boundary)
    assert np.array_equal(apply(logits,x,g,a)[0],apply(changed,xx,gg,a)[0])
    print('PASS: zero-output parity, integer serialization, signed rounding, position locality',flush=True)

if __name__=='__main__':selftest()
