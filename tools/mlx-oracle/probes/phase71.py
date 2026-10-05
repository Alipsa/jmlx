"""In-bounds Phase 7.1 capability probes for the pinned runtime."""
import json
import mlx.core as mx
import mlx.nn as nn
mx.set_default_device(mx.cpu)
def probe(name, fn):
    try:
        value=fn()
        if isinstance(value,mx.array):
            mx.eval(value)
            value={'shape':list(value.shape),'values':value.tolist()}
        print(json.dumps({'name':name,'result':value},allow_nan=True),flush=True)
    except Exception as error:
        print(json.dumps({'name':name,'error':type(error).__name__,'message':str(error)}),flush=True)
for mode in ['constant','edge','reflect','symmetric']:
    probe('pad-'+mode,lambda mode=mode:mx.pad(mx.array([1.,2.,3.]),(1,1),mode=mode))
x=mx.arange(10,dtype=mx.float32)[3:7]
probe('slice-origin',lambda:mx.as_strided(mx.contiguous(x),shape=(4,),strides=(1,),offset=0))
probe('negative-stride',lambda:mx.as_strided(mx.contiguous(x),shape=(4,),strides=(-1,),offset=3))
probe('zero-stride',lambda:mx.as_strided(mx.contiguous(x),shape=(4,),strides=(0,),offset=1))
probe('pool-zero',lambda:nn.MaxPool1d(3,2)(mx.ones((1,2,1))))
probe('pool-negative',lambda:nn.MaxPool1d(4,1)(mx.ones((1,1,1))))
probe('sinusoidal-two',lambda:nn.SinusoidalPositionalEncoding(2)(mx.array([0.,1.])))
for rank in [1,2,3]:
    x=mx.arange(1,1+2**rank,dtype=mx.float32).reshape((1,)+(2,)*rank+(1,))
    w=mx.ones((1,)+(2,)*rank+(1,))
    for prefix in ['conv','conv_transpose']:
        probe(prefix+str(rank),lambda x=x,w=w,rank=rank,prefix=prefix:getattr(mx,prefix+str(rank)+'d')(x,w))
probe('tile-zero',lambda:mx.tile(mx.array([1.,2.]),(2,0)))
probe('repeat-zero',lambda:mx.repeat(mx.array([1.,2.]),0,axis=-1))
probe('clip-inverted',lambda:mx.clip(mx.array([1.,2.]),3,0))
