"""CPU/full-float32 references using the installed, hash-locked MLX 0.31.2."""
import mlx.core as mx
import mlx.nn as nn

def fixture(spec, rounded):
    results=[]
    for case in spec['cases']:
        def arr(key):
            value=case.get(key)
            if value is None: return None
            dtype=mx.int32 if value.get('dtype')=='int32' else mx.float32
            return mx.array([{'-Infinity':float('-inf'),'Infinity':float('inf'),'NaN':float('nan')}.get(v,v) if isinstance(v,str) else v for v in value['values']],dtype=dtype).reshape(value['shape'])
        x=arr('x'); y=arr('y'); w=arr('w'); b=arr('b'); p=case.get('options',{})
        op=case['op']
        if op in ['relu','leaky_relu','elu','selu','tanh','sigmoid','softplus','mish','hardswish']:
            f={'hardswish':nn.hardswish}.get(op,getattr(nn,op,None))
            out=f(x,**p)
        elif op=='dropout':
            layer=nn.Dropout(**p); layer.eval(); out=layer(x)
        elif op=='quick_gelu': out=x*mx.sigmoid(1.702*x)
        elif op in ['conv1d','conv2d','conv3d','conv_transpose1d','conv_transpose2d','conv_transpose3d']:
            out=getattr(mx,op)(x,w,**p)
            if b is not None: out=out+b
        elif op=='conv_general': out=mx.conv_general(x,w,**p)
        elif op=='group_norm':
            layer=nn.GroupNorm(**p,affine=w is not None)
            if w is not None: layer.weight=w; layer.bias=b
            out=layer(x)
        elif op=='instance_norm':
            layer=nn.InstanceNorm(**p,affine=w is not None)
            if w is not None: layer.weight=w; layer.bias=b
            out=layer(x)
        elif op=='batch_norm':
            layer=nn.BatchNorm(**p,affine=w is not None); layer.eval()
            if w is not None: layer.weight=w; layer.bias=b
            if p.get('track_running_stats',True): layer.running_mean=arr('mean'); layer.running_var=arr('variance')
            out=layer(x)
        elif op in ['max_pool1d','avg_pool1d','max_pool2d','avg_pool2d']:
            cls={'max_pool1d':nn.MaxPool1d,'avg_pool1d':nn.AvgPool1d,'max_pool2d':nn.MaxPool2d,'avg_pool2d':nn.AvgPool2d}[op]
            out=cls(**p)(x)
        elif op=='upsample': out=nn.Upsample(**p)(x)
        elif op=='sinusoidal': out=nn.SinusoidalPositionalEncoding(**p)(x)
        elif op=='alibi': out=nn.ALiBi()(x,**p)
        elif op in ['cross_entropy','binary_cross_entropy']:
            out=getattr(nn.losses,op)(x,y,weights=w,**p)
        elif op in ['nll_loss','mse_loss','l1_loss','smooth_l1_loss','kl_div_loss','cosine_similarity_loss']:
            out=getattr(nn.losses,op)(x,y,**p)
        elif op=='minimum': out=mx.minimum(x,y)
        elif op=='log_softmax': out=x-mx.logsumexp(x,axis=p['axis'],keepdims=True)
        elif op=='pad': out=mx.pad(x,**p)
        elif op=='pad_symmetric': out=mx.pad(x,p['width'],constant_values=p['value'],mode=p['mode'])
        elif op=='as_strided': out=mx.as_strided(mx.contiguous(x),**p)
        elif op=='tile': out=mx.tile(x,p['repetitions'])
        elif op=='repeat_axis': out=mx.repeat(x,p['repeats'],axis=p['axis'])
        elif op=='clip': out=mx.clip(x,arr('lower'),arr('upper'))
        elif op in ['max','var','softmax']: out=getattr(mx,op)(x,**p)
        else: out=getattr(mx,op)(x)
        mx.eval(out)
        results.append({'name':case['name'],'shape':list(out.shape),'values':rounded(out.tolist())})
    return {'cases':results}
