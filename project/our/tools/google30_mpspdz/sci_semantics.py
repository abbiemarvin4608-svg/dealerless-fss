"""Deterministic integer semantics of the frozen SCI-OT Google-30 graph.

This is a validation oracle only. Secure inference is generated in secret.mpc.
The exponential and reciprocal constants implement SCI math-functions.cpp;
intermediate truncation follows LinearOT/library_fixed, not SeeDot FLOATEXP.
"""
import math
import re
from functools import lru_cache
from pathlib import Path

SHAPES = {'W1':(32,16),'W2':(16,100),'U1':(100,35),'U2':(35,100),
          'Bg':(1,100),'Bh':(1,100),'FC':(100,30),'FCbias':(1,30)}

def signed(value, bits):
    return (int(value) + (1 << (bits-1))) % (1 << bits) - (1 << (bits-1))

def trunc0(value, shift):
    return value >> shift if value >= 0 else -((-value) >> shift)

def matmul(a, b, bw_a, bw_b, bw_c, shift_total):
    k = len(a); depth = (k-1).bit_length(); width = bw_a+bw_b
    accum = [sum(a[i]*b[i][j] for i in range(k)) for j in range(len(b[0]))]
    scaled = [signed(signed(v,width+depth) >> depth,width) for v in accum]
    shift = shift_total-depth
    return [signed(trunc0(v,shift) if shift>0 else v << -shift,bw_c) for v in scaled]

@lru_cache(maxsize=None)
def exp_table(scale_in,scale_out,digit):
    return [int(math.exp(-v * 2.0**(8*digit-scale_in)) * (1 << scale_out)) for v in range(256)]

@lru_cache(maxsize=None)
def reciprocal_tables(m):
    a0=[]; a1=[]; k=1<<m
    for i in range(k):
        p=1+i/k; z=p*(p+1/k)
        a0.append(int((1/z)*(1<<(m+3))) & ((1<<(m+3))-1))
        a1.append(int((1/(2*k)+math.sqrt(z))/z*(1<<(2*m+2))) & ((1<<(2*m+2))-1))
    return a0,a1

def activation(x,kind,activation_mode='sci'):
    if activation_mode not in ('sci', 'sigmoid-only', 'tanh-only', 'folded'):
        raise ValueError('unknown activation mode: '+str(activation_mode))
    if kind not in ('sigmoid', 'tanh'):
        raise ValueError('activation kind must be sigmoid or tanh')
    if activation_mode == 'folded' or activation_mode == kind+'-only':
        from folded_semantics import activation as folded_activation
        return folded_activation(x,kind)
    s=14 if kind=='sigmoid' else 7
    f=7 if kind=='sigmoid' else 6
    n=abs(signed(x,16)); lo=n&255; hi=n>>8
    e=(exp_table(f,s,0)[lo]*exp_table(f,s,1)[hi]) >> s
    den=(1<<s)+e
    den-=((den>>(s+1))&1)
    m=math.ceil((s-2)/2); a0,a1=reciprocal_tables(m)
    dn_low=den&((1<<s)-1)
    index=dn_low>>(s-m); residue=dn_low&((1<<(s-m))-1)
    approx=(a1[index]*(1<<(s-m+1))-a0[index]*residue)&((1<<(s+m+4))-1)
    w0=approx>>(m+3)
    if kind=='sigmoid':
        numerator=e if x<0 else (1<<s)
        return (numerator*w0)>>s
    nm=(1<<s)-e
    tanh=(nm*w0)>>s
    return -tanh if x<0 else tanh

def load_model(header, variant):
    text=Path(header).read_text(); result={}
    narrow={'W1','U1','FC','FCbias'} | ({'W2','U2','Bg'} if variant=='historical' else set())
    for name,shape in SHAPES.items():
        match=re.search(r'\b'+name+r'_temp\s*\[[^=]+?=\s*\{(.*?)\};',text,re.S)
        if not match: raise ValueError('missing model tensor '+name)
        values=[int(x) for x in re.findall(r'-?\d+',match[1])]
        if len(values)!=shape[0]*shape[1]: raise ValueError('model shape '+name)
        if name in narrow: values=[trunc0(x,8) for x in values]
        result[name]=[values[i*shape[1]:(i+1)*shape[1]] for i in range(shape[0])]
    return result

def step(x,h,model,variant='historical',activation_mode='sci'):
    trace={}
    def keep(name,values): trace[name]=list(values); return values
    a=keep('tmp19',matmul(x,model['W1'],16,8,16,6))
    a=keep('tmp21',matmul(a,model['W2'],16,8 if variant=='historical' else 16,16,7 if variant=='historical' else 15))
    b=keep('tmp23',matmul(h,model['U1'],16,8,8,14))
    b=keep('tmp25',matmul(b,model['U2'],8,8 if variant=='historical' else 16,16,0 if variant=='historical' else 6))
    c=keep('tmp26',[signed(u+trunc0(v,1 if variant=='historical' else 3),16) for u,v in zip(a,b)])
    if variant=='historical':
        gx=[signed((trunc0(u,5)+v)*32,16) for u,v in zip(c,model['Bg'][0])]
    else: gx=[signed(u+trunc0(v,3),16) for u,v in zip(c,model['Bg'][0])]
    gx=keep('tmp27',gx); g=keep('tmp28',[activation(v,'sigmoid',activation_mode) for v in gx])
    hx=keep('tmp29',[signed(u+trunc0(v,3),16) for u,v in zip(c,model['Bh'][0])])
    cand=keep('tmp30',[activation(v,'tanh',activation_mode) for v in hx])
    gh=keep('tmp31',[signed(trunc0(u*v,14),16) for u,v in zip(g,h)])
    ng=keep('tmp33',[signed(16384-v,16) for v in g])
    zg=keep('tmp34',[signed(trunc0(16654*v,14),16) for v in ng])
    zn=keep('tmp35',[signed(v+trunc0(26006,6),16) for v in zg])
    prod=keep('tmp36',[signed(trunc0(u*v,7),16) for u,v in zip(zn,cand)])
    keep('H',[signed(u+trunc0(v,1),16) for u,v in zip(gh,prod)])
    return trace
