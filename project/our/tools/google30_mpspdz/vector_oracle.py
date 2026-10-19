#!/usr/bin/env python3
"""Vectorized *plaintext* SCI integer oracle; never a secure performance result."""
import argparse
import csv
import json
from pathlib import Path
import numpy as np
from sci_semantics import load_model,exp_table,reciprocal_tables
from folded_semantics import correction_table,table_version

ACTIVATION_MODES = ('sci', 'sigmoid-only', 'tanh-only', 'folded')

def wrap(x,bits):return ((x+(1<<(bits-1)))&((1<<bits)-1))-(1<<(bits-1))
def trunc(x,shift):return np.where(x>=0,x>>shift,-((-x)>>shift))
def mm(a,b,bwa,bwb,bwc,total):
    depth=(b.shape[0]-1).bit_length();width=bwa+bwb
    v=wrap(a@b,width+depth)>>depth
    shift=total-depth
    return wrap(trunc(v,shift) if shift>0 else v<<(-shift),bwc)
def activation(x,kind,activation_mode='sci'):
    if activation_mode not in ACTIVATION_MODES:
        raise ValueError('unknown activation mode: '+str(activation_mode))
    if kind not in ('sigmoid', 'tanh'):
        raise ValueError('activation kind must be sigmoid or tanh')
    if activation_mode == 'folded' or activation_mode == kind+'-only':
        x=np.asarray(x,dtype=np.int64)
        if np.any((x < -32768) | (x > 32767)):
            raise ValueError('folded native input must be signed int16/Q7')
        scaled=x*(32 if kind=='sigmoid' else 64)
        indices=np.minimum(np.abs(scaled),65535)//64
        delta=np.asarray(correction_table(),dtype=np.int64)[indices]
        sigma=np.where(scaled>=0,4096-delta,delta)
        return sigma*4 if kind=='sigmoid' else (2*sigma-4096)//32
    s=14 if kind=='sigmoid' else 7;f=7 if kind=='sigmoid' else 6;m=(s-1)//2
    v=np.abs(x);e=(np.asarray(exp_table(f,s,0),dtype=np.int64)[v&255]*np.asarray(exp_table(f,s,1),dtype=np.int64)[v>>8])>>s
    den=(1<<s)+e;den-=((den>>(s+1))&1);low=den&((1<<s)-1);idx=low>>(s-m);residue=low&((1<<(s-m))-1)
    aa,bb=reciprocal_tables(m);a0=np.asarray(aa,dtype=np.int64);a1=np.asarray(bb,dtype=np.int64)
    w0=((a1[idx]*(1<<(s-m+1))-a0[idx]*residue)&((1<<(s+m+4))-1))>>(m+3)
    if kind=='sigmoid':return (np.where(x<0,e,1<<s)*w0)>>s
    t=(((1<<s)-e)*w0)>>s;return np.where(x<0,-t,t)

def step(x,h,m,trace=False,activation_mode='sci'):
    values={}
    def keep(n,v):
        if trace:values[n]=v.copy()
        return v
    a=keep('tmp19',mm(x,m['W1'],16,8,16,6));a=keep('tmp21',mm(a,m['W2'],16,16,16,15))
    b=keep('tmp23',mm(h,m['U1'],16,8,8,14));b=keep('tmp25',mm(b,m['U2'],8,16,16,6))
    c=keep('tmp26',wrap(a+trunc(b,3),16))
    gx=keep('tmp27',wrap(c+trunc(m['Bg'],3),16));g=keep('tmp28',activation(gx,'sigmoid',activation_mode))
    hx=keep('tmp29',wrap(c+trunc(m['Bh'],3),16));cand=keep('tmp30',activation(hx,'tanh',activation_mode))
    gh=keep('tmp31',wrap(trunc(g*h,14),16));ng=keep('tmp33',16384-g)
    zg=keep('tmp34',wrap(trunc(16654*ng,14),16));zn=keep('tmp35',wrap(zg+406,16))
    prod=keep('tmp36',wrap(trunc(zn*cand,7),16));h=keep('H',wrap(gh+trunc(prod,1),16))
    return (h,values) if trace else h

def load(header):return {n:np.asarray(v,dtype=np.int64)for n,v in load_model(header,'independent').items()}
def infer(encoded,m,trace=False,activation_mode='sci',initial_hidden=None):
    if activation_mode not in ACTIVATION_MODES:
        raise ValueError('unknown activation mode: '+str(activation_mode))
    encoded=np.asarray(encoded,dtype=np.int64).reshape(-1,99,32)
    if initial_hidden is None:h=np.zeros((len(encoded),100),dtype=np.int64)
    else:
        h=np.asarray(initial_hidden,dtype=np.int64)
        if h.shape != (len(encoded),100) or np.any((h < -32768) | (h > 32767)):
            raise ValueError('initial hidden state must be batch x 100 signed int16')
        h=h.copy()
    traces=[]
    for t in range(99):
        if trace:h,values=step(encoded[:,t],h,m,True,activation_mode);traces.append(values)
        else:h=step(encoded[:,t],h,m,activation_mode=activation_mode)
    fc=mm(h,m['FC'],16,8,8,16);logits=wrap(fc+trunc(m['FCbias'],3),8);prediction=logits.argmax(axis=1)
    return (prediction,logits,traces) if trace else (prediction,logits)

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--artifact',type=Path,required=True);p.add_argument('--input-csv',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--batch',type=int,default=128);p.add_argument('--limit',type=int,default=0)
    p.add_argument('--activation-mode',choices=ACTIVATION_MODES,default='sci')
    p.add_argument('--diagnostics',action='store_true',help='save the existing 15 first-step semantic tensors')
    p.add_argument('--hidden-state',type=Path,help='optional batch x 100 signed int16 initial state (.npy)')
    a=p.parse_args();m=load(a.artifact/'numeric_artifact/model_fixed.h');a.output.mkdir(parents=True,exist_ok=True)
    if a.batch<=0 or a.limit<0:p.error('batch must be positive and limit nonnegative')
    labels=np.load(a.artifact/'labels.npy');fp=np.load(a.artifact/'fp32_predictions.npy');sd=np.load(a.artifact/'fixed_predictions.npy');allpred=[];alllogits=[];batch=[]
    initial_hidden=np.load(a.hidden_state) if a.hidden_state else None
    diagnostics={}
    def consume(rows):
        encoded=np.asarray(rows,dtype=np.int64).reshape(-1,99,32)
        offset=len(allpred)
        hidden=None if initial_hidden is None else initial_hidden[offset:offset+len(rows)]
        if hidden is not None and hidden.shape != (len(rows),100):
            raise ValueError('hidden-state rows do not cover input samples')
        if a.diagnostics:
            state=np.zeros((len(rows),100),dtype=np.int64) if hidden is None else hidden
            _,values=step(encoded[:,0],state,m,trace=True,activation_mode=a.activation_mode)
            for index in range(len(rows)):
                diagnostics[str(offset+index)]={name:value[index].tolist() for name,value in values.items()}
        pred,logits=infer(encoded,m,activation_mode=a.activation_mode,initial_hidden=hidden)
        allpred.extend(pred.tolist());alllogits.extend(logits.tolist())
    with a.input_csv.open()as f:
        for i,line in enumerate(f):
            if a.limit and i>=a.limit:break
            batch.append([int(float(v)*2048)for v in line.strip().split(',')])
            if len(batch)==a.batch:
                consume(batch);batch=[]
        if batch:
            consume(batch)
    pred=np.asarray(allpred,dtype=np.int64);logits=np.asarray(alllogits,dtype=np.int64);n=len(pred)
    if not n or n>len(labels):raise ValueError('input sample count must be within frozen artifact labels')
    np.save(a.output/'predictions.npy',pred);np.save(a.output/'logits.npy',logits)
    if a.diagnostics:(a.output/'step_diagnostics.json').write_text(json.dumps(diagnostics)+'\n')
    with(a.output/'predictions.csv').open('w')as f:
        writer=csv.writer(f,lineterminator='\n');writer.writerow(['sample_id','label','fp32_prediction','seedot_prediction','sci_integer_prediction' if a.activation_mode=='sci' else 'integer_prediction'])
        writer.writerows((i,int(labels[i]),int(fp[i]),int(sd[i]),int(pred[i]))for i in range(n))
    accuracy=float(np.mean(pred==labels[:n])*100);fp_accuracy=float(np.mean(fp[:n]==labels[:n])*100)
    summary={'execution_mode':'plaintext_integer_oracle_only','activation_mode':a.activation_mode,'numeric_graph':'frozen_google30_independent','table_version':None if a.activation_mode=='sci' else table_version(),'initial_hidden':'external_npy' if a.hidden_state else 'zero','diagnostic_tensors':15 if a.diagnostics else 0,'samples':n,'full_holdout':n==len(labels),'accuracy_pct':accuracy,'fp32_accuracy_pct':fp_accuracy,'loss_pp':fp_accuracy-accuracy,'fp32_agreement_pct':float(np.mean(pred==fp[:n])*100),'seedot_prediction_agreement_pct':float(np.mean(pred==sd[:n])*100),'within_1pp':fp_accuracy-accuracy<=1,'secure_samples':0,'oracle_validation':'SCI graph has prior realSCI validation; folded table and four-mode scalar/vector trace checks are recorded in paper activation ticket02'}
    (a.output/'summary.json').write_text(json.dumps(summary,indent=2)+'\n');print(json.dumps(summary,indent=2))
if __name__=='__main__':main()
