#!/usr/bin/env python3
"""Prepare public program metadata and party-specific private fixture files."""
import argparse
import json
from pathlib import Path
import shutil
from sci_semantics import SHAPES,load_model,exp_table,reciprocal_tables,step

HERE=Path(__file__).resolve().parent

def prepare(out,header,x_csv,variant,diagnostic=True,cases=3):
    out=Path(out); out.mkdir(parents=True,exist_ok=True)
    (out/'Programs/Source').mkdir(parents=True,exist_ok=True)
    (out/'Player-Data').mkdir(exist_ok=True)
    model=load_model(header,variant)
    with Path(x_csv).open() as f: row=next(f)
    first=[int(float(v)*2048) for v in row.strip().split(',')][:32]
    edge=[-32768,-32767,-257,-256,-255,-129,-128,-127,-9,-8,-7,-1,0,1,7,8,9,127,128,129,255,256,257,32766,32767]
    inputs=[(first,[0]*100),([edge[i%len(edge)] for i in range(32)],[edge[(i*7)%len(edge)] for i in range(100)]),([-1]*32,[-1]*100)][:cases]
    shapes={name:list(SHAPES[name]) for name in ['W1','W2','U1','U2','Bg','Bh']}
    tables={}
    for kind,s,f in [('sigmoid',14,7),('tanh',7,6)]:
        a0,a1=reciprocal_tables((s-1)//2)
        tables[kind]={'exp0':exp_table(f,s,0),'exp1':exp_table(f,s,1),'a0':a0,'a1':a1}
    cfg={'variant':variant,'cases':len(inputs),'diagnostic':diagnostic,'shapes':shapes,'tables':tables,
         'model_privacy':'server-private matrices/bias; zeta=16654,nu=26006 public',
         'graph_contract':'SCI integer LUT, accumulation-floor then rescale-toward-zero',
         'validation_output':'intermediates to client ONLY in diagnostic fixture mode' if diagnostic else 'hidden state stays secret'}
    (out/'config.json').write_text(json.dumps(cfg,indent=2)+'\n')
    server=[v for name in shapes for row in model[name] for v in row]
    client=[v for x,h in inputs for vec in [x,h] for v in vec]
    (out/'Player-Data/Input-P0-0').write_text(' '.join(map(str,server))+'\n')
    (out/'Player-Data/Input-P1-0').write_text(' '.join(map(str,client))+'\n')
    (out/'oracle.json').write_text(json.dumps([step(x,h,model,variant) for x,h in inputs],indent=2)+'\n')
    shutil.copyfile(HERE/'secret.mpc',out/'Programs/Source/google30_step.mpc')
    return cfg

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,required=True); p.add_argument('--model-header',type=Path,required=True)
    p.add_argument('--input-csv',type=Path,required=True); p.add_argument('--variant',choices=['historical','independent'],default='historical')
    p.add_argument('--performance',action='store_true');p.add_argument('--cases',type=int,choices=[1,2,3],default=3)
    a=p.parse_args();prepare(a.output,a.model_header,a.input_csv,a.variant,not a.performance,a.cases)
if __name__=='__main__':main()
