#!/usr/bin/env python3
"""Prepare/compile/resume complete private Google-30 inference with real semi2k."""
import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
from sci_semantics import SHAPES,load_model,exp_table,reciprocal_tables
from run import run
HERE=Path(__file__).resolve().parent

def parse_full_output(log):
    predictions={};logits={};traces={}
    for sample,value in re.findall(r'^PRED (\d+) (-?\d+)$',log,re.M):
        sample=int(sample)
        if sample in predictions:raise ValueError('duplicate secure prediction')
        predictions[sample]=int(value)
    for sample,index,value in re.findall(r'^LOGIT (\d+) (\d+) (-?\d+)$',log,re.M):
        values=logits.setdefault(int(sample),[])
        if int(index)!=len(values):raise ValueError('non-contiguous logits')
        values.append(int(value))
    for sample,step,name,index,value in re.findall(r'^TRACE (\d+) (\d+) (\w+) (\d+) (-?\d+)$',log,re.M):
        key=(int(sample),int(step),name);values=traces.setdefault(key,[])
        if int(index)!=len(values):raise ValueError('non-contiguous trace')
        values.append(int(value))
    return {'predictions':predictions,'logits':logits,'traces':traces}

def validate_predictions(predictions,sample_ids):
    if set(predictions)!=set(sample_ids):raise ValueError('incomplete or unexpected secure prediction set')
    if any(not 0<=v<30 for v in predictions.values()):raise ValueError('class out of bounds')
    return True

def digest(path):
    h=hashlib.sha256()
    with Path(path).open('rb')as f:
        for chunk in iter(lambda:f.read(1048576),b''):h.update(chunk)
    return h.hexdigest()

def load_rows(x_csv,sample_ids):
    want=set(sample_ids);found={}
    with Path(x_csv).open()as f:
        for i,line in enumerate(f):
            if i in want:
                values=[int(float(v)*2048) for v in line.strip().split(',')]
                if len(values)!=3168 or any(v<-32768 or v>32767 for v in values):raise ValueError('invalid 16-bit Google-30 input row '+str(i))
                found[i]=values
                if len(found)==len(want):break
    if len(found)!=len(want):raise ValueError('input sample missing')
    return [found[i] for i in sample_ids]

def generate_program():
    # The tested arithmetic library stays identical to ticket08. Full program
    # supplies runtime recurrent loops and public metadata only, never weights.
    code=(HERE/'secret.mpc').read_text().split('def keep(name,x):')[0]
    return code+(HERE/'full_tail.mpc').read_text()

def prepare_full(out,header,x_csv,sample_ids,batch=1,queries=1,diagnostic=False,artifact=None):
    out=Path(out).resolve();out.mkdir(parents=True,exist_ok=True)
    if len(sample_ids)!=batch*queries:raise ValueError('exactly batch * queries sample IDs required')
    if len(set(sample_ids))!=len(sample_ids):raise ValueError('sample IDs must be unique inside one run')
    rows=load_rows(x_csv,sample_ids);model=load_model(header,'independent')
    for d in ['Programs/Source','Player-Data']: (out/d).mkdir(parents=True,exist_ok=True)
    tables={}
    for kind,s,f in [('sigmoid',14,7),('tanh',7,6)]:
        a0,a1=reciprocal_tables((s-1)//2)
        tables[kind]={'exp0':exp_table(f,s,0),'exp1':exp_table(f,s,1),'a0':a0,'a1':a1}
    binding={'model_header_sha256':digest(header),'input_samples':sample_ids,
             'selected_encoded_rows_sha256':hashlib.sha256(json.dumps(rows,separators=(',',':')).encode()).hexdigest(),
             'source_program_sha256':hashlib.sha256(generate_program().encode()).hexdigest(),
             'artifact_contract_sha256':digest(Path(artifact)/'numeric_contract.json') if artifact else None,
             'scope':'ticket02 independent weights/scales + exact SCI integer backend semantics'}
    config={'variant':'independent','program':'google30_full','full_network':True,'steps':99,'batch':batch,'queries':queries,'cases':queries,
            'sample_ids':sample_ids,'diagnostic':False,'trace_intermediates':diagnostic,'shapes':SHAPES,'tables':tables,'binding':binding,
            'input_order':'query, step, batch-index, feature','output':'class only to client; trace/logits client-only in explicit diagnostic fixture mode',
            'ownership':'model matrices+bias private P0, X private P1; zeta16654/nu26006 public'}
    (out/'config.json').write_text(json.dumps(config,indent=2)+'\n')
    (out/'Programs/Source/google30_full.mpc').write_text(generate_program())
    server=[v for name in SHAPES for row in model[name]for v in row]
    client=[rows[q*batch+b][t*32+j]for q in range(queries)for t in range(99)for b in range(batch)for j in range(32)]
    (out/'Player-Data/Input-P0-0').write_text(' '.join(map(str,server))+'\n')
    (out/'Player-Data/Input-P1-0').write_text(' '.join(map(str,client))+'\n')
    return config

def compile_full(runtime,spdz):
    runtime=Path(runtime)
    cmd=[sys.executable,str(Path(spdz)/'compile.py'),'-O','-R','64','google30_full']
    with(runtime/'compile.log').open('w')as f:subprocess.run(cmd,cwd=runtime,stdout=f,stderr=subprocess.STDOUT,check=True)
    (runtime/'compile_command.json').write_text(json.dumps(cmd,indent=2)+'\n')

def run_full(runtime,spdz,timeout=900,memory_mib=3072,ot_batch=128,cpus=(0,2)):
    runtime=Path(runtime);cfg=json.loads((runtime/'config.json').read_text())
    result=run(runtime,spdz,timeout,memory_mib,ot_batch,list(cpus))
    result['full_network']=True;result['timesteps']=99;result['batch']=cfg['batch'];result['queries']=cfg['queries'];result['binding']=cfg['binding']
    result['diagnostic']=cfg['trace_intermediates']
    result['query_accounting']='one model input/session; recurrent state resets per sample; fresh on-demand OT for every operation; setup amortized over %d queries'%cfg['queries']
    if result['status']=='completed':
        outputs=parse_full_output((runtime/'party1.log').read_text());validate_predictions(outputs['predictions'],range(len(cfg['sample_ids'])))
        with(runtime/'secure_predictions.csv').open('w')as f:
            writer=csv.writer(f);writer.writerow(['sample_id','prediction'])
            for local,global_id in enumerate(cfg['sample_ids']):writer.writerow([global_id,outputs['predictions'][local]])
        result['completed_samples']=len(outputs['predictions']);result['output_privacy']=cfg['output']
        result['cpu_pair']=list(cpus)
    (runtime/'run.json').write_text(json.dumps(result,indent=2)+'\n')
    return result

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('action',choices=['prepare','compile','run'])
    p.add_argument('--runtime',type=Path,required=True);p.add_argument('--spdz',type=Path,default=Path('/home/jiang/MP-SPDZ'))
    p.add_argument('--model-header',type=Path);p.add_argument('--input-csv',type=Path);p.add_argument('--artifact',type=Path)
    p.add_argument('--samples',default='0');p.add_argument('--batch',type=int,default=1);p.add_argument('--queries',type=int,default=1);p.add_argument('--diagnostic',action='store_true')
    p.add_argument('--timeout',type=int,default=900);p.add_argument('--memory-mib',type=int,default=3072);p.add_argument('--ot-batch',type=int,default=128);p.add_argument('--cpus',default='0,2')
    a=p.parse_args()
    if a.action=='prepare':
        if not a.model_header or not a.input_csv:p.error('prepare requires --model-header and --input-csv')
        prepare_full(a.runtime,a.model_header,a.input_csv,[int(i)for i in a.samples.split(',')],a.batch,a.queries,a.diagnostic,a.artifact)
    elif a.action=='compile':compile_full(a.runtime,a.spdz)
    else:
        r=run_full(a.runtime,a.spdz,a.timeout,a.memory_mib,a.ot_batch,[int(x)for x in a.cpus.split(',')]);print(json.dumps(r,indent=2))
        if r['status']!='completed':raise SystemExit(1)
if __name__=='__main__':main()
