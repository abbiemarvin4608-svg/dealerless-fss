#!/usr/bin/env python3
"""Prepare immutable full-valid fixtures and resume actual two-party batch execution."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import resource
import subprocess
import time
import numpy as np

MODEL_SHAPES={'W1':512,'W2':1600,'U1':3500,'U2':3500,'Bg':100,'Bh':100,'FC':3000,'FCbias':30}

def digest(path):
 h=hashlib.sha256()
 with Path(path).open('rb')as f:
  for block in iter(lambda:f.read(1024*1024),b''):h.update(block)
 return h.hexdigest()

def validate_ledger(rows,sample_ids,identity):
 seen=set()
 for row in rows:
  index=row['sample_index']
  if row['identity']!=identity:raise ValueError('ledger identity differs')
  if index in seen:raise ValueError('duplicate sample '+str(index))
  if not 0<=row['prediction']<30:raise ValueError('prediction outside 30 classes')
  if not 0<=index<len(sample_ids)or row['sample_id']!=sample_ids[index]:raise ValueError('sample identity/order differs')
  seen.add(index)
 return [i for i in range(len(sample_ids))if i not in seen]

def prepare(output,quality,generated,header):
 output=Path(output);output.mkdir(parents=True,exist_ok=True);quality=Path(quality)
 records=[json.loads(x)for x in (quality/'manifests/valid_samples.jsonl').read_text().splitlines()]
 if len(records)!=6798:raise ValueError('fixed valid manifest must contain 6798 samples')
 csv_path=quality/'heldout_input/X.csv'
 input_path=output/'client_input.i16'
 count=0
 labels_from_csv=np.loadtxt(quality/'heldout_input/Y.csv',delimiter=',',dtype=np.int64).reshape(-1)
 if labels_from_csv.tolist()!=[r['label']for r in records]:raise ValueError('heldout label order differs from manifest')
 with csv_path.open()as source,input_path.open('wb')as target:
  for line in source:
   values=np.fromstring(line,sep=',')
   if values.size!=3168:raise ValueError('wrong input sequence shape')
   # Native CSV encoding: double * 2^11, truncation then int16 wrap.
   np.trunc(values*2048).astype('<i2').tofile(target);count+=1
 if count!=len(records):raise ValueError('input and manifest counts differ')
 text=Path(header).read_text();server=[]
 for name,size in MODEL_SHAPES.items():
  match=re.search(r'\b'+name+r'_temp\s*\[[^=]+?=\s*\{(.*?)\};',text,re.S)
  if not match:raise ValueError('model lacks '+name)
  values=[int(x)for x in re.findall(r'-?\d+',match[1])]
  if len(values)!=size:raise ValueError('model shape '+name)
  if name in {'W1','U1','FC','FCbias'}:values=[x//256 if x>=0 else -((-x)//256)for x in values]
  server+=values
 np.asarray(server,dtype='<i2').tofile(output/'server_model.i16')
 # Binding provenance and numeric rules prevents mixing old/new artifacts on resume.
 contract={'version':'google30-independent-sci-hybrid-v1','samples':len(records),'steps':99,'encoding':'native CSV double*2048 trunc to int16','generated_sha256':digest(generated),'model_sha256':digest(header),'client_input_sha256':digest(input_path),'server_model_sha256':digest(output/'server_model.i16'),'manifest_sha256':digest(quality/'manifests/valid_samples.jsonl'),'quality_source':str(quality.resolve()),'sample_ids':[r['path']for r in records],'labels':[r['label']for r in records]}
 contract['identity']=hashlib.sha256(json.dumps(contract,sort_keys=True).encode()).hexdigest()
 (output/'contract.json').write_text(json.dumps(contract,indent=2)+'\n')
 return contract

def run_batch(binary,fixture,output,start,count,backend,nseg,diagnostic,port,timeout):
 output=Path(output);output.mkdir(parents=True,exist_ok=False);processes=[];logs=[];commands=[];begin=time.monotonic()
 try:
  for role in [1,2]:
   env=dict(os.environ,OMP_NUM_THREADS='1',SEGFSS_SCI_ENABLE=str(int(backend=='SCI_ours')),SEGFSS_SCI_WIDTH='16',SEGFSS_SCI_NSEG=str(nseg),SEGFSS_SCI_PORT=str(port+20),SEGFSS_SCI_HOST='127.0.0.1',SEGFSS_SCI_TRACE=str((output/f'party{role}.jsonl').resolve()))
   owned=Path(fixture)/('server_model.i16'if role==1 else'client_input.i16')
   cmd=[str(Path(binary).resolve()),str(role),str(port),str(owned.resolve()),str(start),str(count),'diagnostic'if diagnostic else'secure']
   cpu_pair=os.environ.get('GOOGLE30_CPUS','10,12').split(',')
   cmd=['taskset','-c',cpu_pair[role-1]]+cmd;commands.append(cmd)
   def bound_memory():resource.setrlimit(resource.RLIMIT_AS,(1024**3,1024**3))
   log=(output/f'party{role}.log').open('w');logs.append(log)
   processes.append(subprocess.Popen(cmd,env=env,stdout=log,stderr=subprocess.STDOUT,preexec_fn=bound_memory))
  deadline=begin+timeout
  for child in processes:
   code=child.wait(timeout=max(.1,deadline-time.monotonic()))
   if code:raise RuntimeError('party exit '+str(code))
 finally:
  for child in processes:
   if child.poll()is None:child.kill()
  for child in processes:child.wait()
  for log in logs:log.close()
  (output/'execution.json').write_text(json.dumps({'commands':commands,'cpu_pair':os.environ.get('GOOGLE30_CPUS','10,12').split(','),'memory_limit_bytes_per_party':1024**3,'wall_seconds':time.monotonic()-begin,'exit_codes':[p.returncode for p in processes],'diagnostic':diagnostic,'backend':backend,'batch':count,'start':start,'fresh_process_and_material':True},indent=2)+'\n')
 client=(output/'party2.log').read_text();server=(output/'party1.log').read_text()
 if not diagnostic and ('TENSOR 'in client or'TENSOR 'in server):raise ValueError('secure mode opened intermediate tensor')
 predictions=[(int(i),int(p))for i,p in re.findall(r'^PRED (\d+) (\d+)$',client,re.M)]
 if [i for i,p in predictions]!=list(range(start,start+count)):raise ValueError('missing/duplicate/out-of-order batch output')
 if backend=='SCI_ours':
  rows=[json.loads(line)for line in (output/'party1.jsonl').read_text().splitlines()]
  other=[json.loads(line)for line in (output/'party2.jsonl').read_text().splitlines()]
  if rows!=other:raise ValueError('FSS metadata mismatch')
  if len(rows)!=297 or sum(r['count']for r in rows)!=29700*count:raise ValueError('full99step replacement coverage mismatch')
  if any(r['input_bits']!=16 or r['comparison_bits']!=15 or r['nseg']!=nseg or not r['fresh_material']for r in rows):raise ValueError('FSS contract mismatch')
 return predictions

def execute(binary,fixture,output,backend,batch_size,nseg,start,count,diagnostic,port,timeout,oracle):
 fixture=Path(fixture);output=Path(output);output.mkdir(parents=True,exist_ok=True)
 contract=json.loads((fixture/'contract.json').read_text())
 claimed=contract['identity'];unsigned={k:v for k,v in contract.items()if k!='identity'}
 if hashlib.sha256(json.dumps(unsigned,sort_keys=True).encode()).hexdigest()!=claimed:raise ValueError('contract metadata identity differs')
 if batch_size<1 or batch_size>256 or count<0 or start<0:raise ValueError('invalid batch or range')
 for filename,key in [('client_input.i16','client_input_sha256'),('server_model.i16','server_model_sha256')]:
  if digest(fixture/filename)!=contract[key]:raise ValueError('prepared private artifact identity differs: '+filename)
 build_manifest=json.loads((Path(binary).parent/'runtime_manifest.json').read_text())
 if build_manifest['generated_sha256']!=contract['generated_sha256']:raise ValueError('binary numerical graph differs from prepared artifact')
 runtime_libraries={}
 for library,expected in build_manifest['dynamic_libraries'].items():
  actual=digest(library)
  if actual!=expected:raise ValueError('runtime library changed: '+library)
  runtime_libraries[library]=actual
 oracle_identity=digest(oracle)
 oracle_record=output/'oracle_reference.json'
 if oracle_record.exists() and json.loads(oracle_record.read_text())['sha256']!=oracle_identity:raise ValueError('oracle reference identity changed')
 oracle_record.write_text(json.dumps({'sha256':oracle_identity,'source':str(Path(oracle).resolve())},indent=2)+'\n')
 oracle=np.load(oracle)
 if len(oracle)!=contract['samples']:raise ValueError('oracle does not cover fixed full sample order')
 identity=hashlib.sha256(json.dumps({'contract':contract['identity'],'binary':digest(binary),'runtime_libraries':runtime_libraries,'runtime_source_identity':build_manifest['source_identity'],'backend':backend,'nseg':nseg,'diagnostic':diagnostic},sort_keys=True).encode()).hexdigest()
 ledger=output/'predictions.jsonl'
 rows=[json.loads(line)for line in ledger.read_text().splitlines()]if ledger.exists()else[]
 missing=validate_ledger(rows,contract['sample_ids'],identity)
 selected=[i for i in missing if start<=i<min(contract['samples'],start+count)]
 while selected:
  first=selected[0];chunk=[]
  for index in selected:
   if len(chunk)>=batch_size or index!=first+len(chunk):break
   chunk.append(index)
  run_id=f'batch{first:05d}-{len(chunk):03d}-attempt{time.time_ns()}'
  run_out=output/run_id
  try:predictions=run_batch(binary,fixture,run_out,first,len(chunk),backend,nseg,diagnostic,port,timeout)
  except Exception as error:
   (output/'last_failure.json').write_text(json.dumps({'start':first,'batch':len(chunk),'reason':str(error),'run':run_id,'committed_samples':len(rows)},indent=2)+'\n');raise
  new=[{'identity':identity,'contract':contract['identity'],'backend':backend,'sample_index':i,'sample_id':contract['sample_ids'][i],'label':contract['labels'][i],'prediction':pred,'oracle_prediction':int(oracle[i]),'oracle_match':pred==int(oracle[i]),'run':run_id,'diagnostic':diagnostic}for i,pred in predictions]
  # Commit a complete independently replayable batch; a crash before replace
  # causes a fresh-material rerun, never duplicates a partially appended batch.
  rows+=new;temp=ledger.with_suffix('.tmp');temp.write_text(''.join(json.dumps(r)+'\n'for r in rows));temp.replace(ledger)
  selected=selected[len(chunk):]
  progress={'backend':backend,'committed_samples':len(rows),'required_samples':contract['samples'],'latest_completed_batch':run_id,'complete':len(rows)==contract['samples']}
  (output/'progress.json').write_text(json.dumps(progress,indent=2)+'\n')
  print(json.dumps({'backend':backend,'completed':len(rows),'required':contract['samples'],'latest_batch':first,'batch_size':len(chunk)}),flush=True)
 missing=validate_ledger(rows,contract['sample_ids'],identity)
 summary={'backend':backend,'identity':identity,'contract':contract['identity'],'required_samples':contract['samples'],'completed_samples':len(rows),'missing_samples':missing,'complete':not missing,'diagnostic':diagnostic,'full_secure_evaluation':not missing and not diagnostic,'partial_subset_top1_pct':100*sum(r['prediction']==r['label']for r in rows)/len(rows)if rows and missing else None,'full_top1_pct':100*sum(r['prediction']==r['label']for r in rows)/len(rows)if rows and not missing else None,'oracle_match_count':sum(r['oracle_match']for r in rows),'oracle_mismatches':[r['sample_index']for r in rows if not r['oracle_match']], 'full_oracle_match':not missing and all(r['oracle_match']for r in rows)}
 (output/'summary.json').write_text(json.dumps(summary,indent=2)+'\n');return summary

if __name__=='__main__':
 p=argparse.ArgumentParser(description=__doc__);sub=p.add_subparsers(dest='command',required=True)
 q=sub.add_parser('prepare');q.add_argument('--output',type=Path,required=True);q.add_argument('--quality',type=Path,required=True);q.add_argument('--generated',type=Path,required=True);q.add_argument('--model-header',type=Path,required=True)
 q=sub.add_parser('run');q.add_argument('--binary',type=Path,required=True);q.add_argument('--fixture',type=Path,required=True);q.add_argument('--output',type=Path,required=True);q.add_argument('--oracle',type=Path,required=True);q.add_argument('--backend',choices=['SCI','SCI_ours'],required=True);q.add_argument('--batch',type=int,default=4);q.add_argument('--nseg',type=int,default=4);q.add_argument('--start',type=int,default=0);q.add_argument('--count',type=int,default=6798);q.add_argument('--diagnostic',action='store_true');q.add_argument('--port',type=int,default=42400);q.add_argument('--timeout',type=int,default=7200)
 a=p.parse_args()
 if a.command=='prepare':prepare(a.output,a.quality,a.generated,a.model_header)
 else:execute(a.binary,a.fixture,a.output,a.backend,a.batch,a.nseg,a.start,a.count,a.diagnostic,a.port,a.timeout,a.oracle)
