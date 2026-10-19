#!/usr/bin/env python3
"""Run real SCI and SCI + ours on identical private single-step fixtures."""
import argparse
import collections
import hashlib
import json
import itertools
import os
from pathlib import Path
import re
import socket
import statistics
import subprocess
import time
from step import SHAPES,compare_traces,model_values,read_values

CASES=['recorded_x0_zero_hidden','signed_and_truncation_boundaries','negative_one']

def validate_variant(schema,variant):
 expected=15 if variant=='independent'else 16
 if len(schema)!=expected:raise ValueError('generated graph and requested model variant disagree: '+variant+' requires '+str(expected)+' recurrent calls')

def validate_sources(schema_path,header,variant,contract_path=None):
 metadata_path=Path(schema_path).parent/'source_metadata.json'
 metadata=json.loads(metadata_path.read_text())if metadata_path.exists()else None
 if metadata and metadata['variant']!=variant:raise ValueError('build source metadata and requested variant disagree')
 if contract_path:
  if not metadata:raise ValueError('artifact contract validation requires build source_metadata.json')
  contract=json.loads(Path(contract_path).read_text())
  if contract['generated_sha256']!=metadata['generated_sha256']:raise ValueError('generated graph differs from artifact contract')
  if contract['model_sha256']!=hashlib.sha256(Path(header).read_bytes()).hexdigest():raise ValueError('model header differs from artifact contract')
 return {'build':metadata,'model_header':str(Path(header).resolve()),'artifact_contract':str(Path(contract_path).resolve())if contract_path else None}

def prepare(out,header,x_csv,variant,batch=1,cases=3,diagnostic=True):
 out=Path(out);out.mkdir(parents=True,exist_ok=True)
 model=model_values(header,variant)
 with Path(x_csv).open()as f:
  first_rows=[[int(float(v)*2048)for v in row.strip().split(',')[:32]]for row in itertools.islice(f,batch)]
 if len(first_rows)!=batch or any(len(row)!=32 for row in first_rows):raise ValueError('input lacks requested Google-30 first timesteps')
 first=first_rows[0]
 edge=[-32768,-32767,-257,-256,-255,-129,-128,-127,-9,-8,-7,-1,0,1,7,8,9,127,128,129,255,256,257,32766,32767]
 inputs=[(first,[0]*100),([edge[i%len(edge)]for i in range(32)],[edge[(i*7)%len(edge)]for i in range(100)]),([-1]*32,[-1]*100)]
 inputs=inputs[:cases] if diagnostic else [inputs[0]]*cases
 query_samples=[[(x,h)]*batch for x,h in inputs]if diagnostic else [[(x,[0]*100)for x in first_rows]for _ in range(cases)]
 (out/'Input-P0-0').write_text(' '.join(str(v)for name in SHAPES for v in model[name])+'\n')
 (out/'Input-P1-0').write_text(' '.join(str(v)for query in query_samples for x,h in query for vec in [x,h]for v in vec)+'\n')
 (out/'fixtures.json').write_text(json.dumps({'variant':variant,'case_names':CASES[:cases]if diagnostic else ['recorded_first_batch_zero_hidden']*cases,'batch':batch,'source_row_indices':list(range(batch))if not diagnostic else None,'input_owner':'client','previous_hidden_owner':'client','matrix_bias_owner':'server','public_learned_scalars':{'zeta':16654,'nu':26006},'inputs':[{'x':x,'hidden':h}for x,h in inputs],'query_samples':[[{'x':x,'hidden':h}for x,h in query]for query in query_samples]},indent=2)+'\n')

def parse_records(text,prefix):
 records=[]
 for line in text.splitlines():
  if not line.startswith(prefix+' '):continue
  record={}
  for token in line.split()[1:]:
   if '='not in token:raise ValueError('invalid metric token '+token)
   key,value=token.split('=',1)
   try:record[key]=float(value)if '.'in value or 'e'in value.lower()else int(value)
   except ValueError:record[key]=value
  records.append(record)
 return records

def parse_performance(logs,cases):
 phases=[parse_records(log,'PHASE')for log in logs]
 activations=[parse_records(log,'ACTIVATION')for log in logs]
 finished=[parse_records(log,'STEP_FINISHED')for log in logs]
 if any(len(rows)!=1 or rows[0]['cases']!=cases for rows in finished):raise ValueError('missing completed step record')
 if finished[0][0]['mode']!=finished[1][0]['mode']:raise ValueError('party modes disagree')
 online_name='online'if finished[0][0]['mode']=='secure'else'diagnostic_online'
 result={}
 for name in ('sci_setup','bridge_setup','offline','ready',online_name):
  grouped=[]
  for query in ([-1]if name.endswith('setup')else range(cases)):
   pair=[[row for row in rows if row['name']==name and row['query']==query]for rows in phases]
   if any(len(rows)!=1 for rows in pair):raise ValueError('missing or duplicate phase '+name)
   left,right=[rows[0]for rows in pair]
   grouped.append({'query':query,'wall_ms':max(left['wall_ms'],right['wall_ms']),
    'sci_sent_bytes':left['sci_sent_bytes']+right['sci_sent_bytes'],
    'bridge_sent_bytes':left['bridge_sent_bytes']+right['bridge_sent_bytes'],
    'combined_sent_bytes':sum(row['sci_sent_bytes']+row['bridge_sent_bytes']for row in (left,right)),
    'sci_rounds':max(left['sci_rounds'],right['sci_rounds']),
    'bridge_rounds':max(left['bridge_rounds'],right['bridge_rounds'])})
  if grouped:result[name]=grouped
 result['activations']=[]
 for query in range(cases):
  pair=[[row for row in rows if row['query']==query]for rows in activations]
  if any(len(rows)!=1 for rows in pair):raise ValueError('missing activation metric')
  left,right=[rows[0]for rows in pair]
  result['activations'].append({key:max(left[key],right[key])for key in left})
 for name in ('offline','online'):
  if name in result:
   result[name+'_median_ms']=statistics.median(row['wall_ms']for row in result[name])
 result['activation_median_ms']=statistics.median(row['total_ms']for row in result['activations'])
 result['lut_median_ms']=statistics.median(row['lut_ms']for row in result['activations'])
 if 'online'in result:
  result['fresh_query_median_ms']=statistics.median(sum(result[name][query]['wall_ms']for name in ('offline','ready','online'))for query in range(cases))
 result['peak_rss_kib']=max(rows[0]['peak_rss_kib']for rows in finished)
 return result

def run_pair(binary,out,fixture,backend,mode,nseg,port,timeout,batch=1,cases=3):
 out=Path(out);out.mkdir(parents=True,exist_ok=True);logs=[];processes=[];commands=[]
 for role in [1,2]:
  trace=out/f'party{role}.jsonl'
  if trace.exists():trace.unlink()
 try:
  for role in [1,2]:
   log=(out/f'party{role}.log').open('w');logs.append(log)
   env=dict(os.environ,OMP_NUM_THREADS='1',SEGFSS_SCI_ENABLE=str(int(backend=='hybrid')),SEGFSS_FSS_ACTIVATION=str(int(backend=='hybrid')),SEGFSS_SCI_WIDTH='16',SEGFSS_SCI_NSEG=str(nseg),SEGFSS_FSS_LUT_NSEG=str(nseg),SEGFSS_SCI_PORT=str(port+20),SEGFSS_SCI_HOST='127.0.0.1',SEGFSS_SCI_TRACE=str((out/f'party{role}.jsonl').resolve()))
   if mode=='secure':env.pop('SEGFSS_SCI_TRACE',None)
   cmd=[str(Path(binary).resolve()),str(role),str(port),str((Path(fixture)/'Input').resolve()),str(cases),mode,str(batch)];commands.append(cmd)
   processes.append(subprocess.Popen(cmd,env=env,stdout=log,stderr=subprocess.STDOUT))
  deadline=time.monotonic()+timeout
  for child in processes:
   if child.wait(timeout=max(.1,deadline-time.monotonic())):raise RuntimeError('party failed; inspect '+str(out))
 finally:
  for child in processes:
   if child.poll()is None:child.kill()
  for child in processes:child.wait()
  for log in logs:log.close()
 (out/'commands.json').write_text(json.dumps(commands,indent=2)+'\n')
 logtexts=[(out/f'party{role}.log').read_text()for role in [1,2]]
 performance=parse_performance(logtexts,cases)
 if mode=='secure':
  for role in [1,2]:
   if re.search(r'^VALUE ',(out/f'party{role}.log').read_text(),re.M):raise ValueError('secure step unexpectedly reveals a tensor')
 if backend=='hybrid':
  counts=collections.Counter()
  log=(out/'party2.log').read_text()
  bridge=parse_records(log,'BRIDGE')[0]
  if mode=='diagnostic':
   events=[json.loads(x)for x in (out/'party1.jsonl').read_text().splitlines()]
   other=[json.loads(x)for x in (out/'party2.jsonl').read_text().splitlines()]
   if events!=other:raise ValueError('FSS traces differ between parties')
   dispatch=re.findall(r'^MSB_CALL case=(\d+) operator=(\w+) input_bits=(\d+) comparison_bits=(\d+) count=(\d+)$',log,re.M)
   if len(events)!=len(dispatch):raise ValueError('FSS generation and SCI dispatch counts differ')
   for event,row in zip(events,dispatch):
    case,operator,width,comparison,count=row
    if event['input_bits']!=int(width)or event['count']!=int(count)or event['nseg']!=nseg:raise ValueError('dispatch contract differs')
    counts[(operator,int(width),int(comparison),nseg)]+=int(count)
   if len(events)!=bridge['calls']:raise ValueError('FSS traces and bridge call counter differ')
  if bridge['calls']!=0 or bridge['elements']!=0:raise ValueError('unexpected native MSB hook call after whole activation replacement')
  if bridge['activation_calls']!=2*cases or bridge['activation_elements']!=200*cases*batch:raise ValueError('unexpected replaced activation coverage')
  return {'msb_invocations':bridge['calls'],'msb_elements':bridge['elements'],'segment_keys':bridge['segment_keys'],'activation_calls':bridge['activation_calls'],'activation_elements':bridge['activation_elements'],'activation_keys':bridge['activation_keys'],'activation_offline_ms':bridge['activation_offline_ms'],'activation_online_ms':bridge['activation_online_ms'],'by_operator':[dict(operator=k[0],input_bits=k[1],comparison_bits=k[2],nseg=k[3],elements=v)for k,v in sorted(counts.items())],'coverage_evidence':'per-operator traces and bridge counters'if mode=='diagnostic'else'public precomputed schedule, bridge counters and material consumption checks','performance':performance}
 return {'invocations':0,'elements':0,'segment_keys':0,'performance':performance}

def main():
 p=argparse.ArgumentParser(description=__doc__)
 p.add_argument('--binary',type=Path,required=True);p.add_argument('--schema',type=Path,required=True)
 p.add_argument('--output',type=Path,required=True);p.add_argument('--model-header',type=Path,required=True);p.add_argument('--input-csv',type=Path,required=True)
 p.add_argument('--variant',choices=['historical','independent'],required=True);p.add_argument('--nseg',type=int,default=4);p.add_argument('--port',type=int,default=41200);p.add_argument('--timeout',type=float,default=120)
 p.add_argument('--batch',type=int,default=1);p.add_argument('--repeats',type=int,default=3);p.add_argument('--diagnostic-cases',type=int,choices=[0,1,2,3],default=3)
 p.add_argument('--artifact-contract',type=Path)
 a=p.parse_args();a.output.mkdir(parents=True,exist_ok=True)
 if not 1<=a.batch<=128 or a.repeats<1:raise ValueError('invalid batch or repeat count')
 schema=json.loads(a.schema.read_text());(a.output/'schema.json').write_text(json.dumps(schema,indent=2)+'\n')
 validate_variant(schema,a.variant)
 sources=validate_sources(a.schema,a.model_header,a.variant,a.artifact_contract)
 sources['input_csv']=str(a.input_csv.resolve())
 coverage=None;comparison=None
 if a.diagnostic_cases:
  fixture=a.output/'fixture';prepare(fixture,a.model_header,a.input_csv,a.variant,cases=a.diagnostic_cases)
  run_pair(a.binary,a.output/'SCI',fixture,'sci','diagnostic',a.nseg,a.port,a.timeout,cases=a.diagnostic_cases)
  coverage=run_pair(a.binary,a.output/'SCI_ours',fixture,'hybrid','diagnostic',a.nseg,a.port,a.timeout,cases=a.diagnostic_cases)
  left=read_values(a.output/'SCI/party2.log');right=read_values(a.output/'SCI_ours/party2.log')
  comparison=compare_traces(left,right,schema,CASES[:a.diagnostic_cases])
  comparison['activation_deltas']=[]
  for row in schema:
   if row['function']not in ('Sigmoid','TanH'):continue
   pairs=[(x,y)for query in range(a.diagnostic_cases)for x,y in zip(left[str(query)][row['tensor']],right[str(query)][row['tensor']])]
   comparison['activation_deltas'].append({'operator':row['operator'],'tensor':row['tensor'],'output_frac':row['frac'],'equal':sum(x==y for x,y in pairs),'count':len(pairs),'max_absolute_integer_difference':max(abs(x-y)for x,y in pairs)})
  for name,values in [('SCI',left),('SCI_ours',right)]:(a.output/(name+'_integer_outputs.json')).write_text(json.dumps(values,indent=2)+'\n')
 fixture=a.output/'secure_fixture';prepare(fixture,a.model_header,a.input_csv,a.variant,a.batch,a.repeats,False)
 baseline=run_pair(a.binary,a.output/'SCI_secure',fixture,'sci','secure',a.nseg,a.port,a.timeout,a.batch,a.repeats)
 secure=run_pair(a.binary,a.output/'SCI_ours_secure',fixture,'hybrid','secure',a.nseg,a.port,a.timeout,a.batch,a.repeats)
 left=baseline['performance'];right=secure['performance']
 report={'backend':'SCI + ours','variant':a.variant,'cases':CASES[:a.diagnostic_cases],'integer_comparison':comparison,'replacement_coverage':coverage,'SCI_secure':baseline,'SCI_ours_secure':secure,'batch':a.batch,'repeat_queries':a.repeats,'online_speedup':left['online_median_ms']/right['online_median_ms'],'activation_online_speedup':left['activation_median_ms']/right['activation_median_ms'],'numeric_algorithm':'SCI graph with complete cGGM FSS Sigmoid/TanH replacements, including their own sign/clipping BitDec; native MSB hooks are not called after activation replacement','quality_caveat':'FSS q=10/f=12 activation changes SCI EXP/div approximation; this is a performance comparison on the same step graph, not an equal-accuracy end-to-end result.','unreplaced':'all SCI operations outside Sigmoid/TanH, including mixed linear layers, extensions and truncation','application_threads':1,'network':'loopback','inference_scope':'one complete recurrent step; no classifier; first timestep of first batch Google-30 rows with zero hidden state; same frozen batch across queries','timing_contract':'SCI setup/baseOT and bridge setup separate; per-query fresh FSS activation preprocessing before client loads x/h; fused online includes SCI and bridge protocol, input loading and final synchronization; activation subtotal includes sign extension; LUT subtotal includes masked opening and precomputed local lookup only; reconstruction excluded from measured secure queries.','communication_contract':'sum sent bytes from both parties over SCI and bridge once; received bytes are not added again; SCI and bridge round counters reported separately','performance_claim':False}
 report['numeric_sources']=sources
 (a.output/'result.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))
 # The replacement intentionally changes SCI's native EXP/div activation
 # approximation to the frozen q=10 FSS oracle.  Intermediate integers are
 # therefore expected to differ; correctness is checked against sigmaPlain /
 # tanhPlain in the activation tests, while this report preserves the
 # SCI-vs-FSS difference counts for attribution.
if __name__=='__main__':main()
