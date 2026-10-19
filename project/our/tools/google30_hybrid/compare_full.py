#!/usr/bin/env python3
"""Compare actual secure ledgers and diagnose full-graph integer differences."""
import argparse,json
from pathlib import Path
import numpy as np
from full import validate_ledger

def compare_ledgers(left,right,contract):
 def read(path):
  rows=[json.loads(line)for line in Path(path).read_text().splitlines()]
  if not rows:return {}
  validate_ledger(rows,contract['sample_ids'],rows[0]['identity'])
  if any(row['contract']!=contract['identity']or row['diagnostic']for row in rows):raise ValueError('requires same-contract non-diagnostic secure ledgers')
  return {row['sample_index']:row for row in rows}
 sci=read(left);hybrid=read(right);common=sorted(set(sci)&set(hybrid))
 differences=[dict(sample_index=i,sample_id=contract['sample_ids'][i],SCI=sci[i]['prediction'],SCI_ours=hybrid[i]['prediction'])for i in common if sci[i]['prediction']!=hybrid[i]['prediction']]
 return {'required_samples':contract['samples'],'SCI_secure_samples':len(sci),'SCI_ours_secure_samples':len(hybrid),'compared_secure_samples':len(common),'prediction_differences':differences,'first_difference':differences[0]if differences else None,'full_secure_agreement':len(common)==contract['samples']and not differences,'agreement_on_executed_overlap_pct':100*(1-len(differences)/len(common))if common else None,'complete':len(common)==contract['samples']and not differences}

def compare_tensors(actual_log,oracle_steps,oracle_logits,schema,sample_index=0):
 schema_by_name={row['tensor']:row for row in schema};checked=0;first=None;count=0;seen=set()
 for line in Path(actual_log).read_text().splitlines():
  if not line.startswith('TENSOR '):continue
  _,step,operator,tensor,bits,rows,cols,*values=line.split();step=int(step);rows=int(rows);cols=int(cols)
  actual=np.array(list(map(int,values))).reshape(rows,cols)
  if step<99:expected=oracle_steps[step][tensor]
  elif tensor in {'tmp40','argmax_input'}:expected=oracle_logits
  else:continue
  if actual.shape!=expected.shape:raise ValueError('trace shape mismatch '+tensor)
  seen.add((step,tensor));mismatches=np.flatnonzero(actual!=expected);count+=len(mismatches);checked+=actual.size
  if len(mismatches)and first is None:
   flat=int(mismatches[0]);row,col=divmod(flat,cols);meta=schema_by_name.get(tensor,{'frac':2})
   first={'sample_index':sample_index+row,'step':step,'operator':operator,'tensor':tensor,'bits':int(bits),'frac':meta['frac'],'element':col,'secure_integer':int(actual.flat[flat]),'SCI_integer_reference':int(expected.flat[flat])}
 expected_keys={(step,row['tensor'])for step in range(99)for row in schema if row['tensor']not in {'tmp39','tmp40'}}|{(99,'tmp40'),(99,'argmax_input')}
 if seen!=expected_keys:raise ValueError('incomplete/duplicate tensor trace')
 return {'sample_start':sample_index,'checked_values':checked,'difference_count':count,'first_difference':first,'equal':count==0,'trace_timesteps':99}

if __name__=='__main__':
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--SCI',type=Path,required=True);p.add_argument('--hybrid',type=Path,required=True);p.add_argument('--contract',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
 result=compare_ledgers(a.SCI,a.hybrid,json.loads(a.contract.read_text()));a.output.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
