#!/usr/bin/env python3
"""Run two actual SCI + dealerless FSS parties; preserve logs and public trace."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import time

p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--binary',type=Path,required=True)
p.add_argument('--output',type=Path,required=True)
p.add_argument('--port',type=int,default=35400)
p.add_argument('--timeout',type=float,default=180)
a=p.parse_args()
a.output.mkdir(parents=True,exist_ok=True)
processes=[]
files=[]
try:
    for role in (1,2):
        trace=a.output/f'party{role}.jsonl'
        if trace.exists():trace.unlink()
        log=(a.output/f'party{role}.log').open('w')
        files.append(log)
        env=dict(os.environ,OMP_NUM_THREADS='1',SEGFSS_SCI_TRACE=str((a.output/f'party{role}.jsonl').resolve()))
        processes.append(subprocess.Popen([str(a.binary.resolve()),str(role),str(a.port),'1'],stdout=log,stderr=subprocess.STDOUT,env=env))
    deadline=time.monotonic()+a.timeout
    codes=[]
    for child in processes:codes.append(child.wait(timeout=max(.1,deadline-time.monotonic())))
finally:
    for child in processes:
        if child.poll() is None:child.kill()
    for child in processes:child.wait()
    for log in files:log.close()
rows=[json.loads(line) for line in (a.output/'party1.jsonl').read_text().splitlines()]
other=[json.loads(line) for line in (a.output/'party2.jsonl').read_text().splitlines()]
assert rows==other,'party traces disagree'
assert all(row['fresh_material'] and row['keys_generated']==row['nseg']*row['count'] for row in rows)
assert all(code==0 for code in codes),codes
report={'status':'pass','party_exit_codes':codes,'calls':len(rows),'scalar_instances':sum(row['count'] for row in rows),'segment_keys':sum(row['keys_generated'] for row in rows),'actual_width_nseg':sorted({(r['input_bits'],r['comparison_bits'],r['nseg']) for r in rows}),'private_run_opened_inputs':0,'private_run_opened_outputs':0,'test_only_output_reconstruction':True}
(a.output/'correctness.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps(report,indent=2))
