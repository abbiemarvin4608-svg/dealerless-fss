#!/usr/bin/env python3
"""Compile independent SCI two-party step harness, compare private client outputs."""
import argparse
import json
from pathlib import Path
import socket
import subprocess
from run import parse_values
HERE=Path(__file__).resolve().parent

def validate(runtime,sci):
    runtime=Path(runtime).resolve();sci=Path(sci).resolve();libs=sci/'build/install/lib'
    cmd=['g++','-O3','-std=c++17','-pthread','-maes','-msse4.1','-mavx2','-mrdseed','-fopenmp','-DSCI_OT=1','-Wno-deprecated-declarations','-I'+str(sci/'src'),'-I'+str(sci/'extern/eigen'),str(HERE/'sci_step.cpp'),str(sci/'src/library_fixed.cpp'),'-Wl,--start-group']+[str(libs/('libSCI-'+x+'.a')) for x in ['OT','LinearOT','GC','Math','BuildingBlocks','FloatingPoint']]+['-Wl,--end-group','-lssl','-lcrypto','-lgmp','-o',str(runtime/'sci_step')]
    (runtime/'sci_build_command.json').write_text(json.dumps(cmd,indent=2)+'\n')
    with(runtime/'sci_compile.log').open('w')as f:subprocess.run(cmd,stdout=f,stderr=subprocess.STDOUT,check=True)
    cfg=json.loads((runtime/'config.json').read_text());ps=[];logs=[]
    with socket.socket()as s:s.bind(('127.0.0.1',0));port=s.getsockname()[1]
    try:
        for party in [1,2]:
            f=(runtime/('sci_party%d.log'%party)).open('w');logs.append(f)
            ps.append(subprocess.Popen([str(runtime/'sci_step'),str(party),str(port),str(runtime/'Player-Data/Input'),cfg['variant'],str(cfg['cases'])],stdout=f,stderr=subprocess.STDOUT))
        for p in ps:
            if p.wait(timeout=180):raise RuntimeError('SCI validation failed')
    finally:
        for p in ps:
            if p.poll() is None:p.kill()
            p.wait()
        for f in logs:f.close()
    actual=parse_values((runtime/'sci_party2.log').read_text());expected=json.loads((runtime/'oracle.json').read_text())
    result={'equal':actual==expected,'cases':len(actual),'output_values':sum(len(v)for c in actual for v in c.values()),'oracle':'real SCI two-party reconstructed integer outputs, client only'}
    (runtime/'sci_validation.json').write_text(json.dumps(result,indent=2)+'\n')
    if actual!=expected:raise ValueError('SCI integer mismatch')
    return result
if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--runtime',type=Path,required=True);p.add_argument('--sci',type=Path,default=Path('/home/jiang/EzPC/SCI'))
    a=p.parse_args();print(json.dumps(validate(a.runtime,a.sci),indent=2))
