#!/usr/bin/env python3
"""Run actual two-party semi2k with fresh live OT preprocessing and bounded resources."""
import argparse
import json
import os
from pathlib import Path
import re
import resource
import signal
import socket
import subprocess
import time

PHASES={60:'argmax_signed_16',100:'queries_including_live_ot',1:'server_private_model_input',2:'client_private_input_and_hidden',20:'secret_matrix_multiply',21:'matrix_floor_rescale_reduce',30:'secret_exp_lut',31:'secret_reciprocal_and_activation',40:'signed_rescale_state_update',50:'client_output_reconstruction'}

def limits(memory_mib):
    def apply():
        resource.setrlimit(resource.RLIMIT_AS,(memory_mib*1024**2,)*2)
    return apply

def parse_log(log):
    timers={int(i):float(s) for i,s in re.findall(r'Time(\d+)\s*=\s*([\d.eE+-]+)',log)}
    data=re.search(r'^Data sent = ([\d.eE+-]+) MB in ~?(\d+) rounds',log,re.M)
    setup=re.search(r'Setup took ([\d.eE+-]+) seconds',log)
    return {'phase_inclusive_s':{PHASES.get(i,str(i)):v for i,v in timers.items() if i},
            'runtime_s':float(re.search(r'^Time = ([\d.eE+-]+)',log,re.M)[1]) if re.search(r'^Time = ([\d.eE+-]+)',log,re.M) else None,'setup_s':float(setup[1]) if setup else None,
            'communication_sent_bytes':float(data[1])*1e6 if data else None,
            'reported_send_rounds':int(data[2]) if data else None}

def parse_values(text):
    cases=[]; current={}
    for name,index,value in re.findall(r'^VALUE (\w+) (\d+) (-?\d+)$',text,re.M):
        index=int(index)
        if name=='tmp19' and index==0 and current:
            cases.append(current);current={}
        values=current.setdefault(name,[])
        if index!=len(values):raise ValueError('non-contiguous output '+name)
        values.append(int(value))
    if current:cases.append(current)
    return cases

def run(runtime,spdz,timeout=900,memory_mib=1536,batch=128,cpu_pair=None):
    runtime=Path(runtime).resolve();spdz=Path(spdz).resolve()
    config=json.loads((runtime/'config.json').read_text())
    program_name=config.get('program','google30_step')
    with socket.socket()as s:s.bind(('127.0.0.1',0));port=s.getsockname()[1]
    processes=[];commands=[];logs=[];start=time.monotonic();status='completed';failure=None
    try:
        for party in [0,1]:
            log=(runtime/('party%d.log'%party)).open('w');logs.append(log)
            cmd=['/usr/bin/time','-v','-o',str(runtime/('party%d.resource.log'%party)),str(spdz/'semi2k-party.x'),'-N','2','-R','64','-b',str(batch),'-pn',str(port),'-v','-OF','.',str(party),program_name]
            if cpu_pair: cmd=cmd[:4]+['taskset','-c',str(cpu_pair[party])]+cmd[4:]
            commands.append(cmd)
            processes.append(subprocess.Popen(cmd,cwd=runtime,stdout=log,stderr=subprocess.STDOUT,start_new_session=True,preexec_fn=limits(memory_mib)))
        while any(p.poll() is None for p in processes):
            failed=[p.returncode for p in processes if p.poll() is not None and p.returncode]
            if failed:status='unavailable';failure='party exited: '+str(failed);break
            if time.monotonic()-start>timeout:status='unavailable';failure='timeout';break
            time.sleep(0.2)
    finally:
        for p in processes:
            if p.poll() is None:os.killpg(p.pid,signal.SIGKILL)
            p.wait()
        for f in logs:f.close()
    elapsed=time.monotonic()-start;parties=[]
    failed=[p.returncode for p in processes if p.returncode]
    if failed and status=='completed':status='unavailable';failure='party exited: '+str(failed)
    for party in [0,1]:
        log=(runtime/('party%d.log'%party)).read_text();info=parse_log(log)
        rss=re.search(r'Maximum resident set size \(kbytes\): (\d+)',(runtime/('party%d.resource.log'%party)).read_text())
        info['peak_rss_kib']=int(rss[1]) if rss else None;info['exit_code']=processes[party].returncode
        parties.append(info)
    config=json.loads((runtime/'config.json').read_text())
    result={'status':status,'unavailable_reason':failure,'backend':'MP-SPDZ semi2k','threat_model':'two-party semi-honest; live OT, no dealer/no file preprocessing',
            'commands':commands,'spdz_commit':subprocess.check_output(['git','-C',str(spdz),'rev-parse','HEAD'],text=True).strip(),
            'total_wall_s':elapsed,'offline_s':None,'online_s':None,
            'offline_online_reason':'semi2k generates genuine OT material on demand inside arithmetic phases; no separately materialized offline pass. total includes both.',
            'dependency_rounds':None,'dependency_rounds_reason':'MP-SPDZ reports party communication rounds and compiler VM rounds; these are not a critical-path network-round counter',
            'ring_bits':64,'graph_variant':config['variant'],'cases':config['cases'],'diagnostic':config['diagnostic'],
            'network':'localhost unshaped','application_threads':1,'cpu_pair':cpu_pair,'ot_batch':batch,'timeout_s':timeout,'memory_limit_mib_per_party':memory_mib,
            'fresh_material':True,'query_accounting':'fresh process first query; no cross-query preprocessing reuse','parties':parties}
    if status=='completed' and config['diagnostic']:
        actual=parse_values((runtime/'party1.log').read_text());expected=json.loads((runtime/'oracle.json').read_text())
        differences=[]
        for case in range(max(len(actual),len(expected))):
            if case>=len(actual) or case>=len(expected):differences.append({'case':case,'error':'missing case'});continue
            for name,values in expected[case].items():
                if name not in actual[case]:differences.append({'case':case,'tensor':name,'error':'missing tensor'});continue
                for i,(a,b) in enumerate(zip(actual[case][name],values)):
                    if a!=b:differences.append({'case':case,'tensor':name,'index':i,'actual':a,'expected':b})
                if len(values)!=len(actual[case][name]):differences.append({'case':case,'tensor':name,'error':'length'})
        result['integer_comparison']={'differences':len(differences),'output_values':sum(len(v) for c in expected for v in c.values()),'equal':not differences}
        (runtime/'differences.json').write_text(json.dumps(differences,indent=2)+'\n')
    (runtime/'run.json').write_text(json.dumps(result,indent=2)+'\n')
    return result

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--runtime',type=Path,required=True)
    p.add_argument('--spdz',type=Path,default=Path('/home/jiang/MP-SPDZ'));p.add_argument('--timeout',type=int,default=900);p.add_argument('--memory-mib',type=int,default=1536);p.add_argument('--ot-batch',type=int,default=128)
    p.add_argument('--cpus',default=None,help='Comma-separated party CPU pair, e.g. 0,2')
    a=p.parse_args();cpu_pair=[int(x) for x in a.cpus.split(',')] if a.cpus else None
    if cpu_pair and len(cpu_pair)!=2:p.error('--cpus needs two CPU IDs')
    r=run(a.runtime,a.spdz,a.timeout,a.memory_mib,a.ot_batch,cpu_pair);print(json.dumps(r,indent=2))
    if r['status']!='completed' or r.get('integer_comparison',{}).get('equal') is False:raise SystemExit(1)
if __name__=='__main__':main()
