#!/usr/bin/env python3
"""Execute fresh, metered SCI parties through a real calibrated link relay."""
import argparse
from contextlib import ExitStack
import json
import os
from pathlib import Path
import re
import resource
import socket
import subprocess
import sys
import time
from metrics import summarize_parties, distribution

HERE=Path(__file__).resolve().parent

def stop(process):
    if process.poll() is None:process.terminate()
    try:process.wait(timeout=2)
    except subprocess.TimeoutExpired:process.kill();process.wait()

def reserve_ports(threads):
    for base in range(18000,48000,137):
        sockets=[]
        try:
            for offset in [0,50,100,1000,1050,1100]:
                for thread in range(threads):
                    s=socket.socket();s.bind(('127.0.0.1',base+offset+thread));sockets.append(s)
            return base
        except OSError:pass
        finally:
            for s in sockets:s.close()
    raise RuntimeError('could not reserve SCI port range')

def run_pair(build,input_dir,output,network,threads=1,cpus=(0,2,4),timeout=600,expected_prediction=24,queries=1):
    output.mkdir(parents=True,exist_ok=True);port=reserve_ports(threads);commands=[]
    mapping=','.join(f'{port+1000+offset+t}:{port+offset+t}' for offset in [0,50,100] for t in range(threads))
    relay_command=['taskset','-c',str(cpus[2]),sys.executable,str(HERE/'link_proxy.py'),'--ports',mapping,'--rtt-ms',str(network['rtt_ms']),'--mbps',str(network['mbps'])]
    processes=[];started=time.monotonic();status='complete';reason=None
    with ExitStack() as stack:
        relaylog=stack.enter_context((output/'relay.log').open('w'))
        relay=subprocess.Popen(relay_command,stdout=subprocess.PIPE,stderr=relaylog,text=True);stack.callback(stop,relay)
        if relay.stdout.readline().strip()!='READY':raise RuntimeError('relay failed to start')
        relay.stdout.close()
        for role in [1,2]:
            env=dict(os.environ,GOOGLE30_METRICS=str(output/f'party{role}_metrics.json'))
            if role==2:env['GOOGLE30_PROXY_OFFSET']='1000'
            cmd=['taskset','-c',str(cpus[role-1]),str(build/f'google30-party{role}'),f'r={role}',f'port={port}',f'nt={threads}',f'q={queries}','inputDir='+str(input_dir)+'/']
            commands.append(cmd);log=stack.enter_context((output/f'party{role}.log').open('w'))
            def limits(): resource.setrlimit(resource.RLIMIT_AS,(1024**3,1024**3))
            process=subprocess.Popen(cmd,env=env,stdout=log,stderr=subprocess.STDOUT,preexec_fn=limits);processes.append(process);stack.callback(stop,process)
        deadline=time.monotonic()+timeout
        for process in processes:
            try:code=process.wait(timeout=max(.01,deadline-time.monotonic()))
            except subprocess.TimeoutExpired:status='censored';reason=f'predeclared wall timeout {timeout}s';break
            if code:status='failed';reason=f'party exit {code}';break
        if status!='complete':
            for process in processes:stop(process)
        exits=[p.poll() for p in processes]
    result=dict(status=status,reason=reason,network=network,observed_wall_seconds=time.monotonic()-started,commands=commands,relay_command=relay_command,party_exit_codes=exits)
    if status=='complete':
        parties=[json.loads((output/f'party{p}_metrics.json').read_text()) for p in [1,2]]
        match=re.findall(r'Predicted Label:\s*(-?\d+)\s*,\s*Actual Label:\s*(-?\d+)',(output/'party2.log').read_text())
        if len(match)!=queries or any(int(x[0])!=expected_prediction for x in match):raise ValueError('SCI prediction mismatch')
        result.update(summarize_parties(parties));result['library_io_direction_transitions']=[int(re.search(r'Number of rounds = (\d+)',(output/f'party{p}.log').read_text())[1]) for p in [1,2]]
        result['library_transition_scope']='native IOPack[0].io + io_GC send/receive direction switches after initialize; excludes reverse socket, not dependency rounds'
        result['prediction']=int(match[0][0]);result['label']=int(match[0][1])
    else:
        result.update(total_seconds=None,link_bytes=None,dependent_rounds=None,offline_seconds=None,online_seconds=None,peak_rss_kib_per_party=None)
    (output/'result.json').write_text(json.dumps(result,indent=2)+'\n');return result

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--config',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--profiles',nargs='*');p.add_argument('--smoke',action='store_true')
    a=p.parse_args();config=json.loads(a.config.read_text());a.output.mkdir(parents=True,exist_ok=True)
    all_results=[]
    for profile in config['networks']:
        if a.profiles and profile['name'] not in a.profiles:continue
        iterations=range(1) if a.smoke else range(-profile['warmups'],profile['repetitions'])
        for i in iterations:
            tag='smoke' if a.smoke else ('warmup' if i<0 else f'repeat{i+1}')
            result=run_pair(Path(config['build']),Path(config['input']),a.output/profile['name']/tag,profile,config['threads'],config['cpus'],profile['timeout_seconds'],config['expected_prediction'],profile.get('queries',1))
            result['tag']=tag;all_results.append(result);print(json.dumps({'network':profile['name'],'run':tag,'status':result['status'],'seconds':result['total_seconds']}),flush=True)
    aggregate={}
    for profile in config['networks']:
        selected=[r for r in all_results if r['network']['name']==profile['name'] and r['tag'].startswith('repeat')]
        good=[r for r in selected if r['status']=='complete']
        if selected:aggregate[profile['name']]=dict(completed=len(good),censored=sum(r['status']=='censored' for r in selected),latency_seconds=distribution([r['total_seconds'] for r in good]) if good else None,link_bytes=distribution([r['link_bytes'] for r in good]) if good else None)
    (a.output/'summary.json').write_text(json.dumps(dict(config=str(a.config),profiles=aggregate,runs=all_results),indent=2)+'\n')
