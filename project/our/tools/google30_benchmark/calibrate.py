#!/usr/bin/env python3
"""Measure real relay RTT (1 byte echo) and shared-link bulk transfer throughput."""
import argparse
import json
from pathlib import Path
import socket
import subprocess
import sys
import threading
import time
from metrics import distribution
from run import stop

HERE=Path(__file__).resolve().parent

def measure(rtt,mbps):
    with socket.socket() as server:
        server.bind(('127.0.0.1',0));server.listen();target=server.getsockname()[1]
        with socket.socket() as reserve:reserve.bind(('127.0.0.1',0));listen=reserve.getsockname()[1]
        def echo():
            conn,_=server.accept()
            with conn:
                while True:
                    data=conn.recv(65536)
                    if not data:break
                    conn.sendall(data)
        worker=threading.Thread(target=echo,daemon=True);worker.start()
        proxy=subprocess.Popen([sys.executable,str(HERE/'link_proxy.py'),'--ports',f'{listen}:{target}','--rtt-ms',str(rtt),'--mbps',str(mbps)],stdout=subprocess.PIPE,stderr=subprocess.PIPE,text=True)
        try:
            if proxy.stdout.readline().strip()!='READY':raise RuntimeError('relay calibration startup failed')
            with socket.create_connection(('127.0.0.1',listen)) as conn:
                conn.setsockopt(socket.IPPROTO_TCP,socket.TCP_NODELAY,1)
                rtts=[]
                for i in range(21):
                    start=time.monotonic();conn.sendall(b'X');assert conn.recv(1)==b'X';elapsed=(time.monotonic()-start)*1000
                    if i:rtts.append(elapsed)
                payload=b'x'*(8*1024*1024);received=[0]
                def receive():
                    while received[0]<len(payload):received[0]+=len(conn.recv(65536))
                consumer=threading.Thread(target=receive);consumer.start();start=time.monotonic();conn.sendall(payload);consumer.join(timeout=30)
                if consumer.is_alive():raise TimeoutError('bulk relay calibration')
                elapsed=time.monotonic()-start
            return dict(target_rtt_ms=rtt,target_mbps=mbps,echo_rtt_ms=distribution(rtts),bulk_payload_bytes=len(payload),bulk_elapsed_seconds=elapsed,echo_pipeline_mbps=len(payload)*8/elapsed/1e6)
        finally:stop(proxy);proxy.communicate(timeout=2);worker.join(timeout=2)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    results=[measure(0,1000),measure(2,100),measure(20,100)]
    a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(results,indent=2)+'\n');print(json.dumps(results,indent=2))
