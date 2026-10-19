#!/usr/bin/env python3
"""Detached long-run supervisor: records exit state without claiming partial completion."""
import argparse
from datetime import datetime,timezone
import json
import os
from pathlib import Path
import traceback
from full import execute

def main():
 p=argparse.ArgumentParser(description=__doc__)
 p.add_argument('--binary',type=Path,required=True);p.add_argument('--fixture',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--control',type=Path,required=True)
 p.add_argument('--backend',choices=['SCI','SCI_ours'],required=True);p.add_argument('--oracle',type=Path,required=True);p.add_argument('--batch',type=int,default=128);p.add_argument('--start',type=int,default=160);p.add_argument('--count',type=int,default=6638);p.add_argument('--port',type=int,required=True);p.add_argument('--timeout',type=int,default=1200)
 a=p.parse_args();a.control.mkdir(parents=True,exist_ok=True)
 begun={'state':'running','backend':a.backend,'pid':os.getpid(),'process_group':os.getpgrp(),'started_utc':datetime.now(timezone.utc).isoformat(),'cpu_pair':os.environ.get('GOOGLE30_CPUS'),'required_full_samples':6798,'initial_committed_samples':160,'requested_start':a.start,'requested_count':a.count,'nseg':4,'batch':a.batch}
 status=a.control/'state.json';status.write_text(json.dumps(begun,indent=2)+'\n')
 try:
  summary=execute(a.binary,a.fixture,a.output,a.backend,a.batch,4,a.start,a.count,False,a.port,a.timeout,a.oracle)
 except BaseException as error:
  begun.update(state='failed',ended_utc=datetime.now(timezone.utc).isoformat(),reason=str(error))
  status.write_text(json.dumps(begun,indent=2)+'\n');traceback.print_exc();raise
 begun.update(state='completed'if summary['full_secure_evaluation']and summary['full_oracle_match']else'partial',ended_utc=datetime.now(timezone.utc).isoformat(),summary=summary)
 status.write_text(json.dumps(begun,indent=2)+'\n')
 if begun['state']!='completed':raise SystemExit(1)
if __name__=='__main__':main()
