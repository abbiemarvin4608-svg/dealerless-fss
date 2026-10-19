#!/usr/bin/env python3
"""Real two-party matrix Beaver regression against clear integers and native SCI."""
import argparse
import json
from pathlib import Path
import socket
import subprocess

HERE=Path(__file__).resolve().parent


def run(args):
    out=args.output.resolve();out.mkdir(parents=True,exist_ok=True)
    sci=args.sci.resolve()
    command=["g++","-O2","-std=c++17","-pthread","-maes","-msse4.1","-mavx2","-mrdseed","-fopenmp","-DSCI_OT=1","-Wno-deprecated-declarations",
             "-I"+str(sci/"src"),"-I"+str(sci/"extern/eigen"),str(HERE/"test_beaver.cpp"),str(HERE/"beaver.cpp"),"-Wl,--start-group"]
    command += [str(sci/"build/lib"/("libSCI-"+name+".a")) for name in ["OT","LinearOT","GC","Math","BuildingBlocks","FloatingPoint"]]
    command += ["-Wl,--end-group","-lssl","-lcrypto","-lgmp","-o",str(out/"beaver-test")]
    (out/"build_command.json").write_text(json.dumps(command,indent=2)+"\n")
    with (out/"build.log").open("w") as log:subprocess.run(command,stdout=log,stderr=subprocess.STDOUT,check=True)
    with socket.socket() as sock:sock.bind(("127.0.0.1",0));port=sock.getsockname()[1]
    processes=[];logs=[]
    try:
        for role in (1,2):
            log=(out/f"party{role}.log").open("w");logs.append(log)
            processes.append(subprocess.Popen([str(out/"beaver-test"),str(role),str(port)],stdout=log,stderr=subprocess.STDOUT))
        codes=[p.wait(timeout=args.timeout) for p in processes]
    finally:
        for p in processes:
            if p.poll() is None:p.kill()
            p.wait()
        for log in logs:log.close()
    report={"status":"passed" if codes==[0,0] else "failed","party_exit_codes":codes,
            "scope":"raw ring MatMul/MulCir/private-ScalarMul, native SCI floor and div_pow2, consumption; all algebra/layout configurations and aliases; real Batch=128 shapes",
            "timeout_seconds_per_party":args.timeout}
    (out/"summary.json").write_text(json.dumps(report,indent=2)+"\n")
    print(json.dumps(report,indent=2))
    if codes!=[0,0]:raise SystemExit(1)


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__);p.add_argument("--sci",type=Path,default=Path("/home/jiang/EzPC/SCI"));p.add_argument("--output",type=Path,required=True);p.add_argument("--timeout",type=int,default=300)
    run(p.parse_args())
