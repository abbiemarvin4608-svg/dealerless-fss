#!/usr/bin/env python3
"""Build isolated, metered native SCI binaries with separate client/server model images."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import re
import shutil
import subprocess
import tarfile

HERE = Path(__file__).resolve().parent
SOURCES = ['library_fixed.cpp','BuildingBlocks/aux-protocols.cpp','BuildingBlocks/truncation.cpp',
           'BuildingBlocks/value-extension.cpp','Math/math-functions.cpp','LinearOT/linear-ot.cpp']

def build(sources, sci, output, jobs):
    output.mkdir(parents=True, exist_ok=True)
    src=output/'sci_src'; network=output/'network'
    with tarfile.open(sources/'sci_src.tar.gz') as archive:
        archive.extractall(output/'frozen')
    shutil.copytree(output/'frozen/src',src,dirs_exist_ok=True)
    shutil.copytree(sources/'FastGRNN/Google-30',network,dirs_exist_ok=True)
    shutil.copy2(HERE/'measurement.h',output/'measurement.h')
    net=src/'utils/net_io_channel.h'; text=net.read_text()
    text=text.replace('  void send_data(const void *data, int len) {','  void send_data(const void *data, int len) {\n    google30_measure::sent += len; ++google30_measure::sends;')
    text=text.replace('  void recv_data(void *data, int len) {','  void recv_data(void *data, int len) {\n    google30_measure::received += len; ++google30_measure::recvs;')
    text=text.replace('      addr = string(address);','      addr = string(address);\n      const char* offset = std::getenv("GOOGLE30_PROXY_OFFSET");\n      if(offset) port += std::atoi(offset);')
    net.write_text(text)
    path=network/'sirnn_fixed.cpp'; text=path.read_text()
    text=text.replace('\tinitialize();','').replace('\tfinalize();','')
    text=re.sub(r'(\tfor \(uint32_t i =[^\n]+)', '\tgoogle30_measure::phase("online_compute");\n'+r'\1',text)
    text=text.replace('\treconstruct(tmp41, res,','\tgoogle30_measure::phase("output_reconstruction");\n\treconstruct(tmp41, res,')
    text=text.replace('\tdelete[] W1;','\tgoogle30_measure::phase("teardown");\n\tdelete[] W1;')
    path.write_text(text)
    model=network/'model_fixed.h'; text=model.read_text()
    pattern=r'(static const MYINT \w+\[[^\]]+\] = )\{[^}]*\};'
    text=re.sub(pattern,lambda m:'#ifdef GOOGLE30_CLIENT_MODEL\n'+m[1]+'{0};\n#else\n'+m[0]+'\n#endif',text)
    model.write_text(text)
    main=network/'main.cpp'; text=main.read_text().split('int main(int argc, char *argv[])')[0]
    text='#include "library_fixed.h"\n'+text+(HERE/'driver_main.cpp').read_text()
    main.write_text(text)
    flags=['g++','-O3','-std=c++17','-pthread','-maes','-msse4.1','-mavx2','-mrdseed','-fopenmp','-DSCI_OT=1',
           '-Wno-deprecated-declarations','-include',str(output/'measurement.h'),'-I'+str(network),'-I'+str(src),'-I'+str(sci/'extern/eigen')]
    commands=[]; objects=[]
    for i,path in enumerate([src/name for name in SOURCES]+[main]):
        obj=output/('object%d.o'%i); objects.append(str(obj)); commands.append(flags+['-c',str(path),'-o',str(obj)])
    for role in [1,2]:
        cmd=flags+(['-DGOOGLE30_CLIENT_MODEL'] if role==2 else [])+['-c',str(network/'sirnn_fixed.cpp'),'-o',str(output/('network%d.o'%role))]
        commands.append(cmd)
    def compile_one(item):
        i,cmd=item; result=subprocess.run(cmd,text=True,capture_output=True)
        (output/('compile%d.log'%i)).write_text(result.stdout+result.stderr); result.check_returncode()
    with ThreadPoolExecutor(max_workers=jobs) as executor: list(executor.map(compile_one,enumerate(commands)))
    libs=sci/'build/install/lib'
    for role in [1,2]:
        cmd=flags+objects+[str(output/('network%d.o'%role)),'-o',str(output/('google30-party%d'%role)),'-Wl,--start-group']
        cmd += [str(libs/('libSCI-'+name+'.a')) for name in ['OT','LinearOT','GC','Math','BuildingBlocks','FloatingPoint']]
        cmd += ['-Wl,--end-group','-lssl','-lcrypto','-lgmp']; commands.append(cmd)
        result=subprocess.run(cmd,text=True,capture_output=True); (output/('link%d.log'%role)).write_text(result.stdout+result.stderr); result.check_returncode()
    (output/'build_commands.json').write_text(json.dumps(commands,indent=2)+'\n')

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__); p.add_argument('--sources',type=Path,required=True); p.add_argument('--sci',type=Path,required=True); p.add_argument('--output',type=Path,required=True); p.add_argument('--jobs',type=int,default=2)
    a=p.parse_args(); build(a.sources.resolve(),a.sci.resolve(),a.output.resolve(),a.jobs)
