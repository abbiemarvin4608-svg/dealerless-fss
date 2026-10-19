#!/usr/bin/env python3
import argparse,json,shutil,subprocess
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--build',type=Path,required=True);p.add_argument('--sci',type=Path,default=Path('/home/jiang/EzPC/SCI'));a=p.parse_args();a.build=a.build.resolve();here=Path(__file__).resolve().parent
src=a.build/'sweep_sci';shutil.copytree(a.sci/'src',src,dirs_exist_ok=True)
f=src/'utils/net_io_channel.h';t=f.read_text().replace('  uint64_t counter = 0;','  uint64_t counter = 0;\n  uint64_t sweep_received = 0;').replace('  void recv_data(void *data, int len) {','  void recv_data(void *data, int len) {\n    sweep_received += len;');f.write_text(t)
flags=['g++','-O3','-std=c++17','-pthread','-maes','-msse4.1','-mavx2','-mrdseed','-fopenmp','-ffunction-sections','-fdata-sections','-Wno-deprecated-declarations','-I'+str(here.parent.parent/'src'),'-I'+str(src)]
cmd=flags+[str(here/'sweep_point.cpp'),str(src/'BuildingBlocks/aux-protocols.cpp'),'-L'+str(a.build),'-lsegfss_sci_bridge','-Wl,-rpath,'+str(a.build),'-Wl,--gc-sections','-lssl','-lcrypto','-lgmp','-o',str(a.build/'SCI_MSB_SWEEP')]
r=subprocess.run(cmd,capture_output=True,text=True);(a.build/'sweep_compile.log').write_text(r.stdout+r.stderr);r.check_returncode()
subprocess.run(['gcc','-shared','-fPIC','-O2',str(here/'connect_offset.c'),'-o',str(a.build/'connect_offset.so'),'-ldl'],check=True)
(a.build/'sweep_build_command.json').write_text(json.dumps(cmd,indent=2)+'\n')
