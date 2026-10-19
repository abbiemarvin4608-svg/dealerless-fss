#!/usr/bin/env python3
"""Validate the complete independent graph against original generated SCI code."""
import argparse
import json
from pathlib import Path
import re
import shutil
import socket
import subprocess
import sys
import numpy as np
from full import parse_full_output,load_rows
import vector_oracle as vo
HERE=Path(__file__).resolve().parent

MAIN=r'''
#include "datatypes.h"
#include "library_fixed.h"
#include <fstream>
#include <iostream>
#include <sstream>
#include <vector>
#include <string>
#include <cmath>
int party=1,port=18590,num_threads=1;int32_t bitlength=32;
std::string address="127.0.0.1";
extern void sirnnFixed(MYINT*,int64_t*);
int main(int argc,char**argv){
 party=std::stoi(argv[1]);port=std::stoi(argv[2]);std::ifstream in(argv[3]);std::string line,v;
 std::getline(in,line);std::istringstream row(line);std::vector<MYINT>x(3168);
 for(int i=0;i<3168;++i){std::getline(row,v,',');x[i]=(MYINT)(std::stod(v)*2048);}
 int64_t pred=-1;sirnnFixed(x.data(),&pred);if(party==2)std::cout<<"PRED 0 "<<pred<<"\n";
}
'''
HEADER=r'''
#include "library_fixed.h"
extern int party;
static int observed_step=-1;
inline void record_full(const char*name,int64_t*value,int n,int bw){
 std::vector<int64_t>v(n);reconstruct(value,v.data(),1,n,bw);
 if(party==2)for(int i=0;i<n;++i)std::cout<<"TRACE 0 "<<observed_step<<" "<<name<<" "<<i<<" "<<v[i]<<"\n";
}
'''

def build(out,network,sci):
    out=Path(out).resolve();out.mkdir(parents=True,exist_ok=True);network=Path(network);sci=Path(sci);src=out/'source';src.mkdir(exist_ok=True)
    for n in ['model_fixed.h','datatypes.h','vars_fixed.h','predictors.h']:shutil.copy2(network/n,src/n)
    text=(network/'sirnn_fixed.cpp').read_text();text=text.replace('void sirnnFixed(MYINT *X_temp, int64_t* res)', 'void sirnnFixed(MYINT *X_temp, int64_t* res)')
    text=text.replace('i <  (int32_t)99; i++){','i <  (int32_t)99; i++){\n observed_step=i;')
    output_re=re.compile(r'^(\s*)((?:MatMul|MatAdd|Sigmoid|TanH|MulCir|MatSubBroadCastA|ScalarMul|MatAddBroadCastB)\([^;]+;)',re.M)
    def capture(m):
        call=m[2];args=call[call.index('(')+1:call.rindex(')')].split(',');name=call[:call.index('(')]
        out_arg=args[-2 if name=='MatMul' else -1].strip();base=re.sub(r'bw\d+$','',out_arg)
        if base=='tmp11':base='H'
        count=16 if base=='tmp19' else 35 if base=='tmp23' else 30 if base in ['tmp39','tmp40'] else 100
        bits=8 if base in ['tmp23','tmp39','tmp40'] else 16
        return m[0]+'\n record_full("%s",%s,%d,%d);'%(base,out_arg,count,bits)
    text=output_re.sub(capture,text)
    text=text.replace('\tMatMul( (int32_t)1,  (int32_t)100,  (int32_t)30,','\tobserved_step=99;\n\tMatMul( (int32_t)1,  (int32_t)100,  (int32_t)30,')
    (src/'instrumented.cpp').write_text(text);(src/'main.cpp').write_text(MAIN);(src/'trace.h').write_text(HEADER)
    libs=sci/'build/install/lib';cmd=['g++','-O3','-std=c++17','-pthread','-maes','-msse4.1','-mavx2','-mrdseed','-fopenmp','-DSCI_OT=1','-Wno-deprecated-declarations','-I'+str(src),'-I'+str(sci/'src'),'-I'+str(sci/'extern/eigen'),'-include',str(src/'trace.h'),str(src/'main.cpp'),str(src/'instrumented.cpp'),str(sci/'src/library_fixed.cpp'),'-Wl,--start-group']+[str(libs/('libSCI-'+x+'.a'))for x in ['OT','LinearOT','GC','Math','BuildingBlocks','FloatingPoint']]+['-Wl,--end-group','-lssl','-lcrypto','-lgmp','-o',str(out/'sci_full')]
    (out/'build_command.json').write_text(json.dumps(cmd,indent=2)+'\n')
    with(out/'compile.log').open('w')as f:subprocess.run(cmd,stdout=f,stderr=subprocess.STDOUT,check=True)

def validate(out,x_csv,sample,header,skip_build,network,sci,cpus=(0,2)):
    out=Path(out).resolve();out.mkdir(parents=True,exist_ok=True)
    if not skip_build:build(out,network,sci)
    with Path(x_csv).open()as f:line=next(v for i,v in enumerate(f)if i==sample)
    (out/'X.csv').write_text(line);logs=[];ps=[]
    with socket.socket()as s:s.bind(('127.0.0.1',0));port=s.getsockname()[1]
    try:
        for party in [1,2]:
            f=(out/('party%d.log'%party)).open('w');logs.append(f)
            ps.append(subprocess.Popen(['taskset','-c',str(cpus[party-1]),str(out/'sci_full'),str(party),str(port),str(out/'X.csv')],stdout=f,stderr=subprocess.STDOUT))
        for p in ps:
            if p.wait(timeout=180):raise RuntimeError('SCI full failed')
    finally:
        for p in ps:
            if p.poll() is None:p.kill()
            p.wait()
        for f in logs:f.close()
    actual=parse_full_output((out/'party2.log').read_text());pred,logits,steps=vo.infer(load_rows(x_csv,[sample]),vo.load(header),True);diff=[];count=0
    for t,tr in enumerate(steps):
        for name,v in tr.items():
            key=(0,t,name);expected=v[0].tolist();got=actual['traces'].get(key)
            count+=len(expected)
            if got!=expected:diff.append({'step':t,'tensor':name,'actual':got,'expected':expected})
    got_logits=actual['traces'].get((0,99,'tmp40'))
    if got_logits!=logits[0].tolist():diff.append({'step':99,'tensor':'tmp40','actual':got_logits,'expected':logits[0].tolist()})
    if actual['predictions'].get(0)!=int(pred[0]):diff.append({'tensor':'prediction','actual':actual['predictions'].get(0),'expected':int(pred[0])})
    result={'sample_id':sample,'status':'equal'if not diff else'mismatch','checked_intermediate_values':count+30,'prediction':int(pred[0]),'differences':len(diff),'source':'original ticket02 generated SCI graph; independent integer oracle'}
    (out/'validation.json').write_text(json.dumps(result,indent=2)+'\n');(out/'differences.json').write_text(json.dumps(diff,indent=2)+'\n')
    if diff:raise ValueError('full SCI oracle difference')
    return result
if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True);p.add_argument('--network',type=Path,required=True);p.add_argument('--sci',type=Path,default=Path('/home/jiang/EzPC/SCI'));p.add_argument('--input-csv',type=Path,required=True);p.add_argument('--sample',type=int,default=0);p.add_argument('--model-header',type=Path,required=True);p.add_argument('--skip-build',action='store_true');p.add_argument('--build-only',action='store_true');p.add_argument('--cpus',default='0,2')
    a=p.parse_args()
    if a.build_only:build(a.output,a.network,a.sci)
    else:print(json.dumps(validate(a.output,a.input_csv,a.sample,a.model_header,a.skip_build,a.network,a.sci,[int(x)for x in a.cpus.split(',')]),indent=2))
