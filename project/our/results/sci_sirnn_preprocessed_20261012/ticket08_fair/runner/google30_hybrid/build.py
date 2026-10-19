#!/usr/bin/env python3
"""Build a source-extracted complete recurrent step with an isolated SCI hook."""
import argparse,hashlib,json,re,subprocess
from pathlib import Path
from step import extract_calls
HERE=Path(__file__).resolve().parent
TRACE='''#pragma once
#include <iostream>
extern bool diagnostic;
namespace step_trace { inline const char* op="input"; inline int case_index=0; }
'''
TEMPLATE='''#include "library_fixed.h"
#include "globals.h"
#include "sci_hook.h"
#include <fstream>
#include <vector>
#include <stdexcept>
#include <chrono>
#include <iomanip>
#include <cstdlib>
#include <algorithm>
#include <sys/resource.h>
using namespace std;
int party=1,port=19580,num_threads=1;int32_t bitlength=32;
string address="127.0.0.1";bool diagnostic=false;
using Clock=chrono::steady_clock;
double milliseconds(Clock::time_point start){return chrono::duration<double,milli>(Clock::now()-start).count();}
struct Meter {uint64_t sci_sent,sci_rounds,bridge_sent,bridge_received,bridge_rounds;};
Meter meter(){
 auto bridge=sci_segmented::stats();
 return {iopack?iopack->get_comm():0,iopack?iopack->get_rounds():0,
  bridge.setup_sent_bytes+bridge.offline_sent_bytes+bridge.online_sent_bytes,
  bridge.setup_received_bytes+bridge.offline_received_bytes+bridge.online_received_bytes,
  bridge.setup_rounds+bridge.offline_rounds+bridge.online_rounds};
}
void phase(const char* name,int query,double wall_ms,const Meter& before){
 auto after=meter();
 cout<<"PHASE name="<<name<<" query="<<query<<" wall_ms="<<wall_ms
 <<" sci_sent_bytes="<<after.sci_sent-before.sci_sent<<" sci_rounds="<<after.sci_rounds-before.sci_rounds
 <<" bridge_sent_bytes="<<after.bridge_sent-before.bridge_sent
 <<" bridge_received_bytes="<<after.bridge_received-before.bridge_received
 <<" bridge_rounds="<<after.bridge_rounds-before.bridge_rounds<<"\\n";
}
extern void reconstruct(int64_t*,int64_t*,int32_t,int32_t,int);
void show(const char* name,int64_t* values,int count,int bits){
 if(!diagnostic)return;
 vector<int64_t> out(count);reconstruct(values,out.data(),1,count,bits);
 if(party==2)for(int i=0;i<count;++i)cout<<"VALUE "<<step_trace::case_index<<" "<<name<<" "<<i<<" "<<out[i]<<"\\n";
}
void read_private(ifstream& input,int64_t* values,int count,int owner){
 if(party==owner)for(int i=0;i<count;++i)if(!(input>>values[i]))throw runtime_error("private input ended");
}
int main(int argc,char**argv){
 if(argc!=6&&argc!=7)throw runtime_error("party port input-prefix cases diagnostic|secure [batch]");
 party=stoi(argv[1]);port=stoi(argv[2]);int cases=stoi(argv[4]);diagnostic=string(argv[5])=="diagnostic";
 const int batch=argc==7?stoi(argv[6]):1;
 if(cases<1||batch<1||batch>128)throw runtime_error("invalid cases or batch");
 cout<<setprecision(12);
 ifstream input(string(argv[3])+"-P"+to_string(party-1)+"-0");
 int64_t W1[512]={},W2[1600]={},U1[3500]={},U2[3500]={},Bg[100]={},Bh[100]={};
 read_private(input,W1,512,1);read_private(input,W2,1600,1);read_private(input,U1,3500,1);
 read_private(input,U2,3500,1);read_private(input,Bg,100,1);read_private(input,Bh,100,1);
 auto setup_start=Clock::now();initialize();
 phase("sci_setup",-1,milliseconds(setup_start),Meter{});
 const bool hybrid=sci_segmented::activation_enabled();
 auto before_bridge=meter();auto bridge_start=Clock::now();
 if(hybrid){
  sci_segmented::ensure(party);
 }
 phase("bridge_setup",-1,milliseconds(bridge_start),before_bridge);
 for(int cse=0;cse<cases;++cse){
 step_trace::case_index=cse;
 DECLARATIONS
 int64_t tmp9bw16=16654,tmp10bw16=26006,tmp32=16384;
 auto before_offline=meter();auto offline_start=Clock::now();
 if(hybrid){
  sci_segmented::precompute_activations(batch,1,100,32);
 }
 phase("offline",cse,milliseconds(offline_start),before_offline);
 auto before_ready=meter();auto ready_start=Clock::now();io->sync();
 phase("ready",cse,milliseconds(ready_start),before_ready);
 auto before_online=meter();auto before_stats=sci_segmented::stats();auto online_start=Clock::now();
 double sigmoid_ms=0,tanh_ms=0;
 for(int sample=0;sample<batch;++sample){
  read_private(input,tmp15bw16+sample*100,32,2);read_private(input,tmp11bw16+sample*100,100,2);
 }
 CALLS
 io->sync();
 phase(diagnostic?"diagnostic_online":"online",cse,milliseconds(online_start),before_online);
 auto after_stats=sci_segmented::stats();
 if(hybrid&&after_stats.offline_ms!=before_stats.offline_ms)throw runtime_error("offline material generated inside online phase");
 cout<<"ACTIVATION query="<<cse<<" sigmoid_ms="<<sigmoid_ms<<" tanh_ms="<<tanh_ms
 <<" total_ms="<<sigmoid_ms+tanh_ms<<" bridge_ms="<<after_stats.activation_online_ms-before_stats.activation_online_ms
 <<" lut_ms="<<after_stats.activation_lut_online_ms-before_stats.activation_lut_online_ms
 <<" calls="<<after_stats.activation_calls-before_stats.activation_calls
 <<" elements="<<after_stats.activation_elements-before_stats.activation_elements<<"\\n";
 }
 auto stats=sci_segmented::stats();
 cout<<"BRIDGE calls="<<stats.calls<<" elements="<<stats.elements<<" segment_keys="<<stats.segment_keys<<" offline_ms="<<stats.offline_ms<<" online_ms="<<stats.online_ms<<" activation_calls="<<stats.activation_calls<<" activation_elements="<<stats.activation_elements<<" activation_keys="<<stats.activation_keys<<" activation_offline_ms="<<stats.activation_offline_ms<<" activation_online_ms="<<stats.activation_online_ms<<" offline_sent_bytes="<<stats.offline_sent_bytes<<" offline_received_bytes="<<stats.offline_received_bytes<<" online_sent_bytes="<<stats.online_sent_bytes<<" online_received_bytes="<<stats.online_received_bytes<<"\\n";
 sci_segmented::finish();finalize();
 struct rusage usage;getrusage(RUSAGE_SELF,&usage);
 cout<<"STEP_FINISHED mode="<<(diagnostic?"diagnostic":"secure")<<" cases="<<cases<<" batch="<<batch<<" peak_rss_kib="<<usage.ru_maxrss<<"\\n";
}
'''

def source_for(schema):
 names={name for row in schema for name in re.findall(r'\btmp\d+bw\d+\b',row['source_call'])}-{'tmp9bw16','tmp10bw16'}
 declarations='\n'.join('vector<int64_t> '+name+'_storage(batch*100,0); int64_t* '+name+'='+name+'_storage.data();'for name in sorted(names))
 calls=[]
 for row in schema:
  if row['function'] in ('Sigmoid','TanH'):
   kind=0 if row['function']=='Sigmoid' else 1
   output=row['output'];bits=row['bits']
   input_name=re.findall(r'\btmp\d+bw\d+\b',row['source_call'])[0]
   args=row['source_call'].split('(',1)[1].rstrip('; )').split(', ')
   if args[:2]!=['1','100']:raise RuntimeError('unexpected activation shape')
   args[1]='100*batch'
   native_call=row['function']+'('+', '.join(args)+');'
   calls += ['step_trace::op="'+row['operator']+'";','{auto activation_start=Clock::now();',
    'if(sci_segmented::activation_enabled()){',
    ' vector<uint64_t> fss_in(100*batch),fss_out(100*batch);',
    ' xt->s_extend(100*batch,reinterpret_cast<uint64_t*>('+input_name+'),fss_in.data(),'+str(bits)+',32);',
    ' sci_segmented::activation('+str(kind)+',fss_in.data(),fss_out.data(),100*batch,32);',
    ' for(int i=0;i<100*batch;++i) '+output+'[i]=fss_out[i]&((uint64_t(1)<<'+str(bits)+')-1);',
    '}else{'+native_call+'}',('sigmoid_ms'if kind==0 else 'tanh_ms')+'+=milliseconds(activation_start);}']
  else:
   sample_call=re.sub(r'\btmp\d+bw\d+\b',lambda match:'('+match[0]+'+sample*100)'if match[0]in names else match[0],row['source_call'])
   calls+=['step_trace::op="'+row['operator']+'";','for(int sample=0;sample<batch;++sample){'+sample_call+'}']
  count=16 if row['tensor']=='tmp19'else 35 if row['tensor']=='tmp23'else 100
  calls+=['for(int sample=0;sample<batch;++sample)show("'+row['tensor']+'",'+row['output']+'+sample*100,'+str(count)+','+str(row['bits'])+');']
 return TEMPLATE.replace('DECLARATIONS',declarations).replace('CALLS','\n'.join(calls))

def build(out,sci,our,bridge_lib,generated):
 out=Path(out).resolve();out.mkdir(parents=True,exist_ok=True);sci=Path(sci).resolve();our=Path(our).resolve();bridge_lib=Path(bridge_lib).resolve()
 schema=extract_calls(Path(generated).read_text());(out/'schema.json').write_text(json.dumps(schema,indent=2)+'\n')
 (out/'source_metadata.json').write_text(json.dumps({'generated':str(Path(generated).resolve()),'generated_sha256':hashlib.sha256(Path(generated).read_bytes()).hexdigest(),'variant':'independent'if len(schema)==15 else'historical'},indent=2)+'\n')
 (out/'step.cpp').write_text(source_for(schema));(out/'step_trace.h').write_text(TRACE)
 aux=sci/'src/BuildingBlocks/aux-protocols.cpp';text=aux.read_text()
 needle='  assert(bw_x <= 64);\n  int32_t shift = bw_x - 1;'
 if text.count(needle)!=1:raise RuntimeError('SCI MSB source shape changed')
 dispatch='''  if (sci_segmented::msb(party, x, msb_x, size, bw_x)) {
    if(diagnostic)std::cout << "MSB_CALL case=" << step_trace::case_index << " operator=" << step_trace::op << " input_bits=" << bw_x << " comparison_bits=" << bw_x-1 << " count=" << size << "\\n";
    return;
  }
'''
 text=text.replace(needle,dispatch+needle,1)
 patched=out/'aux-protocols-hook.cpp';patched.write_text('#include "sci_hook.h"\n'+text)
 libs=sci/'build/install/lib'
 flags=['g++','-O3','-std=c++17','-pthread','-maes','-msse4.1','-mavx2','-mrdseed','-fopenmp','-DSCI_OT=1','-Wno-deprecated-declarations','-I'+str(sci/'src'),'-I'+str(sci/'extern/eigen'),'-I'+str(our/'src'),'-I'+str(our/'tools/sci_bridge'),'-include',str(out/'step_trace.h')]
 cmd=flags+[str(out/'step.cpp'),str(sci/'src/library_fixed.cpp'),str(patched),str(bridge_lib),'-Wl,-rpath,'+str(bridge_lib.parent),'-Wl,--start-group']+[str(libs/('libSCI-'+x+'.a'))for x in ['OT','LinearOT','GC','Math','BuildingBlocks','FloatingPoint']]+['-Wl,--end-group','-lssl','-lcrypto','-lgmp','-o',str(out/'sci-step-hook')]
 (out/'build_command.json').write_text(json.dumps(cmd,indent=2)+'\n')
 result=subprocess.run(cmd,text=True,capture_output=True);(out/'compile.log').write_text(result.stdout+result.stderr)
 if result.returncode:raise RuntimeError('compile failed; inspect '+str(out/'compile.log'))
 return out/'sci-step-hook'
if __name__=='__main__':
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--out',type=Path,required=True);p.add_argument('--sci',type=Path,default=Path('/home/jiang/EzPC/SCI'));p.add_argument('--our',type=Path,default=HERE.parent.parent);p.add_argument('--bridge-lib',type=Path,required=True);p.add_argument('--generated',type=Path,required=True)
 a=p.parse_args();print(build(a.out,a.sci,a.our,a.bridge_lib,a.generated))
