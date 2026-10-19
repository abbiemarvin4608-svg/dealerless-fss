#!/usr/bin/env python3
"""Build the native SCI 99-step graph, vectorized across independent examples."""
import argparse,hashlib,json,re,subprocess
from pathlib import Path
from step import extract_calls
HERE=Path(__file__).resolve().parent
LIBS=['OT','LinearOT','GC','Math','BuildingBlocks','FloatingPoint']

def generate(source):
 recurrent=extract_calls(source)
 if len(recurrent)!=15:raise ValueError('requires independent02 graph with 15 recurrent calls')
 classification=[]
 for match in re.finditer(r'^\s*(MatMul|MatAdd)\(([^;]+)\);',source,re.M):
  call=match[1]+'('+re.sub(r'\(\s*int(?:32|64)_t\s*\)','',match[2])+');'
  if 'FC,'in call or'FCbias,'in call:
   output='tmp39bw8'if match[1]=='MatMul'else'tmp40bw8'
   classification.append(dict(operator='MatMul_05'if match[1]=='MatMul'else'MatAdd_07',function=match[1],source_call=call,output=output,tensor=output[:-3],bits=8,frac=2))
 if len(classification)!=2:raise ValueError('classifier source extraction incomplete')
 schema=recurrent+classification
 names={name for row in schema for name in re.findall(r'\btmp\d+bw\d+\b',row['source_call'])}-{'tmp9bw16','tmp10bw16'}
 widths={name:100 for name in names};widths.update(tmp15bw16=32,tmp19bw16=16,tmp23bw8=35,tmp39bw8=30,tmp40bw8=30)
 declarations='\n'.join(f'vector<int64_t> storage_{name}(batch*{widths[name]},0); int64_t* {name}=storage_{name}.data();'for name in sorted(names))
 fills='\n'.join(f'fill(storage_{name}.begin(),storage_{name}.end(),0);' for name in sorted(names))
 def calls(rows):
  result=[]
  for row in rows:
   args=row['source_call'][row['source_call'].index('(')+1:-2].split(',');args[0]='batch'
   call=row['function']+'('+','.join(args)+');'
   cols=widths[row['output']]
   if row['function'] in ('Sigmoid','TanH'):
    kind=0 if row['function']=='Sigmoid' else 1
    input_name=re.findall(r'\btmp\d+bw\d+\b',row['source_call'])[0]
    result += [
      'if(sci_segmented::activation_enabled()){',
      ' vector<uint64_t> fss_in(batch*100),fss_out(batch*100);',
      ' xt->s_extend(batch*100,reinterpret_cast<uint64_t*>('+input_name+'),fss_in.data(),'+str(row['bits'])+',32);',
      ' sci_segmented::activation('+str(kind)+',fss_in.data(),fss_out.data(),batch*100,32);',
      ' for(int i=0;i<batch*100;++i) '+row['output']+'[i]=fss_out[i]&((uint64_t(1)<<'+str(row['bits'])+')-1);',
      '}else{'+call+'}',
    ]
   else:
    result += [call]
   result += [f'inspect("{row["operator"]}","{row["tensor"]}",{row["output"]},batch,{cols},{row["bits"]});']
  return '\n'.join(result)
 text=(HERE/'full_driver.cpp.in').read_text().replace('FILL_GENERATED',fills).replace('DECLARATIONS',declarations).replace('RECURRENT_CALLS',calls(recurrent)).replace('CLASSIFIER_CALLS',calls(classification))
 return text,schema

def build(out,sci,generated,our,bridge_lib):
 out=Path(out).resolve();out.mkdir(parents=True,exist_ok=True);sci=Path(sci).resolve();our=Path(our).resolve();bridge_lib=Path(bridge_lib).resolve()
 driver,schema=generate(Path(generated).read_text());(out/'driver.cpp').write_text(driver);(out/'schema.json').write_text(json.dumps(schema,indent=2)+'\n')
 # Compile against a private metered NetIO header.  Patching the source tree
 # in place would make the binary impossible to reproduce and would also
 # affect unrelated SCI builds.
 measurement=(our/'tools/google30_benchmark/measurement.h').resolve()
 (out/'measurement.h').write_text(measurement.read_text())
 net_override=out/'include_override/utils';net_override.mkdir(parents=True,exist_ok=True)
 net=(sci/'src/utils/net_io_channel.h').read_text()
 net=net.replace('#include "utils/io_channel.h"', '#include "utils/io_channel.h"\n#include "measurement.h"', 1)
 net=net.replace('  void send_data(const void *data, int len) {', '  void send_data(const void *data, int len) {\n    google30_measure::sent += len; ++google30_measure::sends;', 1)
 net=net.replace('  void recv_data(void *data, int len) {', '  void recv_data(void *data, int len) {\n    google30_measure::received += len; ++google30_measure::recvs;', 1)
 (net_override/'net_io_channel.h').write_text(net)
 text=(sci/'src/BuildingBlocks/aux-protocols.cpp').read_text();needle='  assert(bw_x <= 64);\n  int32_t shift = bw_x - 1;'
 if text.count(needle)!=1:raise ValueError('SCI MSB source changed')
 text=text.replace(needle,'  if (sci_segmented::msb(party, x, msb_x, size, bw_x)) return;\n'+needle,1)
 (out/'aux-hook.cpp').write_text('#include "sci_hook.h"\n'+text)
 libs=sci/'build/install/lib';flags=['g++','-O3','-std=c++17','-pthread','-maes','-msse4.1','-mavx2','-mrdseed','-fopenmp','-DSCI_OT=1','-Wno-deprecated-declarations','-I'+str(out),'-I'+str(out/'include_override'),'-I'+str(sci/'src'),'-I'+str(sci/'extern/eigen'),'-I'+str(our/'src'),'-I'+str(our/'tools/sci_bridge')]
 sources_to_compile = [
  (out/'driver.cpp', 'driver.o'),
  (sci/'src/library_fixed.cpp', 'library_fixed.o'),
  (out/'aux-hook.cpp', 'aux-hook.o'),
  # These translation units instantiate SCI's inline OT/GC/extension calls;
  # compiling them with the private NetIO header keeps metering complete even
  # when the corresponding archive object was built without instrumentation.
  (sci/'src/library_fixed_uniform.cpp', 'library_fixed_uniform.o'),
  (sci/'src/globals.cpp', 'globals.o'),
  (sci/'src/cleartext_library_fixed.cpp', 'cleartext_library_fixed.o'),
  (sci/'src/LinearOT/linear-ot.cpp', 'linear-ot.o'),
  (sci/'src/GC/emp-tool.cpp', 'emp-tool.o'),
  (sci/'src/GC/halfgate_gen.cpp', 'halfgate_gen.o'),
  (sci/'src/GC/halfgate_eva.cpp', 'halfgate_eva.o'),
  (sci/'src/BuildingBlocks/truncation.cpp', 'truncation.o'),
  (sci/'src/BuildingBlocks/value-extension.cpp', 'value-extension.o'),
  (sci/'src/Math/math-functions.cpp', 'math-functions.o'),
 ]
 objects=[]; compile_logs=[]
 for source,name in sources_to_compile:
  object=out/name
  compile_cmd=flags+['-c',str(source),'-o',str(object)]
  result=subprocess.run(compile_cmd,text=True,capture_output=True)
  (out/(name+'.log')).write_text(result.stdout+result.stderr)
  if result.returncode: raise RuntimeError('compile failed: '+str(out/(name+'.log')))
  objects.append(str(object)); compile_logs.append(compile_cmd)
 cmd=flags+objects+[str(bridge_lib),'-Wl,-rpath,'+str(bridge_lib.parent),'-Wl,--start-group']+[str(libs/('libSCI-'+x+'.a'))for x in LIBS]+['-Wl,--end-group','-lssl','-lcrypto','-lgmp','-o',str(out/'google30-full')]
 (out/'build_command.json').write_text(json.dumps({'compile':compile_logs,'link':cmd},indent=2)+'\n')
 result=subprocess.run(cmd,text=True,capture_output=True);(out/'compile.log').write_text(result.stdout+result.stderr)
 if result.returncode:raise RuntimeError('compile failed: '+str(out/'compile.log'))
 # Record dynamic runtime versions as well as linked SCI source/archive versions.
 def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
 linked=subprocess.check_output(['ldd',str(out/'google30-full')],text=True)
 libraries={match[1]:sha(match[1])for match in re.finditer(r'=> (/\S+)',linked)}
 compiled_sources=[source for source,_ in sources_to_compile]
 sources=[generated,HERE/'full_driver.cpp.in',measurement,net_override/'net_io_channel.h']+compiled_sources+[libs/('libSCI-'+name+'.a')for name in LIBS]
 # Preserve order while avoiding duplicate driver/library entries in the
 # manifest. Every translation unit compiled above is therefore auditable.
 sources=list(dict.fromkeys(str(Path(path).resolve()) for path in sources))
 identities={path:sha(path)for path in sources}
 manifest={'generated_sha256':sha(generated),'dynamic_libraries':libraries,'linked_sources':identities,'source_identity':hashlib.sha256(json.dumps(identities,sort_keys=True).encode()).hexdigest()}
 (out/'runtime_manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
 return out/'google30-full'
if __name__=='__main__':
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--out',type=Path,required=True);p.add_argument('--sci',type=Path,default=Path('/home/jiang/EzPC/SCI'));p.add_argument('--generated',type=Path,required=True);p.add_argument('--our',type=Path,default=HERE.parent.parent);p.add_argument('--bridge-lib',type=Path,required=True);a=p.parse_args();print(build(a.out,a.sci,a.generated,a.our,a.bridge_lib))
