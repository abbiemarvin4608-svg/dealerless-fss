#!/usr/bin/env python3
"""Export a numerical contract checked against generated calls and quantized weights."""
import argparse
import csv
import gzip
import json
import math
from pathlib import Path
import re
import shutil
import tarfile
import numpy as np


def dump(path, obj):
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False) + '\n')

def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--archive', type=Path, required=True)
    p.add_argument('--runtime', type=Path, required=True)
    p.add_argument('--sci', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args(); out = a.output.resolve(); out.mkdir(parents=True, exist_ok=True)
    snapshot = out / 'sources'; snapshot.mkdir(exist_ok=True)
    for sub in ['SeeDot/temp/Predictor', 'FastGRNN/Google-30']:
        target = snapshot / sub; target.mkdir(parents=True, exist_ok=True)
        for source in (a.archive / sub).iterdir():
            if source.is_file(): shutil.copy2(source, target / source.name)
    for file in ['seedot_preparation.json', 'summary.json', 'fingerprints_google30.json',
                 'secure_samples.csv', 'DATA_TRAINING.md', 'LICENSE_EdgeML.txt', 'LICENSE_EzPC',
                 'models/Google-30-FastGRNN-r16-u35-h100/training_config.json',
                 'SeeDot/model/fastgrnn/Google-30/input.sd',
                 'data/Google-30-processed/preprocessing.json',
                 'data/Google-30-processed/test_samples.jsonl']:
        target = snapshot / file; target.parent.mkdir(parents=True, exist_ok=True); shutil.copy2(a.archive / file, target)
    # Full original headers/sources prevent future installed defaults changing the instrumented program.
    with tarfile.open(snapshot / 'sci_src.tar.gz', 'w:gz') as tar:
        tar.add(a.sci / 'src', arcname='src')
    model_header = (a.archive / 'SeeDot/temp/Predictor/model_fixed.h').read_text()
    weights = {}
    for name in ['W1','W2','U1','U2','Bg','Bh','FC','FCbias']:
        match = re.search(r'\b' + name + r'_temp[^=]+=\s*([^;]+);', model_header)
        integers = np.array([int(x) for x in re.findall(r'-?\d+', match[1])])
        original = np.load(a.archive / 'models/Google-30-FastGRNN-r16-u35-h100' / (name + '.npy')).reshape(-1)
        matches = [f for f in range(-4, 28) if np.array_equal(np.trunc(original.astype('float64') * 2.**f).astype('int64'), integers)]
        if len(matches) != 1: raise ValueError('non-unique model scale: ' + name)
        demote = 0 if name == 'Bh' else 8
        weights[name] = dict(bits=16 if name == 'Bh' else 8, frac=matches[0]-demote,
                             quantization='trunc(float64(weight) * 2^%d) then signed divide by 2^%d' % (matches[0], demote),
                             elements=int(integers.size), scale_evidence='all archived encoded weights match uniquely')
    # Each fraction follows the frozen source's multiplicative scaling, checked weight encodings,
    # exact sigmoid/tanh interface constants, and recurrent scale consistency.
    tensors = {name:dict(bits=bits, frac=frac) for name,bits,frac in [
        ('X',16,11),('tmp12',16,11),('tmp15',16,11),('tmp19',16,10),('tmp21',16,7),('H',16,16),
        ('tmp23',8,5),('tmp25',16,8),('tmp26',16,7),('tmp27_before_shl',16,2),
        ('tmp27',16,7),('tmp28',16,14),('tmp29',16,7),('tmp30',16,7),
        ('tmp31',16,16),('one',16,14),('tmp33',16,14),('zeta',16,17),
        ('tmp34',16,17),('nu',16,23),('tmp35',16,17),('tmp36',16,17),
        ('tmp39',8,2),('tmp40',8,2),('argmax_input',16,2),('class_index',8,0) ]}
    tensors.update(weights)
    specs = [
      ('MatMul_01',['tmp15','W1'],'tmp19',32), ('MatMul_02',['tmp19','W2'],'tmp21',16),
      ('MatMul_03',['H','U1'],'tmp23',100), ('MatMul_04',['tmp23','U2'],'tmp25',35),
      ('MatAdd_01',['tmp21','tmp25'],'tmp26',2), ('MatAdd_02',['tmp26','Bg'],'tmp27_before_shl',2),
      ('AdjustScaleShl_01',['tmp27_before_shl'],'tmp27',1), ('Sigmoid_01',['tmp27'],'tmp28',1),
      ('MatAdd_03',['tmp26','Bh'],'tmp29',2), ('TanH_01',['tmp29'],'tmp30',1),
      ('MulCir_01',['tmp28','H'],'tmp31',1), ('MatSubBroadCastA_01',['one','tmp28'],'tmp33',2),
      ('ScalarMul_01',['zeta','tmp33'],'tmp34',1), ('MatAddBroadCastB_01',['tmp34','nu'],'tmp35',2),
      ('MulCir_02',['tmp35','tmp30'],'tmp36',1), ('MatAdd_04',['tmp31','tmp36'],'H',2),
      ('MatMul_05',['H','FC'],'tmp39',100), ('MatAdd_05',['tmp39','FCbias'],'tmp40',2),
      ('ArgMax_01',['argmax_input'],'class_index',30)]
    source = (a.archive / 'FastGRNN/Google-30/sirnn_fixed.cpp').read_text()
    pattern = re.compile(r'^\s*((?:MatMul|MatAdd|MatSubBroadCastA|MatAddBroadCastB|Sigmoid|TanH|MulCir|ScalarMul|ArgMax|AdjustScaleShl))\(([^\n]+)\);', re.M)
    calls = list(pattern.finditer(source)); assert len(calls) == len(specs)
    observed = json.loads((out / 'sci_trace/primitive_counts.json').read_text())
    operators = []
    for call, (name, inputs, output, length) in zip(calls, specs):
        if call[1] != name.rsplit('_',1)[0]: raise ValueError('generated operation order changed')
        arguments = [x.strip() for x in re.sub(r'\(int(?:32|64)_t\)', '', call[2]).split(',')]
        scalars = [int(x) for x in arguments if re.fullmatch(r'\d+',x)]
        repeated = name not in ['MatMul_05','MatAdd_05','ArgMax_01']
        op = dict(id=name, inputs=[dict(name=n, **tensors[n]) for n in inputs],
                  output=dict(name=output, **tensors[output]), calls_per_inference=99 if repeated else 1,
                  steps=list(range(99)) if repeated else [99], accumulation_length=length,
                  source_line=source[:call.start()].count('\n') + 1, generated_call=call.group(0).strip(),
                  overflow='two-complement modulo the stated destination ring; no saturation',
                  sign='signed two-complement except sigmoid output and class index are nonnegative',
                  observed_primitives=[{k:v for k,v in g.items() if k not in ['operator','steps']} for g in observed if g['operator']==name])
        if name.startswith('MatMul'):
            _,K,_,shrA,shrB,H1,H2,demote,bwA,bwB,bwTemp,bwC = scalars
            depth = math.ceil(math.log2(K)); shift = int(math.log2(shrA*shrB*demote)) + H1-depth
            expected_frac = sum(tensors[n]['frac'] for n in inputs) - int(math.log2(shrA*shrB*demote)) - H1
            if expected_frac != tensors[output]['frac']: raise ValueError('scale mismatch at '+name)
            if [tensors[n]['bits'] for n in inputs] != [bwA,bwB] or tensors[output]['bits'] != bwC:
                raise ValueError('bitwidth mismatch at '+name)
            op.update(product_bits=bwA+bwB, accumulator_bits=bwA+bwB+depth, source_declared_temp_bits=bwTemp,
                      accumulator_frac=sum(tensors[n]['frac'] for n in inputs),
                      see_dot_rule=dict(temp_bits=bwTemp, tree_divide_levels=H1, tree_unscaled_levels=H2,
                                        final_divisors=[shrA,shrB,demote], rounding='signed division toward zero at every tree stage'),
                      sci_rule=dict(extended_sum_bits=bwA+bwB+depth, first_shift=depth,
                                    first_rounding='floor via exact truncate_and_reduce, output modulo 2^(bwA+bwB)',
                                    second_shift=shift, second_rounding='signed division toward zero if positive; exact left shift if nonpositive',
                                    output_reduction_bits=bwC))
        elif name.startswith(('MulCir','ScalarMul')):
            shift=int(math.log2(scalars[2]*scalars[3]*scalars[4]))
            if sum(tensors[n]['frac'] for n in inputs)-shift != tensors[output]['frac']:
                raise ValueError('scale mismatch at '+name)
            op.update(product_bits=32,accumulator_bits=32,truncate_after='elementwise product',truncate_shift=shift,
                      rounding='signed divide by 2^shift toward zero, then destination reduction')
        elif name.startswith(('Sigmoid','TanH')):
            op.update(product_bits=None,accumulator_bits=None,rounding='SCI integer LUT/division approximation; see frozen math-functions.cpp and dynamic inner trace',
                      function_io_bits=[16,16],function_io_frac=[7,14 if name.startswith('Sigmoid') else 7],
                      see_dot_rule='FLOATEXP float exp/tanh then truncating cast; not a bit-exact SCI oracle')
        elif name.startswith('ArgMax'):
            op.update(product_bits=None,accumulator_bits=16,rounding='none; strict comparison of signed differences, earlier class wins ties',
                      logical_comparisons=29, comparison_input_bits=15,
                      preprocessing='8-bit logits sign-extended to 16 bits before comparison; index output is ring 2^8')
        elif name.startswith('AdjustScale'):
            op.update(product_bits=None,accumulator_bits=16,rounding='exact multiply by 32; low 16 bits interpreted by subsequent interface',truncate_shift=-5)
        else:
            for n,divisor in zip(inputs,[scalars[2]*scalars[4],scalars[3]*scalars[4]]):
                if tensors[n]['frac']-int(math.log2(divisor*scalars[5])) != tensors[output]['frac']:
                    raise ValueError('scale mismatch at '+name)
            op.update(product_bits=None,accumulator_bits=scalars[8],
                      input_divisors=[scalars[2]*scalars[4], scalars[3]*scalars[4]],output_divisor=scalars[5],
                      rounding='sign-extend to temporary ring; signed division toward zero of each input, add/subtract modulo temporary ring, divide then reduce')
        operators.append(op)
    # Make the classification input conversion explicit; it is not part of the preceding bias addition.
    operators.insert(-1, dict(id='ArgMaxInputSignExtend_01',calls_per_inference=1,steps=[99],
          inputs=[dict(name='tmp40',**tensors['tmp40'])],output=dict(name='argmax_input',**tensors['argmax_input']),
          product_bits=None,accumulator_bits=None,accumulation_length=1,rounding='none; exact signed extension 8 to 16',
          observed_primitives=[{k:v for k,v in g.items() if k not in ['operator','steps']} for g in observed if g['operator']=='ArgMaxInputSignExtend_01']))
    contract = dict(contract_id='google30-historical-20261007-sci-source-v1',
        scope='local retraining of official EdgeML/SeeDot/SCI; original historical model unchanged',
        reference_contract='SeeDot historical replay is a separate approximation contract, predictions compared only',
        representation='real value = signed(integer_mod_2^bits) * 2^-frac; fraction can equal/exceed total width',
        timesteps=99,inputs_per_step=32,hidden=100,classes=30,tensors=tensors,operators=operators,
        data_movement=[{'operation':'input encode','shape':[3168,1],'rule':'trunc(float64(CSV)*2^11), exact 16-bit signed range'},
                       {'operation':'slice X[32*i:32*i+32] to tmp12 then reshape to tmp15','steps':list(range(99)),
                        'input_bits':16,'output_bits':16,'input_frac':11,'output_frac':11,
                        'rule':'local share copies, no truncation or arithmetic'}],
        initialization={'H':'100 zero integers at bits=16, frac=16','zeta':16654,'nu':26006,'one':16384},
        sharing={'arithmetic':'additive modulo 2^k', 'boolean':'XOR bit shares',
                 'ownership':'party1 privately holds matrix weights and biases; party2 holds inputs; output class reconstructed only to party2; learned zeta/nu are public compiled constants',
                 'server_private_parameters':['W1','W2','U1','U2','Bg','Bh','FC','FCbias'],
                 'public_model_scalars':{'zeta':16654,'nu':26006},
                 'public_program_constants':{'one':16384},
                 'scalar_multiplication':'SCI ScalarMul uses the public scalar A[0] locally; this graph does not protect every learned parameter'},
        fss={'DPF':'not used by native SCI path','DCF':'not used by native SCI path',
             'OT_LUT':'actual domains 2/3/6/8, see dynamic counts; lookup_table is sender table/receiver selection OT, not shared-input FSS'},
        truncation_randomness='recorded div_pow2 and truncate_and_reduce compute deterministic integer functions; sharing randomness is not numerical randomized rounding',
        unresolved=[{'item':'SeeDot versus SCI all intermediate integers','status':'not equivalent by source design; no claim of equality',
                     'ticket05_blocker':False,'action':'use original SCI reconstructed output as the oracle at replacement boundaries'},
                    {'item':'whole-test-set SCI predictions','status':'not executed; history has 30 and tracing adds one repeated sample',
                     'ticket05_blocker':False,'action':'ticket07 performs full fixed prediction agreement on independent model'},
                    {'item':'global real-number error certificate for SCI reciprocal/LUT approximation',
                     'status':'not derived here; exact coefficient generation and integer formula frozen in source snapshot',
                     'ticket05_blocker':False,'action':'retain implementation in ticket05; separate if replacing activations'}],
        ticket05_handoff='MSB/strict comparison input-output semantics confirmed; implement same SCI integer oracle, preserve arithmetic, Boolean conversion and fresh dealerless preprocessing')
    dump(out / 'numeric_contract.json', contract)
    fields=['id','calls_per_inference','inputs','output','product_bits','accumulator_bits','accumulation_length','rounding']
    with (out / 'operator_contract.csv').open('w') as f:
        w=csv.DictWriter(f,fieldnames=fields);w.writeheader()
        for op in operators:
            w.writerow({k:json.dumps(op[k],ensure_ascii=False) if isinstance(op.get(k), (list,dict)) else op.get(k,'see sci_rule') for k in fields})
    # Width histograms keep nested levels separate: MSB, wrap and their compare children cannot be added together.
    hist=[]
    for kind in sorted(set(g['primitive'] for g in observed)):
        groups=[g for g in observed if g['primitive']==kind]; total=sum(g['scalar_instances'] for g in groups)
        for bits in sorted(set(g['input_bits'] for g in groups)):
            matching=[g for g in groups if g['input_bits']==bits]; n=sum(g['scalar_instances'] for g in matching)
            hist.append(dict(primitive=kind,input_bits=bits,invocations=sum(g['invocations'] for g in matching),scalar_instances=n,fraction_within_primitive=n/total))
    with (out / 'primitive_widths.csv').open('w') as f:
        w=csv.DictWriter(f,fieldnames=list(hist[0]));w.writeheader();w.writerows(hist)
    metadata=dict(artifact_version=1,date='2026-10-08',archive=str(a.archive.resolve()),runtime=str(a.runtime.resolve()),sci=str(a.sci.resolve()),
         source_commits=json.loads((a.archive/'summary.json').read_text())['commits'],
         calibration='512 indexed training rows, archived seedot_preparation.json; historical fixer used full train/test ranges',
         checkpoint='seed42, epoch205 selected by compressed-phase test accuracy; not independent evaluation',
         preprocessing='full train mean/std; pad16000 samples; log fbank nfilt32 numpy.hamming; 99x32 row-major; permutation seed42',
         input_files={name:str((a.runtime/'SeeDot/temp/Predictor/input'/name).resolve()) for name in ['X.csv','Y.csv']},
         input_order='sources/data/Google-30-processed/test_samples.jsonl',
         scope='No model training/quantization modified, no historical files written; no performance claim from instrumented run')
    dump(out/'manifest.json',metadata)
    print(json.dumps(dict(operator_classes=len(operators),weights=weights),ensure_ascii=False))

if __name__ == '__main__': main()
