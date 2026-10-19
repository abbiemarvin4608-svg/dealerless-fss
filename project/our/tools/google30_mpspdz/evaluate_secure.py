#!/usr/bin/env python3
"""Resumable real semi2k evaluation; never treats partial coverage as full quality."""
import argparse
import csv
import fcntl
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import time
import numpy as np
from full import prepare_full,compile_full,run_full,digest
HERE=Path(__file__).resolve().parent

def write_json(path,data):
    path=Path(path);temp=path.with_suffix(path.suffix+'.tmp')
    temp.write_text(json.dumps(data,indent=2)+'\n');temp.replace(path)

def write_manifest(output,data):
    """Atomically publish the declared target and per-invocation budget."""
    write_json(Path(output)/'evaluation_manifest.json',data)

def compiler_digest(spdz):
    root=Path(spdz);h=hashlib.sha256()
    for source in sorted([root/'compile.py']+list((root/'Compiler').rglob('*.py'))):
        h.update(str(source.relative_to(root)).encode());h.update(source.read_bytes())
    return h.hexdigest()

def summarize_coverage(predictions,labels,fp32,oracle):
    labels=np.asarray(labels);fp32=np.asarray(fp32);oracle=np.asarray(oracle)
    if len(labels)!=len(fp32) or len(labels)!=len(oracle):raise ValueError('oracle/dataset length mismatch')
    ids=sorted(predictions);n=len(labels)
    if any(i<0 or i>=n for i in ids):raise ValueError('sample ID outside evaluation set')
    mismatches=[{'sample_id':i,'actual':int(predictions[i]),'SCI_oracle':int(oracle[i])}for i in ids if predictions[i]!=oracle[i]]
    full=ids==list(range(n));complete=full and not mismatches
    result={'coverage_complete':full,'complete':complete,'quality_gate_pass':None,'mismatches':mismatches}
    if complete:
        actual=np.asarray([predictions[i]for i in ids]);accuracy=float(np.mean(actual==labels)*100);fp_accuracy=float(np.mean(fp32==labels)*100)
        result.update(full_top1_pct=accuracy,full_loss_pp=fp_accuracy-accuracy,full_prediction_agreement_pct=float(np.mean(actual==fp32)*100),quality_gate_pass=fp_accuracy-accuracy<=1)
    return result

def evaluate(artifact,input_csv,output,spdz,oracle_dir,batch=1,queries=1,limit=None,timeout=900,memory_mib=3072,cpus=(0,2),max_seconds=None,stop_file=None):
    artifact=Path(artifact).resolve();output=Path(output).resolve();output.mkdir(parents=True,exist_ok=True)
    lock_handle=(output/'evaluation.lock').open('w')
    try:
        fcntl.flock(lock_handle.fileno(),fcntl.LOCK_EX|fcntl.LOCK_NB)
    except BlockingIOError as exc:
        lock_handle.close();raise RuntimeError('evaluation output is already locked by another writer') from exc
    oracle_dir=Path(oracle_dir);oracle=np.load(oracle_dir/'predictions.npy');oracle_summary=json.loads((oracle_dir/'summary.json').read_text())
    if not oracle_summary.get('full_holdout'):raise ValueError('full heldout SCI oracle is required')
    model=artifact/'numeric_artifact/model_fixed.h';labels=np.load(artifact/'labels.npy');fp=np.load(artifact/'fp32_predictions.npy')
    n=len(labels)
    if len(oracle)!=n or len(fp)!=n:raise ValueError('oracle/dataset length mismatch')
    count=n if limit is None else min(limit,n);path=output/'ledger.json'
    binding={'contract':digest(artifact/'numeric_contract.json'),'model':digest(model),'labels':digest(artifact/'labels.npy'),'fp32_predictions':digest(artifact/'fp32_predictions.npy'),
             'input_csv':digest(input_csv),'secret_library':digest(HERE/'secret.mpc'),'secret_full':digest(HERE/'full_tail.mpc'),
             'spdz_git_commit':subprocess.check_output(['git','-C',str(spdz),'rev-parse','HEAD'],text=True).strip(),
             'runtime_sha256':digest(Path(spdz)/'semi2k-party.x'),'compiler_source_sha256':compiler_digest(spdz),
             'oracle_source_sha256':digest(HERE/'vector_oracle.py'),'oracle_predictions_sha256':digest(oracle_dir/'predictions.npy'),
             'oracle_summary_sha256':digest(oracle_dir/'summary.json'),'batch':batch,'queries':queries,'backend':'MP-SPDZ semi2k real liveOT use_split2/R64','dataset_size':n}
    manifest_path=output/'evaluation_manifest.json'
    if manifest_path.exists():
        manifest=json.loads(manifest_path.read_text())
        if manifest.get('binding')!=binding:raise ValueError('resume manifest binding differs: start a separate evaluation directory')
        if manifest.get('target_samples')!=count or manifest.get('batch')!=batch or manifest.get('queries')!=queries:
            raise ValueError('resume manifest target or vector shape differs: start a separate evaluation directory')
    else:
        manifest={'binding':binding,'target_samples':count,'dataset_size':n,'batch':batch,'queries':queries,
                  'timeout_s':timeout,'memory_limit_mib_per_party':memory_mib,'cpus':list(cpus),
                  'slice_budget_s':max_seconds,'status':'running','started_utc':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),
                  'attempts_completed':0,'samples_completed':0}
        write_manifest(output,manifest)
    if path.exists():
        ledger=json.loads(path.read_text())
        if ledger['binding']!=binding:raise ValueError('resume binding differs: start a separate evaluation directory')
    else:ledger={'binding':binding,'samples':{},'attempts':[],'status':'partial'}
    started=time.monotonic();chunk=batch*queries
    missing=[i for i in range(count)if str(i)not in ledger['samples']]
    for off in range(0,len(missing),chunk):
        if stop_file is not None and Path(stop_file).exists():break
        if max_seconds is not None and time.monotonic()-started>=max_seconds:break
        ids=missing[off:off+chunk];actual_batch=batch if len(ids)==chunk else 1;actual_queries=queries if len(ids)==chunk else len(ids)
        dest=output/('run_%06d_%06d_attempt%d'%(ids[0],ids[-1],len(ledger['attempts'])))
        cfg=prepare_full(dest,model,input_csv,ids,actual_batch,actual_queries,False,artifact)
        cache=output/('compiled_B%d_Q%d'%(actual_batch,actual_queries))
        if cache.exists():shutil.copytree(cache/'Programs',dest/'Programs',dirs_exist_ok=True)
        else:
            compile_full(dest,spdz);cache.mkdir(exist_ok=True);shutil.copytree(dest/'Programs',cache/'Programs',dirs_exist_ok=True)
        result=run_full(dest,spdz,timeout,memory_mib,128,cpus)
        ledger['attempts'].append({'directory':dest.name,'sample_ids':ids,'status':result['status'],'seconds':result['total_wall_s']})
        if result['status']=='completed':
            with(dest/'secure_predictions.csv').open()as f:
                for row in csv.DictReader(f):
                    sample=int(row['sample_id']);prediction=int(row['prediction'])
                    ledger['samples'][str(sample)]={'prediction':prediction,'run':dest.name,'oracle_equal':prediction==int(oracle[sample])}
                    if prediction!=int(oracle[sample]):ledger.setdefault('mismatches',[]).append({'sample_id':sample,'actual':prediction,'SCI_oracle':int(oracle[sample]),'first_operator':'final prediction; rerun this ID with diagnostic trace for localization'})
        write_json(path,ledger)
        manifest.update({'attempts_completed':sum(a['status']=='completed' for a in ledger['attempts']),
                         'samples_completed':len(ledger['samples']),'last_attempt':dest.name,
                         'last_attempt_status':result['status'],'updated_utc':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime())})
        write_manifest(output,manifest)
        if result['status']!='completed' or ledger.get('mismatches'):break
    ids=sorted(int(i)for i in ledger['samples']);pred=np.asarray([ledger['samples'][str(i)]['prediction']for i in ids],dtype=np.int64)
    coverage_complete=ids==list(range(n));complete=coverage_complete and not ledger.get('mismatches');ledger['status']='complete'if complete else'mismatch'if ledger.get('mismatches')else'partial';write_json(path,ledger)
    summary={'execution_mode':'two_party_semi2k_real_OT','secure_samples':len(ids),'required_samples':n,'complete':complete,'coverage_complete':coverage_complete,'integer_predictions_equal':not ledger.get('mismatches'),'mismatches':ledger.get('mismatches',[]),
             'full_top1_pct':None,'full_fp32_top1_pct':float(np.mean(fp==labels)*100),'full_loss_pp':None,'full_prediction_agreement_pct':None,'quality_gate_pass':None,
             'partial_subset_accuracy_pct':float(np.mean(pred==labels[ids])*100)if ids else None,
             'partial_subset_agreement_pct':float(np.mean(pred==fp[ids])*100)if ids else None,
             'partial_results_note':'Partial subset is evidence of completed secure executions only; no full-dataset quality claim or ranking.',
             'binding':binding}
    summary.update({'target_samples':count,'budget_seconds':max_seconds,
                    'budget_exhausted':bool(max_seconds is not None and time.monotonic()-started>=max_seconds and not complete),
                    'attempts_completed':sum(a['status']=='completed' for a in ledger['attempts']),
                    'attempts_total':len(ledger['attempts']),
                    'next_sample_id':next((i for i in range(count) if str(i) not in ledger['samples']),None)})
    summary.update(summarize_coverage({i:int(v)for i,v in zip(ids,pred)},labels,fp,oracle))
    with(output/'secure_predictions.csv').open('w')as f:
        w=csv.writer(f);w.writerow(['sample_id','label','fp32_prediction','secure_prediction'])
        w.writerows((i,int(labels[i]),int(fp[i]),int(v))for i,v in zip(ids,pred))
    manifest.update({'status':'complete' if complete else 'mismatch' if ledger.get('mismatches') else 'partial',
                     'samples_completed':len(ids),'attempts_completed':sum(a['status']=='completed' for a in ledger['attempts']),
                     'updated_utc':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime())})
    write_manifest(output,manifest)
    write_json(output/'summary.json',summary);return summary

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--artifact',type=Path,required=True);p.add_argument('--input-csv',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--spdz',type=Path,default=Path('/home/jiang/MP-SPDZ'));p.add_argument('--batch',type=int,default=1);p.add_argument('--queries',type=int,default=1);p.add_argument('--limit',type=int);p.add_argument('--timeout',type=int,default=900);p.add_argument('--memory-mib',type=int,default=3072);p.add_argument('--cpus',default='0,2');p.add_argument('--max-seconds',type=int);p.add_argument('--stop-file',type=Path);p.add_argument('--oracle-dir',type=Path,required=True)
    a=p.parse_args();print(json.dumps(evaluate(a.artifact,a.input_csv,a.output,a.spdz,a.oracle_dir,a.batch,a.queries,a.limit,a.timeout,a.memory_mib,[int(x)for x in a.cpus.split(',')],a.max_seconds,a.stop_file),indent=2))
