#!/usr/bin/env python3
"""Check reconstructed secure step tensors against the frozen four-mode oracle."""
import argparse
import json
from pathlib import Path
import sys
import numpy as np
sys.path.append(str(Path(__file__).resolve().parents[1]/'google30_mpspdz'))
import vector_oracle as vector
from folded_semantics import table_version
from sci_semantics import SHAPES

STAGES=('input_adapter','shared_sign_prepare','shared_bitxa_prepare','sign',
    'abs_bitxa','clip_compare','clip_bitxa','index_truncation','lut_mask','lut_keygen',
    'lut_materialize','lut_tensor_and','lut_leaf_b2a','lut_payload','lut_open',
    'lut_product','restore_bitxa','output_adapter')

def protocol_report(logs,config,activation_mode):
    def records(log,prefix):
        return [json.loads(line[len(prefix)+1:]) for line in log.splitlines()
            if line.startswith(prefix+' ')]
    checks=dict(phase_sent_received_balance=True,stage_sent_received_balance=True,
        recorded_inventory_balanced=True,recorded_online_generation_zero=True,
        folded_lifecycle_complete=True,activation_mode_matches_runtime=True)
    phases=[records(log,'METRIC') for log in logs]
    paired=[]
    def indexed(rows,fields):
        result={}
        for row in rows:
            key=tuple(row[field] for field in fields)
            if key in result:raise ValueError('duplicate protocol record: '+str(key))
            result[key]=row
        return result
    def traffic(left,right):
        return left.get('sent_bytes')==right.get('received_bytes') and \
            right.get('sent_bytes')==left.get('received_bytes')
    indexed_phases=[indexed(rows,('query','phase')) for rows in phases]
    for key in sorted(set(indexed_phases[0])|set(indexed_phases[1])):
        left,right=[rows.get(key) for rows in indexed_phases]
        balanced=left is not None and right is not None and traffic(left,right)
        checks['phase_sent_received_balance'] &= balanced
        paired.append(dict(query=key[0],phase=key[1],balanced=balanced,per_party=[left,right]))
    for rows in phases:
        for row in rows:
            if row['phase'] in ('input','ready','online','output'):
                for field,value in row.items():
                    if field.endswith('generation_epoch_delta'):
                        checks['recorded_online_generation_zero'] &= value==0
    inventory=[];stages=[]
    for prefix in ('NATIVE_MATERIAL','NATIVE_TRIPLE','DPF_LUT','FOLDED_ACTIVATION'):
        parties=[]
        for log in logs:
            prepared=records(log,prefix+'_PREPARED')
            consumed=records(log,prefix+'_CONSUMED')
            balances=records(log,prefix+'_BALANCE')
            for row in prepared+consumed+balances:
                for field in ('online_generation_epoch_delta','online_generated','online_loaded_bytes'):
                    if field in row:checks['recorded_online_generation_zero'] &= row[field]==0
            for row in balances:
                if 'remaining' in row:checks['recorded_inventory_balanced'] &= row['remaining']==0
                if 'generated' in row and 'consumed' in row:
                    padding=row.get('padding_discarded',0)
                    checks['recorded_inventory_balanced'] &= padding>=0 and row['generated']==row['consumed']+padding
                if 'generated_elements' in row and 'consumed_elements' in row:
                    checks['recorded_inventory_balanced'] &= row['generated_elements']==row['consumed_elements']
            parties.append(dict(prepared_records=len(prepared),consumed_records=len(consumed),
                balance_records=len(balances),prepared=prepared,consumed=consumed,balances=balances))
        inventory.append(dict(protocol=prefix,per_party=parties))
        if prefix!='FOLDED_ACTIVATION':continue
        if config.get('backend')=='SCI_Beaver_segmented_PaperLUT':
            selection={'sci':'none','sigmoid-only':'sigmoid','tanh-only':'tanh','folded':'both'}
            kinds={'sci':set(),'sigmoid-only':{0},'tanh-only':{1},'folded':{0,1}}[activation_mode]
            checks['activation_mode_matches_runtime'] &= config.get('folded_activations')==selection[activation_mode]
            queries=range(config['queries']+config.get('warmup_queries',0))
            shape_fields=('query','layer','slot','call','kind','count','input_bits','output_bits',
                'input_scale','output_scale','table_version')
            for log,party in zip(logs,parties):
                configurations=records(log,'FOLDED_ACTIVATION_CONFIG')
                checks['folded_lifecycle_complete'] &= bool(configurations)
                for row in configurations:
                    for kind,name in enumerate(('sigmoid','tanh')):
                        algorithm='folded_sigma_f12_q10' if kind in kinds else 'native_sci'
                        checks['activation_mode_matches_runtime'] &= row.get(name)==algorithm
                    checks['folded_lifecycle_complete'] &= all(row.get(field)==value for field,value in
                        (('index_bits',10),('table_entries',1024),('table_version',1)))
                planned=indexed(records(log,'FOLDED_ACTIVATION_PLAN'),('query','layer','slot','call','kind'))
                prepared=indexed(party['prepared'],('query','layer','slot','call','kind'))
                consumed=indexed(party['consumed'],('query','layer','slot','call','kind'))
                balances=indexed(party['balances'],('query',))
                checks['folded_lifecycle_complete'] &= set(planned)==set(prepared)==set(consumed)
                checks['folded_lifecycle_complete'] &= set(balances)=={(query,) for query in queries}
                for key,plan in planned.items():
                    ready,used=prepared.get(key,{}),consumed.get(key,{})
                    checks['folded_lifecycle_complete'] &= key[0] in queries and key[4] in kinds and plan.get('count',0)>0
                    checks['folded_lifecycle_complete'] &= all(row.get(field)==plan.get(field)
                        for row in (ready,used) for field in shape_fields)
                    checks['folded_lifecycle_complete'] &= ready.get('pre_input') is True
                    checks['folded_lifecycle_complete'] &= all(plan.get(field)==value for field,value in
                        (('input_bits',16),('output_bits',16),('input_scale',7),
                         ('output_scale',7 if key[4] else 14),('table_version',1)))
                    checks['recorded_online_generation_zero'] &= all(used.get(field)==0 for field in
                        ('online_generated','online_loaded_bytes'))
                for query in queries:
                    calls=[row for row in planned.values() if row['query']==query]
                    checks['folded_lifecycle_complete'] &= all(sum(row['count'] for row in calls
                        if row['kind']==kind)==config['batch']*100 for kind in kinds)
                    balance=balances.get((query,),{})
                    count=sum(row['count'] for row in calls)
                    checks['folded_lifecycle_complete'] &= all(balance.get(field)==value for field,value in
                        (('generated',len(calls)),('consumed',len(calls)),('remaining',0),
                         ('generated_elements',count),('consumed_elements',count)))
                    checks['recorded_online_generation_zero'] &= all(balance.get(field)==0 for field in
                        ('online_generation_epoch_delta','online_generated','online_loaded_bytes'))
        for phase,field in (('offline','prepared'),('online','consumed')):
            calls=[indexed(party[field],('query','layer','slot','call','kind')) for party in parties]
            for key in sorted(set(calls[0])|set(calls[1])):
                left,right=[rows.get(key) for rows in calls]
                if left is None or right is None:
                    checks['stage_sent_received_balance']=False
                    continue
                if any(len(row.get(phase,[]))!=len(STAGES) for row in (left,right)):
                    raise ValueError('folded activation stage coverage is incomplete')
                details=[]
                for name,a,b in zip(STAGES,left[phase],right[phase]):
                    balanced=traffic(a,b)
                    checks['stage_sent_received_balance'] &= balanced
                    details.append(dict(stage=name,balanced=balanced,per_party=[a,b]))
                for row in (left,right):
                    for field in ('sent_bytes','received_bytes'):
                        if field in row:
                            checks['stage_sent_received_balance'] &= row[field]==sum(s[field] for s in row[phase])
                stages.append(dict(query=key[0],layer=key[1],slot=key[2],call=key[3],kind=key[4],
                    phase=phase,stages=details))
    return dict(checks=checks,phase_records_per_party=[len(rows) for rows in phases],
        phases=paired,inventory=inventory,folded_stages=stages,
        scope='recorded diagnostic evidence; complete residency qualification is separate')

def fixture_model(path):
    values=np.fromfile(path,dtype='<i2').astype(np.int64)
    if values.size!=sum(rows*cols for rows,cols in SHAPES.values()):
        raise ValueError('actual server model has an invalid public shape')
    result={};offset=0
    for name,(rows,cols) in SHAPES.items():
        count=rows*cols
        result[name]=values[offset:offset+count].reshape(rows,cols)
        if name in ('W1','U1','FC','FCbias'):
            result[name]=vector.wrap(result[name],8)
        offset+=count
    return result

def tensors(path,queries,batch):
    records={};query=None
    for line in path.read_text().splitlines():
        if line.startswith('METRIC '):
            metric=json.loads(line[7:])
            if metric['phase']=='ready':query=metric['query']
        if line.startswith('TENSOR '):
            fields=line.split();step,operator,name,bits,rows,cols=fields[1:7]
            if int(step)!=0 or query not in range(queries):raise ValueError('tensor outside qualified step')
            key=(query,name)
            if key in records:raise ValueError('duplicate semantic tensor: '+str(key))
            if int(rows)!=batch:raise ValueError('tensor batch mismatch')
            values=np.asarray([int(x) for x in fields[7:]],dtype=np.int64)
            if values.size!=int(rows)*int(cols):raise ValueError('tensor public shape mismatch')
            records[key]=(int(bits),values.reshape(int(rows),int(cols)))
    return records

def audit(directory,artifact,fixture,activation_mode,hidden=None):
    result=json.loads((directory/'result.json').read_text());config=result['config']
    if result['status']!='complete' or result['party_exit_codes']!=[0,0] or not config['diagnostic']:
        raise ValueError('secure diagnostic row is incomplete')
    if config['steps']!=1:raise ValueError('audit requires one recurrent step')
    batch,queries,start=(config[name] for name in ('batch','queries','start'))
    queries+=config.get('warmup_queries',0)
    x=np.memmap(fixture/'client_input.i16',dtype='<i2',mode='r').reshape(-1,99,32)[start:start+batch,0].astype(np.int64)
    hidden = hidden if hidden is not None else config.get('hidden_file')
    h=np.zeros((batch,100),dtype=np.int64)
    if hidden is not None:
        h=np.fromfile(hidden,dtype='<i2',count=batch*100).astype(np.int64).reshape(batch,100)
    model=fixture_model(fixture/'server_model.i16')
    _,expected=vector.step(x,h,model,trace=True,activation_mode=activation_mode)
    actual=tensors(directory/'party2.log',queries,batch)
    protocol=protocol_report([(directory/('party%d.log'%role)).read_text() for role in (1,2)],config,activation_mode)
    if set(actual)!={(query,name) for query in range(queries) for name in expected}:
        raise ValueError('diagnostic must cover all 15 tensors in every fresh query')
    rows=[]
    for query in range(queries):
        count=0;details={}
        for name,reference in expected.items():
            bits,values=actual[query,name]
            if values.shape!=reference.shape:raise ValueError('shape mismatch: '+name)
            bad=np.argwhere(values!=reference)
            detail=dict(integer_values=int(values.size),mismatches=int(len(bad)))
            if bad.size:
                sample,column=map(int,bad[0])
                detail['first_mismatch']=dict(sample=sample,column=column,
                    actual=int(values[sample,column]),expected=int(reference[sample,column]))
            if bits!=(8 if name=='tmp23' else 16):raise ValueError('width mismatch: '+name)
            count+=values.size
            details[name]=detail
        rows.append(dict(query=query,tensors=len(expected),integer_values=int(count),
            mismatches=sum(detail['mismatches'] for detail in details.values()),
            final_hidden_mismatches=details['H']['mismatches'],tensors_detail=details))
    passed=not any(row['mismatches'] for row in rows) and all(protocol['checks'].values())
    return dict(status='passed' if passed else 'failed',protocol=protocol,
        activation_mode=activation_mode,batch=batch,queries=queries,
        initial_hidden='external_i16' if hidden is not None else 'zero',table_version=table_version(),
        sources=dict(model=str(fixture/'server_model.i16'),input=str(fixture/'client_input.i16'),
            start=start,hidden=None if hidden is None else str(hidden)),
        reference='actual binary model and independent Decimal LUT/vector integer graph',rows=rows)

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('directory','artifact','fixture','output'):parser.add_argument('--'+name,type=Path,required=True)
    parser.add_argument('--activation-mode',choices=vector.ACTIVATION_MODES,required=True)
    parser.add_argument('--hidden',type=Path)
    args=parser.parse_args()
    try:report=audit(args.directory,args.artifact,args.fixture,args.activation_mode,args.hidden)
    except (ValueError,KeyError,OSError) as error:
        report=dict(status='failed',error=str(error),activation_mode=args.activation_mode)
    args.output.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(dict(status=report['status'],activation_mode=args.activation_mode,
        output=str(args.output),queries=report.get('queries'),
        mismatches=sum(row['mismatches'] for row in report.get('rows',[])),
        protocol_checks=report.get('protocol',{}).get('checks'),error=report.get('error'))))
    return 0 if report['status']=='passed' else 1
if __name__=='__main__':raise SystemExit(main())
