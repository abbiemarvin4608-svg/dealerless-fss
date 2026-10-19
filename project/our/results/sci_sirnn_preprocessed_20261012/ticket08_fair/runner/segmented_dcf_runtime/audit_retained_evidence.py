"""Audit retained runtime samples and integer logs without rerunning protocols.

Run from a checkout containing the retained runtime_optimization_20261010 data.
The audit writes FINAL_RAW_AUDIT.json in that results directory. Recorded build
identities and earlier test counts are carried forward; binaries are not rehashed
and this script does not rerun tests or benchmarks.
"""

import datetime
import json
from pathlib import Path
import statistics
import sys

REPO = Path(__file__).resolve().parents[4]
ROOT = REPO / 'project/our/results/runtime_optimization_20261010'
sys.path.insert(0, str(REPO / 'project/our/tools/segmented_dcf_runtime'))
import run as contracts

def read(p):
    return json.loads(p.read_text())

def records(p, prefix):
    return [json.loads(line[len(prefix)+1:]) for line in p.read_text().splitlines() if line.startswith(prefix+' ')]

def tensors(p):
    return [line for line in p.read_text().splitlines() if line.startswith('TENSOR ')]

def near(a, b):
    return abs(a-b) <= 1e-12 * max(1, abs(a), abs(b))

report = dict(scope='Read-only re-audit of retained raw samples and integer logs; Batch128 single step; SDCF-09 deferred',
              audited_at=datetime.datetime.now(datetime.timezone.utc).isoformat(), source_commit='a39164f',
              execution='No benchmarks rerun; frozen binaries formerly in /tmp are absent after host restart. Recorded build identities are checked for consistency, not rehashed.',
              formal=[], diagnostic=[], wide=[], tests=dict(runtime='34 run, 29 passed, 5 compiled-fixture skips', primitive_runner='13 run, 10 passed, 3 compiled-fixture skips'))

for diagnostic, cohort in [(False, 'final2'), (True, 'final2_diagnostics')]:
    directory = ROOT/cohort
    summary = read(directory/'summary.json')
    manifest = read(directory/'manifest.json')
    declared = {r['id']:r for r in manifest['command_rows']}
    assert len(summary['rows']) == len(declared) == (16 if diagnostic else 8)
    for row in summary['rows']:
        output=directory/row['id']
        result=read(output/'result.json'); cfg=read(output/'config.json'); parties=read(output/'parties.json'); schedule=read(output/'schedule.json')
        n=2 if diagnostic else 10; checks={}
        checks['complete_and_frozen_scope'] = (result['status']=='complete' and result['party_exit_codes']==[0,0] and result['config']==cfg and cfg['batch']==128 and cfg['steps']==1 and cfg['queries']==n and cfg['warmup_queries']==0 and cfg['diagnostic']==diagnostic)
        checks['frozen_build_identity'] = cfg['runtime_manifest']==manifest['artifact_identity'] and cfg['binary_sha256']==manifest['artifact_identity']['binary_sha256']
        checks['resources'] = cfg['cpu_affinity_per_party']==[[0,2,4,10],[6,8,12,16]] and cfg['memory_limit_bytes_per_party']==1610612736 and all(p['peak_rss_kib']*1024<1610612736 for p in parties)
        phases = ('offline','input','ready','online','output','query_total')
        aligned = result['measurements']['queries']
        checks['phase_count']=len(aligned)==n and result['measurements']['warmup_queries']==[]
        for phase in phases:
            values=[]; byte_values=[]
            for q in range(n):
                raw=[r for p in parties for r in p['phases'] if r['phase']==phase and r['query']==q]
                assert len(raw)==2
                maximum=max(r['seconds'] for r in raw); sends=sum(r['sent_bytes'] for r in raw)
                assert near(aligned[q][phase]['seconds'],maximum) and aligned[q][phase]['sent_bytes']==sends
                values.append(maximum);byte_values.append(sends)
            for field, numbers in [('seconds',values),('bytes',byte_values)]:
                reported=row['measurements'][phase+'_'+field]
                assert reported['values']==numbers
                assert near(reported['median'],statistics.median(numbers))
                assert reported['min']==min(numbers) and reported['max']==max(numbers) and near(reported['pstdev'],statistics.pstdev(numbers))
        checks['party_max_and_send_sum_statistics']=True
        all_integer_counts=[]
        for party in parties:
            qids={(q,l['index']) for q in range(n) for l in schedule}
            coverage=records(output/('party%d.log'%party['party']),'COVERAGE')
            checks['party%d_exact_calls'%party['party']]=len(coverage)==len(qids) and {(r['query'],r['index']) for r in coverage}==qids
            contracts.validate_conversions(party['conversion_offline_calls'],party['conversion_calls'],party['conversion_preprocess_details'],party['conversion_groups'],schedule,n,cfg['backend'],False)
            checks['party%d_material_contract'%party['party']]=all(r.get('fresh') is True and r.get('dealerless') is True for r in party['conversion_preprocess_details']+party['beaver_offline_calls']) and all(r.get('consumed') is True for r in party['beaver_calls'])
            checks['party%d_caps'%party['party']]=all(r['observed_compute_participants']<=cfg['branch_compute_limit'] for r in party['graph_tasks']) and all(r['observed_compute_upper_bound']<=4 for r in party['graph_groups']) and all(r['compute_participants']<=cfg['activation_workers_per_party'] for r in party['activations']) and all(r['generation_participants']<=cfg['cggm_worker_limit_per_party'] for r in party['conversion_preprocess_details'])
            actual=tensors(output/('party%d.log'%party['party']))
            if diagnostic and party['party']==2:
                hidden='nonzero' if row['id'].endswith('nonzero') else 'default'
                reference=tensors(ROOT/'baseline/diagnostic'/hidden/'SCI-n4-t4/party2.log')
                checks['integer_reference']=actual==reference*n and len(reference)==15 and sum(len(line.split())-7 for line in reference)==172928
                all_integer_counts.append(sum(len(line.split())-7 for line in actual))
            else:
                checks['party%d_no_tensor_disclosure'%party['party']]=not actual
        contracts.validate_beaver_traffic(parties);checks['owned_beaver_receive_matches_remote_send']=True
        assert all(checks.values()), (row['id'], checks)
        report['diagnostic' if diagnostic else 'formal'].append(dict(id=row['id'],passed=True,checks=checks,integer_values=sum(all_integer_counts),retained_result=str((output/'result.json').relative_to(ROOT))))

wide=ROOT/'wide_confirm';manifest=read(wide/'manifest.json')
assert manifest['resource_contract']['memory_bytes_per_party']==1610612736
for row in read(wide/'summary.json')['rows']:
    output=wide/row['id']; result=read(output/'result.json');execution=read(output/'execution.json')
    assert result['status']==row['status'] and row['runs']==10 and row['timeout_seconds']==600
    assert execution['memory_limit_bytes']==1610612736 and execution['requested_cpu_affinity_per_party']==[[0,2,4,10],[6,8,12,16]]
    if row['status']!='complete':
        assert result['measurements'] is None and row['measurements'] is None
        report['wide'].append(dict(id=row['id'],status=row['status'],passed_failure_accounting=True,completed_cost=None));continue
    measured=result['measurements'];parties=measured['parties']; assert measured['repetitions']==10 and execution['returncodes']==[0,0]
    assert all(p['correctness_passed'] is True and len(p['samples'])==10 for p in parties)
    for phase in ('offline','online','total'):
        values=[max(p['samples'][q][phase+'_seconds'] for p in parties) for q in range(10)]
        assert measured[phase+'_seconds_distribution']['values']==values and near(measured[phase+'_seconds'],statistics.median(values))
    peak_rss=max(s['peak_rss_kib'] for p in parties for s in p['samples']); assert peak_rss*1024<1610612736
    max_participants=max(s['generation_participants'] for p in parties for s in p['samples']);assert max_participants<=4
    gens=[]
    for p in parties:
        gs=records(output/('party%d.stdout.log'%p['party']),'DCF_GENERATION');assert len(gs)==10
        for g in gs:
            plan=g['tile_plan'];assert len(plan)==g['tiles'] and sum(t['keys'] for t in plan)==row['count']*len(p['layout'])
            cursor=0
            for tile in plan:
                assert tile['begin']==cursor;cursor+=tile['keys']
            assert g['peak_working_bytes']+g['resident_bytes']+g['reserve_bytes']<=g['budget_bytes']
            assert g['extra_hashes']==0 and g['participant_limit']<=4
        gens.append(gs[0])
    g=gens[0]
    report['wide'].append(dict(id=row['id'],status='complete',correctness=True,fresh_queries=10,peak_rss_kib=peak_rss,
                              actual_widths=parties[0]['layout'],tiles=g['tiles'],tile_keys=g['tile_keys'],tail_keys=g['tail_keys'],
                              peak_working_bytes=g['peak_working_bytes'],resident_bytes=g['resident_bytes'],internal_hashes=g['internal_hashes'],leaf_hashes=g['leaf_hashes'],extra_hashes=g['extra_hashes'],
                              added_cot_batches=g['added_cot_batches'],added_opening_batches=g['added_opening_batches'],
                              offline_seconds=measured['offline_seconds'],online_seconds=measured['online_seconds'],direct_total_seconds=measured['total_seconds']))
failed=read(ROOT/'wide_batch/summary.json')['rows'][0]; assert failed['status']=='timeout' and failed['measurements'] is None
report['wide_batch']=dict(status='timeout',completed_cost=None,declared_row_timeout_seconds=600,scope='fixed2 tile8 full Batch128 single-step attempt')
report['integer_values_checked']=sum(r['integer_values'] for r in report['diagnostic']);report['passed']=True
(ROOT/'FINAL_RAW_AUDIT.json').write_text(json.dumps(report,indent=2,sort_keys=True)+'\n')
print(json.dumps(dict(passed=True,formal_rows=len(report['formal']),diagnostic_rows=len(report['diagnostic']),integer_values=report['integer_values_checked'],wide_rows=len(report['wide'])),indent=2))
