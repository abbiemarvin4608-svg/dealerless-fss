#!/usr/bin/env python3
"""Qualify the complete serial folded step, with all query materials resident."""
import argparse
from collections import Counter, defaultdict
import json
import math
from pathlib import Path

from run import records, validate_native_plan_summary

PREFETCH_BYTES = 512 << 20
MEMORY_BYTES = 1536 << 20
IDENTITY = ('query', 'index', 'slot')


def indexed(rows):
    result = {tuple(row.get(field) for field in IDENTITY): row for row in rows}
    return result, len(result) == len(rows)


def party_checks(log, party, config, queries):
    calls, unique = indexed(party['conversion_calls'])
    details, detail_unique = indexed(party['conversion_details'])
    checks = dict(exact_conversion_coverage=unique and detail_unique and bool(calls),
                  no_online_load_or_wait=True, joint_budget_accounted=True,
                  prefix_before_input=True, complete_context_prefixes=True,
                  prefix_resources_bounded=True, resident_consumed_in_order=True)
    for name in ('conversion_offline_calls', 'conversion_details'):
        rows, single = indexed(party[name])
        checks['exact_conversion_coverage'] &= single and set(rows) == set(calls)
    checks['no_online_load_or_wait'] = all(
        type(row.get('prefetched')) is int and row['prefetched'] == 1
        and row.get('material_load_ms') == row.get('prefetch_wait_ms') == 0
        and math.isfinite(row.get('material_preload_ms', float('nan')))
        and row['material_preload_ms'] >= 0 for row in details.values())
    budgets = party['conversion_prefetch_budgets']
    allocations = {row.get('context'): row.get('context_budget_bytes') for row in budgets}
    sign_segments = config.get('folded_sign_segments', 0)
    per_element = 8192 + (1024 + 256 * sign_segments
                          if config.get('folded_sign_backend') == 'segmented_fss' else 0)
    reserve = config['batch'] * 200 * per_element + (8 << 20)
    declaration = records(log, 'FOLDED_ACTIVATION_RESERVE')
    checks['joint_budget_accounted'] = (
        config.get('prefetch_bytes') == PREFETCH_BYTES
        and config.get('prefetch_mode') == 'prepared_prefix'
        and config.get('memory_limit_bytes_per_party') == MEMORY_BYTES
        and len(declaration) == 1 and all(declaration[0].get(field) == value for field, value in (
            ('resident_budget_bytes', reserve), ('sigmoid_count', config['batch'] * 100),
            ('tanh_count', config['batch'] * 100), ('per_party_address_space_bytes', MEMORY_BYTES),
            ('includes_allocator_headroom', True)))
        and len(budgets) == len(allocations) == 1 and set(allocations) == {0}
        and all(row.get('contexts') == 1 and row.get('dpf_reserved_bytes') == reserve
                and row.get('configured_party_budget_bytes') == PREFETCH_BYTES
                and row.get('party_budget_bytes') == PREFETCH_BYTES - reserve
                and type(row.get('context_budget_bytes')) is int
                and row['context_budget_bytes'] > 0 for row in budgets)
        and sum(allocations.values()) == PREFETCH_BYTES - reserve)
    ready, prefixes, preprocessing = defaultdict(list), defaultdict(list), defaultdict(list)
    query = 0
    for position, line in enumerate(log.splitlines()):
        prefix, _, payload = line.partition(' ')
        if prefix not in ('CONVERSION_PREFETCH', 'CONVERSION_PREPROCESS_DETAIL', 'MATERIAL_READY', 'METRIC'):
            continue
        row = json.loads(payload)
        if prefix == 'CONVERSION_PREFETCH':
            prefixes[query].append((position, row))
        elif prefix == 'CONVERSION_PREPROCESS_DETAIL':
            preprocessing[query].append((position, row))
        elif prefix == 'MATERIAL_READY':
            ready[row['query']].append(position)
        elif row['phase'] == 'query_total':
            query = row['query'] + 1
    checks['complete_context_prefixes'] &= set(prefixes) == set(range(queries))
    checks['prefix_before_input'] &= set(ready) == set(range(queries))
    checks['exact_conversion_coverage'] &= set(preprocessing) == set(range(queries)) and [
        row for query in range(queries) for _, row in preprocessing[query]] == party['conversion_preprocess_details']
    evidence = []
    for query in range(queries):
        groups = defaultdict(list)
        for key, call in calls.items():
            if key[0] == query and key in details:
                groups[call['context']].append(details[key])
        prepared = prefixes.get(query, [])
        boundaries = ready.get(query, [])
        checks['prefix_before_input'] &= len(boundaries) == 1 and bool(prepared) and all(
            position < boundaries[0] for position, _ in prepared)
        generated = preprocessing.get(query, [])
        count = sum(key[0] == query for key in calls)
        checks['exact_conversion_coverage'] &= len(generated) == count and [
            row.get('ordinal') for _, row in generated] == list(range(count)) and all(
            row.get('fresh') is True and row.get('dealerless') is True for _, row in generated)
        checks['prefix_before_input'] &= len(boundaries) == 1 and all(
            position < boundaries[0] for position, _ in generated)
        expected = Counter((len(rows), allocations.get(context), rows[0].get('resident_bytes'))
                           for context, rows in groups.items())
        actual = Counter((row.get('calls'), row.get('budget_bytes'), row.get('resident_bytes'))
                         for _, row in prepared)
        checks['complete_context_prefixes'] &= len(prepared) == 1 and set(groups) == {0} and expected == actual
        checks['prefix_resources_bounded'] &= all(
            row.get('mode') == 'prepared_prefix' and row.get('charged_phase') == 'offline'
            and type(row.get('resident_bytes')) is int and 0 < row['resident_bytes'] <= row['budget_bytes']
            and type(row.get('address_space_bytes')) is int and 0 < row['address_space_bytes'] <= MEMORY_BYTES
            and type(row.get('workspace_reserve_bytes')) is int and row['workspace_reserve_bytes'] > 0
            for _, row in prepared)
        for context, rows in groups.items():
            live = [row.get('resident_bytes') for row in rows]
            budget = allocations.get(context, 0)
            checks['resident_consumed_in_order'] &= all(type(value) is int and 0 < value <= budget for value in live)
            checks['resident_consumed_in_order'] &= all(a > b for a, b in zip(live, live[1:]))
            checks['resident_consumed_in_order'] &= all(row.get('prefetch_budget_bytes') == budget for row in rows)
        evidence.append(dict(query=query, actual_conversion_calls=sum(len(rows) for rows in groups.values()),
                             prefixes=[row for _, row in prepared]))
    return checks, dict(folded_reserved_bytes=reserve, outer_allocations_bytes=allocations, queries=evidence)


def audit(directory, expected_queries=2):
    result = json.loads((directory / 'result.json').read_text())
    config = result['config']
    parties = json.loads((directory / 'parties.json').read_text())
    queries = config['queries'] + config.get('warmup_queries', 0)
    checks = dict(runtime_complete=result.get('status') == 'complete' and result.get('party_exit_codes') == [0, 0],
        serial_joint_configuration=config.get('backend') == 'SCI_Beaver_segmented_PaperLUT'
            and config.get('folded_activations') == 'both' and config.get('graph_mode') == 'sequential'
            and config.get('activation_workers_per_party') == 1 and config.get('batch') == 128
            and config.get('steps') == 1 and queries == expected_queries,
        address_space_limit_enforced=result.get('address_space_hard_limit_bytes_per_party') == MEMORY_BYTES,
        four_cpu_affinity=len(config.get('cpu_affinity_per_party', [])) == 2
            and all(len(set(cpus)) == 4 for cpus in config.get('cpu_affinity_per_party', [])),
        both_parties_present=len(parties) == 2)
    reports = []
    for index, party in enumerate(parties):
        log = (directory / ('party%d.log' % (index + 1))).read_text()
        local, evidence = party_checks(log, party, config, queries)
        phases = party['phases']
        local['zero_postprepare_generation'] = all(
            len(rows := [row for row in phases if row.get('query') == q and row.get('phase') == phase]) == 1
            and all(rows[0].get(field) == 0 for field in ('native_generation_epoch_delta',
                'bridge_generation_epoch_delta', 'dpf_generation_epoch_delta', 'folded_generation_epoch_delta'))
            for q in range(queries) for phase in ('input', 'ready', 'online', 'output'))
        local['input_ready_accounted'] = all(
            row['sent_bytes'] == row['received_bytes'] == (1 if row['phase'] == 'ready' else 0)
            for row in phases if row['phase'] in ('input', 'ready'))
        local['explicit_empty_native_inventory'] = validate_native_plan_summary(
            party['native_public_plan_summary'], party['native_public_plan'], party['native_triple_plan'], queries)
        local['explicit_empty_native_inventory'] &= not any(party[name] for name in (
            'native_public_plan', 'native_material_prepared', 'native_material_consumed', 'native_material_balances',
            'native_triple_plan', 'native_triple_prepared', 'native_triple_consumed', 'native_triple_balances'))
        local['complete_folded_inventory'] = party['folded_activation_verified']
        folded = []
        for q in range(queries):
            material = [row for row in party['folded_activation_prepared'] if row['query'] == q]
            resident = sum(row['resident_bytes'] for row in material)
            counts = [sum(row['count'] for row in material if row['kind'] == kind) for kind in (0, 1)]
            local['complete_folded_inventory'] &= counts == [12800, 12800] and 0 < resident <= evidence['folded_reserved_bytes']
            folded.append(dict(query=q, prepared_counts=counts, folded_capacity_bytes=resident))
        selected = {row['layer'] for row in party['folded_activation_plan']}
        local['complete_sci_activation_replacement'] = not any(party[name] for name in (
            'dpf_lut_plan', 'dpf_lut_prepared', 'dpf_lut_consumed', 'dpf_lut_balances'))
        local['complete_sci_activation_replacement'] &= all(
            row.get('conversion') == 'folded_sigma_f12_q10' for row in party['activations'] if row['index'] in selected)
        local['complete_sci_activation_replacement'] &= all(
            row.get('native_conversion_calls') == row.get('sent_bytes') == 0
            for row in party['activation_workers'] if row['index'] in selected)
        local['complete_sci_activation_replacement'] &= all(
            row.get('native_ot_consumed') == row.get('bit_triples_consumed') == 0
            for row in party['folded_activation_consumed'])
        local['complete_sci_activation_replacement'] &= not any(
            row.get('index') in selected for row in party['conversion_remaining_sci'])
        worker = party['native_worker_config']
        local['compute_budget_bounded'] = len(worker) == 1 and all(worker[0].get(field) == value for field, value in (
            ('requested_activation_workers', 1), ('protocol_contexts', 1), ('graph_branches', 1), ('eigen_threads', 1)))
        local['compute_budget_bounded'] &= all(row.get('max_compute_participants', 5) <= 4 for row in worker)
        local['compute_budget_bounded'] &= all(1 <= row.get('compute_participants', 5) <= 4 for row in party['activations'])
        local['compute_budget_bounded'] &= all(1 <= row.get(field, 5) <= 4
            for row in party['conversion_details'] for field in ('eval_participants', 'keygen_participants'))
        local['compute_budget_bounded'] &= all(1 <= row.get('generation_participants', 5) <= 4
            for row in party['conversion_preprocess_details'])
        sampled = result['process_resource_peaks'][index]
        final_rss = party['peak_rss_kib'] * 1024
        local['joint_process_resources_bounded'] = all(0 < value <= MEMORY_BYTES for value in (
            sampled['vm_peak_bytes'], sampled['rss_peak_bytes'], party['vm_peak_bytes'], final_rss))
        resources = dict(sampled=sampled, final=dict(vm_peak_bytes=party['vm_peak_bytes'], rss_peak_bytes=final_rss),
            conservative_vm_peak_bytes=max(sampled['vm_peak_bytes'], party['vm_peak_bytes']),
            conservative_rss_peak_bytes=max(sampled['rss_peak_bytes'], final_rss))
        reports.append(dict(party=index + 1, checks=local, prefix=evidence, folded=folded, resources=resources))
    passed = all(checks.values()) and all(all(row['checks'].values()) for row in reports)
    return dict(status='passed' if passed else 'failed', checks=checks, parties=reports,
        lut_family=config['lut_key_family'], lut_segments=config['lut_segments'],
        lut_segment_widths=config['lut_segment_widths'], sign_backend=config['folded_sign_backend'],
        scope=f'batch128, one complete serial step, {queries} fresh queries; not a 99-step qualification',
        material_wait_evidence='outer FSS measured prefetch_wait_ms=0; folded online directly consumes resident entries without a loader or producer queue')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--directory', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--queries', type=int, default=2)
    args = parser.parse_args()
    try:
        if args.queries < 2:
            raise ValueError('joint residency requires at least two fresh queries')
        report = audit(args.directory, args.queries)
    except (ValueError, KeyError, OSError) as error:
        report = dict(status='failed', error=str(error))
    args.output.write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(dict(status=report['status'], output=str(args.output), error=report.get('error'))))
    return 0 if report['status'] == 'passed' else 1


if __name__ == '__main__':
    raise SystemExit(main())
