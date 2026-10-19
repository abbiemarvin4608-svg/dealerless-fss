#!/usr/bin/env python3
"""Qualify fresh a2 resident benchmarks and complete causal round traces."""
import argparse
import copy
import json
from pathlib import Path
import sys

import audit_preprocessed as base
from audit_resident import resident_party_checks, PREFETCH_BYTES

CONVENTION = 'two-party exchange; causal send-flight clocks'
BACKENDS = {'SCI_preprocessed', 'SCI_Beaver_segmented'}
PHASE_BYTES = {'SCI_preprocessed': {'offline': 195480822, 'online': 92204166},
               'SCI_Beaver_segmented': {'offline': 2236223038, 'online': 38210530}}
_original_actual_contract = base.actual_run_contract_matches


def trace_path(directory, role):
    raw = directory / ('party%d.roundtrace.jsonl' % role)
    for path in (raw, raw.with_suffix(raw.suffix+'.gz')):
        if path.is_file():
            return path
    raise FileNotFoundError('missing raw or gzip trace for party %d' % role)


def fresh_integer_checks(directory, declarations):
    controls = {(row.get('hidden'), row.get('cohort')): row['id'] for row in declarations
                if row['backend'] == 'SCI'}
    checks = {}
    for row in declarations:
        try:
            control = controls[(row.get('hidden'), row.get('cohort'))]
            tensors = [line for line in (directory/row['id']/'party2.log').read_text().splitlines()
                       if line.startswith('TENSOR ')]
            expected = [line for line in (directory/control/'party2.log').read_text().splitlines()
                        if line.startswith('TENSOR ')]
            server_log = (directory/row['id']/'party1.log').read_text()
            checks[row['id']] = (len(tensors) == len(expected) == 30
                and sum(len(line.split())-7 for line in tensors) == 345856 and tensors == expected
                and not any(line.startswith('TENSOR ') for line in server_log.splitlines()))
        except (OSError, KeyError):
            checks[row['id']] = False
    return checks


def phase_round_coverage(rounds, queries):
    expected = {(query, phase) for query in range(queries) for phase in ('offline', 'online')}
    return expected <= rounds.keys() and all(
        type(rounds[identity].get('logical_rounds')) is int and rounds[identity]['logical_rounds'] >= 0
        and (rounds[identity]['logical_rounds'] == 0) == (sum(rounds[identity]['party_sent_bytes']) == 0)
        for identity in expected)


def traced_actual_contract(config, actual, row_id):
    if config.get('round_trace') is not True or config.get('round_convention') != CONVENTION:
        return False
    adjusted = copy.deepcopy(actual)
    environments = adjusted.get('environment', [])
    if len(environments) != 2:
        return False
    for role, environment in enumerate(environments, 1):
        temporary = environment.get('TMPDIR')
        if not isinstance(temporary, str):
            return False
        expected = str(Path(temporary).parent / ('party%d.roundtrace.jsonl' % role))
        if environment.pop('SDCF_ROUND_TRACE', None) != expected:
            return False
    return _original_actual_contract(config, adjusted, row_id)


def row_audit(directory, declaration, diagnostic, contract, references):
    base.actual_run_contract_matches = traced_actual_contract
    try:
        row, config = base.audit_row(directory, declaration, diagnostic, contract, references)
    finally:
        base.actual_run_contract_matches = _original_actual_contract
    if config is None:
        return row, None, None
    output = directory / declaration['id']
    parties = base.read(output / 'parties.json')
    result = base.read(output / 'result.json')
    queries = 2 if diagnostic else 10
    checks = row['checks']
    checks['frozen_optimizer_defaults'] = (config.get('beaver_mapped_layout') is False
        and config.get('aes_offline') is False and config.get('prefetch_bytes') == PREFETCH_BYTES)
    checks['declared_resident_trace_policy'] = (base.arg(declaration['command'], '--prefetch-bytes') == str(PREFETCH_BYTES)
        and '--round-trace' in declaration['command'] and config.get('round_trace') is True
        and config.get('round_convention') == CONVENTION
        and config['binary_description'].get('round_trace_api') == 1)
    row['evidence']['resident_materials_per_party'] = []
    for role, (party, audit) in enumerate(zip(parties, row['parties']), 1):
        resident_checks, evidence = resident_party_checks((output / ('party%d.log' % role)).read_text(),
                                                          party, config, queries)
        audit['checks'].update(resident_checks)
        audit['eligible'] = all(audit['checks'].values())
        row['evidence']['resident_materials_per_party'].append(evidence)
    from round_trace import analyze_pair
    traces = [trace_path(output, role) for role in (1, 2)]
    fresh_rounds = analyze_pair(*traces, expected_phases=parties)
    checks['complete_trace_qualification'] = fresh_rounds.get('status') == 'passed'
    checks['recomputed_trace_identical'] = fresh_rounds == base.read(output / 'rounds.json') == result.get('round_measurements')
    rounds = {(item['query'], item['phase']): item for item in fresh_rounds.get('phases', [])}
    checks['offline_online_round_coverage'] = phase_round_coverage(rounds, queries)
    if not diagnostic:
        checks['unchanged_phase_communication'] = all(
            row['measurements'][phase + '_bytes']['values'] == [PHASE_BYTES[config['backend']][phase]] * queries
            for phase in ('offline', 'online'))
        checks['stable_fresh_query_round_counts'] = all(len({rounds[(query, phase)]['logical_rounds']
            for query in range(queries)}) == 1 for phase in ('offline', 'online'))
    row['evidence']['round_trace_files'] = {path.name: base.sha(path) for path in traces}
    row['evidence']['round_measurements'] = fresh_rounds
    row['eligible'] = all(checks.values()) and all(party['eligible'] for party in row['parties'])
    if not row['eligible']:
        row['measurements'] = None
    return row, config, rounds


def audit(results, contract_path, references):
    sys.path.insert(0, str(contract_path.parent))
    contract = base.load_contract_module(contract_path)
    report = dict(schema=1, scope='a2 batch128 single step; resident materials; passive rounds; SDCF-09 deferred',
                  convention=CONVENTION, formal=[], diagnostic=[], checks={})
    configurations = []
    plans = {}
    formal_rounds = {}
    for kind, count in (('diagnostic', 6), ('formal', 2)):
        directory = results / ('diagnostics' if kind == 'diagnostic' else kind)
        manifest = base.read(directory / 'manifest.json')
        declarations = manifest.get('command_rows', manifest['rows'])
        plans[kind] = (manifest, declarations)
        wanted = {(backend, hidden) for backend in (BACKENDS | {'SCI'} if kind == 'diagnostic' else BACKENDS)
                  for hidden in (('default', 'nonzero') if kind == 'diagnostic' else (None,))}
        actual = [(row['backend'], row.get('hidden')) for row in declarations]
        report['checks'][kind + '_declaration_coverage'] = (len(declarations) == count
            and len(set(actual)) == count and set(actual) == wanted
            and all(row['cohort'] == 'a2-parallel' for row in declarations))
        report['checks'][kind + '_unique_row_ids'] = len({row['id'] for row in declarations}) == count
        report['checks'][kind + '_exact_output_rows'] = ({path.name for path in directory.iterdir() if path.is_dir()}
            == {row['id'] for row in declarations})
        for hidden in (('default', 'nonzero') if kind == 'diagnostic' else (None,)):
            cohort = [row for row in declarations if row.get('hidden') == hidden]
            report['checks'][kind + '_shared_command_environment_' + str(hidden)] = bool(cohort) and all(
                base.normalized_command(row['command']) == base.normalized_command(cohort[0]['command'])
                and row.get('environment') == cohort[0].get('environment') for row in cohort)
        summary = base.read(directory / 'summary.json')['rows']
        report['checks'][kind + '_completed_summary'] = (len(summary) == count
            and {row['id'] for row in summary} == {row['id'] for row in declarations}
            and all(row['status'] == 'complete' for row in summary))
        report['checks'][kind + '_frozen_build'] = (base.sha(Path(manifest['binary'])) == manifest['binary_sha256']
            and base.sha(Path(manifest['bridge'])) == manifest['bridge_sha256'])
        for declaration in declarations:
            try:
                row, config, rounds = row_audit(directory, declaration, kind == 'diagnostic', contract, references)
                if config:
                    row['checks']['declared_frozen_binary'] = (config['binary'] == manifest['binary']
                        and config['binary_sha256'] == manifest['binary_sha256'])
                    bridges = [(path, digest) for path, digest in config['runtime_manifest']['sources'].items()
                               if Path(path).name == 'libsegfss_sci_bridge.so']
                    row['checks']['declared_frozen_bridge'] = bridges == [(manifest['bridge'], manifest['bridge_sha256'])]
                    row['checks']['frozen_generated_source'] = (base.sha(Path(config['runtime_manifest']['generated']))
                        == config['runtime_manifest']['generated_sha256'])
                    row['checks']['hidden_state_matches_declaration'] = (config['hidden_file']
                        == base.arg(declaration['command'], '--hidden')
                        and ((config['hidden_file'] is None) if declaration.get('hidden') != 'nonzero' else
                            base.sha(Path(config['hidden_file'])) == '96d0e2436ca32b5fd5d333ba75866540a2eb03f8e7ee0a2f2fd0a5fcbf31f400'))
                    row['eligible'] &= all(row['checks'].values())
                    configurations.append(config)
                if kind == 'formal':
                    formal_rounds[row['id']] = rounds
            except Exception as error:
                row = dict(id=declaration['id'], backend=declaration['backend'], cohort=declaration['cohort'],
                           hidden=declaration.get('hidden'), eligible=False, status='invalid_evidence',
                           reason=str(error), measurements=None)
            report[kind].append(row)
    fields = base.SHARED_FIELDS + ('round_trace', 'round_convention')
    shared = [{key: config.get(key) for key in fields if key not in ('queries', 'diagnostic', 'hidden_file')}
              for config in configurations]
    report['checks']['shared_build_fixture_and_resources'] = len(shared) == 8 and all(config == shared[0] for config in shared)
    report['checks']['single_runtime_manifest'] = len(configurations) == 8 and all(
        config['runtime_manifest'] == configurations[0]['runtime_manifest'] for config in configurations)
    report['checks']['same_frozen_binary_bridge'] = all(plans['formal'][0][key] == plans['diagnostic'][0][key]
        for key in ('binary', 'binary_sha256', 'bridge', 'bridge_sha256'))
    if configurations:
        report['public_scalar_source'] = base.public_scalar_source(Path(configurations[0]['binary']))
        report['checks']['public_scalar_contract'] = all(report['public_scalar_source'][key] is True
            for key in ('native_scalar_contract_matches_original', 'beaver_scalar_contract_matches_original'))
    else:
        report['checks']['public_scalar_contract'] = False
    integers = fresh_integer_checks(results/'diagnostics', plans['diagnostic'][1])
    report['fresh_sci_integer_checks'] = integers
    report['checks']['fresh_sci_integer_diagnostics'] = len(integers) == 6 and all(integers.values())
    for row in report['diagnostic']:
        row.setdefault('checks', {})['fresh_sci_integer_equality'] = integers.get(row['id']) is True
        row['eligible'] &= all(row['checks'].values())
    for row in report['formal']:
        row.setdefault('checks', {})
        row['checks']['default_nonzero_integer_diagnostics'] = base.matching_diagnostics(row, report['diagnostic'])
        declared = next(item for item in plans['formal'][1] if item['id'] == row['id'])
        diagnostics = [item for item in plans['diagnostic'][1] if item['backend'] == row['backend']]
        row['checks']['matching_diagnostic_commands'] = len(diagnostics) == 2 and all(
            base.diagnostic_command(item['command']) == base.diagnostic_command(declared['command'])
            and item.get('environment') == declared.get('environment') for item in diagnostics)
        row['eligible'] &= all(row['checks'].values())
    report['eligible'] = all(report['checks'].values()) and all(row['eligible']
        for row in report['formal'] + report['diagnostic'])
    for row in report['formal'] + report['diagnostic']:
        if not row['eligible']:
            row['measurements'] = None
    projection = dict(schema=1, eligible=report['eligible'], cohort='a2-parallel', convention=CONVENTION, rows=[])
    for row in report['formal']:
        item = dict(id=row['id'], backend=row['backend'], eligible=row['eligible'], queries=[])
        if row['eligible']:
            for query in range(10):
                phases = {}
                for phase in ('offline', 'online'):
                    evidence = formal_rounds[row['id']][(query, phase)]
                    phases[phase] = dict(rounds=evidence['logical_rounds'],
                        sent_bytes=evidence['party_sent_bytes'], received_bytes=evidence['party_received_bytes'],
                        trace_coverage=True)
                item['queries'].append(dict(query=query, phases=phases))
        else:
            row['measurements'] = None
        projection['rows'].append(item)
    return report, projection


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--results', type=Path, required=True)
    parser.add_argument('--contract', type=Path, required=True)
    parser.add_argument('--reference-dir', type=Path, required=True)
    args = parser.parse_args()
    report, projection = audit(args.results, args.contract, args.reference_dir)
    for name, value in (('ELIGIBILITY.json', report), ('ROUND_ELIGIBILITY.json', projection)):
        (args.results / name).write_text(json.dumps(value, indent=2, sort_keys=True) + '\n')
    print(json.dumps(dict(eligible=report['eligible'], formal=sum(row['eligible'] for row in report['formal']),
                         diagnostic=sum(row['eligible'] for row in report['diagnostic']))))
    return 0 if report['eligible'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
