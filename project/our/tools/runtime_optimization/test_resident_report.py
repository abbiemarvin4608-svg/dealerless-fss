"""Qualification of actual resident inventories, including fallback rejection."""
import copy
import json
from pathlib import Path
import tempfile
import unittest

import audit_preprocessed as base
import audit_resident as audit


ARCHIVED = base.RESULTS


def resident_fixture():
    """Use a real archived schedule to exercise each context/query inventory."""
    output = ARCHIVED / 'diagnostics/default-a2-parallel-SCI_ours'
    config = base.read(output / 'config.json')
    party = base.read(output / 'parties.json')[0]
    lines = (output / 'party1.log').read_text().splitlines()
    config.update(prefetch_bytes=audit.PREFETCH_BYTES, prefetch_mode='prepared_prefix')
    allocations = {context: audit.PREFETCH_BYTES // config['graph_branches']
                   for context in range(config['graph_branches'])}
    calls = {(row['query'], row['index'], row['slot']): row['context']
             for row in party['conversion_calls']}
    grouped = {}
    for detail in party['conversion_details']:
        grouped.setdefault((detail['query'], calls[(detail['query'], detail['index'], detail['slot'])]), []).append(detail)
    remaining = {key: sum(detail['material_bytes'] + 1024 for detail in details)
                 for key, details in grouped.items()}
    prefix = {key: dict(mode='prepared_prefix', calls=len(grouped[key]),
        resident_bytes=size, budget_bytes=allocations[key[1]],
        workspace_reserve_bytes=64 * 2**20, address_space_bytes=900 * 2**20,
        charged_phase='offline') for key, size in remaining.items()}
    transformed = []
    for line in lines:
        name, _, text = line.partition(' ')
        if name == 'CONVERSION_PREFETCH_BUDGET':
            row = json.loads(text)
            row.update(context_budget_bytes=allocations[row['context']], party_budget_bytes=audit.PREFETCH_BYTES)
            line = name + ' ' + json.dumps(row)
        elif name == 'MATERIAL_READY':
            query = json.loads(text)['query']
            transformed += ['CONVERSION_PREFETCH ' + json.dumps(prefix[(query, context)])
                            for context in allocations]
        elif name == 'CONVERSION_DETAIL':
            row = json.loads(text)
            context = calls[(row['query'], row['index'], row['slot'])]
            key = (row['query'], context)
            row.update(prefetched=1, material_load_ms=0, material_preload_ms=1,
                resident_bytes=remaining[key], prefetch_budget_bytes=allocations[context])
            remaining[key] -= row['material_bytes'] + 1024
            line = name + ' ' + json.dumps(row)
        transformed.append(line)
    log = '\n'.join(transformed) + '\n'
    party.update(conversion_details=base.records(log, 'CONVERSION_DETAIL'),
        conversion_prefetch=base.records(log, 'CONVERSION_PREFETCH'),
        conversion_prefetch_budgets=base.records(log, 'CONVERSION_PREFETCH_BUDGET'))
    return log, party, config


class ResidentReportTests(unittest.TestCase):
    def test_full_context_inventories_qualify_before_each_input(self):
        checks, evidence = audit.resident_party_checks(*resident_fixture(), 2)
        self.assertTrue(all(checks.values()), checks)
        self.assertEqual([row['calls'] for row in evidence['queries']], [19, 19])
        self.assertEqual([row['contexts'] for row in evidence['queries']], [2, 2])

    def test_partial_inventory_online_fallback_and_late_load_are_rejected(self):
        log, party, config = resident_fixture()
        for change in ('partial', 'fallback', 'load', 'late', 'budget', 'hidden_input_traffic',
                       'resident_total', 'duplicate_prefix'):
            with self.subTest(change=change):
                changed = copy.deepcopy(party)
                changed_log = log
                if change in ('partial', 'resident_total', 'duplicate_prefix'):
                    row = changed['conversion_prefetch'][0]
                    new = dict(row, calls=row['calls'] - 1) if change == 'partial' else dict(
                        row, resident_bytes=row['resident_bytes'] + 1)
                    replacement = ('CONVERSION_PREFETCH ' + json.dumps(row) + '\nCONVERSION_PREFETCH ' + json.dumps(row)
                                   if change == 'duplicate_prefix' else 'CONVERSION_PREFETCH ' + json.dumps(new))
                    changed_log = changed_log.replace('CONVERSION_PREFETCH ' + json.dumps(row),
                        replacement, 1)
                elif change in ('fallback', 'load'):
                    row = changed['conversion_details'][0]
                    row['prefetched' if change == 'fallback' else 'material_load_ms'] = 0 if change == 'fallback' else 0.1
                elif change == 'late':
                    lines = changed_log.splitlines()
                    position = next(i for i, line in enumerate(lines) if line.startswith('CONVERSION_PREFETCH '))
                    prefix = lines.pop(position)
                    ready = next(i for i, line in enumerate(lines) if line.startswith('MATERIAL_READY '))
                    lines.insert(ready + 1, prefix)
                    changed_log = '\n'.join(lines)
                elif change == 'budget':
                    changed['conversion_prefetch_budgets'][0]['context_budget_bytes'] += 1
                else:
                    next(row for row in changed['phases'] if row['phase'] == 'input')['sent_bytes'] = 10
                checks, _ = audit.resident_party_checks(changed_log, changed, config, 2)
                self.assertFalse(all(checks.values()), checks)

    def test_fresh_actual_child_contract_disagreements_have_no_cost(self):
        directory = audit.RESULTS / 'diagnostics'
        if not directory.exists():
            directory = Path('/tmp/sci-resident-diagnostics')
        row_id = 'default-a4-sequential-SCI_ours'
        source = directory / row_id
        manifest = base.read(directory / 'manifest.json')
        declaration = next(row for row in manifest['command_rows'] if row['id'] == row_id)
        contract = base.load_contract_module(ARCHIVED / 'runner/segmented_dcf_runtime/run.py')
        reference = ARCHIVED / 'reference'
        valid, _ = audit.audit_row(directory, declaration, True, contract, reference)
        self.assertTrue(valid['eligible'], valid['checks'])
        for change in ('backend_argument', 'preprocessing_environment', 'prefetch_environment', 'declared_budget'):
            with self.subTest(change=change), tempfile.TemporaryDirectory() as tmp:
                output = Path(tmp) / row_id
                output.mkdir()
                for item in source.iterdir():
                    if item.name != 'commands.json':
                        (output / item.name).symlink_to(item)
                commands = base.read(source / 'commands.json')
                declaration = copy.deepcopy(declaration)
                if change == 'backend_argument':
                    commands['commands'][0][12] = 'SCI'
                elif change == 'preprocessing_environment':
                    commands['environment'][0]['SDCF_NATIVE_PREPROCESSING'] = '0'
                elif change == 'prefetch_environment':
                    commands['environment'][0]['SEGFSS_PREFETCH_BYTES'] = '0'
                else:
                    index = declaration['command'].index('--prefetch-bytes')
                    declaration['command'][index + 1] = '0'
                (output / 'commands.json').write_text(json.dumps(commands))
                invalid, _ = audit.audit_row(Path(tmp), declaration, True, contract, reference)
                self.assertFalse(invalid['eligible'])
                self.assertIsNone(invalid['measurements'])

    def test_old_file_loading_result_is_ineligible_and_has_no_cost(self):
        output = next(path for path in (ARCHIVED / 'formal').iterdir()
                      if path.is_dir() and path.name == 'SCI_ours-a4-sequential')
        manifest = base.read(output.parent / 'manifest.json')
        declaration = next(row for row in manifest['command_rows'] if row['id'] == output.name)
        contract = base.load_contract_module(ARCHIVED / 'runner/segmented_dcf_runtime/run.py')
        row, _ = audit.audit_row(output.parent, declaration, False, contract, ARCHIVED / 'reference')
        self.assertFalse(row['eligible'])
        self.assertIsNone(row['measurements'])

    def test_failed_or_missing_prepare_never_has_completion_cost(self):
        declaration = dict(id='pending', backend='SCI_ours', cohort='a4-sequential')
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            for status in (None, 'timeout', 'failed'):
                if status:
                    output = directory / declaration['id']
                    output.mkdir(exist_ok=True)
                    (output / 'result.json').write_text(json.dumps(dict(status=status, reason='partial prepare')))
                row, _ = audit.audit_row(directory, declaration, False, None, None)
                self.assertFalse(row['eligible'])
                self.assertIsNone(row['measurements'])

    def test_report_reuses_direct_aligned_measurements_without_subtraction(self):
        parties = []
        for role, times in ((1, [1, 9]), (2, [8, 2])):
            phases = [dict(query=query, phase=phase, seconds=seconds if phase == 'online' else seconds + 5,
                           sent_bytes=role * 10, received_bytes=(3-role) * 10)
                      for query, seconds in enumerate(times) for phase in ('online', 'query_total')]
            phases += [dict(query=-1, phase=phase, seconds=2, sent_bytes=role * 10,
                            received_bytes=(3-role) * 10) for phase in ('setup', 'first_query')]
            parties.append(dict(phases=phases, total_seconds=30, peak_rss_kib=100))
        summary = base.measured_summary(parties, 2)
        self.assertEqual(summary['online_seconds']['values'], [8, 9])
        self.assertEqual(summary['query_total_seconds']['values'], [13, 14])

    def test_report_distinguishes_complete_method_and_conversion_ablation(self):
        report = base.read(ARCHIVED / 'ELIGIBILITY.json')
        for row in report['formal']:
            row['evidence']['resident_materials_per_party'] = [dict(queries=[dict(
                query=0, calls=21 if row['backend'] == 'SCI_Beaver_segmented' else 19,
                resident_bytes=180000000)]) for _ in range(2)]
        text = audit.chinese_report(report, base.comparisons(report))
        self.assertIn('SCI+ours（完整方法）', text)
        self.assertIn('仅转换替换（消融）', text)
        self.assertIn('没有事后扣除任何计时', text)


if __name__ == '__main__':
    unittest.main()
