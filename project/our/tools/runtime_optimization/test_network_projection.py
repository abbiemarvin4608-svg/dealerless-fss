"""Public contract for projections from qualified, freshly measured rounds."""
import copy
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import network_projection as model


def evidence():
    rows = []
    for backend, sent, rounds in (
            ('SCI_preprocessed', [62500, 62500], 4),
            ('SCI_Beaver_segmented', [31250, 31250], 2)):
        phases = {phase: dict(rounds=rounds, sent_bytes=sent,
                             received_bytes=list(reversed(sent)), trace_coverage=True)
                  for phase in ('offline', 'online')}
        rows.append(dict(id=backend+'-a2-parallel', backend=backend, eligible=True,
                         queries=[dict(query=query, phases=copy.deepcopy(phases))
                                  for query in range(10)]))
    return dict(schema=1, eligible=True, cohort='a2-parallel',
                convention='two-party exchange; causal send-flight clocks', rows=rows)


class NetworkProjectionTests(unittest.TestCase):
    def test_known_byte_and_round_example_uses_decimal_bandwidth_and_rtt(self):
        projected = model.project(evidence())
        online = projected['rows'][0]['queries'][0]['phases']['online']
        lan, wan = (online['networks'][name] for name in ('LAN', 'WAN'))
        self.assertAlmostEqual(lan['transmission_seconds'], 0.001)
        self.assertAlmostEqual(lan['propagation_seconds'], 0.002)
        self.assertAlmostEqual(lan['communication_seconds'], 0.003)
        self.assertAlmostEqual(wan['transmission_seconds'], 0.0025)
        self.assertAlmostEqual(wan['propagation_seconds'], 0.016)
        self.assertAlmostEqual(wan['communication_seconds'], 0.0185)
        self.assertAlmostEqual(projected['comparisons'][0]['phases']['online']['speedup'], 2)

    def test_unqualified_or_incomplete_measurements_never_produce_a_projection(self):
        for change in ('run', 'row', 'coverage', 'queries', 'missing_backend', 'duplicate_backend',
                       'schema', 'cohort', 'convention'):
            with self.subTest(change=change):
                changed = evidence()
                if change == 'run':
                    changed['eligible'] = False
                elif change == 'row':
                    changed['rows'][0]['eligible'] = False
                elif change == 'coverage':
                    changed['rows'][0]['queries'][0]['phases']['offline']['trace_coverage'] = False
                elif change == 'queries':
                    changed['rows'][0]['queries'].pop()
                elif change == 'missing_backend':
                    changed['rows'].pop()
                elif change == 'duplicate_backend':
                    changed['rows'][1]['backend'] = changed['rows'][0]['backend']
                elif change == 'schema':
                    changed['schema'] = 0
                elif change == 'cohort':
                    changed['cohort'] = 'a4-sequential'
                else:
                    changed['convention'] = 'send calls'
                with self.assertRaises(ValueError):
                    model.project(changed)

    def test_noninteger_rounds_unbalanced_bytes_and_invalid_query_ids_are_rejected(self):
        for field, value in (('rounds', None), ('rounds', 2.5), ('rounds', -1), ('rounds', True),
                             ('sent_bytes', [1, -1]), ('sent_bytes', [1.5, 2]),
                             ('sent_bytes', [True, 2]), ('sent_bytes', [1]),
                             ('received_bytes', [62500, 62501]),
                             ('received_bytes', None)):
            with self.subTest(field=field, value=value):
                changed = evidence()
                changed['rows'][0]['queries'][0]['phases']['online'][field] = value
                with self.assertRaises(ValueError):
                    model.project(changed)
        for value in (True, 0.0, '0', 10):
            with self.subTest(query=value):
                changed = evidence()
                changed['rows'][0]['queries'][0]['query'] = value
                with self.assertRaises(ValueError):
                    model.project(changed)

    def test_query_sums_are_summarized_before_medians_and_input_order_is_irrelevant(self):
        changed = evidence()
        baseline = changed['rows'][0]
        for query, (total_bytes, rounds) in zip(baseline['queries'],
                [(0, 100)]*4 + [(100000000, 0)]*4 + [(200000000, 200)]*2):
            query['phases']['online'].update(sent_bytes=[total_bytes, 0],
                received_bytes=[0, total_bytes], rounds=rounds)
        baseline['queries'].reverse()
        changed['rows'].reverse()
        result = model.project(changed)
        self.assertEqual(result['rows'][0]['backend'], 'SCI_preprocessed')
        self.assertEqual([query['query'] for query in result['rows'][0]['queries']], list(range(10)))
        lan = result['rows'][0]['summary']['online']['networks']['LAN']
        self.assertAlmostEqual(lan['communication_seconds']['median'], 0.8)
        self.assertAlmostEqual(lan['transmission_seconds']['median'], 0.8)
        self.assertAlmostEqual(lan['propagation_seconds']['median'], 0.05)

    def test_cli_writes_executable_model_and_chinese_report_without_localhost_addition(self):
        changed = evidence()
        changed['localhost_online_seconds'] = 12345
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            source, output = directory/'rounds.json', directory/'results'
            source.write_text(json.dumps(changed))
            completed = subprocess.run([sys.executable, model.__file__, '--round-evidence', str(source),
                '--output-directory', str(output)], capture_output=True, text=True)
            self.assertEqual(completed.returncode, 0, completed.stderr)
            projected = json.loads((output/'NETWORK_MODEL.json').read_text())
            self.assertEqual(projected['kind'], 'communication-time-model')
            self.assertFalse(projected['local_computation_included'])
            self.assertEqual(len(projected['source']['sha256']), 64)
            online = projected['rows'][0]['summary']['online']['networks']['LAN']
            self.assertAlmostEqual(online['communication_seconds']['median'], 0.003)
            report = (output/'NETWORK_MODEL.zh.md').read_text()
            self.assertIn('0.003000000', report)
            self.assertIn('RTT', report)
            self.assertNotIn('12345', report)

    def test_phase_with_no_communication_has_zero_cost_and_no_undefined_speedup(self):
        changed = evidence()
        for row in changed['rows']:
            for query in row['queries']:
                query['phases']['online'].update(rounds=0, sent_bytes=[0, 0], received_bytes=[0, 0])
        projected = model.project(changed)
        for comparison in projected['comparisons']:
            self.assertIsNone(comparison['phases']['online']['speedup'])
            self.assertIsNone(comparison['phases']['online']['ratio'])
        self.assertIn('不可定义', model.chinese_report(projected))

    def test_rejected_cli_evidence_does_not_write_completion_costs(self):
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            source, output = directory/'rounds.json', directory/'results'
            source.write_text(json.dumps(dict(evidence(), eligible=False)))
            completed = subprocess.run([sys.executable, model.__file__, '--round-evidence', str(source),
                '--output-directory', str(output)], capture_output=True, text=True)
            self.assertEqual(completed.returncode, 1)
            self.assertIn('rejected', completed.stderr)
            self.assertFalse(output.exists())


if __name__ == '__main__':
    unittest.main()
