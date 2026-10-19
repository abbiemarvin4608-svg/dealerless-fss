"""Public qualification/report seams agreed in the preprocessing spec."""
import json
from pathlib import Path
import tempfile
import unittest

import audit_preprocessed as audit


class PreprocessedReportTests(unittest.TestCase):
    def test_actual_run_contract_disagreement_rejects_artifacts(self):
        local = Path(__file__).resolve().parent
        archived = local if (local / 'formal').exists() else local.parents[1] / 'results/sci_sirnn_preprocessed_20261012/ticket08_fair'
        target = 'SCI_ours-a2-parallel'
        for change in ('backend_argument', 'preprocessing_environment'):
            with self.subTest(change=change), tempfile.TemporaryDirectory() as tmp:
                formal = Path(tmp) / 'formal'
                formal.mkdir()
                for source in (archived / 'formal').iterdir():
                    if source.name != target:
                        (formal / source.name).symlink_to(source)
                output = formal / target
                output.mkdir()
                for source in (archived / 'formal' / target).iterdir():
                    if source.name != 'commands.json':
                        (output / source.name).symlink_to(source)
                commands = json.loads((archived / 'formal' / target / 'commands.json').read_text())
                if change == 'backend_argument':
                    commands['commands'][0][12] = 'SCI'
                else:
                    commands['environment'][0]['SDCF_NATIVE_PREPROCESSING'] = '0'
                (output / 'commands.json').write_text(json.dumps(commands))
                report = audit.audit_matrix(formal, archived / 'diagnostics', archived / 'reference',
                    archived / 'runner/segmented_dcf_runtime/run.py')
                row = next(row for row in report['formal'] if row['id'] == target)
                self.assertFalse(report['eligible'])
                self.assertFalse(row['eligible'])
                self.assertIsNone(row['measurements'])
                self.assertTrue(all(item['changes'] is None for item in audit.comparisons(report)['comparisons']))

    def test_ours_without_shared_preprocessing_cannot_qualify(self):
        config = dict(backend="SCI_ours", native_preprocessing_requested=False,
                      native_preprocessing_effective=False)
        self.assertFalse(audit.native_mode_matches(config, ["run.py", "--backend", "SCI_ours"]))

    def test_missing_nonzero_diagnostic_rejects_formal_comparison(self):
        formal = dict(backend="SCI_ours", cohort="a2-parallel", eligible=True)
        diagnostics = [dict(backend="SCI_ours", cohort="a2-parallel", hidden="default", eligible=True)]
        self.assertFalse(audit.matching_diagnostics(formal, diagnostics))

    def test_report_uses_aligned_party_maxima_and_direct_query_totals(self):
        parties = []
        for role, online, offline, totals in ((1, [1, 9, 2], [9, 1, 2], [12, 15, 7]),
                                               (2, [8, 2, 3], [2, 8, 3], [13, 14, 8])):
            phases = []
            for query in range(3):
                for phase, values in (("online", online), ("offline", offline), ("query_total", totals)):
                    phases.append(dict(query=query, phase=phase, seconds=values[query],
                                       sent_bytes=role * 10, received_bytes=(3-role) * 10))
            phases += [dict(query=-1, phase=phase, seconds=role * 2,
                           sent_bytes=role * 10, received_bytes=(3-role) * 10)
                       for phase in ("setup", "first_query")]
            parties.append(dict(party=role, phases=phases, total_seconds=20+role, peak_rss_kib=100+role))
        report = audit.measured_summary(parties, 3)
        self.assertEqual(report["online_seconds"]["values"], [8, 9, 3])
        self.assertEqual(report["online_seconds"]["median"], 8)
        self.assertEqual(report["query_total_seconds"]["median"], 13)
        self.assertEqual(report["online_bytes"]["values"], [30, 30, 30])
        self.assertEqual(report["session_total_seconds"], 22)

    def test_session_communication_counts_setup_queries_and_teardown_once(self):
        parties = []
        for role in (1, 2):
            sent = {"application_io": 0, "setup": role * 10, "offline": role * 40,
                    "online": role * 60, "query_total": role * 100,
                    "first_query": role * 110, "teardown": 8 if role == 2 else 0}
            other = 3 - role
            received = {"application_io": 0, "setup": other * 10, "offline": other * 40,
                        "online": other * 60, "query_total": other * 100,
                        "first_query": other * 110, "teardown": 8 if role == 1 else 0}
            phases = [dict(phase=phase, query=0 if phase in ("offline", "online", "query_total") else -1,
                           seconds=1, sent_bytes=value, received_bytes=received[phase])
                      for phase, value in sent.items()]
            parties.append(dict(party=role, phases=phases, total_seconds=5, peak_rss_kib=100))
        report = audit.measured_summary(parties, 1)
        self.assertEqual(report["session_total_bytes"], 338)
        self.assertEqual(report["session_sent_bytes_per_party"], [110, 228])
        self.assertEqual(report["session_received_bytes_per_party"], [228, 110])


if __name__ == "__main__":
    unittest.main()
