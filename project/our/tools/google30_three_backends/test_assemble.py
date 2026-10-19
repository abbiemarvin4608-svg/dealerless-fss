import unittest

from assemble import make_table


class TableTests(unittest.TestCase):
    def test_partial_mpspdz_never_inherits_plaintext_quality_or_a_speed_rank(self):
        quality = dict(samples=6798, complete=True, full_secure_evaluation=True,
            full_oracle_match=True, full_secure_agreement=True, quality_gate_pass=True,
            fp32_top1_pct=89.3, fixed_top1_pct=88.9, loss_pp=0.4,
            prediction_agreement_pct=96.2, contract='graph')
        rows = [dict(backend=backend, track='same_graph', network='lan', run_kind='repeat',
            eligible=True, status='complete', query_count=1, total_seconds=seconds,
            first_query_seconds=seconds, link_bytes=100, measurement_artifact='run'+str(i),
            config_id=backend+'-native', comparison_group_id='batch1-same-numeric-and-resource-configuration')
            for backend, seconds in [('SCI', 20), ('SCI_ours', 25)] for i in range(3)]
        result = make_table(rows, quality, dict(complete=False, secure_samples=4,
            required_samples=6798, full_top1_pct=None), [])
        self.assertFalse(result['three_backend_ranking_available'])
        same_graph = [row for row in result['rows'] if row['track'] == 'same_graph']
        self.assertEqual({row['backend'] for row in same_graph}, {'SCI', 'SCI_ours', 'MP-SPDZ semi2k'})
        mp = next(row for row in same_graph if row['backend'] == 'MP-SPDZ semi2k')
        self.assertIsNone(mp['top1_pct'])
        self.assertIsNone(mp['total_median_seconds'])
        self.assertFalse(mp['eligible'])
        self.assertEqual(result['two_backend_comparisons'][0]['ours_over_sci_latency_ratio'], 1.25)

    def test_incompatible_batch_or_missing_resource_binding_never_produces_ratio(self):
        quality = dict(samples=6798, complete=True, full_secure_evaluation=True,
            full_oracle_match=True, full_secure_agreement=True, loss_pp=0.4)
        rows = [dict(backend=backend, track='same_graph', network='lan', run_kind='repeat',
            eligible=True, status='complete', query_count=1, total_seconds=20, link_bytes=100,
            measurement_artifact=backend+str(i), config_id=backend+'-native',
            comparison_group_id=backend+'-different-cpus')
            for backend in ('SCI', 'SCI_ours') for i in range(3)]
        result = make_table(rows, quality, {}, [])
        self.assertEqual(result['two_backend_comparisons'], [])

    def test_resource_failure_is_retained_without_becoming_a_measurement(self):
        failed = dict(backend='SCI_ours', network='wan_2ms', status='censored',
            run_kind='repeat', track='same_graph', query_count=1,
            measurement_artifact='run/result.json', reason='wall timeout',
            timeout_seconds=600, memory_limit_mib=1024)
        result = make_table([failed], {}, {}, [])
        row = next(row for row in result['rows']
                   if row['backend'] == 'SCI_ours' and row['track'] == 'same_graph')
        self.assertFalse(row['eligible'])
        self.assertIsNone(row['total_median_seconds'])
        self.assertEqual(row['observed_runs'][0]['reason'], 'wall timeout')
        self.assertEqual(row['observed_runs'][0]['timeout_seconds'], 600)


if __name__ == '__main__':
    unittest.main()
