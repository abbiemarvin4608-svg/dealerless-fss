import unittest
from schema import normalize_comparison, rankable_repeat_group, result_row, summarize_rows

class SchemaTests(unittest.TestCase):
    def test_partial_secure_row_is_typed_but_not_rankable(self):
        row = result_row(
            {'status': 'complete', 'backend': 'SCI', 'track': 'same_graph',
             'network': {'name': 'lan'}, 'run_kind': 'repeat', 'total_seconds': 2},
            {'samples': 6798, 'loss_pp': .4},
            {'complete': False, 'compared_secure_samples': 160,
             'full_secure_agreement': False})
        self.assertFalse(row['eligible'])
        self.assertIn('full 6798 actual secure backend agreement is pending', row['ineligibility_reasons'])
        self.assertEqual(row['secure_compared_samples'], 160)

    def test_secure_summary_requires_all_samples(self):
        c = normalize_comparison({'complete': True, 'completed_samples': 6798,
                                  'oracle_mismatches': []})
        self.assertTrue(c['complete'])
        self.assertTrue(c['full_oracle_match'])

    def test_summary_never_ranks_partial(self):
        rows = [result_row({'status': 'complete', 'backend': 'SCI',
                            'network': {'name': 'lan'}, 'run_kind': 'repeat',
                            'total_seconds': 1}, {'samples': 6798, 'loss_pp': 0.2},
                           {'complete': False, 'compared_secure_samples': 6798})]
        self.assertFalse(next(iter(summarize_rows(rows).values()))['rankable'])

    def test_quality_backend_binding_is_required_when_quality_lists_identities(self):
        row = result_row(
            {'status': 'complete', 'backend': 'SCI', 'network': {'name': 'lan'},
             'run_kind': 'repeat', 'total_seconds': 1},
            {'samples': 6798, 'loss_pp': .2,
             'backend_identities': {'SCI': 'secure-id'}},
            {'complete': True, 'compared_secure_samples': 6798,
             'full_secure_agreement': True, 'full_oracle_match': True})
        self.assertFalse(row['eligible'])
        self.assertIn('quality identity binding', ' '.join(row['ineligibility_reasons']))

    def test_quality_backend_identity_must_match(self):
        row = result_row(
            {'status': 'complete', 'backend': 'SCI', 'network': {'name': 'lan'},
             'run_kind': 'repeat', 'quality_backend_identity': 'wrong',
             'total_seconds': 1},
            {'samples': 6798, 'loss_pp': .2,
             'backend_identities': {'SCI': 'secure-id'}},
            {'complete': True, 'compared_secure_samples': 6798,
             'full_secure_agreement': True, 'full_oracle_match': True})
        self.assertFalse(row['eligible'])
        self.assertIn('does not match', ' '.join(row['ineligibility_reasons']))

    def test_rankable_group_requires_three_distinct_measurements(self):
        rows = []
        for index in range(2):
            rows.append({'eligible': True, 'status': 'complete', 'run_kind': 'repeat',
                         'query_count': 1, 'config_id': 'cfg', 'comparison_group_id': 'grp',
                         'measurement_artifact': 'run%d' % index,
                         'total_seconds': 1, 'link_bytes': 2})
        self.assertFalse(rankable_repeat_group(rows))
        rows.append(dict(rows[-1], measurement_artifact='run2'))
        self.assertTrue(rankable_repeat_group(rows))

    def test_summary_does_not_pool_independently_complete_configurations(self):
        rows = [dict(backend='SCI', track='same_graph', network='lan',
            eligible=True, status='complete', run_kind='repeat', query_count=1,
            config_id=config, comparison_group_id='resources-'+config,
            measurement_artifact=config+str(index), total_seconds=seconds, link_bytes=100)
            for config, seconds in [('batch1', 10), ('batch128', 100)]
            for index in range(3)]
        self.assertFalse(rankable_repeat_group(rows))
        summary = next(iter(summarize_rows(rows).values()))
        self.assertFalse(summary['rankable'])
        self.assertIsNone(summary['latency_seconds'])
        self.assertEqual(len(summary['configuration_groups']), 2)

if __name__ == '__main__': unittest.main()
