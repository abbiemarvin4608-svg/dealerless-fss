"""Reject incomplete measurements and count each transmitted byte once."""
import unittest
from run_step import parse_performance, validate_variant


def log(sci_sent, bridge_sent, online_ms):
    rows = []
    for name, query, elapsed in [('sci_setup', -1, 100),
                                 ('bridge_setup', -1, 50),
                                 ('offline', 0, 30),
                                 ('ready', 0, 1),
                                 ('online', 0, online_ms)]:
        rows.append(
            f'PHASE query={query} bridge_rounds=2 name={name} '
            f'bridge_received_bytes=100000 sci_sent_bytes={sci_sent} '
            f'wall_ms={elapsed} bridge_sent_bytes={bridge_sent} sci_rounds=3')
    rows.append('ACTIVATION tanh_ms=4 query=0 total_ms=7 sigmoid_ms=3 '
                'bridge_ms=5 lut_ms=0.1 calls=2 elements=200')
    rows.append('STEP_FINISHED mode=secure cases=1 batch=1 peak_rss_kib=100')
    return '\n'.join(rows)


class StepMeteringTests(unittest.TestCase):
    def test_sums_sends_once_and_accepts_reordered_fields(self):
        result = parse_performance([log(11, 3, 20), log(17, 5, 15)], 1)
        self.assertEqual(result['online'][0]['combined_sent_bytes'], 36)
        self.assertEqual(result['online_median_ms'], 20)
        self.assertEqual(result['fresh_query_median_ms'], 51)
        self.assertEqual(result['lut_median_ms'], 0.1)

    def test_missing_phase_is_rejected(self):
        incomplete = '\n'.join(row for row in log(11, 3, 20).splitlines()
                               if 'name=online ' not in row)
        with self.assertRaisesRegex(ValueError, 'missing or duplicate phase online'):
            parse_performance([incomplete, incomplete], 1)

    def test_unfinished_run_is_rejected(self):
        incomplete = log(11, 3, 20).replace('STEP_FINISHED', 'FAILED')
        with self.assertRaisesRegex(ValueError, 'missing completed step record'):
            parse_performance([incomplete, incomplete], 1)

    def test_historical_graph_cannot_use_independent_weight_narrowing(self):
        with self.assertRaisesRegex(ValueError, 'generated graph and requested model variant disagree'):
            validate_variant([{}] * 16, 'independent')
        validate_variant([{}] * 15, 'independent')


if __name__ == '__main__':
    unittest.main()
