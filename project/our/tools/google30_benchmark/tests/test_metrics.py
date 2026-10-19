import importlib.util
import pathlib
import unittest

MODULE = pathlib.Path(__file__).parents[1] / 'metrics.py'

class MetricsTest(unittest.TestCase):
    def test_two_party_total_counts_each_link_byte_once(self):
        spec = importlib.util.spec_from_file_location('metrics', MODULE)
        metrics = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(metrics)
        phases = ['application_io', 'initialization_base_ot', 'input_sharing', 'online_compute', 'output_reconstruction', 'teardown']
        parties = []
        for role, sent, received, total in [(1, 10, 20, 6.0), (2, 20, 10, 7.0)]:
            parties.append({'party':role, 'total_seconds':total, 'peak_rss_kib':100*role,
                'phases':{name:{'seconds':1.0, 'sent_bytes':sent, 'received_bytes':received,
                    'send_calls':2, 'recv_calls':3} for name in phases}})
        result = metrics.summarize_parties(parties)
        self.assertEqual(result['link_bytes'], 180)
        self.assertEqual(result['first_query_seconds'], 7.0)
        self.assertIsNone(result['offline_seconds'])
        self.assertIsNone(result['dependent_rounds'])
        self.assertEqual(result['io_calls'], 60)
        self.assertEqual(result['amortization']['queries'], 1)
        self.assertEqual(result['peak_rss_kib_per_party'], [100, 200])

    def test_unbalanced_bytes_rejects_incomplete_execution(self):
        spec = importlib.util.spec_from_file_location('metrics', MODULE)
        metrics = importlib.util.module_from_spec(spec); spec.loader.exec_module(metrics)
        p = {'party':1,'total_seconds':1,'peak_rss_kib':1,'phases':{'online_compute':{'seconds':1,'sent_bytes':2,'received_bytes':0,'send_calls':1,'recv_calls':0}}}
        with self.assertRaises(ValueError):
            metrics.summarize_parties([p, dict(p,party=2)])

if __name__ == '__main__': unittest.main()
