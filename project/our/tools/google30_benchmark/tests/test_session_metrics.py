import importlib.util
import pathlib
import unittest

class SessionMetricsTest(unittest.TestCase):
    def test_persistent_query_cost_uses_observed_two_queries(self):
        spec=importlib.util.spec_from_file_location('metrics',pathlib.Path(__file__).parents[1]/'metrics.py')
        module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
        parties=[]
        for role,sent,received,total in [(1,100,200,10),(2,200,100,11)]:
            parties.append(dict(party=role,total_seconds=total,peak_rss_kib=100,phases={'initialization_base_ot':dict(seconds=2,sent_bytes=sent,received_bytes=received,send_calls=1,recv_calls=1), 'online_compute':dict(seconds=7,sent_bytes=40 if role==1 else 60,received_bytes=60 if role==1 else 40,send_calls=2,recv_calls=2)},queries=[dict(seconds=3,sent_bytes=10 if role==1 else 20,received_bytes=20 if role==1 else 10),dict(seconds=4,sent_bytes=30 if role==1 else 40,received_bytes=40 if role==1 else 30)]))
        result=module.summarize_parties(parties)
        self.assertEqual(result['amortization']['queries'],2)
        self.assertEqual(result['first_query_seconds'],5)
        self.assertEqual(result['amortization']['measured_session_bytes_per_query'],200)
        self.assertEqual(result['amortization']['measured_session_seconds_per_query'],5.5)
        self.assertEqual(result['amortization']['measured_second_query_seconds'],4)
        self.assertAlmostEqual(result['throughput_queries_per_second'],2/11)

    def test_different_party_query_counts_are_rejected(self):
        spec=importlib.util.spec_from_file_location('metrics',pathlib.Path(__file__).parents[1]/'metrics.py')
        module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
        p=dict(party=1,total_seconds=1,peak_rss_kib=1,phases={},queries=[dict(seconds=1,sent_bytes=0,received_bytes=0)])
        with self.assertRaises(ValueError):
            module.summarize_parties([p,dict(p,party=2,queries=[])])

if __name__=='__main__':unittest.main()
