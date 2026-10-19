"""Public measurement/result eligibility contract."""
import unittest
from metrics import combine, quality_eligibility

class MetricsTests(unittest.TestCase):
 def test_count_link_bytes_once(self):
  a={'party':1,'total_seconds':2,'peak_rss_kib':10,'phases':{'compute':{'seconds':1,'sent_bytes':100,'received_bytes':200}},'queries':[{'seconds':1,'sent_bytes':100,'received_bytes':200}],'bridge':{}}
  b={'party':2,'total_seconds':2.1,'peak_rss_kib':12,'phases':{'compute':{'seconds':1.1,'sent_bytes':200,'received_bytes':100}},'queries':[{'seconds':1.1,'sent_bytes':200,'received_bytes':100}],'bridge':{}}
  self.assertEqual(combine([a,b])['link_bytes'],300)
 def test_partial_accuracy_never_eligible(self):
  q={'samples':6798,'loss_pp':.4,'within_1pp':True}
  a={'complete':False,'full_secure_agreement':False,'compared_secure_samples':160}
  self.assertFalse(quality_eligibility(q,a)['eligible'])
 def test_equal_top1_does_not_replace_prediction_equality(self):
  q={'samples':6798,'loss_pp':.4,'within_1pp':True}
  a={'complete':True,'full_secure_agreement':False,'compared_secure_samples':6798}
  self.assertFalse(quality_eligibility(q,a)['eligible'])
if __name__=='__main__':unittest.main()

class BridgeMetricsTests(unittest.TestCase):
    def _party(self, role):
        return {'party': role, 'total_seconds': 2, 'peak_rss_kib': 10,
                'phases': {'compute': {'seconds': 1, 'sent_bytes': 100 if role == 1 else 200,
                                        'received_bytes': 200 if role == 1 else 100}},
                'queries': [{'seconds': 1, 'sent_bytes': 100 if role == 1 else 200,
                             'received_bytes': 200 if role == 1 else 100}],
                'bridge': {'sent_bytes': 40 if role == 1 else 60,
                           'received_bytes': 60 if role == 1 else 40}}

    def test_cumulative_bridge_is_added_once(self):
        result = combine([self._party(1), self._party(2)])
        self.assertEqual(result['sci_link_bytes'], 300)
        self.assertEqual(result['fss_link_bytes'], 100)
        self.assertEqual(result['link_bytes'], 400)

if __name__ == '__main__': unittest.main()
