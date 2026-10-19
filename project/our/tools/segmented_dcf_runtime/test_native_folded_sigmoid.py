"""Complete native folded Sigmoid, independently checked under random sharing."""
import sys
import os
import unittest
from unittest.mock import patch
from pathlib import Path
from test_native_dpf_activations import NativeDpfActivationTests
from run import records
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'google30_mpspdz'))
from folded_semantics import activation

VALUES=(0,1,-1,127,-128,128,-129,1023,-1024,32767,-32768,32766,-32767,
    255,-256,256,-257,4095,-4096,4096,-4097,16383,-16384,8191,-8192)
class NativeFoldedSigmoidTests(unittest.TestCase):
    setUpClass=classmethod(NativeDpfActivationTests.setUpClass.__func__)
    tearDownClass=classmethod(NativeDpfActivationTests.tearDownClass.__func__)
    parties=NativeDpfActivationTests.parties
    def check_outputs(self,count,queries,exhaustive=False):
        logs=self.parties('folded_sigmoid',count,queries,exhaustive)
        outputs=records(logs[1],'ACTIVATION_OUTPUT')
        self.assertEqual(len(outputs),queries*2)
        for row in outputs:
            if row['activation']!='sigmoid':continue
            inputs=range(row['query']*count,(row['query']+1)*count) if exhaustive else (
                VALUES[(i+row['query']*7)%len(VALUES)] for i in range(count))
            self.assertEqual(row['values'],[activation(((x+32768)&65535)-32768,'sigmoid')&65535 for x in inputs])
        for log in logs:
            consumed=records(log,'FOLDED_ACTIVATION_CONSUMED')
            self.assertEqual(len(consumed),queries)
            self.assertTrue(all(row['kind']==0 and row['count']==count and row['online_generated']==0 and
                row['online_loaded_bytes']==0 for row in consumed))
            for row in consumed:
                self.assertEqual(sum(stage['sent_bytes'] for stage in row['online']),row['sent_bytes'])
                self.assertEqual(sum(stage['received_bytes'] for stage in row['online']),row['received_bytes'])
            self.assertFalse(records(log,'DPF_LUT_CONSUMED'))
            self.assertTrue(all(row['generated']==row['consumed'] and row['remaining']==0 and
                row['generated_elements']==row['consumed_elements'] for row in records(log,'FOLDED_ACTIVATION_BALANCE')))
            coverage=[row for row in records(log,'ACTIVATION_MATERIAL_COVERAGE') if row['activation']=='sigmoid']
            self.assertTrue(all(row['native_ot_consumed']==0 and row['bit_triples_consumed']==0 for row in coverage))
        return logs
    def test_random_shares_boundaries_two_fresh_queries(self):self.check_outputs(25,2)
    def test_all_uint16_residues(self):self.check_outputs(4096,16,True)
    def test_segmented_family_has_same_native_results(self):
        with patch.dict(os.environ,SDCF_FOLDED_LUT_FAMILY='1'):
            logs=self.check_outputs(25,2)
        for log in logs:
            rows=records(log,'FOLDED_ACTIVATION_PREPARED')
            self.assertTrue(all(row['lut_family']==1 and row['offline'][11]['sent_bytes']>0 and
                row['offline'][12]['sent_bytes']>0 for row in rows))
if __name__=='__main__':unittest.main()
