"""Public resume/artifact contract; secure computations remain integration tests."""
import unittest
from full import validate_ledger
from compare_full import compare_ledgers
from pathlib import Path
import json
import tempfile

class LedgerTests(unittest.TestCase):
 def test_rejects_duplicate_sample_even_with_equal_predictions(self):
  rows=[{'sample_index':0,'sample_id':'a','prediction':2,'identity':'v1'}]*2
  with self.assertRaisesRegex(ValueError,'duplicate'):validate_ledger(rows,['a'],'v1')
 def test_rejects_mixed_contracts(self):
  with self.assertRaisesRegex(ValueError,'identity'):validate_ledger([{'sample_index':0,'sample_id':'a','prediction':2,'identity':'old'}],['a'],'v1')
 def test_resume_reports_missing_samples(self):
  rows=[{'sample_index':1,'sample_id':'b','prediction':2,'identity':'v1'}]
  self.assertEqual(validate_ledger(rows,['a','b','c'],'v1'),[0,2])
class CompareTests(unittest.TestCase):
 def test_equal_overlap_does_not_close_incomplete_full_run(self):
  contract={'identity':'contract1','samples':2,'sample_ids':['a','b']}
  row={'identity':'backend1','contract':'contract1','sample_index':0,'sample_id':'a','prediction':2,'diagnostic':False}
  with tempfile.TemporaryDirectory()as directory:
   left=Path(directory)/'SCI.jsonl';right=Path(directory)/'ours.jsonl'
   left.write_text(json.dumps(row)+'\n');row['identity']='backend2';right.write_text(json.dumps(row)+'\n')
   result=compare_ledgers(left,right,contract)
  self.assertEqual(result['agreement_on_executed_overlap_pct'],100)
  self.assertFalse(result['full_secure_agreement'])
  self.assertFalse(result['complete'])

if __name__=='__main__':unittest.main()
