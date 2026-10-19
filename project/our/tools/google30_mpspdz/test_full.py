"""Public full-network execution ledger and client-output seam."""
import unittest
from full import validate_predictions, parse_full_output
from evaluate_secure import summarize_coverage

class FullPredictionTests(unittest.TestCase):
    def test_full_output_preserves_client_sample_and_step_identity(self):
        log='TRACE 0 98 H 0 -32768\nLOGIT 0 0 -12\nLOGIT 0 1 7\nPRED 0 1\n'
        out=parse_full_output(log)
        self.assertEqual(out['predictions'],{0:1})
        self.assertEqual(out['logits'],{0:[-12,7]})
        self.assertEqual(out['traces'][(0,98,'H')],[-32768])
    def test_incomplete_or_duplicate_secure_prediction_does_not_close_evaluation(self):
        with self.assertRaises(ValueError):validate_predictions({0:3},[0,1])
        with self.assertRaises(ValueError):parse_full_output('PRED 0 3\nPRED 0 3\n')
    def test_full_coverage_with_oracle_mismatch_never_passes_quality_gate(self):
        result=summarize_coverage({0:3,1:4},[3,4],[3,4],[3,5])
        self.assertTrue(result['coverage_complete'])
        self.assertFalse(result['complete'])
        self.assertIsNone(result['quality_gate_pass'])
        self.assertEqual(result['mismatches'][0]['sample_id'],1)
    def test_matching_predictions_validated_without_releasing_intermediates(self):
        self.assertTrue(validate_predictions({0:3,1:4},[0,1]))
if __name__=='__main__':unittest.main()
