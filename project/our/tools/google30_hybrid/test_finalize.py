"""Full quality publication requires actual secure outputs and an independent oracle."""
import json
import tempfile
import unittest
from pathlib import Path

from finalize import quality_from_ledgers


class FullQualityTests(unittest.TestCase):
    def test_complete_quality_uses_labels_and_fp32_separately(self):
        contract = {'identity': 'contract', 'samples': 4, 'sample_ids': list('abcd'), 'labels': [2, 3, 1, 0]}
        with tempfile.TemporaryDirectory() as directory:
            paths = {}
            for backend in ('SCI', 'SCI_ours'):
                paths[backend] = Path(directory) / (backend + '.jsonl')
                rows = [dict(identity=backend, contract='contract', backend=backend,
                    sample_index=i, sample_id=contract['sample_ids'][i], label=contract['labels'][i],
                    prediction=prediction, oracle_prediction=prediction, oracle_match=True,
                    diagnostic=False) for i, prediction in enumerate([2, 2, 1, 0])]
                paths[backend].write_text(''.join(json.dumps(row) + '\n' for row in rows))
            result = quality_from_ledgers(paths, contract, [2, 3, 1, 0], [2, 3, 1, 2], [2, 2, 1, 0])
            self.assertTrue(result['complete'])
            self.assertEqual(result['fixed_top1_pct'], 75)
            self.assertEqual(result['loss_pp'], 0)
            self.assertEqual(result['prediction_agreement_pct'], 50)
            paths['SCI'].write_text(json.dumps(dict(rows[0], backend='SCI', identity='SCI')) + '\n')
            partial = quality_from_ledgers(paths, contract, [2, 3, 1, 0], [2, 3, 1, 2], [2, 2, 1, 0])
            self.assertIsNone(partial['fixed_top1_pct'])
            self.assertFalse(partial['quality_gate_pass'])

    def test_equal_backends_cannot_hide_a_shared_oracle_mismatch(self):
        contract = {'identity': 'contract', 'samples': 2, 'sample_ids': ['a', 'b'], 'labels': [2, 3]}
        with tempfile.TemporaryDirectory() as directory:
            paths = {}
            for backend in ('SCI', 'SCI_ours'):
                paths[backend] = Path(directory) / (backend + '.jsonl')
                rows = [dict(identity=backend, contract='contract', backend=backend,
                             sample_index=i, sample_id=sample, label=label,
                             prediction=2, oracle_prediction=2, oracle_match=True,
                             diagnostic=False) for i, (sample, label) in enumerate(zip(['a', 'b'], [2, 3]))]
                paths[backend].write_text(''.join(json.dumps(row) + '\n' for row in rows))
            result = quality_from_ledgers(paths, contract, [2, 3], [2, 3], [2, 3])
        self.assertFalse(result['complete'])
        self.assertFalse(result['full_oracle_match'])
        self.assertIsNone(result['fixed_top1_pct'])
        self.assertEqual(result['oracle_mismatches']['SCI'], [1])


if __name__ == '__main__':
    unittest.main()
