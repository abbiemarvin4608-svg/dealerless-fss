"""Integrity checks at the existing plaintext oracle report boundary."""
import unittest

import numpy as np

from summarize_folded_quality import compare_predictions


class FoldedQualityTests(unittest.TestCase):
    def test_partial_holdout_cannot_pass_the_model_quality_gate(self):
        with self.assertRaisesRegex(ValueError, '6798'):
            compare_predictions(np.zeros(3, dtype=np.int64), np.zeros(3, dtype=np.int64),
                                np.zeros(3, dtype=np.int64), np.zeros(3, dtype=np.int64),
                                np.zeros((3, 30), dtype=np.int64))

    def test_full_holdout_failure_is_reported_without_tuning(self):
        labels = np.zeros(6798, dtype=np.int64)
        fp32 = labels.copy()
        sci = labels.copy()
        sci[:34] = 1
        folded = labels.copy()
        folded[:68] = 1
        logits = np.zeros((6798, 30), dtype=np.int64)
        logits[np.arange(6798), folded] = 1
        report = compare_predictions(labels, fp32, sci, folded, logits)
        self.assertEqual(report['folded_correct'], 6730)
        self.assertEqual(report['sci_correct'], 6764)
        self.assertEqual(report['fp32_correct'], 6798)
        self.assertAlmostEqual(report['folded_loss_pp'], 68 * 100 / 6798)
        self.assertAlmostEqual(report['folded_sci_prediction_agreement_pct'], 6764 * 100 / 6798)
        self.assertFalse(report['model_quality_qualified'])
        self.assertEqual(report['secure_samples'], 0)


if __name__ == '__main__':
    unittest.main()
