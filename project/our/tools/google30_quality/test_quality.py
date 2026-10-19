"""Public CLI tests for the agreed data-separation and quality-artifact seams."""
import json
import os
import shutil
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
import numpy as np

CLI = Path(__file__).with_name('quality.py')

class QualityCLI(unittest.TestCase):
    def test_rejects_evaluation_speaker_seen_in_model_selection(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            for split, path in [('train', 'yes/train_nohash_0.wav'),
                                ('test', 'no/shared_nohash_0.wav'),
                                ('valid', 'yes/shared_nohash_1.wav')]:
                (root / (split + '_samples.jsonl')).write_text(json.dumps({'path': path, 'label': 0}) + '\n')
            result = subprocess.run([sys.executable, str(CLI), 'audit-splits', '--data', str(root)],
                                    stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
            self.assertNotEqual(result.returncode, 0)
            self.assertIn('speaker overlap', result.stderr)

    def test_reports_negative_loss_and_logit_error_without_changing_gate(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            arrays = {'float': [[2, 0], [3, 1], [0, 2], [0, 3]],
                      'fixed': [[2, 0], [0, 3], [0, 2], [3, 0]],
                      'labels': [0, 1, 1, 0]}
            for name, values in arrays.items():
                np.save(str(root / (name + '.npy')), values)
            result = subprocess.run([sys.executable, str(CLI), 'summarize',
                                     '--float-logits', str(root / 'float.npy'),
                                     '--fixed-logits', str(root / 'fixed.npy'),
                                     '--labels', str(root / 'labels.npy'),
                                     '--output', str(root / 'quality.json')],
                                    stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
            self.assertEqual(result.returncode, 0, result.stderr)
            report = json.loads((root / 'quality.json').read_text())
            self.assertEqual(report['loss_pp'], -50.0)
            self.assertEqual(report['prediction_agreement_pct'], 50.0)
            self.assertEqual(report['fp32_correct_fixed_wrong'], 0)
            self.assertEqual(report['fixed_correct_fp32_wrong'], 2)
            self.assertEqual(report['logit_error']['mae'], 1.375)
            self.assertTrue(report['passes_1pp'])

    @unittest.skipUnless(os.environ.get('GOOGLE30_REPRODUCTION'), 'requires local Google-30 source bundle')
    def test_freezing_never_reads_test_or_holdout_feature_arrays(self):
        source = Path(os.environ['GOOGLE30_REPRODUCTION']).resolve()
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            repro = root / 'reproduction'
            data = repro / 'data/Google-30-processed'
            data.mkdir(parents=True)
            (repro / 'models').symlink_to(source / 'models', target_is_directory=True)
            (root / 'EdgeML-sirnn').symlink_to(source.parent / 'EdgeML-sirnn', target_is_directory=True)
            original_data = source / 'data/Google-30-processed'
            (data / 'train.npy').symlink_to(original_data / 'train.npy')
            for name in ['train_samples.jsonl', 'test_samples.jsonl', 'valid_samples.jsonl', 'preprocessing.json']:
                shutil.copy2(str(original_data / name), str(data))
            for name in ['test.npy', 'valid.npy']:
                (data / name).write_text('Deliberately invalid: final features must not be opened during freeze.')
            output = root / 'run'
            result = subprocess.run([sys.executable, str(CLI.with_name('prepare.py')),
                                     '--reproduction', str(repro), '--output', str(output)],
                                    stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
            self.assertEqual(result.returncode, 0, result.stderr)
            cal = output / 'SeeDot/datasets/fastgrnn/Google-30'
            self.assertTrue(np.array_equal(np.load(str(cal / 'train.npy')), np.load(str(cal / 'test.npy'))))
            self.assertEqual(np.load(str(cal / 'test.npy')).shape, (512, 3169))
            self.assertFalse(json.loads((output / 'freeze_policy.json').read_text())['final_evaluation_features_read_during_preparation'])

if __name__ == '__main__':
    unittest.main()
