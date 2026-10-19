"""Public CLI regression against archived, independently recorded predictions."""
import csv
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

SCRIPT = Path(__file__).resolve().parents[1] / 'replay.py'
ARCHIVE = Path(os.environ.get('GOOGLE30_ARCHIVE', '/home/jiang/dealerless-lunwen/reproduction_results/sirnn_google30_20261007'))
RUNTIME = Path(os.environ.get('GOOGLE30_RUNTIME', '/home/jiang/EzPC/SIRNN/reproduction'))

class NativeReplayTest(unittest.TestCase):
    def test_archived_predictions_are_recomputed_including_known_error(self):
        with tempfile.TemporaryDirectory() as output:
            subprocess.run([sys.executable, str(SCRIPT), '--archive', str(ARCHIVE),
                            '--runtime', str(RUNTIME), '--output', output,
                            '--rows', '0,16,28'], check=True, capture_output=True, text=True)
            with open(Path(output) / 'predictions.csv') as f:
                rows = list(csv.DictReader(f))
            self.assertEqual([(int(r['row']), int(r['prediction'])) for r in rows],
                             [(0, 24), (16, 4), (28, 27)])
            self.assertEqual(int(rows[1]['label']), 3)
            self.assertTrue(all(r['matches_archived_seedot'] == 'True' for r in rows))

if __name__ == '__main__':
    unittest.main()
