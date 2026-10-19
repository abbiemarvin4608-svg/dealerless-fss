"""Call-count artifact consumer: primitive families must not be double-counted."""
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

SCRIPT = Path(__file__).resolve().parents[1] / 'summarize_trace.py'

class TraceSummaryTest(unittest.TestCase):
    def test_nested_msb_and_comparator_are_distinct_cost_families(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            source=root/'trace.csv'
            source.write_text('Sigmoid_01,0,/sigmoid/MSB,MSB,100,16,1,0\n'
                              'Sigmoid_01,0,/sigmoid/MSB/compare,compare,100,15,1,1\n'
                              'ArgMax_01,99,/compare,compare,3,15,1,1\n')
            subprocess.run([sys.executable,str(SCRIPT),'--input',str(source),'--output',str(root/'summary.json')],check=True,capture_output=True)
            result=json.loads((root/'summary.json').read_text())
            self.assertEqual(result['compare']['scalar_instances'],103)
            self.assertEqual(result['compare']['invocations'],2)
            self.assertEqual(result['MSB']['scalar_instances'],100)
            self.assertEqual(result['compare']['widths'],{'15':103})

if __name__=='__main__': unittest.main()
