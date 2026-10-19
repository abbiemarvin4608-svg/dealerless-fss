"""Metrics CLI seam, using unmodified installed MP-SPDZ log syntax."""
import unittest
from run import parse_log,parse_values
class RunnerTests(unittest.TestCase):
    def test_native_runtime_metrics_keep_units_and_reported_rounds(self):
        log='Setup took 0.015 seconds.\nTime = 0.909713 seconds\nTime20 = 0.102773 seconds (9.72 MB, 640 rounds)\nData sent = 76.3455 MB in ~5947 rounds (party 0 only)\n'
        r=parse_log(log)
        self.assertEqual(r['runtime_s'],0.909713)
        self.assertEqual(r['communication_sent_bytes'],76345500)
        self.assertEqual(r['reported_send_rounds'],5947)
    def test_reconstructed_integer_output_preserves_signs_and_case_boundaries(self):
        self.assertEqual(parse_values('VALUE tmp19 0 -1\nVALUE H 0 -32768\nVALUE tmp19 0 0\nVALUE H 0 32767\n'),[{'tmp19':[-1],'H':[-32768]},{'tmp19':[0],'H':[32767]}])
    def test_invalid_output_is_rejected(self):
        with self.assertRaises(ValueError):parse_values('VALUE tmp19 1 0\n')
if __name__=='__main__':unittest.main()
