"""Complete native TanH behavior against the frozen paper integer reference."""
from pathlib import Path
import sys
import unittest

import test_native_dpf_activations as native
from run import records

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'google30_mpspdz'))
from folded_semantics import activation


class NativeFoldedTanhTests(unittest.TestCase):
    setUpClass = classmethod(native.NativeDpfActivationTests.setUpClass.__func__)
    tearDownClass = classmethod(native.NativeDpfActivationTests.tearDownClass.__func__)
    parties = native.NativeDpfActivationTests.parties

    def check_tanh(self, count, queries, exhaustive=False):
        logs = self.parties('folded_tanh', count, queries, exhaustive)
        outputs = [row for row in records(logs[1], 'ACTIVATION_OUTPUT')
                   if row['activation'] == 'tanh']
        self.assertEqual(len(outputs), queries)
        boundaries = (0, 1, -1, 127, -128, 128, -129, 1023, -1024,
                      32767, -32768, 32766, -32767, 255, -256, 256,
                      -257, 4095, -4096, 4096, -4097, 16383, -16384,
                      8191, -8192)
        for row in outputs:
            query = row['query']
            expected = []
            for element in range(count):
                residue = ((query * count + element) & 65535) if exhaustive else (
                    boundaries[(element + query * 7) % len(boundaries)] & 65535)
                signed = residue if residue < 32768 else residue - 65536
                expected.append(activation(signed, 'tanh') & 65535)
            self.assertEqual(row['values'], expected)
        for log in logs:
            prepared = records(log, 'FOLDED_ACTIVATION_PREPARED')
            consumed = records(log, 'FOLDED_ACTIVATION_CONSUMED')
            balance = records(log, 'FOLDED_ACTIVATION_BALANCE')
            self.assertEqual(len(prepared), queries)
            self.assertEqual(len(consumed), queries)
            self.assertEqual(len(balance), queries)
            for row in prepared:
                self.assertEqual((row['kind'], row['count'], row['input_bits'],
                                  row['output_bits'], row['input_scale'],
                                  row['output_scale'], row['lut_family'],
                                  row['sign_backend']), (1, count, 16, 16, 7, 7, 0, 0))
                self.assertTrue(row['pre_input'])
                self.assertGreater(row['resident_bytes'], 0)
                self.assertGreater(row['offline'][0]['sent_bytes'], 0)
                self.assertGreater(row['offline'][17]['sent_bytes'], 0)
            for row in consumed:
                self.assertEqual(row['online_generated'], 0)
                self.assertEqual(row['online_loaded_bytes'], 0)
                self.assertGreater(row['online'][0]['sent_bytes'], 0)
                self.assertGreater(row['online'][17]['sent_bytes'], 0)
            for row in balance:
                self.assertEqual(row['generated'], row['consumed'])
                self.assertEqual(row['remaining'], 0)
                self.assertEqual(row['online_generated'], 0)
                self.assertEqual(row['online_loaded_bytes'], 0)
            for row in records(log, 'METRIC'):
                if row['phase'] in ('input', 'ready', 'online', 'output'):
                    self.assertEqual(row['folded_generation_epoch_delta'], 0)
            tanh_coverage = [row for row in records(log, 'ACTIVATION_MATERIAL_COVERAGE')
                             if row['activation'] == 'tanh']
            self.assertEqual(len(tanh_coverage), queries)
            self.assertTrue(all(row['native_ot_consumed'] == 0 and
                                row['bit_triples_consumed'] == 0
                                for row in tanh_coverage))

    def test_complete_tanh_native_boundaries_and_fresh_queries(self):
        self.check_tanh(25, 2)

    def test_complete_tanh_full_uint16_domain(self):
        self.check_tanh(4096, 16, True)


if __name__ == '__main__':
    unittest.main()
