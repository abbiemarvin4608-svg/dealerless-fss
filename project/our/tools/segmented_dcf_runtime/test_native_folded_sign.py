"""Private segmented sign tests preserve both complete native activations."""
import os
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

import test_native_dpf_activations as harness
import test_native_folded_joint as joint
from run import records
sys.path.append(str(Path(__file__).resolve().parents[1] / 'google30_mpspdz'))
from folded_semantics import activation

VALUES = (0, 1, -1, 127, -128, 128, -129, 1023, -1024, 32767, -32768,
          32766, -32767, 255, -256, 256, -257, 4095, -4096, 4096, -4097,
          16383, -16384, 8191, -8192)


class NativeFoldedSignTests(unittest.TestCase):
    setUpClass = classmethod(harness.NativeDpfActivationTests.setUpClass.__func__)
    tearDownClass = classmethod(harness.NativeDpfActivationTests.tearDownClass.__func__)
    parties = harness.NativeDpfActivationTests.parties
    check_family = joint.NativeFoldedJointTests.check_family

    def test_private_sign_preserves_native_boundaries(self):
        with patch.dict(os.environ, SDCF_FOLDED_SIGN_BACKEND='1', SDCF_FOLDED_SIGN_SEGMENTS='4'):
            logs = self.parties('folded', 25, 2)
        for row in records(logs[1], 'ACTIVATION_OUTPUT'):
            expected = [activation(VALUES[(i + row['query'] * 7) % len(VALUES)], row['activation']) & 65535
                        for i in range(25)]
            self.assertEqual(row['values'], expected)
        for log in logs:
            consumed = records(log, 'FOLDED_ACTIVATION_CONSUMED')
            self.assertEqual(len(consumed), 4)
            self.assertTrue(all(row['sign_backend'] == 1 and row['sign_segments'] == 4
                                and row['online_generated'] == row['online_loaded_bytes'] == 0
                                for row in consumed))

    def compare_family(self, family):
        reference = None
        evidence = os.environ.get('SDCF_FOLDED_SIGN_EVIDENCE')
        for backend, segments in ((0, 0), (1, 4)):
            env = {'SDCF_FOLDED_JOINT_EVIDENCE': str(Path(evidence) / ('sign' + str(backend)))} if evidence else {}
            with patch.dict(os.environ, env):
                output = self.check_family(family, backend, segments)
            if reference is None:
                reference = output
            else:
                self.assertEqual(output, reference)

    def test_full_q_private_sign_and_bitdec_cover_native_domain(self):
        self.compare_family(0)

    def test_segmented_private_sign_and_bitdec_cover_native_domain(self):
        self.compare_family(1)


if __name__ == '__main__':
    unittest.main()
