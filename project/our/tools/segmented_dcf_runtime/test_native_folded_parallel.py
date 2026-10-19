"""Uneven complete-activation partitions at the real native worker boundary."""
import os
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

import test_native_dpf_activations as native_harness
from run import records

sys.path.append(str(Path(__file__).resolve().parents[1] / "google30_mpspdz"))
from folded_semantics import activation

BOUNDARIES = [0, 1, -1, 127, -128, 128, -129, 1023, -1024, 32767, -32768, 32766, -32767,
              255, -256, 256, -257, 4095, -4096, 4096, -4097, 16383, -16384, 8191, -8192]


class NativeFoldedParallelTests(unittest.TestCase):
    setUpClass = classmethod(native_harness.NativeDpfActivationTests.setUpClass.__func__)
    tearDownClass = classmethod(native_harness.NativeDpfActivationTests.tearDownClass.__func__)
    parties = native_harness.NativeDpfActivationTests.parties

    def test_uneven_two_worker_fresh_activations_preserve_both_lut_oracles_and_budget(self):
        for family in (0, 1):
            environment = dict(SDCF_FOLDED_LUT_FAMILY=str(family), SDCF_ACTIVATION_WORKERS="2")
            if os.environ.get("SDCF_ACTIVATION_EVIDENCE_DIR"):
                environment["SDCF_ACTIVATION_EVIDENCE_DIR"] = str(
                    Path(os.environ["SDCF_ACTIVATION_EVIDENCE_DIR"]) / ("full" if family == 0 else "segmented"))
            with patch.dict(os.environ, environment):
                logs = self.parties("folded", 259, 2)
            output = records(logs[1], "ACTIVATION_OUTPUT")
            self.assertEqual(len(output), 4)
            for row in output:
                clear = [BOUNDARIES[(i + row["query"] * 7) % len(BOUNDARIES)] for i in range(259)]
                self.assertEqual(row["values"], [activation(x, row["activation"]) & 65535 for x in clear])
            for log in logs:
                details = records(log, "ACTIVATION_DETAIL")
                self.assertEqual(len(details), 4)
                for row in details:
                    self.assertEqual(row["actual_workers"], 2)
                    self.assertEqual(row["compute_participants"], 2)
                workers = records(log, "ACTIVATION_WORKER")
                self.assertEqual(len(workers), 8)
                for query in (0, 1):
                    for layer in (0, 1):
                        chunks = sorted((row for row in workers if (row["query"], row["index"]) == (query, layer)), key=lambda row: row["begin"])
                        self.assertEqual([(row["begin"], row["count"]) for row in chunks], [(0, 130), (130, 129)])
                for row in workers:
                    self.assertEqual(row["cggm_participant_limit"], 1)
                    self.assertEqual(row["compute_participants"], 1)
                    self.assertLessEqual(row["observed_cggm_participants"], 1)
                    self.assertEqual(row["observed_transport_participants"], 1)


if __name__ == "__main__":
    unittest.main()
