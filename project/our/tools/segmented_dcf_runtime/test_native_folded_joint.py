"""Both complete paper activations cover every native input in both LUT families."""
import json
import os
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

import test_native_dpf_activations as native_harness
from run import records, validate_folded_inventory

sys.path.append(str(Path(__file__).resolve().parents[1] / "google30_mpspdz"))
from folded_semantics import activation


class NativeFoldedJointTests(unittest.TestCase):
    setUpClass = classmethod(native_harness.NativeDpfActivationTests.setUpClass.__func__)
    tearDownClass = classmethod(native_harness.NativeDpfActivationTests.tearDownClass.__func__)
    parties = native_harness.NativeDpfActivationTests.parties

    def check_family(self, family, sign_backend=0, sign_segments=0):
        count, queries = 4096, 16
        directory = os.environ.get("SDCF_FOLDED_JOINT_EVIDENCE")
        env = {"SDCF_FOLDED_LUT_FAMILY": str(family), "SDCF_FOLDED_SIGN_BACKEND": str(sign_backend),
               "SDCF_FOLDED_SIGN_SEGMENTS": str(sign_segments)}
        if directory:
            env["SDCF_ACTIVATION_EVIDENCE_DIR"] = str(Path(directory) / str(family))
        with patch.dict(os.environ, env):
            logs = self.parties("folded", count, queries, True)
        outputs = records(logs[1], "ACTIVATION_OUTPUT")
        self.assertEqual(len(outputs), 2 * queries)
        checked = {"sigmoid": 0, "tanh": 0}
        for row in outputs:
            name, query = row["activation"], row["query"]
            clear = range(query * count, (query + 1) * count)
            expected = [activation(x if x < 32768 else x - 65536, name) & 65535
                        for x in clear]
            self.assertEqual(row["values"], expected)
            checked[name] += len(expected)
        self.assertEqual(checked, {"sigmoid": 65536, "tanh": 65536})
        for log in logs:
            self.assertFalse(records(log, "DPF_LUT_PLAN"))
            self.assertTrue(validate_folded_inventory(records(log, 'FOLDED_ACTIVATION_PLAN'),
                records(log, 'FOLDED_ACTIVATION_PREPARED'), records(log, 'FOLDED_ACTIVATION_CONSUMED'),
                records(log, 'FOLDED_ACTIVATION_BALANCE'), records(log, 'METRIC'), queries, True, log))
            prepared = records(log, "FOLDED_ACTIVATION_PREPARED")
            consumed = records(log, "FOLDED_ACTIVATION_CONSUMED")
            self.assertEqual(len(prepared), 2 * queries)
            self.assertEqual(len(consumed), 2 * queries)
            for row in prepared:
                self.assertTrue(row["pre_input"])
                self.assertEqual(row["lut_family"], family)
                self.assertEqual(row["sign_backend"], sign_backend)
                self.assertEqual(row["sign_segments"], sign_segments)
                self.assertEqual(row["count"], count)
            for row in consumed:
                self.assertEqual(row["online_generated"], 0)
                self.assertEqual(row["online_loaded_bytes"], 0)
                self.assertEqual(row['native_ot_consumed'], 0)
                self.assertEqual(row['bit_triples_consumed'], 0)
            for row in records(log, "ACTIVATION_MATERIAL_COVERAGE"):
                self.assertEqual(row["native_ot_consumed"], 0)
                self.assertEqual(row["bit_triples_consumed"], 0)
            for row in records(log, "FOLDED_ACTIVATION_BALANCE"):
                self.assertEqual((row["generated"], row["consumed"], row["remaining"]), (2, 2, 0))
                self.assertEqual(row["generated_elements"], 2 * count)
                self.assertEqual(row["consumed_elements"], 2 * count)
                self.assertEqual(row["online_generation_epoch_delta"], 0)
            for row in records(log, "ACTIVATION_RESOURCES"):
                self.assertLessEqual(row["vm_peak_bytes"], 1536 << 20)
        if directory:
            report = dict(status="passed",lut_family=family,sign_backend=sign_backend,
                          sign_segments=sign_segments,checked_inputs=checked,
                          count=count,queries=queries,random_sharing=True,
                          zero_online_generation=True,zero_online_load=True,
                          no_native_sci_activation_inventory=True,
                          resources=[records(log,"ACTIVATION_RESOURCES") for log in logs])
            (Path(directory) / str(family) / "qualification.json").write_text(
                json.dumps(report, indent=2) + "\n")
        return outputs

    def test_full_q_native_sigmoid_and_tanh_domains(self):
        self.check_family(0)

    def test_segmented_native_sigmoid_and_tanh_domains(self):
        self.check_family(1)


if __name__ == "__main__":
    unittest.main()
