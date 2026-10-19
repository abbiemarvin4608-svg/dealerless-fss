"""The compiled public runtime must expose every conversion control."""
import json
import os
import subprocess
import unittest


class BackendContract(unittest.TestCase):
    def test_conversion_backend_controls_are_available(self):
        result = json.loads(subprocess.check_output(
            [os.environ["SDCF_TEST_BINARY"], "--describe"], text=True))
        self.assertEqual(set(result["backends"]), {
            "SCI", "SCI_ours", "SCI_Beaver", "SCI_Beaver_full", "SCI_Beaver_segmented"})


if __name__ == "__main__":
    unittest.main()
