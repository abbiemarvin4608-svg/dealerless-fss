"""Native activation controls at the compiled runtime boundary."""
import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


@unittest.skipUnless(os.environ.get("SDCF_TEST_BINARY"), "set SDCF_TEST_BINARY")
class NativeWorkerContract(unittest.TestCase):
    def test_runtime_exposes_bounded_resident_activation_workers(self):
        description = json.loads(subprocess.check_output(
            [os.environ["SDCF_TEST_BINARY"], "--describe"], text=True))
        self.assertEqual(description["activation_workers"], [1, 2, 3, 4])
        self.assertEqual(description["max_compute_participants"], 4)
        self.assertTrue(description["resident_activation_workers"])

    @unittest.skipUnless(os.environ.get("SDCF_TEST_FIXTURE"), "set SDCF_TEST_FIXTURE")
    def test_repeated_queries_preserve_native_tensors_with_parallel_and_tail_chunks(self):
        with tempfile.TemporaryDirectory() as directory:
            observed = []
            for count in (1, 4, 3):
                output = Path(directory) / str(count)
                command = [sys.executable, str(Path(__file__).with_name("run.py")),
                    "--binary", os.environ["SDCF_TEST_BINARY"],
                    "--fixture", os.environ["SDCF_TEST_FIXTURE"], "--output", str(output),
                    "--backend", "SCI", "--steps", "1", "--batch", "5", "--queries", "2",
                    "--diagnostic", "--activation-workers", str(count), "--cggm-threads", "1",
                    "--server-cpus", "0,2,4,10", "--client-cpus", "6,8,12,16", "--timeout", "120"]
                child = subprocess.run(command, text=True, capture_output=True)
                self.assertEqual(child.returncode, 0, child.stdout + child.stderr)
                result = json.loads((output / "result.json").read_text())
                self.assertEqual(result["status"], "complete")
                log = (output / "party2.log").read_text().splitlines()
                tensors = [line for line in log if line.startswith("TENSOR ")]
                self.assertEqual(len(tensors), 30)
                observed.append(tensors)
                activations = [json.loads(line.split(" ", 1)[1]) for line in log
                               if line.startswith("ACTIVATION_DETAIL ")]
                self.assertEqual(len(activations), 4)
                self.assertTrue(all(row["actual_workers"] == count for row in activations))
                self.assertTrue(all(row["compute_participants"] <= 4 for row in activations))
            self.assertEqual(observed[0], observed[1])
            self.assertEqual(observed[0], observed[2])


if __name__ == "__main__":
    unittest.main()
