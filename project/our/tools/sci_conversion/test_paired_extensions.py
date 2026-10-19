"""Live paired-extension regression through the public two-party runtime."""
import json
import os
from pathlib import Path
import subprocess
import tempfile
import unittest


@unittest.skipUnless(os.environ.get("SDCF_TEST_BINARY") and os.environ.get("SDCF_TEST_FIXTURE"),
                     "requires a compiled runtime and frozen fixture")
class PairedExtensions(unittest.TestCase):
    def test_pairing_preserves_tensors_and_reduces_extension_depth(self):
        runner = Path(__file__).parents[1] / "segmented_dcf_runtime/run.py"
        with tempfile.TemporaryDirectory(prefix="sdcf-paired-") as directory:
            root = Path(directory)
            tensors = []
            extensions = []
            for paired in (False, True):
                output = root / ("paired" if paired else "separate")
                command = ["python3", str(runner), "--binary", os.environ["SDCF_TEST_BINARY"],
                           "--fixture", os.environ["SDCF_TEST_FIXTURE"], "--output", str(output),
                           "--backend", "SCI_Beaver_segmented", "--batch", "5", "--steps", "1",
                           "--queries", "1", "--warmup", "0", "--diagnostic", "--timeout", "120",
                           "--cpus", "10,12,14", "--server-cpus", "0,2,4,10",
                           "--client-cpus", "6,8,12,16", "--cggm-threads", "4"]
                if paired:
                    command.append("--pair-extensions")
                result = subprocess.run(command, capture_output=True, text=True, timeout=150)
                self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
                log = (output / "party2.log").read_text().splitlines()
                tensors.append([line for line in log if line.startswith("TENSOR ")])
                calls = [json.loads(line.split(" ", 1)[1]) for line in log
                         if line.startswith("CONVERSION_ONLINE ")]
                details = [json.loads(line.split(" ", 1)[1]) for line in log
                           if line.startswith("CONVERSION_DETAIL ")]
                indices = {row["index"] for row in json.loads((output / "schedule.json").read_text())
                           if row["function"] == "MulCir"}
                keys = {(row["index"], row["slot"]) for row in calls
                        if row["index"] in indices and row["call"] == 3}
                extensions.append([row for row in details if (row["index"], row["slot"]) in keys])
            self.assertEqual(len(tensors[0]), 15)
            self.assertEqual(tensors[0], tensors[1])
            self.assertEqual(len(extensions[0]), 4)
            self.assertEqual(len(extensions[1]), 2)
            self.assertEqual(sum(row["logical_rounds"] for row in extensions[0]), 24)
            self.assertEqual(sum(row["logical_rounds"] for row in extensions[1]), 12)


if __name__ == "__main__":
    unittest.main()
