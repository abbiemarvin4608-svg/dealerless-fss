"""Complete batch-128 native preprocessing through the frozen public runner."""
import json
import os
from pathlib import Path
import subprocess
import tempfile
import unittest

HERE = Path(__file__).resolve().parent


@unittest.skipUnless(os.environ.get("SDCF_SHARED_STEP_BINARY"), "set SDCF_SHARED_STEP_BINARY to the combined runtime")
class SharedStepTests(unittest.TestCase):
    def check_all_backends(self, hidden_file):
        binary = Path(os.environ["SDCF_SHARED_STEP_BINARY"])
        description = json.loads(subprocess.check_output([str(binary), "--describe"], text=True))
        self.assertIn("SCI_preprocessed", description["backends"])
        self.assertTrue(description["supports_native_preprocessing"])
        self.assertEqual(description["native_preprocessing_steps"], [1])
        directory = Path(os.environ.get("SDCF_SHARED_STEP_EVIDENCE", tempfile.mkdtemp(prefix="shared-step-evidence-")))
        directory.mkdir(parents=True, exist_ok=True)
        observed = []
        for backend in ("SCI", "SCI_preprocessed", "SCI_ours", "SCI_Beaver_segmented"):
            out = directory / ("nonzero_hidden" if hidden_file else "default_hidden") / backend
            command = ["python3", str(HERE / "run.py"), "--binary", str(binary), "--fixture",
                       os.environ.get("SDCF_SHARED_STEP_FIXTURE", "/tmp/sci-preprocessed-fixture"),
                       "--output", str(out), "--backend", backend, "--steps", "1", "--batch", "128", "--queries", "2",
                       "--warmup", "0", "--network", "localhost", "--activation-workers", "1", "--graph-mode", "sequential",
                       "--server-cpus", "0,2,4,10", "--client-cpus", "6,8,12,16", "--cpus", "0,6,18",
                       "--memory-gib", "1.5", "--timeout", "600", "--diagnostic", "--cggm-threads", "4",
                       "--fuse-domains", "--pair-extensions", "--pack-bits", "--material-layout", "contiguous",
                       "--aes-online", "--segment-policy", "fixed-4",
                       "--generation-tile-keys", "0", "--generation-budget-bytes", "1610612736",
                       "--generation-reserve-bytes", "536870912", "--prefetch-bytes", "0", "--beaver-algebra"]
            if backend in ("SCI_ours", "SCI_Beaver_segmented"): command.append("--native-preprocessing")
            if hidden_file: command.extend(("--hidden", str(hidden_file)))
            child = subprocess.run(command, capture_output=True, text=True)
            self.assertEqual(child.returncode, 0, child.stdout + child.stderr)
            result = json.loads((out / "result.json").read_text())
            self.assertEqual(result["status"], "complete", result)
            tensors = [line for line in (out / "party2.log").read_text().splitlines() if line.startswith("TENSOR ")]
            self.assertEqual(len(tensors), 30)
            self.assertEqual(sum(len(line.split()) - 7 for line in tensors), 2 * 172928)
            observed.append(tensors)
            if backend != "SCI":
                self.assertTrue(result["config"]["native_preprocessing_effective"])
                parties = json.loads((out / "parties.json").read_text())
                self.assertTrue(all(party["native_preprocessing_verified"] and party["native_triples_verified"] for party in parties))
        for tensors in observed[1:]:
            self.assertEqual(observed[0], tensors)

    def test_default_hidden_preserves_all_fresh_query_integers(self):
        self.check_all_backends(None)

    def test_nonzero_hidden_preserves_all_fresh_query_integers(self):
        hidden_file = os.environ.get("SDCF_SHARED_STEP_HIDDEN_FILE",
            "/home/jiang/dealerless-lunwen/project/our/results/runtime_optimization_20261010/baseline/diagnostic/nonzero_hidden.i16")
        self.check_all_backends(hidden_file)


if __name__ == "__main__":
    unittest.main()
