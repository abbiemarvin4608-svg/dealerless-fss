"""Frozen benchmark contracts through the existing plan-only runner CLI."""
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest


@unittest.skipUnless(os.environ.get("SDCF_BASELINE_DIR"), "set SDCF_BASELINE_DIR for frozen plan integration")
class FrozenBaselineTests(unittest.TestCase):
    def test_baseline_freezes_fresh_queries_and_matched_resources(self):
        baseline = Path(os.environ["SDCF_BASELINE_DIR"])
        fixture = Path(os.environ["SDCF_BASELINE_FIXTURE"])
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "plan"
            child = subprocess.run([
                sys.executable, str(Path(__file__).with_name("benchmark_pool.py")),
                "--profile", "runtime-opt-01", "--phase", "after",
                "--new-binary", str(baseline / "primitive_cost_sweep"),
                "--runtime-binary", str(baseline / "sdcf-runtime"),
                "--fixture", str(fixture), "--output", str(output), "--plan-only"],
                text=True, capture_output=True)
            self.assertEqual(child.returncode, 0, child.stderr)
            manifest = json.loads((output / "manifest.json").read_text())
            rows = manifest["runtime_rows"]
            self.assertEqual(len(rows), 6)
            self.assertEqual({row["backend"] for row in rows},
                             {"SCI", "SCI_ours", "SCI_Beaver", "SCI_Beaver_segmented"})
            self.assertTrue(all(row["queries"] == 10 and row["max_segments"] == 4 and
                                row["timeout_seconds"] == 600 for row in rows))
            self.assertEqual(manifest["party_cpu_affinities"], [[0, 2, 4, 10], [6, 8, 12, 16]])
            self.assertEqual(manifest["memory_limit_bytes_per_party"], 1610612736)
            self.assertEqual(manifest["warmup_queries"], 0)
            primitives = manifest["primitive_rows"]
            self.assertEqual({row["runs"] for row in primitives if row["kind"] == "dcf" and row["count"] == 128}, {30})
            self.assertEqual({row["runs"] for row in primitives if row["kind"] == "dcf" and row["count"] == 12800}, {10})
            self.assertEqual({row["runs"] for row in primitives if row["kind"] == "conversion"}, {5})
            self.assertEqual(len(manifest["diagnostic_commands"]), 8)
            self.assertFalse((output / "summary.json").exists())


if __name__ == "__main__":
    unittest.main()
