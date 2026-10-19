"""Graph isolation and branch scheduling at the runtime boundary."""
import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


@unittest.skipUnless(os.environ.get("SDCF_TEST_BINARY"), "set SDCF_TEST_BINARY")
class GraphWorkerContract(unittest.TestCase):
    def test_runtime_exposes_all_graph_modes_and_fixed_branch_quota(self):
        description = json.loads(subprocess.check_output(
            [os.environ["SDCF_TEST_BINARY"], "--describe"], text=True))
        self.assertEqual(description["graph_modes"], ["sequential", "isolated", "parallel"])
        self.assertEqual(description["graph_openings"], ["independent", "batched"])
        self.assertEqual(description["graph_branches"], 2)
        self.assertEqual(description["branch_compute_limit"], 2)
        self.assertTrue(description["observed_ot_teams"])

    @unittest.skipUnless(os.environ.get("SDCF_TEST_FIXTURE"), "set SDCF_TEST_FIXTURE")
    def test_repeated_queries_preserve_all_tensors_for_three_graph_modes(self):
        with tempfile.TemporaryDirectory() as directory:
            tensors = []
            for mode, openings in ((mode, openings) for mode in ("sequential", "isolated", "parallel")
                                   for openings in ("independent", "batched")):
                output = Path(directory) / (mode + "-" + openings)
                command = [sys.executable, str(Path(__file__).with_name("run.py")),
                    "--binary", os.environ["SDCF_TEST_BINARY"], "--fixture", os.environ["SDCF_TEST_FIXTURE"],
                    "--output", str(output), "--backend", os.environ.get("SDCF_TEST_BACKEND", "SCI"),
                    "--steps", "1", "--batch", "5", "--queries", "2", "--diagnostic",
                    "--activation-workers", "2", "--cggm-threads", "2", "--graph-mode", mode,
                    "--graph-openings", openings, "--material-layout", "contiguous", "--prefetch-bytes", "8388609",
                    "--server-cpus", "0,2,4,10", "--client-cpus", "6,8,12,16", "--timeout", "120"]
                child = subprocess.run(command, text=True, capture_output=True)
                self.assertEqual(child.returncode, 0, child.stdout + child.stderr + str(output))
                result = json.loads((output / "result.json").read_text())
                self.assertEqual(result["status"], "complete")
                lines = (output / "party2.log").read_text().splitlines()
                tensors.append([line for line in lines if line.startswith("TENSOR ")])
                self.assertEqual(len(tensors[-1]), 30)
                tasks = [json.loads(line.split(" ",1)[1]) for line in lines if line.startswith("GRAPH_TASK ")]
                self.assertEqual(len(tasks), 30)
                self.assertEqual(len({(row["query"],row["index"]) for row in tasks}), 30)
                self.assertTrue(all(row["observed_ot_participants"] <= row["branch_compute_limit"] for row in tasks))
                parties = json.loads((output / "parties.json").read_text())
                budgets = parties[0]["conversion_prefetch_budgets"]
                if budgets:
                    self.assertEqual(sum(row["context_budget_bytes"] for row in budgets), 8388609)
            self.assertTrue(all(rows == tensors[0] for rows in tensors))


if __name__ == "__main__":
    unittest.main()
