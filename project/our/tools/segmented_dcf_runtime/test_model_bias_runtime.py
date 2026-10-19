"""Exact model bias preparation at the frozen single-step runner boundary."""
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

from runtime import call_arguments, dependency_schedule


class ModelBiasScheduleTests(unittest.TestCase):
    def test_prepared_model_removes_only_the_two_owned_bias_divisions(self):
        schema = [dict(operator="MatAdd_01", function="MatAdd", output="tmp26bw16",
                       source_call="MatAdd(1, 100, 1, 8, 1, 1, 16, 16, 16, 16, tmp21bw16, tmp25bw16, tmp26bw16);"),
                  dict(operator="MatAdd_02", function="MatAdd", output="tmp27bw16",
                       source_call="MatAdd(1, 100, 1, 8, 1, 1, 16, 16, 16, 16, tmp26bw16, Bg, tmp27bw16);"),
                  dict(operator="MatAdd_03", function="MatAdd", output="tmp29bw16",
                       source_call="MatAdd(1, 100, 1, 8, 1, 1, 16, 16, 16, 16, tmp26bw16, Bh, tmp29bw16);")]
        original = dependency_schedule(schema, 128, 1, classifier=False)
        prepared = dependency_schedule(schema, 128, 1, classifier=False,
                                       model_bias_precomputed=True)
        self.assertEqual([call_arguments(row)[3] for row in original], ["8", "8", "8"])
        self.assertEqual([call_arguments(row)[3] for row in prepared], ["8", "1", "1"])
        self.assertEqual([row["output_elements"] for row in prepared], [12800] * 3)
        self.assertEqual([row["inputs"] for row in prepared], [row["inputs"] for row in original])
        self.assertEqual([row["dependencies"] for row in prepared], [row["dependencies"] for row in original])
        self.assertEqual(call_arguments(schema[1])[3], "8")


@unittest.skipUnless(os.environ.get("SDCF_MODEL_BIAS_BINARY"), "set SDCF_MODEL_BIAS_BINARY")
class ModelBiasRunnerTests(unittest.TestCase):
    def test_runtime_exposes_common_model_preparation_permission(self):
        description = json.loads(subprocess.check_output(
            [os.environ["SDCF_MODEL_BIAS_BINARY"], "--describe"], text=True))
        self.assertTrue(description["supports_model_bias_precomputation"])

    def check_query_integers(self, hidden):
        directory = Path(os.environ.get("SDCF_MODEL_BIAS_EVIDENCE", tempfile.mkdtemp(prefix="model-bias-evidence-")))
        directory.mkdir(parents=True, exist_ok=True)
        all_tensors = []
        for backend in ("SCI_preprocessed", "SCI_Beaver_segmented"):
            results = []
            for prepared in (False, True):
                out = directory / ("nonzero" if hidden else "default") / backend / ("prepared" if prepared else "original")
                command = [sys.executable, str(Path(__file__).with_name("run.py")),
                    "--binary", os.environ["SDCF_MODEL_BIAS_BINARY"], "--fixture",
                    os.environ.get("SDCF_MODEL_BIAS_FIXTURE", "/tmp/sci-preprocessed-fixture"),
                    "--output", str(out), "--backend", backend, "--steps", "1", "--batch", "128",
                    "--queries", "2", "--warmup", "0", "--diagnostic",
                    "--network", "localhost", "--activation-workers", "2", "--graph-mode", "parallel",
                    "--cggm-threads", "2", "--server-cpus", "0,2,4,10", "--client-cpus", "6,8,12,16",
                    "--cpus", "0,6,18", "--memory-gib", "1.5", "--timeout", "600",
                    "--fuse-domains", "--pair-extensions", "--pack-bits", "--material-layout", "contiguous",
                    "--aes-online", "--segment-policy", "fixed-4", "--prefetch-bytes", "0", "--beaver-algebra"]
                if backend == "SCI_Beaver_segmented":
                    command.append("--native-preprocessing")
                if os.environ.get("SDCF_MODEL_BIAS_ROUND_TRACE") == "1":
                    command.append("--round-trace")
                if prepared:
                    command.append("--model-bias-precompute")
                if hidden:
                    command.extend(("--hidden", str(hidden)))
                child = subprocess.run(command, text=True, capture_output=True)
                self.assertEqual(child.returncode, 0, child.stdout + child.stderr + str(out))
                result = json.loads((out / "result.json").read_text())
                self.assertEqual(result["status"], "complete", result.get("reason"))
                tensors = [line for line in (out / "party2.log").read_text().splitlines() if line.startswith("TENSOR ")]
                self.assertEqual(len(tensors), 30)
                self.assertEqual(sum(len(line.split()) - 7 for line in tensors), 2 * 172928)
                all_tensors.append(tensors)
                results.append(result)
                self.assertEqual(result["config"]["model_bias_precomputed"], prepared)
                if prepared:
                    self.assertEqual(result["measurements"]["model_prepare"]["sent_bytes"], 0)
                    self.assertIn("model_bias_preparation", result["measurements"])
            for query in range(2):
                for phase in ("offline", "online"):
                    before, after = (r["measurements"]["queries"][query][phase] for r in results)
                    self.assertLess(after["sent_bytes"], before["sent_bytes"])
        self.assertTrue(all(tensors == all_tensors[0] for tensors in all_tensors))

    def test_default_hidden_matches_all_original_tensors_for_both_methods(self):
        self.check_query_integers(None)

    def test_nonzero_hidden_matches_all_original_tensors_for_both_methods(self):
        self.check_query_integers(Path(os.environ.get("SDCF_MODEL_BIAS_HIDDEN",
            "/home/jiang/dealerless-lunwen/project/our/results/runtime_optimization_20261010/baseline/diagnostic/nonzero_hidden.i16")))


if __name__ == "__main__":
    unittest.main()
