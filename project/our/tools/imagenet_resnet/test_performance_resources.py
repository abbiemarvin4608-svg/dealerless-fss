"""Resource preflight must reject infeasible native spools before any protocol."""

from argparse import Namespace
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import run_performance as runner
from summarize import extract_run


class NativeResourceTests(unittest.TestCase):
    def graph(self):
        return dict(model="resnet50", batch=1, bits=32, frac=12, input="input", output="fc",
                    tensors={"input": [1, 4], "fc": [1, 2]}, nodes=[dict(name="fc", op="linear",
                    inputs=["input"], output="fc", shape=[1, 2], weight_shape=[2, 4])])

    def identity(self):
        return dict(model="resnet50", batch=1, bits=32, frac=12, queries=1, diagnostic=False,
                    backend="SCI_preprocessed", tile=64, optimization_profile="optimized",
                    round_trace="off", spool_budget_bytes_per_cnn_or_beaver_store=0,
                    native_spool_budget_bytes=0)

    def disk(self, free):
        return dict(path=str(Path("/tmp").resolve()), free_bytes=free, total_bytes=100000, device=1)

    def test_both_party_spools_share_the_disk_limit(self):
        result = runner.native_resource_preflight(self.graph(), self.identity(), self.disk(5000))
        self.assertEqual(result["estimate"]["native_linear_spool_bytes_party1_lower_bound"], 4120)
        self.assertEqual(result["estimate"]["native_linear_spool_bytes_party2_lower_bound"], 2200)
        self.assertEqual(result["violations"], [dict(constraint="both_party_spools_exceed_available_disk",
                                                   required_bytes=6320, available_bytes=5000)])
        self.assertEqual(result["status"], "resource_unavailable")
        self.assertEqual(runner.native_resource_preflight(self.graph(), self.identity(), self.disk(6320))["status"],
                         "lower_bound_fits")

    def test_explicit_native_budget_is_per_party_and_overrides_inherited_budget(self):
        identity = dict(self.identity(), spool_budget_bytes_per_cnn_or_beaver_store=2500)
        result = runner.native_resource_preflight(self.graph(), identity, self.disk(100000))
        self.assertEqual(result["native_spool_budget_bytes_per_party"], 2500)
        self.assertEqual(result["violations"], [dict(constraint="party_spool_exceeds_explicit_budget",
                                                   party=1, required_bytes=4120, available_bytes=2500)])
        identity["native_spool_budget_bytes"] = 5000
        self.assertEqual(runner.native_resource_preflight(self.graph(), identity, self.disk(100000))["status"],
                         "lower_bound_fits")

    def test_capacity_uses_libc_tmp_directory_and_unprivileged_available_blocks(self):
        filesystem = SimpleNamespace(f_bavail=2, f_frsize=4096, f_blocks=100)
        with patch.dict("os.environ", {"TMPDIR": "/unused-python-temporary-directory"}), \
                patch.object(runner.os, "statvfs", return_value=filesystem) as observed:
            result = runner.native_resource_preflight(self.graph(), self.identity())
        observed.assert_called_once_with(Path("/tmp").resolve())
        self.assertEqual(result["disk"]["free_bytes"], 8192)

    def fixture(self, root):
        model = root / "models/resnet50_w32_f12"
        (model / "audit").mkdir(parents=True)
        runner.write_json(model / "graph.json", self.graph())
        (model / "parameters.bin").write_bytes(b"parameters")
        (model / "audit/input.bin").write_bytes(b"input")
        binary = root / "resnet_secure"
        binary.write_bytes(b"binary")
        (root / "libcnn_ours.so").write_bytes(b"bridge")
        directory = root / "performance/resnet50_SCI_preprocessed_cost_q1"
        identity = runner.expected_identity(binary, model, "SCI_preprocessed", "off", "optimized")
        preflight = runner.native_resource_preflight(self.graph(), identity, self.disk(5000))
        runner.record_resource_unavailable(directory, identity, preflight, {}, binary, model)
        return directory, identity, model, binary

    def test_resume_preserves_unavailable_evidence_without_analyzer_or_party_files(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory, identity, model, _ = self.fixture(Path(temporary))
            originals = {path.name: path.read_bytes() for path in directory.iterdir()}
            with patch.object(runner.os, "statvfs", side_effect=AssertionError("retain original capacity snapshot")):
                action = runner.qualify_existing(directory, identity, self.graph(), model / "audit",
                                                 lambda *_: self.fail("analyzer must not start"))
            self.assertEqual(action, "skipped_resource_unavailable")
            self.assertEqual(extract_run(directory)["status"], "resource_unavailable")
            self.assertEqual({path.name: path.read_bytes() for path in directory.iterdir()}, originals)
            with self.assertRaisesRegex(ValueError, "identity"):
                runner.qualify_existing(directory, dict(identity, native_spool_budget_bytes=1),
                                        self.graph(), model / "audit")

    def test_resume_rejects_started_protocol_or_unsupported_resource_claim(self):
        for defect in ("started", "unsupported"):
            with self.subTest(defect=defect), tempfile.TemporaryDirectory() as temporary:
                directory, identity, model, _ = self.fixture(Path(temporary))
                if defect == "started":
                    (directory / "party1.log").write_text("protocol started\n")
                else:
                    record = runner.read_json(directory / "RESOURCE_PREFLIGHT.json")
                    record["violations"] = []
                    runner.write_json(directory / "RESOURCE_PREFLIGHT.json", record)
                with self.assertRaises(ValueError):
                    runner.qualify_existing(directory, identity, self.graph(), model / "audit")

    def test_batch_continues_to_ours_after_unavailable_native_preprocessing(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            model = root / "models/resnet50_w32_f12"
            (model / "audit").mkdir(parents=True)
            runner.write_json(model / "graph.json", self.graph())
            (model / "parameters.bin").write_bytes(b"parameters")
            (model / "audit/input.bin").write_bytes(b"input")
            binary = root / "resnet_secure"
            binary.write_bytes(b"binary")
            (root / "libcnn_ours.so").write_bytes(b"bridge")
            args = Namespace(results=root, binary=binary, models_root=None, resume=False, port=42000,
                             round_trace="off", optimization_profile="optimized", refresh_guarded_baseline=False,
                             spool_budget=0, native_spool_budget=2500)
            rows = [("resnet50", "SCI_preprocessed", 1, "native"), ("resnet50", "SCI_ours_beaver", 1, "ours")]
            commands = []

            def subprocess_run(command):
                commands.append(command)
                if "--backend" in command:
                    backend = command[command.index("--backend") + 1]
                    self.assertEqual(backend, "SCI_ours_beaver")
                    destination = Path(command[command.index("--output") + 1])
                    destination.mkdir()
                    record = dict(runner.expected_identity(binary, model, backend, "off", "optimized", 0, 2500),
                                  status="complete")
                    for name in ("PLAN.json", "STATUS.json"):
                        runner.write_json(destination / name, record)
                return SimpleNamespace(returncode=0)

            with patch.object(runner, "require_memory_guard", return_value={}), \
                    patch.object(runner, "guarded_baseline_rows", return_value=rows), \
                    patch.object(runner.subprocess, "run", side_effect=subprocess_run), \
                    patch.object(runner, "validate_existing"), patch.object(runner, "extract_run"), \
                    patch("builtins.print"):
                self.assertEqual(runner.run(args), 0)
            batch = runner.read_json(root / "performance/COST_BATCH_STATUS.json")
            self.assertEqual(batch["status"], "complete_with_resource_failures")
            self.assertEqual([row["status"] for row in batch["rows"]], ["resource_unavailable", "qualified"])
            self.assertEqual(len(commands), 2)
            native = root / "performance/native"
            self.assertFalse(list(native.glob("party*")))
            self.assertFalse(runner.read_json(native / "STATUS.json")["protocol_started"])
            self.assertEqual(commands[0][commands[0].index("--native-spool-budget") + 1], "2500")


if __name__ == "__main__":
    unittest.main()
