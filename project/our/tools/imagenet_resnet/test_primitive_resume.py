"""Recovery must preserve completed evidence and the original private inputs."""

import json
import os
from pathlib import Path
import tempfile
import time
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import resume_primitive_ablation as recovery


class PrimitiveResumeTests(unittest.TestCase):
    def fixture(self, output):
        root = output / "missing_width_runs"
        resumed = root / "resume_wan_n8"
        resumed.mkdir(parents=True)
        original = root / "runs/low_latency_wan_model/ours_w28_n8/repeat1/party1.log"
        original.parent.mkdir(parents=True)
        original.write_text("interrupted original protocol\n")
        first = root / "runs/lan_model/sci_w28_n1/warmup/result.json"
        first.parent.mkdir(parents=True)
        first.write_text("{}\n")
        future = time.time() + 100
        os.utime(first, (future, future))
        shares = output / "private"
        shares.mkdir(mode=0o700)
        for role in (1, 2):
            path = shares / ("party%d.bin" % role)
            path.write_bytes(bytes([role]) * 1024)
            path.chmod(0o600)
        identity = recovery.input_identity(shares, 0, future)
        plan = dict(source_files=[recovery.file_identity(original)], input_identity=identity,
                    recovered_input_directory=str(shares))
        recovery.write_json(resumed / "plan.json", plan)
        config = dict(build=str(output / "build"), timeout_seconds=60,
                      address_space_limit_bytes_per_party=768 << 20,
                      networks=[dict(name=recovery.NETWORK, rtt_ms=2, mbps=100)])
        original_plan = dict(resource_guard=dict(started_epoch=0))
        return root, config, original_plan, original, shares

    def execute(self, output, fixture, run_point):
        root, config, plan, _, _ = fixture
        with patch("run_performance.require_memory_guard", return_value={}), \
                patch.object(recovery, "original_state", return_value=(root, config, plan)), \
                patch.object(recovery, "original_records", return_value=[]), \
                patch.object(recovery, "load_runtime", return_value=SimpleNamespace(run_point=run_point)):
            recovery.execute_resume(output)

    def test_resume_runs_only_missing_repeats_and_keeps_original_attempt(self):
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary)
            fixture = self.fixture(output)
            root, config, _, original, shares = fixture
            before = original.read_bytes()
            calls = []

            def run_point(build, destination, backend, bits, segments, network, inputs, count, mode, timeout):
                calls.append((destination.name, backend, bits, segments, inputs, count, mode))
                destination.mkdir(parents=True)
                return dict(status="complete", backend=backend, input_bits=bits, nseg=segments,
                            network=network, total_ms=1.)

            self.execute(output, fixture, run_point)
            self.assertEqual([call[0] for call in calls], recovery.TAGS)
            self.assertTrue(all(call[1:4] == ("ours", 28, 8) and call[5:] == (128, "perf") for call in calls))
            self.assertTrue(all(call[4] == [shares / "party1.bin", shares / "party2.bin"] for call in calls))
            self.assertEqual(original.read_bytes(), before)
            summary = json.loads((root / "summary.json").read_text())
            self.assertEqual(summary["status"], "complete_with_resource_failures")
            self.assertEqual(summary["rows"][0]["completed"], 5)

    def test_changed_input_or_evidence_blocks_launch(self):
        for changed in ("input", "evidence"):
            with self.subTest(changed=changed), tempfile.TemporaryDirectory() as temporary:
                output = Path(temporary)
                fixture = self.fixture(output)
                root, _, _, original, shares = fixture
                path = shares / "party1.bin" if changed == "input" else original
                path.write_bytes(b"x" * path.stat().st_size)
                with self.assertRaisesRegex(ValueError, "identity changed|evidence changed"):
                    self.execute(output, fixture, lambda *_: self.fail("protocol must not start"))
                self.assertFalse((root / "resume_wan_n8/STATUS.json").exists())

    def test_existing_resume_status_blocks_overwrite(self):
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary)
            fixture = self.fixture(output)
            status = fixture[0] / "resume_wan_n8/STATUS.json"
            status.write_text('{"status":"running"}\n')
            with self.assertRaisesRegex(FileExistsError, "already started"):
                self.execute(output, fixture, lambda *_: self.fail("protocol must not start"))
            self.assertEqual(status.read_text(), '{"status":"running"}\n')


if __name__ == "__main__":
    unittest.main()
