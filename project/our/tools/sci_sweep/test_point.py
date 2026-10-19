"""Regression tests for the ticket10 runner's public seams."""

import importlib.util
import json
import os
import struct
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("sweep_runtime", HERE / "run.py")
runtime = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runtime)
formal_spec = importlib.util.spec_from_file_location("sweep_formal", HERE / "formal.py")
formal = importlib.util.module_from_spec(formal_spec)
formal_spec.loader.exec_module(formal)


class PointCLI(unittest.TestCase):
    def test_invalid_point_exits_without_network(self):
        binary = Path(os.environ.get(
            "SWEEP_BINARY",
            "/tmp/dealerless-tickets-04-12-20261008/ticket10-build/SCI_MSB_SWEEP"))
        result = subprocess.run(
            [str(binary), "1", "18000", "sci", "1", "1", "100", "perf"],
            capture_output=True, timeout=3)
        self.assertEqual(result.returncode, 2)

    def test_private_files_use_full_ring_and_fresh_randomness(self):
        # The sequence makes the ring arithmetic independently checkable and
        # verifies both generated values are requested as 32-bit quantities.
        with tempfile.TemporaryDirectory() as tmp:
            with mock.patch.object(runtime.secrets, "randbits",
                                   side_effect=[0xF0000001, 0x12345678,
                                                0x00000002, 0xFFFFFFFF]) as rb:
                paths = runtime.private_files(Path(tmp), count=2)
            self.assertEqual([call.args for call in rb.call_args_list],
                             [(32,), (32,), (32,), (32,)])
            left = struct.unpack("<QQ", paths[0].read_bytes())
            right = struct.unpack("<QQ", paths[1].read_bytes())
            self.assertEqual((left[0] + right[0]) & 0xFFFFFFFF, 0xF0000001)
            self.assertEqual((left[1] + right[1]) & 0xFFFFFFFF, 0x00000002)

    def test_aggregate_excludes_warmup_and_keeps_failed_repeat(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            point = out / "runs" / "lan_model" / "ours_w16_n4"
            base = {
                "backend": "ours", "input_bits": 16,
                "comparison_bits": 15, "nseg": 4, "count": 100,
                "network": {"name": "lan_model"},
            }
            for tag, status, ms in [
                    ("warmup", "complete", 999),
                    ("repeat1", "complete", 11),
                    ("repeat2", "failed", None),
                    ("repeat3", "complete", 13)]:
                path = point / tag
                path.mkdir(parents=True)
                row = dict(base, status=status, reason=None, total_ms=ms,
                           setup_ms=ms, offline_ms=ms, online_ms=1,
                           link_bytes=10)
                (path / "result.json").write_text(json.dumps(row))
            summaries = runtime.aggregate(out)
            self.assertEqual(len(summaries), 1)
            row = summaries[0]
            self.assertEqual(row["expected_repeats"], 3)
            self.assertEqual(row["completed"], 2)
            self.assertEqual(row["failed"], 1)
            self.assertEqual(row["total_ms"]["median"], 12)
            self.assertEqual(row["statuses"], ["complete", "failed"])

    def test_complete_profile_has_all_legal_width_segment_points(self):
        cfg = {"widths": [2, 9, 16, 20, 24, 25, 32],
               "nseg": [1, 2, 4, 8]}
        points = formal.points_for("complete", cfg)
        self.assertEqual(len(points), 2 + 6 * 5)
        ours = {(width, nseg) for width, backend, nseg, *_ in points
                if backend == "ours"}
        self.assertEqual(ours, {(2, 1), (9, 1), (9, 2), (9, 4), (9, 8),
                                (16, 1), (16, 2), (16, 4), (16, 8),
                                (20, 1), (20, 2), (20, 4), (20, 8),
                                (24, 1), (24, 2), (24, 4), (24, 8),
                                (25, 1), (25, 2), (25, 4), (25, 8),
                                (32, 1), (32, 2), (32, 4), (32, 8)})

    def test_boundary_profile_checks_every_legal_wide_domain_segment_point(self):
        cfg = {"widths": [2, 9, 16, 20, 24, 25, 32],
               "nseg": [1, 2, 4, 8]}
        points = formal.points_for("boundary", cfg)
        self.assertEqual(len(points), 32)
        for width in [9, 16, 20, 24, 25, 32]:
            self.assertEqual(
                [(backend, nseg, mode, repeats)
                 for w, backend, nseg, mode, repeats in points if w == width],
                [("sci", 1, "check", 0), ("ours", 1, "check", 0),
                 ("ours", 2, "check", 0), ("ours", 4, "check", 0),
                 ("ours", 8, "check", 0)])
        self.assertEqual([point for point in points if point[0] == 2],
                         [(2, "sci", 1, "check", 0),
                          (2, "ours", 1, "check", 0)])

    def test_boundary_cli_rejects_count_that_omits_final_sign_boundary(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)
            config = path / "config.json"
            config.write_text(json.dumps({
                "widths": [32], "nseg": [4], "batch": 100,
                "build": str(path / "absent-build"), "networks": [],
                "timeout_seconds": 60,
            }))
            result = subprocess.run(
                [sys.executable, str(HERE / "formal.py"), "--config",
                 str(config), "--output", str(path / "output"),
                 "--profile", "boundary"], capture_output=True, text=True,
                timeout=3)
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("boundary checks require count >= 101", result.stderr)

    def test_boundary_acceptance_keeps_resource_failure_unverified(self):
        rows = [
            {"backend": "sci", "input_bits": 32, "comparison_bits": 31,
             "nseg": None, "count": 128, "status": "complete", "mode": "check",
             "completed_calls": 1, "opened_outputs": 256, "mismatches": 0},
            {"backend": "ours", "input_bits": 32, "comparison_bits": 31,
             "nseg": 1, "count": 128, "status": "failed", "mode": "check",
             "completed_calls": 0, "reason": "party_nonzero_exit:std::bad_alloc"},
        ]
        summary = formal.boundary_acceptance(rows)
        self.assertEqual(summary["requested_points"], 2)
        self.assertEqual(summary["verified_points"], 1)
        self.assertEqual(summary["verified_scalar_instances"], 128)
        self.assertEqual(summary["unverified_points"], 1)
        self.assertFalse(summary["all_requested_points_verified"])
        self.assertFalse(summary["points"][1]["verified"])
        self.assertIsNone(summary["points"][1]["mismatches"])

    def test_boundary_acceptance_retains_failed_larger_batch_after_valid_retry(self):
        common = {"backend": "ours", "input_bits": 20, "comparison_bits": 19,
                  "nseg": 1, "mode": "check"}
        rows = [dict(common, count=128, status="failed", completed_calls=0,
                     reason="party_nonzero_exit:std::bad_alloc"),
                dict(common, count=100, status="complete", completed_calls=1,
                     opened_outputs=200, mismatches=0)]
        summary = formal.boundary_acceptance(rows)
        self.assertEqual(summary["requested_points"], 1)
        self.assertEqual(summary["attempted_checks"], 2)
        self.assertEqual(summary["verified_points"], 1)
        self.assertEqual(summary["verified_scalar_instances"], 100)
        self.assertEqual(summary["points"][0]["attempts"][0]["status"], "failed")


if __name__ == "__main__":
    unittest.main()
