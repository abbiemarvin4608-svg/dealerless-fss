import json
import contextlib
import io
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

from summarize_final import BACKENDS, MODELS, WRAP_SCOPES, accuracy_rows, performance_rows, primitive_rows, run


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


def resource_status(model="resnet50"):
    return dict(model=model, backend="SCI_preprocessed", bits=32, frac=12,
        batch=1, queries=1, status="resource_unavailable", diagnostic=False,
        protocol_started=False, evidence_scope="public_shape_resource_preflight_not_measurement",
        reason="unit-test disk preflight exceeds available bytes",
        resource_preflight=dict(status="resource_unavailable", estimate={"unit_test": True},
            disk={"path": "/tmp", "free_bytes": 700 * (1 << 30)},
            violations=[dict(constraint="filesystem_free_bytes", required_bytes=3150000000000,
                             available_bytes=700 * (1 << 30))]))


class FinalSummaryTests(unittest.TestCase):
    def test_incomplete_accuracy_is_not_reported_as_formal(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            save(root / "scan/resnet50/selected.json", {"rows": [{"bits": 20, "frac": 8, "calibration_top1_pct": 80.0}]})
            save(root / "scan/resnet50/validation/w20_f8/accuracy.json", dict(
                model="resnet50", bits=20, frac=8, status="running", evidence_scope="diagnostic",
                samples=32, fixed={"top1_pct": 99.0, "top5_pct": 100.0}))
            save(root / "scan/resnet50/calibration/w20_f0/metrics.json", {"complete": True, "status": "complete"})
            save(root / "scan/resnet50/calibration/w20_f1/metrics.json", {"complete": False, "status": "pruned"})
            rows = accuracy_rows(root / "scan", root / "anchors")
            row = next(row for row in rows if row["model"] == "resnet50" and row["bits"] == 20)
            self.assertEqual((row["status"], row["samples"]), ("running", 32))
            self.assertIsNone(row["fixed_top1_pct"])
            self.assertEqual((row["completed_calibration_candidates"], row["pruned_calibration_candidates"],
                              row["resolved_calibration_candidates"]), (1, 1, 2))
            self.assertEqual(len(rows), 10)

    def test_anchor_missing_overflow_is_not_zero(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            save(root / "anchors/resnet50_w32_f12/accuracy.json", dict(
                model="resnet50", bits=32, frac=12, status="complete", evidence_scope="formal", samples=50000,
                fp32={"top1_pct": 76.146, "top5_pct": 92.872}, fixed={"top1_pct": 76.128, "top5_pct": 92.878},
                top1_agreement_pct=99.61))
            row = next(row for row in accuracy_rows(root / "scan", root / "anchors")
                       if row["model"] == "resnet50" and row["bits"] == 32)
            self.assertEqual(row["status"], "reused_complete")
            self.assertEqual(row["overflow_status"], "missing_historical_run")
            self.assertIsNone(row["dot_before_truncation_wrapped"])

    def test_missing_primitive_points_are_pending_not_failures(self):
        with tempfile.TemporaryDirectory() as directory:
            reused, new, status = primitive_rows(Path(directory))
            self.assertEqual((len(reused), len(new), status), (0, 10, "pending"))
            self.assertTrue(all(row["status"] == "pending" and row["completed"] is None for row in new))

    def test_cnn_uses_extractor_and_keeps_failed_validation_empty(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            save(root / "r50/STATUS.json", dict(model="resnet50", backend="SCI", bits=32, frac=12,
                                                batch=1, queries=1, status="complete", diagnostic=False))
            save(root / "diagnostic/STATUS.json", dict(model="resnet152", backend="SCI", diagnostic=True))
            with mock.patch("summarize_final.extract_run", side_effect=ValueError("output audit incomplete")) as extract:
                rows, phases, details = performance_rows(root)
            self.assertEqual(extract.call_count, 1)
            self.assertEqual(len(rows), 6)
            row = next(row for row in rows if row["model"] == "resnet50" and row["backend"] == "SCI")
            self.assertEqual(row["status"], "invalid_or_incomplete")
            self.assertIsNone(row["total_sent_bytes"])
            self.assertEqual((phases, details), ([], []))

    def test_resource_preflight_is_preserved_without_measured_costs(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            evidence = resource_status()
            save(root / "r50/STATUS.json", evidence)
            with mock.patch("summarize_final.extract_run") as extract:
                rows, phases, details = performance_rows(root)
            extract.assert_not_called()
            row = next(row for row in rows if row["backend"] == "SCI_preprocessed" and row["model"] == "resnet50")
            self.assertEqual(row["status"], "resource_unavailable")
            self.assertEqual(row["resource_preflight"], evidence["resource_preflight"])
            self.assertIs(row["protocol_started"], False)
            for metric in ("total_wall_seconds", "total_sent_bytes", "total_process_cpu_seconds", "output_values_checked"):
                self.assertIsNone(row[metric])
            self.assertEqual((phases, details), ([], []))

    def test_unavailable_without_explicit_preflight_is_invalid(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            evidence = resource_status()
            del evidence["resource_preflight"]
            save(root / "r50/STATUS.json", evidence)
            with mock.patch("summarize_final.extract_run") as extract:
                rows, _, _ = performance_rows(root)
            extract.assert_not_called()
            row = next(row for row in rows if row["source"] is not None)
            self.assertEqual(row["status"], "invalid_or_incomplete")

    def test_final_resource_completion_keeps_measured_count_separate(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            args = SimpleNamespace(root=root, output_dir=None, width_scan_dir=None,
                                   anchor_dir=root / "anchors", performance_dir=None)
            accuracy = accuracy_rows(root / "scan", root / "anchors")
            for row in accuracy:
                row.update(status="complete", samples=50000)
                for scope in WRAP_SCOPES:
                    row[scope + "_total"] = 1
            save(root / "experiment2/analysis.json", {"status": "complete_existing_record_analysis", "domains": []})
            for model in MODELS:
                for backend in BACKENDS:
                    evidence = resource_status(model) if backend == "SCI_preprocessed" else dict(
                        model=model, backend=backend, bits=32, frac=12, batch=1,
                        queries=1, status="complete", diagnostic=False)
                    save(root / "performance" / (model + "_" + backend) / "STATUS.json", evidence)

            def measured(path):
                return dict(bits=32, frac=12, status="complete", samples=[dict(query=0, phase="online",
                    wall_seconds=1.0, sent_bytes=2, cpu_seconds_total=3.0)],
                    statistics={"online": {"wall_seconds": {"median": 1.0}}}, output_audit={"values_checked": 1000})

            with mock.patch("summarize_final.accuracy_rows", return_value=accuracy), \
                    mock.patch("summarize_final.primitive_rows", return_value=([{"unit_test": True}], [], "complete")), \
                    mock.patch("summarize_final.extract_run", side_effect=measured) as extract, \
                    contextlib.redirect_stdout(io.StringIO()):
                state = run(args)
            self.assertEqual(extract.call_count, 4)
            self.assertEqual(state["status"], "complete_with_resource_failures")
            result = state["experiment3"]
            self.assertEqual((result["completed_cnn"], result["resource_unavailable_cnn"], result["resolved_cnn"]), (4, 2, 6))
            self.assertEqual(len(result["qualified_run_details"]), 4)
            report = (root / "results/REPORT.zh.md").read_text()
            self.assertIn("4/6", report)
            self.assertIn("SCI_preprocessed | resource_unavailable | N/A | N/A", report)

            def incomplete(path):
                result = measured(path)
                if path.name == "resnet152_SCI_ours_beaver":
                    result["status"] = "failed"
                return result

            with mock.patch("summarize_final.accuracy_rows", return_value=accuracy), \
                    mock.patch("summarize_final.primitive_rows", return_value=([{"unit_test": True}], [], "complete")), \
                    mock.patch("summarize_final.extract_run", side_effect=incomplete), \
                    contextlib.redirect_stdout(io.StringIO()):
                state = run(args)
            self.assertEqual(state["status"], "pending")
            result = state["experiment3"]
            self.assertEqual((result["completed_cnn"], result["resource_unavailable_cnn"], result["resolved_cnn"]), (3, 2, 5))


if __name__ == "__main__":
    unittest.main()
