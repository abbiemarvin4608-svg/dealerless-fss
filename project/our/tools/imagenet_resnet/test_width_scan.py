"""Focused tests for independent selection and exact overflow accounting."""
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import torch

import calibration
from model import IntegerResNet
from width_scan import frozen_json, select_candidate
from width_stats import RingOverflowStats, merge_counts


class WidthScanTests(unittest.TestCase):
    def test_selection_requires_all_legal_scales_and_breaks_ties_down(self):
        rows = [dict(bits=20, frac=s, samples=2000, complete=True, status="complete",
                     integer_top1_correct=1600 if s in (8, 12) else 1400,
                     integer_top1_pct=80.0 if s in (8, 12) else 70.0) for s in range(20)]
        self.assertEqual(select_candidate(rows, 20, 2000)["frac"], 8)
        with self.assertRaisesRegex(ValueError, "every legal"):
            select_candidate(rows[:-1], 20, 2000)
        rows[0]["samples"] = 1999
        with self.assertRaisesRegex(ValueError, "equal-sample"):
            select_candidate(rows, 20, 2000)

    def test_manifest_selection_rejects_validation_images_and_is_balanced(self):
        classes = ["n%08d" % i for i in range(1000)]
        mapping = {name: i for i, name in enumerate(classes)}
        train = SimpleNamespace(classes=classes, class_to_idx=mapping,
            samples=[("/train/%s/%s_%d.JPEG" % (name, name, k), i)
                     for i, name in enumerate(classes) for k in range(3)])
        validation = SimpleNamespace(classes=classes, class_to_idx=mapping,
                                     samples=[("/val/ILSVRC2012_val_00000001.JPEG", 0)])
        with mock.patch.object(calibration, "ImageFolder", side_effect=[train, validation, train, validation]):
            first = calibration.build_manifest("/train", "/val")
            second = calibration.build_manifest("/train", "/val")
        self.assertEqual(first, second)
        self.assertEqual(first["samples"], 2000)
        self.assertTrue(all(sum(e["label"] == i for e in first["entries"]) == 2 for i in range(1000)))
        train.samples.append(("/train/ILSVRC2012_val_00000001.JPEG", 0))
        with mock.patch.object(calibration, "ImageFolder", side_effect=[train, validation]), \
                self.assertRaisesRegex(ValueError, "validation image"):
            calibration.build_manifest("/train", "/val")

    def test_frozen_selection_cannot_be_replaced(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "selection.json"
            frozen_json(path, {"frac": 7})
            frozen_json(path, {"frac": 7})
            with self.assertRaisesRegex(ValueError, "frozen configuration differs"):
                frozen_json(path, {"frac": 8})
            self.assertEqual(json.loads(path.read_text()), {"frac": 7})

    def test_cpu_dot_count_uses_true_sum_not_residue_or_prefix(self):
        stats = RingOverflowStats(8)
        stats.observe_cpu_dot(torch.tensor([[100, 100], [100, -100]]), torch.tensor([[2], [2]]))
        counts = stats.snapshot()
        self.assertEqual(counts["dot_before_truncation"]["wrapped"], 1)
        self.assertEqual(counts["dot_before_truncation"]["total"], 2)
        # The individual products overflow, but the second complete dot is 0.
        large = RingOverflowStats(32)
        large.observe_cpu_dot(torch.tensor([[2**31 - 1] * 3]), torch.tensor([[2**31 - 1]] * 3))
        self.assertEqual(large.snapshot()["dot_before_truncation"]["wrapped"], 1)

    def test_combiner_diagnostic_is_wide_unscaled_and_does_not_change_output(self):
        calls = []
        def combine(dots, shifts, bias, ell, scale, registers, proof, ring, wrap):
            calls.append((scale, registers, ring, wrap))
            return torch.tensor([[9]]), torch.tensor([1 if not ring else 0, 0, 9, 9])
        stats = RingOverflowStats(8)
        output, _ = stats.wrap_combiner(combine)(torch.tensor([[[400]]], dtype=torch.int32),
            torch.tensor([0]), torch.tensor([], dtype=torch.int64), 8, 2, 8, 10, True, True)
        self.assertEqual(calls, [(0, 128, False, True), (2, 8, True, True)])
        self.assertEqual(output.item(), 9)
        self.assertEqual(stats.snapshot()["dot_before_truncation"]["wrapped"], 1)

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA unavailable")
    def test_cuda_exact_overflow_count_preserves_ring_outputs(self):
        from uniform_fixed_point import _native_ops
        combine = _native_ops().combine_limb_dots
        dots = torch.tensor([[[1073741824, 1073741824]],
                             [[-1073741824, 1073741824]]], dtype=torch.int32, device="cuda")
        shifts = torch.tensor([62, 62], device="cuda")
        bias = torch.tensor([], dtype=torch.int64, device="cuda")
        for bits in (20, 28, 32):
            expected, _ = combine(dots, shifts, bias, bits, 7, bits, 95, True, True)
            stats = RingOverflowStats(bits)
            actual, _ = stats.wrap_combiner(combine)(dots, shifts, bias, bits, 7, bits, 95, True, True)
            self.assertTrue(torch.equal(actual, expected))
            self.assertEqual(actual.cpu().tolist(), [[0, 0]])
            row = stats.snapshot()["dot_before_truncation"]
            self.assertEqual((row["wrapped"], row["total"]), (1, 2))

    def test_encoding_residual_pool_counts_and_merge(self):
        model = IntegerResNet.__new__(IntegerResNet)
        torch.nn.Module.__init__(model)
        model.bits = 8
        model.overflow = RingOverflowStats(8)
        model.overflow.observe_encoding("input_encoding", torch.tensor([-128.5, 127.5, 2.5]), 0)
        result = model.execute_node({"op": "add", "inputs": ["x", "y"]},
                                    {"x": torch.tensor([127]), "y": torch.tensor([1])})
        self.assertEqual(result.item(), -128)
        result = model.execute_node({"op": "avgpool", "inputs": ["x"]},
                                    {"x": torch.full((1, 1, 7, 7), 127, dtype=torch.int64)})
        self.assertEqual(result.item(), 1)
        counts = model.overflow.snapshot(reset=True)
        self.assertEqual(counts["input_encoding"]["wrapped"], 1)
        self.assertEqual(counts["residual_add"]["wrapped"], 1)
        self.assertEqual(counts["avgpool_sum_before_division"]["wrapped"], 1)
        self.assertEqual(model.overflow.snapshot(), {})
        merged = {}
        merge_counts(merged, counts)
        merge_counts(merged, counts)
        self.assertEqual(merged["input_encoding"]["total"], 6)
        self.assertEqual(merged["input_encoding"]["wrapped"], 2)


if __name__ == "__main__":
    unittest.main()
