"""Exact calibration pruning, evidence validation, and checkpoint recovery."""
import argparse
import contextlib
import copy
import io
import json
import random
import signal
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import torch

import calibration
import evaluate as evaluator
from evaluate import PRUNING_RULE, pruning_certificate
from width_scan import incumbent_record, select_candidate


def candidate(frac, outcomes, incumbent, bits=4, batch=2):
    for processed in range(batch, len(outcomes) + 1, batch):
        correct = sum(outcomes[:processed])
        proof = pruning_certificate(processed, correct, len(outcomes), incumbent)
        if proof is not None:
            return dict(bits=bits, frac=frac, samples=processed, complete=False, status="pruned",
                requested_samples=len(outcomes), integer_top1_correct=correct, pruning_certificate=proof,
                calibration_pruning=dict(rule=PRUNING_RULE, batch_size=batch, incumbent=incumbent))
    return dict(bits=bits, frac=frac, samples=len(outcomes), complete=True, status="complete",
                integer_top1_correct=sum(outcomes), integer_top1_pct=100.0 * sum(outcomes) / len(outcomes))


class PruningTests(unittest.TestCase):
    def test_stop_during_batch_commits_once_and_resumes_without_duplicate_counts(self):
        class Dataset:
            classes, class_to_idx = ["a", "b"], {"a": 0, "b": 1}
            samples = [("test_image_%d" % i, 1) for i in range(6)]
            def __len__(self):
                return 6
            def __getitem__(self, index):
                return torch.tensor([index], dtype=torch.float32), 1

        class Reference(torch.nn.Module):
            _weights_identity = {"source": "unit_test"}
            def forward(self, inputs):
                output = torch.zeros(len(inputs), 10)
                output[:, 1] = 2
                return output

        class Integer(Reference):
            seen = []
            def __init__(self, *args, **kwargs):
                super().__init__()
            def forward(self, inputs):
                indices = inputs.flatten().int().tolist()
                self.seen.extend(indices)
                if indices[0] == 2:
                    evaluator.request_stop(signal.SIGTERM, None)
                    evaluator.request_stop(signal.SIGTERM, None)
                output = torch.zeros(len(inputs), 10, dtype=torch.int64)
                for row, index in enumerate(indices):
                    output[row, 1 if index % 2 == 0 else 0] = 2
                return output

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest = root / "manifest.json"
            manifest.write_text("{}")
            args = argparse.Namespace(device="cpu", model="resnet50", weights_path=None,
                val_dir=root / "val", limit=None, bits=16, frac=0, output_dir=root / "run",
                batch_size=2, workers=0, checkpoint_every=64, restart=False, calibration_manifest=manifest)
            with mock.patch.object(calibration, "ManifestDataset", return_value=Dataset()), \
                    mock.patch.object(evaluator, "load_reference", side_effect=lambda *a: (Reference(), None)), \
                    mock.patch.object(evaluator, "IntegerResNet", Integer), \
                    mock.patch.object(evaluator, "_STOP_REQUESTED", False), contextlib.redirect_stdout(io.StringIO()):
                with self.assertRaisesRegex(SystemExit, "stop requested"):
                    evaluator.evaluate(args)
                checkpoint = torch.load(args.output_dir / "checkpoint.pt", weights_only=False)
                self.assertEqual((checkpoint["status"], checkpoint["processed"]), ("interrupted", 4))
                self.assertEqual((checkpoint["fp32_top1_correct"], checkpoint["integer_top1_correct"],
                                  checkpoint["top1_agree"], checkpoint["logit_values"]), (4, 2, 2, 40))
                self.assertEqual(checkpoint["fp32_predictions"], [1, 1, 1, 1])
                self.assertEqual(checkpoint["integer_predictions"], [1, 0, 1, 0])
                self.assertEqual(Integer.seen, [0, 1, 2, 3])
                for name in ("metrics.json", "accuracy.json"):
                    saved = json.loads((args.output_dir / name).read_text())
                    self.assertEqual((saved["status"], saved["samples"]), ("interrupted", 4))
                # A new worker starts with no pending stop request.
                with mock.patch.object(evaluator, "_STOP_REQUESTED", False):
                    result = evaluator.evaluate(args)
                checkpoint = torch.load(args.output_dir / "checkpoint.pt", weights_only=False)
                self.assertEqual((result["status"], result["samples"], result["complete"]), ("complete", 6, True))
                self.assertEqual(Integer.seen, [0, 1, 2, 3, 4, 5])
                self.assertEqual((checkpoint["fp32_top1_correct"], checkpoint["integer_top1_correct"],
                                  checkpoint["top1_agree"], checkpoint["logit_values"]), (6, 3, 3, 60))
                self.assertEqual(checkpoint["fp32_predictions"], [1] * 6)
                self.assertEqual(checkpoint["integer_predictions"], [1, 0, 1, 0, 1, 0])

    def test_pruned_grid_selects_same_winner_as_full_grid_including_ties(self):
        rng = random.Random(41)
        cases = [[[1] * 12] * 4, [[0] * 12] * 4]
        cases += [[[rng.randrange(2) for _ in range(12)] for _ in range(4)] for _ in range(100)]
        pruned = 0
        for outcomes in cases:
            rows, best = [], None
            for frac, values in enumerate(outcomes):
                row = candidate(frac, values, incumbent_record(best))
                rows.append(row)
                pruned += row["status"] == "pruned"
                if row["complete"] and (best is None or row["integer_top1_correct"] > best["integer_top1_correct"]):
                    best = row
            expected = max(range(4), key=lambda frac: (sum(outcomes[frac]), -frac))
            chosen = select_candidate(rows, 4, 12)
            self.assertEqual(chosen["frac"], expected)
            self.assertEqual((chosen["samples"], chosen["complete"]), (12, True))
        self.assertGreater(pruned, 0)

    def test_selection_rejects_invalid_or_missing_dominance_evidence(self):
        first = candidate(0, [1] * 12, None)
        rows = [first] + [candidate(frac, [0] * 12, incumbent_record(first)) for frac in range(1, 4)]
        self.assertEqual(select_candidate(rows, 4, 12)["frac"], 0)
        for field in ("processed", "correct", "remaining", "upper_bound"):
            altered = copy.deepcopy(rows)
            altered[1]["pruning_certificate"][field] += 1
            with self.assertRaisesRegex(ValueError, "valid pruning certificate"):
                select_candidate(altered, 4, 12)
        for change in ({"status": "running"}, {"complete": True}, {"integer_top1_correct": 2},
                       {"pruning_certificate": None}, {"calibration_pruning": {}}):
            altered = copy.deepcopy(rows)
            altered[1].update(change)
            with self.assertRaises(ValueError):
                select_candidate(altered, 4, 12)

    def test_evaluator_prunes_after_resumed_batch_and_restores_terminal_artifacts(self):
        class Dataset:
            classes, class_to_idx = ["a", "b"], {"a": 0, "b": 1}
            samples = [("test_image_%d" % i, 1) for i in range(8)]
            def __len__(self):
                return 8
            def __getitem__(self, index):
                return torch.tensor([index], dtype=torch.float32), 1

        class Reference(torch.nn.Module):
            _weights_identity = {"source": "unit_test"}
            def forward(self, inputs):
                output = torch.zeros(len(inputs), 10)
                output[:, 1] = 2
                return output

        class Integer(Reference):
            fail, seen = True, []
            def __init__(self, *args, **kwargs):
                super().__init__()
            def forward(self, inputs):
                indices = inputs.flatten().int().tolist()
                self.seen.append(indices)
                if self.fail and indices[0] == 2:
                    raise RuntimeError("test interruption")
                output = torch.zeros(len(inputs), 10, dtype=torch.int64)
                for row, index in enumerate(indices):
                    output[row, 0 if index < 4 else 1] = 2
                return output

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest = root / "manifest.json"
            manifest.write_text("{}")
            args = argparse.Namespace(device="cpu", model="resnet50", weights_path=None,
                val_dir=root / "val", limit=None, bits=16, frac=1, output_dir=root / "run",
                batch_size=2, workers=0, checkpoint_every=2, restart=False,
                calibration_manifest=manifest, prune_incumbent=dict(frac=0, correct=4, samples=8))
            with mock.patch.object(calibration, "ManifestDataset", return_value=Dataset()), \
                    mock.patch.object(evaluator, "load_reference", side_effect=lambda *a: (Reference(), None)), \
                    mock.patch.object(evaluator, "IntegerResNet", Integer), contextlib.redirect_stdout(io.StringIO()):
                with self.assertRaisesRegex(RuntimeError, "test interruption"):
                    evaluator.evaluate(args)
                Integer.fail = False
                result = evaluator.evaluate(args)
                self.assertEqual((result["status"], result["samples"], result["complete"]), ("pruned", 4, False))
                self.assertEqual(result["pruning_certificate"], pruning_certificate(4, 0, 8, args.prune_incumbent))
                self.assertEqual(Integer.seen, [[0, 1], [2, 3], [2, 3]])
                (args.output_dir / "metrics.json").unlink()
                self.assertEqual(evaluator.evaluate(args), result)
                self.assertEqual(Integer.seen, [[0, 1], [2, 3], [2, 3]])
                self.assertEqual(json.loads((args.output_dir / "metrics.json").read_text()), result)
                accuracy = json.loads((args.output_dir / "accuracy.json").read_text())
                self.assertEqual((accuracy["evidence_scope"], accuracy["samples"]), ("calibration_pruned", 4))
                args.prune_incumbent = dict(frac=0, correct=5, samples=8)
                with self.assertRaisesRegex(ValueError, "checkpoint configuration"):
                    evaluator.evaluate(args)


if __name__ == "__main__":
    unittest.main()
