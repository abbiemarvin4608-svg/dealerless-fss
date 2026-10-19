"""Unit tests for the formal ImageNet-1K E0-Q-fixed driver."""

from pathlib import Path

import torch
from torch.utils.data import DataLoader, TensorDataset

from run_e0_q_fixed_imagenet import (
    _imagenet_dirs,
    balanced_class_indices,
    evaluate,
)


def test_balanced_class_indices_are_complete_deterministic_and_disjoint():
    targets = [0, 1, 2] * 5
    first = balanced_class_indices(targets, per_class=2, seed=17, classes=3)
    second = balanced_class_indices(targets, per_class=2, seed=17, classes=3)
    assert first == second
    assert len(first) == len(set(first)) == 6
    assert [sum(targets[index] == c for index in first) for c in range(3)] == [2, 2, 2]


def test_balanced_class_indices_reject_missing_class():
    try:
        balanced_class_indices([0, 0, 1, 1], per_class=1, seed=1, classes=3)
    except ValueError as exc:
        assert "classes have fewer" in str(exc)
    else:
        raise AssertionError("missing class was accepted")


def test_imagenet_dir_resolution(tmp_path: Path):
    (tmp_path / "train").mkdir()
    (tmp_path / "val").mkdir()
    assert _imagenet_dirs(tmp_path) == (tmp_path / "train", tmp_path / "val")


class _IdentityLogits(torch.nn.Module):
    def forward(self, inputs):
        return inputs


def test_evaluate_reports_top1_top5_and_paired_changes():
    logits = torch.tensor([
        [9.0, 8.0, 7.0, 6.0, 5.0, 4.0],
        [0.0, 1.0, 2.0, 3.0, 4.0, 5.0],
        [6.0, 5.0, 4.0, 3.0, 2.0, 1.0],
    ])
    labels = torch.tensor([0, 4, 1])
    loader = DataLoader(TensorDataset(logits, labels), batch_size=2, shuffle=False)
    reference = torch.tensor([0, 5, 2])
    result = evaluate(
        _IdentityLogits(), loader, torch.device("cpu"),
        reference_top1=reference, collect_top1=True,
    )
    assert result["top1_pct"] == 100.0 / 3.0
    assert result["top5_pct"] == 100.0
    assert result["prediction_agreement_pct"] == 200.0 / 3.0
    assert result["fp32_right_fixed_wrong"] == 0
    assert result["fp32_wrong_fixed_right"] == 0
    assert torch.equal(result["top1_predictions"], torch.tensor([0, 5, 0]))
