#!/usr/bin/env python3
"""Two-dimensional, ring-faithful E0 precision sweep.

For every total width ``ell`` this driver evaluates every legal fractional
width ``s in {0, ..., ell-1}`` on a deterministic calibration split, selects
the highest-accuracy ``s`` (ties prefer the smaller ``s``), and then evaluates
only the selected pair on the complete test split.  The calibration and final
rows are saved separately so the reported optimum cannot silently be chosen
on the final test set.

Use ``--lstm-activation ideal`` to reproduce the original E0-B activation
reference. The default ``sigma`` run includes the q=10 LUT and skips formats
unsupported by the C++ activation API (s < 6 or ell <= s + 4).

The fixed-point implementation is in :mod:`fixed_point`; all runs are CPU
runs because exact PyTorch int64 matmul/convolution is not implemented on CUDA.
"""

from __future__ import annotations

import argparse
import gc
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Sequence, Tuple

import pandas as pd
import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Subset, TensorDataset

from fixed_point import FixedPointConfig, convert_model_to_fixed_point
from models import MLP, ResNet20, SimpleLSTM


ELL_GRID = (12, 16, 20, 24, 28, 32)


def _candidate_fractional_widths(model_kind: str, ell: int,
                                 lstm_activation: str) -> List[int]:
    if model_kind == "lstm" and lstm_activation == "sigma":
        return [s for s in range(ell) if 6 <= s and ell > s + 4]
    return list(range(ell))


def _loader(dataset, batch_size: int, samples: int | None) -> DataLoader:
    if samples is not None:
        samples = min(samples, len(dataset))
        dataset = Subset(dataset, list(range(samples)))
    return DataLoader(dataset, batch_size=batch_size, shuffle=False,
                      num_workers=0)


def _accuracy(model, loader: DataLoader, model_kind: str) -> Tuple[float, int]:
    model.eval()
    correct = total = 0
    with torch.no_grad():
        for inputs, labels in loader:
            if model_kind == "mlp":
                inputs = inputs.view(inputs.size(0), -1)
            outputs = model(inputs)
            if isinstance(outputs, tuple):
                # A direct FixedPointLSTM returns (sequence, (h_n, c_n));
                # SimpleLSTM's classifier returns logits and is not a tuple.
                outputs = outputs[0]
            predictions = outputs.argmax(dim=1)
            correct += (predictions == labels).sum().item()
            total += labels.size(0)
    return (100.0 * correct / total if total else float("nan")), total


def _make_mnist() -> Tuple[DataLoader, torch.utils.data.Dataset]:
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)),
    ])
    test_data = torchvision.datasets.MNIST(
        root="./data", train=False, download=True, transform=transform)
    calibration_data = torchvision.datasets.MNIST(
        root="./data", train=True, download=True, transform=transform)
    return _loader(test_data, 512, None), calibration_data


def _make_cifar() -> Tuple[DataLoader, torch.utils.data.Dataset]:
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465),
                             (0.2023, 0.1994, 0.2010)),
    ])
    test_data = torchvision.datasets.CIFAR10(
        root="./data", train=False, download=True, transform=transform)
    calibration_data = torchvision.datasets.CIFAR10(
        root="./data", train=True, download=True, transform=transform)
    return _loader(test_data, 128, None), calibration_data


def _make_lstm() -> Tuple[DataLoader, TensorDataset]:
    # Keep calibration and test sequences independent.  The test seed matches
    # run_e0.py; the calibration seed is deliberately different.
    from run_e0 import generate_sequence_sum_task

    x_test, y_test = generate_sequence_sum_task(2000, seed=1234)
    x_cal, y_cal = generate_sequence_sum_task(2000, seed=4321)
    test_data = TensorDataset(x_test, y_test)
    calibration_data = TensorDataset(x_cal, y_cal)
    return _loader(test_data, 256, None), calibration_data


def _build(model_kind: str):
    if model_kind == "mlp":
        model = MLP()
        model.load_state_dict(torch.load("models/mlp_mnist.pth",
                                         map_location="cpu"))
    elif model_kind == "resnet":
        model = ResNet20()
        model.load_state_dict(torch.load("models/resnet20_cifar10.pth",
                                         map_location="cpu"))
    elif model_kind == "lstm":
        model = SimpleLSTM(vocab_size=100, embed_dim=32,
                           hidden_dim=32, num_classes=2)
        model.load_state_dict(torch.load("models/lstm_synthetic.pth",
                                         map_location="cpu"))
    else:
        raise ValueError(f"unknown model kind: {model_kind}")
    return model.cpu().eval()


def _sweep_model(model_kind: str, ell_grid: Sequence[int],
                 calibration_samples: int, results_dir: Path,
                 lstm_activation: str = "sigma") -> List[dict]:
    activation_mode = lstm_activation if model_kind == "lstm" else "not_applicable"
    for filename in ("e0_precision_2d_candidates.csv", "e0_precision_2d_final.csv"):
        result_path = results_dir / filename
        if result_path.exists() and "activation_mode" not in pd.read_csv(
                result_path, nrows=0).columns:
            raise ValueError(
                f"{result_path} is a legacy artifact without activation_mode; "
                "use a new --results-dir to preserve the recorded experiment"
            )
    model = _build(model_kind)
    if model_kind == "mlp":
        full_loader, calibration_dataset = _make_mnist()
        name = "MLP"
    elif model_kind == "resnet":
        full_loader, calibration_dataset = _make_cifar()
        name = "ResNet20"
    else:
        full_loader, calibration_dataset = _make_lstm()
        name = "LSTM"

    calibration_loader = _loader(
        calibration_dataset, full_loader.batch_size, calibration_samples)
    float_acc, n_full = _accuracy(model, full_loader, model_kind)
    float_cal_acc, n_cal = _accuracy(model, calibration_loader, model_kind)
    print(f"\n{name}: float32 calibration={float_cal_acc:.2f}% "
          f"({n_cal}), full={float_acc:.2f}% ({n_full})")

    candidate_rows: List[dict] = []
    selected: Dict[int, Tuple[int, float]] = {}
    for ell in ell_grid:
        legal = []
        fractional_widths = _candidate_fractional_widths(
            model_kind, ell, lstm_activation)
        if len(fractional_widths) != ell:
            skipped = sorted(set(range(ell)) - set(fractional_widths))
            print(f"  ell={ell}: skipped unsupported SIGMA s={skipped} "
                  "(requires s >= 6 and ell > s + 4)")
        if not fractional_widths:
            raise ValueError(f"no supported {activation_mode} configurations for ell={ell}")
        for s in fractional_widths:
            config = FixedPointConfig(ell, s)
            fixed = convert_model_to_fixed_point(
                model, config, lstm_activation=lstm_activation)
            acc, count = _accuracy(fixed, calibration_loader, model_kind)
            row = {
                "model": name,
                "activation_mode": activation_mode,
                "ell": ell,
                "s": s,
                "int_bits": ell - s,
                "split": "calibration",
                "accuracy_pct": acc,
                "float_accuracy_pct": float_cal_acc,
                "delta_pp": acc - float_cal_acc,
                "samples": count,
                "ring_semantics": (
                    f"Z_2^ell_{config.truncation_mode}_truncation"
                ),
            }
            candidate_rows.append(row)
            legal.append((acc, s))
            del fixed
            gc.collect()
            print(f"  ell={ell:2d}, s={s:2d}: {acc:6.2f}%")
        # Max accuracy, then minimum s among exact ties.
        best_acc = max(acc for acc, _ in legal)
        best_s = min(s for acc, s in legal if abs(acc - best_acc) < 1e-12)
        selected[ell] = (best_s, best_acc)
        print(f"  -> selected ell={ell}, s={best_s} "
              f"(calibration {best_acc:.2f}%)")

    final_rows: List[dict] = []
    for ell in ell_grid:
        s, calibration_acc = selected[ell]
        config = FixedPointConfig(ell, s)
        fixed = convert_model_to_fixed_point(
            model, config, lstm_activation=lstm_activation)
        acc, count = _accuracy(fixed, full_loader, model_kind)
        final_rows.append({
            "model": name,
            "activation_mode": activation_mode,
            "ell": ell,
            "s": s,
            "int_bits": ell - s,
            "split": "full_test",
            "accuracy_pct": acc,
            "float_accuracy_pct": float_acc,
            "delta_pp": acc - float_acc,
            "calibration_accuracy_pct": calibration_acc,
            "samples": count,
            "ring_semantics": (
                f"Z_2^ell_{config.truncation_mode}_truncation"
            ),
        })
        print(f"  FINAL ell={ell:2d}, s={s:2d}: {acc:6.2f}% "
              f"(delta {acc - float_acc:+.2f} pp)")
        del fixed
        gc.collect()

    results_dir.mkdir(parents=True, exist_ok=True)
    # Replace this model/mode's rows while preserving other modes and models
    # to the shared result directory.  Blind appending makes a rerun look like
    # additional experimental samples and can silently duplicate rows.
    candidate_path = results_dir / "e0_precision_2d_candidates.csv"
    final_path = results_dir / "e0_precision_2d_final.csv"
    candidate_df = pd.DataFrame(candidate_rows)
    final_df = pd.DataFrame(final_rows)
    if candidate_path.exists():
        old = pd.read_csv(candidate_path)
        same_mode = old["activation_mode"] == activation_mode
        candidate_df = pd.concat(
            [old[~((old["model"] == name) & same_mode)], candidate_df], ignore_index=True
        )
    if final_path.exists():
        old = pd.read_csv(final_path)
        same_mode = old["activation_mode"] == activation_mode
        final_df = pd.concat(
            [old[~((old["model"] == name) & same_mode)], final_df], ignore_index=True
        )
    candidate_df.to_csv(candidate_path, index=False)
    final_df.to_csv(final_path, index=False)
    return final_rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models", default="mlp,resnet,lstm",
                        help="comma-separated subset of mlp,resnet,lstm")
    parser.add_argument("--calibration-samples", type=int, default=512)
    parser.add_argument("--results-dir", default="results")
    parser.add_argument("--lstm-activation", choices=("ideal", "sigma"),
                        default="sigma",
                        help="ideal reproduces the original E0-B gates; default sigma includes "
                             "the q=10 LUT and scans only 6 <= s <= ell-5")
    args = parser.parse_args()
    selected = [name.strip().lower() for name in args.models.split(",")
                if name.strip()]
    results_dir = Path(args.results_dir)
    for model_kind in selected:
        _sweep_model(model_kind, ELL_GRID, args.calibration_samples,
                     results_dir, lstm_activation=args.lstm_activation)


if __name__ == "__main__":
    main()
