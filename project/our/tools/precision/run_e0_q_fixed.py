#!/usr/bin/env python3
"""E0-Q-fixed: strict saturating integer fixed-point ResNet inference.

Unlike ``run_e0_q_all.py``, this driver never executes a network operator on
floating-point tensors.  The tested ResNet path uses integer-coded input,
integer convolution/linear products and ell-bit accumulators, integer BN affine
operations, integer residual additions, integer pooling, and integer ReLU.
The only floating-point model is the FP32 reference used for accuracy.

This is a non-wrapping fixed-point experiment. Each scalar multiply is floor-
rescaled immediately to Q(ell,s), and every scalar accumulation step saturates
to ell bits. A host 2*ell-bit product temporary is permitted, but a widened
semantic dot-product accumulator is not. The ring-wrapping counterpart is E0-B
in ``fixed_point.py``.
"""

from __future__ import annotations

import argparse
import gc
import time
from pathlib import Path

import pandas as pd
import torch

from fixed_point_saturated import SaturatingFixedPoint, convert_resnet_to_saturating
from run_e0_q_all import MODEL_SPECS, accuracy, build_model, load_imagenette


ARITHMETIC = "strict_q_mul_then_ell_bit_saturating_add"


def _upsert_csv(path: Path, rows: list[dict], key_columns: list[str]) -> None:
    """Checkpoint long scans without duplicating an already completed key."""
    frame = pd.DataFrame(rows)
    if path.exists():
        old = pd.read_csv(path)
        keys = set(tuple(row[col] for col in key_columns) for row in rows)
        keep = [tuple(row[col] for col in key_columns) not in keys
                for _, row in old.iterrows()]
        frame = pd.concat([old.loc[keep], frame], ignore_index=True)
    frame.sort_values(key_columns).to_csv(path, index=False)


def _stats_columns(fixed) -> dict:
    return fixed.fp.stats()


def _load_completed(path: Path, name: str) -> dict[tuple[int, int], dict]:
    if not path.exists():
        return {}
    frame = pd.read_csv(path)
    if "model" not in frame.columns:
        return {}
    return {
        (int(row["ell"]), int(row["s"])): row.to_dict()
        for _, row in frame[frame.model == name].iterrows()
        if row.get("status", "ok") == "ok"
    }


def _can_reuse_wider(row: dict) -> bool:
    """A wider ell is identical when no smaller-format boundary was touched."""
    required = (
        "offline_clipped", "input_clipped", "mul_saturations",
        "add_saturations",
    )
    return all(int(row.get(column, -1)) == 0 for column in required)


def run_model(name: str, data_root: Path, ell_grid, calibration_samples,
              test_samples, batch_size, workers, results_dir: Path,
              device: torch.device):
    model, weights = build_model(name)
    calibration, test, class_to_imagenet = load_imagenette(
        data_root, weights, batch_size, calibration_samples, test_samples, workers
    )
    results_dir.mkdir(parents=True, exist_ok=True)
    candidate_path = results_dir / "e0_q_fixed_candidates.csv"
    final_path = results_dir / "e0_q_fixed_final.csv"
    model.to(device).eval()
    float_cal, n_cal = accuracy(model, calibration, class_to_imagenet,
                                device)
    float_test, n_test = accuracy(model, test, class_to_imagenet,
                                  device)
    print(f"\n{name}: FP32 calibration={float_cal:.3f}% ({n_cal}), "
          f"test={float_test:.3f}% ({n_test})", flush=True)

    candidates = []
    completed = _load_completed(candidate_path, name)
    selected = {}
    for ell in ell_grid:
        legal = []
        for s in range(ell):
            previous = completed.get((ell, s))
            if previous is not None:
                acc = float(previous["accuracy_pct"])
                legal.append((acc, s))
                candidates.append(previous)
                print(f"  cal ell={ell:2d}, s={s:2d}: {acc:6.3f}% "
                      "(checkpoint)", flush=True)
                continue
            source = None
            for source_ell in sorted((e for e in ell_grid if e < ell), reverse=True):
                candidate = completed.get((source_ell, s))
                if candidate is not None and _can_reuse_wider(candidate):
                    source = candidate
                    break
            if source is not None:
                row = dict(source)
                row.update({
                    "ell": ell,
                    "int_bits": ell - s,
                    "accumulator_bits": ell,
                    "host_product_bits_max": 2 * ell,
                    "elapsed_seconds": 0.0,
                    "measurement": "exact_reuse_no_boundary_event",
                    "reused_from_ell": int(source["ell"]),
                })
                acc = float(row["accuracy_pct"])
                legal.append((acc, s))
                candidates.append(row)
                completed[(ell, s)] = row
                _upsert_csv(candidate_path, [row], ["model", "ell", "s"])
                print(f"  cal ell={ell:2d}, s={s:2d}: {acc:6.3f}% "
                      f"(exact reuse from ell={int(source['ell'])}; zero boundary events)",
                      flush=True)
                continue
            fixed = None
            error = None
            started = time.time()
            try:
                fixed = convert_resnet_to_saturating(
                    model, ell, s, track_stats=True
                ).to(device)
                fixed.fp.reset_runtime_stats()
                acc, count = accuracy(fixed, calibration, class_to_imagenet,
                                      device)
                stat_columns = _stats_columns(fixed)
            except (OverflowError, RuntimeError) as exc:
                acc, count = float("nan"), 0
                error = f"{type(exc).__name__}: {exc}"
                stat_columns = {}
            row = {
                "model": name,
                "dataset": "ImageNette",
                "split": "calibration",
                "ell": ell,
                "s": s,
                "int_bits": ell - s,
                "accuracy_pct": acc,
                "float_accuracy_pct": float_cal,
                "delta_pp": acc - float_cal,
                "samples": count,
                "arithmetic": ARITHMETIC,
                "wraparound": False,
                "wide_accumulator": False,
                "accumulator_bits": ell,
                "host_product_bits_max": 2 * ell,
                "elapsed_seconds": time.time() - started,
                "device": str(device),
                "status": "ok" if error is None else "failed",
                "error": error or "",
                "measurement": "measured",
                "reused_from_ell": "",
                **stat_columns,
            }
            candidates.append(row)
            if error is None:
                completed[(ell, s)] = row
            _upsert_csv(candidate_path, [row], ["model", "ell", "s"])
            if error is None:
                legal.append((acc, s))
                print(f"  cal ell={ell:2d}, s={s:2d}: {acc:6.3f}% "
                      f"({row['elapsed_seconds']:.1f}s, "
                      f"mul-sat={stat_columns['mul_saturation_rate_pct']:.6f}%, "
                      f"add-sat={stat_columns['add_saturation_rate_pct']:.6f}%)",
                      flush=True)
            else:
                print(f"  cal ell={ell:2d}, s={s:2d}: UNSUPPORTED ({error})", flush=True)
            del fixed
            gc.collect()
            if device.type == "cuda":
                torch.cuda.empty_cache()
        if not legal:
            raise RuntimeError(f"no int64-safe candidates for ell={ell}")
        best_acc = max(acc for acc, _ in legal)
        best_s = min(s for acc, s in legal if abs(acc - best_acc) < 1e-12)
        selected[ell] = (best_s, best_acc)
        print(f"  -> selected ell={ell}, s={best_s} ({best_acc:.3f}%)", flush=True)

    final_rows = []
    for ell in ell_grid:
        s, cal_acc = selected[ell]
        fixed = convert_resnet_to_saturating(
            model, ell, s, track_stats=True
        ).to(device)
        fixed.fp.reset_runtime_stats()
        started = time.time()
        acc, count = accuracy(fixed, test, class_to_imagenet,
                              device)
        row = {
            "model": name,
            "dataset": "ImageNette",
            "split": "validation",
            "ell": ell,
            "s": s,
            "int_bits": ell - s,
            "accuracy_pct": acc,
            "float_accuracy_pct": float_test,
            "delta_pp": acc - float_test,
            "calibration_accuracy_pct": cal_acc,
            "samples": count,
            "arithmetic": ARITHMETIC,
            "wraparound": False,
            "wide_accumulator": False,
            "accumulator_bits": ell,
            "host_product_bits_max": 2 * ell,
            "elapsed_seconds": time.time() - started,
            "device": str(device),
            "status": "ok",
            "error": "",
            **_stats_columns(fixed),
        }
        final_rows.append(row)
        _upsert_csv(final_path, [row], ["model", "ell"])
        print(f"FINAL ell={ell:2d}, s={s:2d}: {acc:6.3f}% "
              f"delta={acc-float_test:+.3f} pp", flush=True)
        del fixed
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models", default="resnet50,resnet101,resnet152")
    parser.add_argument("--data-root", default="data")
    parser.add_argument("--results-dir", default="results/e0_q_fixed_imagenette")
    parser.add_argument("--ell", default="12,16,20,24,28,32")
    parser.add_argument("--calibration-samples", type=int, default=512)
    parser.add_argument("--test-samples", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--device", default="auto",
                        help="auto, cuda, cuda:0, or cpu; formal strict scans should use CUDA")
    args = parser.parse_args()
    ell_grid = tuple(int(x) for x in args.ell.split(",") if x.strip())
    calibration = None if args.calibration_samples <= 0 else args.calibration_samples
    test = None if args.test_samples <= 0 else args.test_samples
    device_name = ("cuda" if torch.cuda.is_available() else "cpu") \
        if args.device == "auto" else args.device
    device = torch.device(device_name)
    if device.type == "cpu":
        print("WARNING: strict CPU reference is intended for small correctness tests; "
              "large ResNet scans are practical only with the exact CUDA kernel.",
              flush=True)
    for name in [x.strip().lower() for x in args.models.split(",") if x.strip()]:
        if name not in MODEL_SPECS:
            raise ValueError(f"unknown model: {name}")
        run_model(name, Path(args.data_root), ell_grid, calibration, test,
                  args.batch_size, args.workers, Path(args.results_dir), device)


if __name__ == "__main__":
    main()
