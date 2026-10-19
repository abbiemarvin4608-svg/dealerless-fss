#!/usr/bin/env python3
"""Strict E0-Q-fixed sweep on the complete ImageNet-1K task.

The calibration set is sampled only from the official training split, with an
equal number of images from every one of the 1,000 classes.  For each requested
total width ``ell``, every legal fractional width ``s=0,...,ell-1`` is scored
on that calibration set.  The highest-Top-1 candidate (ties choose the smaller
``s``) is then evaluated exactly once on all 50,000 validation images.

All network operations use the strict integer-domain implementation in
``fixed_point_saturated.py``.  Checkpoints make the multi-day experiment
resumable and prevent completed validation configurations from being rerun.
"""

from __future__ import annotations

import argparse
import gc
import json
import random
import time
from pathlib import Path
from typing import Iterable, Sequence

import pandas as pd
import torch
from torch.utils.data import DataLoader, Subset
from torchvision.datasets import ImageFolder

from fixed_point_saturated import convert_resnet_to_saturating
from run_e0_q_all import MODEL_SPECS, build_model


ARITHMETIC = "strict_q_mul_then_ell_bit_saturating_add"
IMAGENET_TRAIN_IMAGES = 1_281_167
IMAGENET_VAL_IMAGES = 50_000
IMAGENET_CLASSES = 1_000


def balanced_class_indices(targets: Sequence[int], per_class: int,
                           seed: int, classes: int = IMAGENET_CLASSES) -> list[int]:
    """Select ``per_class`` deterministic random examples from every class."""
    if per_class < 1:
        raise ValueError("calibration_per_class must be at least one")
    buckets = [[] for _ in range(classes)]
    for index, target in enumerate(targets):
        target = int(target)
        if target < 0 or target >= classes:
            raise ValueError(f"target {target} is outside [0, {classes})")
        buckets[target].append(index)
    missing = [class_index for class_index, values in enumerate(buckets)
               if len(values) < per_class]
    if missing:
        raise ValueError(
            f"{len(missing)} classes have fewer than {per_class} samples; "
            f"first missing classes: {missing[:10]}"
        )
    selected = []
    for class_index, values in enumerate(buckets):
        # An independent class seed makes the selection stable if filesystem
        # enumeration changes in an unrelated class.
        generator = random.Random(seed + class_index)
        selected.extend(sorted(generator.sample(values, per_class)))
    return selected


def _imagenet_dirs(root: Path) -> tuple[Path, Path]:
    """Resolve the common extracted ImageNet-1K directory layouts."""
    candidates = (
        (root / "train", root / "val"),
        (root / "ILSVRC2012" / "train", root / "ILSVRC2012" / "val"),
        (root / "ILSVRC" / "Data" / "CLS-LOC" / "train",
         root / "ILSVRC" / "Data" / "CLS-LOC" / "val"),
    )
    for train_dir, val_dir in candidates:
        if train_dir.is_dir() and val_dir.is_dir():
            return train_dir, val_dir
    checked = "\n".join(f"  - {train} and {val}" for train, val in candidates)
    raise FileNotFoundError(
        "Could not find extracted ImageNet-1K train/val directories. Checked:\n"
        + checked
    )


def load_imagenet(root: Path, transform, calibration_per_class: int,
                  calibration_seed: int, batch_size: int, workers: int,
                  allow_nonstandard_counts: bool = False):
    """Load a balanced train calibration subset and the complete validation set."""
    train_dir, val_dir = _imagenet_dirs(root)
    train = ImageFolder(train_dir, transform=transform)
    val = ImageFolder(val_dir, transform=transform)

    if len(train.classes) != IMAGENET_CLASSES or len(val.classes) != IMAGENET_CLASSES:
        raise RuntimeError(
            f"expected 1,000 classes, found train={len(train.classes)}, "
            f"val={len(val.classes)}"
        )
    if train.class_to_idx != val.class_to_idx:
        raise RuntimeError("training and validation class-to-index mappings differ")
    if not allow_nonstandard_counts:
        if len(train) != IMAGENET_TRAIN_IMAGES:
            raise RuntimeError(
                f"formal run requires {IMAGENET_TRAIN_IMAGES:,} training images; "
                f"found {len(train):,}"
            )
        if len(val) != IMAGENET_VAL_IMAGES:
            raise RuntimeError(
                f"formal run requires {IMAGENET_VAL_IMAGES:,} validation images; "
                f"found {len(val):,}"
            )

    indices = balanced_class_indices(
        train.targets, calibration_per_class, calibration_seed,
        classes=IMAGENET_CLASSES,
    )
    calibration = Subset(train, indices)
    loader_args = {
        "batch_size": batch_size,
        "shuffle": False,
        "num_workers": workers,
        "pin_memory": torch.cuda.is_available(),
    }
    return (
        DataLoader(calibration, **loader_args),
        DataLoader(val, **loader_args),
        train,
        indices,
        train_dir,
        val_dir,
    )


def write_calibration_manifest(path: Path, dataset: ImageFolder,
                               indices: Iterable[int]) -> None:
    rows = []
    root = Path(dataset.root)
    for index in indices:
        sample_path, class_index = dataset.samples[index]
        rows.append({
            "dataset_index": int(index),
            "class_index": int(class_index),
            "class_id": dataset.classes[int(class_index)],
            "relative_path": str(Path(sample_path).relative_to(root)),
        })
    frame = pd.DataFrame(rows).sort_values(["class_index", "relative_path"])
    frame.to_csv(path, index=False)


@torch.no_grad()
def evaluate(model, loader, device: torch.device,
             reference_top1: torch.Tensor | None = None,
             collect_top1: bool = False) -> dict:
    """Compute Top-1/Top-5 and optional paired changes from FP32 predictions."""
    top1_correct = top5_correct = total = 0
    agree = float_right_fixed_wrong = float_wrong_fixed_right = 0
    predictions = []
    reference_offset = 0
    started = time.time()
    for inputs, labels in loader:
        inputs = inputs.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        outputs = model(inputs)
        top5 = outputs.topk(5, dim=1).indices
        top1 = top5[:, 0]
        fixed_correct = top1.eq(labels)
        top1_correct += int(fixed_correct.sum().item())
        top5_correct += int(top5.eq(labels[:, None]).any(dim=1).sum().item())
        count = int(labels.numel())
        if reference_top1 is not None:
            reference = reference_top1[
                reference_offset:reference_offset + count
            ].to(device, non_blocking=True)
            reference_correct = reference.eq(labels)
            agree += int(top1.eq(reference).sum().item())
            float_right_fixed_wrong += int(
                (reference_correct & ~fixed_correct).sum().item()
            )
            float_wrong_fixed_right += int(
                (~reference_correct & fixed_correct).sum().item()
            )
            reference_offset += count
        if collect_top1:
            predictions.append(top1.cpu())
        total += count
    if reference_top1 is not None and reference_offset != len(reference_top1):
        raise RuntimeError("reference prediction count does not match dataset")
    result = {
        "top1_pct": 100.0 * top1_correct / total,
        "top5_pct": 100.0 * top5_correct / total,
        "samples": total,
        "elapsed_seconds": time.time() - started,
    }
    if reference_top1 is not None:
        result.update({
            "prediction_agreement_pct": 100.0 * agree / total,
            "fp32_right_fixed_wrong": float_right_fixed_wrong,
            "fp32_wrong_fixed_right": float_wrong_fixed_right,
        })
    if collect_top1:
        result["top1_predictions"] = torch.cat(predictions)
    return result


def _upsert_csv(path: Path, rows: list[dict], keys: list[str]) -> None:
    new = pd.DataFrame(rows)
    if path.exists():
        old = pd.read_csv(path)
        new_keys = {tuple(row[key] for key in keys) for row in rows}
        keep = [tuple(row[key] for key in keys) not in new_keys
                for _, row in old.iterrows()]
        new = pd.concat([old.loc[keep], new], ignore_index=True)
    new.sort_values(keys).to_csv(path, index=False)


def _completed_rows(path: Path, model: str) -> dict[tuple[int, int], dict]:
    if not path.exists():
        return {}
    frame = pd.read_csv(path)
    if "model" not in frame:
        return {}
    return {
        (int(row.ell), int(row.s)): row.to_dict()
        for _, row in frame[(frame.model == model) & (frame.status == "ok")].iterrows()
    }


def _can_reuse_wider(row: dict) -> bool:
    return all(int(row.get(column, -1)) == 0 for column in (
        "offline_clipped", "input_clipped", "mul_saturations", "add_saturations"
    ))


def _write_or_check_config(path: Path, config: dict) -> None:
    if path.exists():
        previous = json.loads(path.read_text())
        if previous != config:
            raise RuntimeError(
                f"result directory contains a different experiment config:\n"
                f"existing={previous}\nrequested={config}"
            )
        return
    path.write_text(json.dumps(config, indent=2, sort_keys=True) + "\n")


def run_model(name: str, calibration_loader, validation_loader,
              ell_grid: Sequence[int], results_dir: Path,
              device: torch.device, phase: str) -> None:
    model, _ = build_model(name)
    model.to(device).eval()
    candidate_path = results_dir / "e0_q_fixed_imagenet_candidates.csv"
    selected_path = results_dir / "e0_q_fixed_imagenet_selected.csv"
    final_path = results_dir / "e0_q_fixed_imagenet_final.csv"

    float_cal = evaluate(model, calibration_loader, device)
    print(
        f"\n{name}: FP32 calibration Top-1={float_cal['top1_pct']:.3f}% "
        f"Top-5={float_cal['top5_pct']:.3f}% ({float_cal['samples']})",
        flush=True,
    )

    completed = _completed_rows(candidate_path, name)
    selected = {}
    if phase in ("scan", "all"):
        for ell in ell_grid:
            legal = []
            for s in range(ell):
                previous = completed.get((ell, s))
                if previous is not None:
                    legal.append((float(previous["top1_pct"]), s))
                    print(
                        f"  cal ell={ell:2d}, s={s:2d}: "
                        f"Top-1={float(previous['top1_pct']):6.3f}% (checkpoint)",
                        flush=True,
                    )
                    continue

                source = None
                for source_ell in sorted((e for e in ell_grid if e < ell), reverse=True):
                    narrower = completed.get((source_ell, s))
                    if narrower is not None and _can_reuse_wider(narrower):
                        source = narrower
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
                else:
                    fixed = None
                    started = time.time()
                    try:
                        fixed = convert_resnet_to_saturating(
                            model, ell, s, track_stats=True
                        ).to(device)
                        fixed.fp.reset_runtime_stats()
                        metrics = evaluate(fixed, calibration_loader, device)
                        stats = fixed.fp.stats()
                        error = ""
                    except (OverflowError, RuntimeError) as exc:
                        metrics = {
                            "top1_pct": float("nan"), "top5_pct": float("nan"),
                            "samples": 0, "elapsed_seconds": time.time() - started,
                        }
                        stats = {}
                        error = f"{type(exc).__name__}: {exc}"
                    row = {
                        "model": name,
                        "dataset": "ImageNet-1K",
                        "split": "train_calibration",
                        "ell": ell,
                        "s": s,
                        "int_bits": ell - s,
                        "top1_pct": metrics["top1_pct"],
                        "top5_pct": metrics["top5_pct"],
                        "float_top1_pct": float_cal["top1_pct"],
                        "float_top5_pct": float_cal["top5_pct"],
                        "samples": metrics["samples"],
                        "arithmetic": ARITHMETIC,
                        "wraparound": False,
                        "wide_accumulator": False,
                        "accumulator_bits": ell,
                        "host_product_bits_max": 2 * ell,
                        "elapsed_seconds": metrics["elapsed_seconds"],
                        "device": str(device),
                        "status": "ok" if not error else "failed",
                        "error": error,
                        "measurement": "measured",
                        "reused_from_ell": "",
                        **stats,
                    }
                    del fixed
                    gc.collect()
                    if device.type == "cuda":
                        torch.cuda.empty_cache()

                _upsert_csv(candidate_path, [row], ["model", "ell", "s"])
                if row["status"] == "ok":
                    completed[(ell, s)] = row
                    legal.append((float(row["top1_pct"]), s))
                    print(
                        f"  cal ell={ell:2d}, s={s:2d}: "
                        f"Top-1={float(row['top1_pct']):6.3f}% "
                        f"Top-5={float(row['top5_pct']):6.3f}% "
                        f"({row['measurement']})",
                        flush=True,
                    )
                else:
                    print(f"  cal ell={ell:2d}, s={s:2d}: {row['error']}", flush=True)
            if not legal:
                raise RuntimeError(f"no successful calibration candidate for ell={ell}")
            best_top1 = max(accuracy for accuracy, _ in legal)
            best_s = min(s for accuracy, s in legal if abs(accuracy - best_top1) < 1e-12)
            selected[ell] = (best_s, best_top1)
            selected_row = completed[(ell, best_s)]
            _upsert_csv(selected_path, [{
                "model": name,
                "ell": ell,
                "s": best_s,
                "int_bits": ell - best_s,
                "calibration_top1_pct": selected_row["top1_pct"],
                "calibration_top5_pct": selected_row["top5_pct"],
                "calibration_samples": selected_row["samples"],
                "selection_metric": "top1_pct",
                "tie_break": "smaller_s",
            }], ["model", "ell"])
            print(f"  -> selected ell={ell}, s={best_s}", flush=True)

    if phase == "scan":
        return

    if not selected_path.exists():
        raise RuntimeError("selection file is missing; run --phase scan first")
    selection_frame = pd.read_csv(selected_path)
    for ell in ell_grid:
        rows = selection_frame[
            (selection_frame.model == name) & (selection_frame.ell == ell)
        ]
        if len(rows) != 1:
            raise RuntimeError(f"missing unique selected s for {name}, ell={ell}")
        row = rows.iloc[0]
        selected[ell] = (int(row.s), float(row.calibration_top1_pct))

    existing_final = pd.read_csv(final_path) if final_path.exists() else pd.DataFrame()
    completed_final = set()
    if not existing_final.empty:
        completed_final = {
            (str(row.model), int(row.ell))
            for _, row in existing_final[existing_final.status == "ok"].iterrows()
        }

    # Compute the reference once and retain predictions for paired diagnostics.
    model.to(device).eval()
    float_val = evaluate(
        model, validation_loader, device, collect_top1=True
    )
    reference_top1 = float_val.pop("top1_predictions")
    print(
        f"{name}: FP32 validation Top-1={float_val['top1_pct']:.3f}% "
        f"Top-5={float_val['top5_pct']:.3f}% ({float_val['samples']})",
        flush=True,
    )

    for ell in ell_grid:
        if (name, ell) in completed_final:
            print(f"FINAL {name}, ell={ell}: checkpoint; not rerun", flush=True)
            continue
        s, calibration_top1 = selected[ell]
        fixed = convert_resnet_to_saturating(
            model, ell, s, track_stats=True
        ).to(device)
        fixed.fp.reset_runtime_stats()
        metrics = evaluate(
            fixed, validation_loader, device, reference_top1=reference_top1
        )
        row = {
            "model": name,
            "dataset": "ImageNet-1K",
            "split": "validation",
            "ell": ell,
            "s": s,
            "int_bits": ell - s,
            "top1_pct": metrics["top1_pct"],
            "top5_pct": metrics["top5_pct"],
            "float_top1_pct": float_val["top1_pct"],
            "float_top5_pct": float_val["top5_pct"],
            "top1_delta_pp": metrics["top1_pct"] - float_val["top1_pct"],
            "top5_delta_pp": metrics["top5_pct"] - float_val["top5_pct"],
            "prediction_agreement_pct": metrics["prediction_agreement_pct"],
            "fp32_right_fixed_wrong": metrics["fp32_right_fixed_wrong"],
            "fp32_wrong_fixed_right": metrics["fp32_wrong_fixed_right"],
            "calibration_top1_pct": calibration_top1,
            "samples": metrics["samples"],
            "arithmetic": ARITHMETIC,
            "wraparound": False,
            "wide_accumulator": False,
            "accumulator_bits": ell,
            "host_product_bits_max": 2 * ell,
            "elapsed_seconds": metrics["elapsed_seconds"],
            "device": str(device),
            "status": "ok",
            "error": "",
            **fixed.fp.stats(),
        }
        _upsert_csv(final_path, [row], ["model", "ell"])
        print(
            f"FINAL {name}, ell={ell}, s={s}: "
            f"Top-1={metrics['top1_pct']:.3f}% "
            f"({row['top1_delta_pp']:+.3f} pp), "
            f"Top-5={metrics['top5_pct']:.3f}% "
            f"({row['top5_delta_pp']:+.3f} pp)",
            flush=True,
        )
        del fixed
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", required=True,
                        help="extracted ImageNet-1K root containing train/ and val/")
    parser.add_argument("--results-dir", default="results/e0_q_fixed_imagenet1k")
    parser.add_argument("--models", default="resnet50,resnet101,resnet152")
    parser.add_argument("--ell", default="20,24,32")
    parser.add_argument("--calibration-per-class", type=int, default=1)
    parser.add_argument("--calibration-seed", type=int, default=20260910)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--phase", choices=("scan", "validate", "all"), default="all")
    parser.add_argument("--allow-nonstandard-counts", action="store_true",
                        help="development only; formal runs require official split counts")
    args = parser.parse_args()

    models = tuple(part.strip().lower() for part in args.models.split(",") if part.strip())
    ell_grid = tuple(int(part) for part in args.ell.split(",") if part.strip())
    if not models or any(model not in MODEL_SPECS for model in models):
        raise ValueError(f"models must be chosen from {sorted(MODEL_SPECS)}")
    if not ell_grid or any(ell < 2 or ell > 32 for ell in ell_grid):
        raise ValueError("ell values must lie in [2, 32]")
    if tuple(sorted(set(ell_grid))) != ell_grid:
        raise ValueError("ell values must be unique and increasing")

    device_name = ("cuda" if torch.cuda.is_available() else "cpu") \
        if args.device == "auto" else args.device
    device = torch.device(device_name)
    if device.type != "cuda":
        raise RuntimeError("the formal strict ImageNet-1K experiment requires CUDA")

    results_dir = Path(args.results_dir).resolve()
    results_dir.mkdir(parents=True, exist_ok=True)
    _, first_weights = build_model(models[0])
    (calibration_loader, validation_loader, train_dataset, calibration_indices,
     train_dir, val_dir) = load_imagenet(
        Path(args.data_root).resolve(), first_weights.transforms(),
        args.calibration_per_class, args.calibration_seed,
        args.batch_size, args.workers, args.allow_nonstandard_counts,
    )
    config = {
        "arithmetic": ARITHMETIC,
        "dataset": "ImageNet-1K",
        "train_dir": str(train_dir.resolve()),
        "val_dir": str(val_dir.resolve()),
        "train_images": len(train_dataset),
        "validation_images": len(validation_loader.dataset),
        "classes": len(train_dataset.classes),
        "calibration_per_class": args.calibration_per_class,
        "calibration_images": len(calibration_indices),
        "calibration_seed": args.calibration_seed,
        "models": list(models),
        "ell": list(ell_grid),
        "weights": "torchvision IMAGENET1K_V1",
        "selection_metric": "Top-1 accuracy; smaller s breaks ties",
    }
    _write_or_check_config(results_dir / "run_config.json", config)
    write_calibration_manifest(
        results_dir / "calibration_manifest.csv", train_dataset,
        calibration_indices,
    )
    print(json.dumps(config, indent=2), flush=True)

    for model_name in models:
        run_model(
            model_name, calibration_loader, validation_loader, ell_grid,
            results_dir, device, args.phase,
        )


if __name__ == "__main__":
    main()
