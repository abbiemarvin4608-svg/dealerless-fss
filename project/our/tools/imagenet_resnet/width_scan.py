"""Calibrate uniform ring widths, freeze selection, and validate selected points."""
from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path
import signal

import torch

from calibration import ManifestDataset
from evaluate import PRUNING_RULE, _write_json, evaluate, pruning_certificate, request_stop
from model import ARITHMETIC, MODEL_SPECS, file_digest, load_reference


WIDTHS = (16, 20, 24, 28)
SELECTION = "maximum calibration Top-1 correct count; ties choose smaller fractional width"


def select_candidate(rows, bits, expected_samples):
    if len(rows) != bits or {row["frac"] for row in rows} != set(range(bits)):
        raise ValueError("selection requires every legal fractional width")
    best = None
    for row in sorted(rows, key=lambda row: row["frac"]):
        correct, processed = row.get("integer_top1_correct"), row["samples"]
        if (row["bits"] != bits or type(correct) is not int or type(processed) is not int
                or not 0 <= correct <= processed <= expected_samples):
            raise ValueError("selection requires valid equal-sample calibration counts")
        if row["complete"] and processed == expected_samples and row.get("status") == "complete":
            if best is None or correct > best["integer_top1_correct"]:
                best = row
        else:
            incumbent = incumbent_record(best)
            expected = pruning_certificate(processed, correct, expected_samples, incumbent)
            policy = row.get("calibration_pruning", {})
            batch = policy.get("batch_size")
            if (row.get("status") != "pruned" or row["complete"] or expected is None
                    or row.get("pruning_certificate") != expected or policy.get("rule") != PRUNING_RULE
                    or policy.get("incumbent") != incumbent or type(batch) is not int or batch < 1
                    or processed % batch or row.get("requested_samples") != expected_samples):
                raise ValueError("selection requires complete equal-sample results or a valid pruning certificate")
    return best


def incumbent_record(row):
    return None if row is None else dict(frac=row["frac"], correct=row["integer_top1_correct"], samples=row["samples"])


def frozen_json(path, document):
    if path.exists():
        if json.loads(path.read_text()) != document:
            raise ValueError("frozen configuration differs: " + str(path))
    else:
        _write_json(path, document)


def evaluation_args(args, model, bits, frac, output, calibration=False, incumbent=None):
    return argparse.Namespace(model=model, bits=bits, frac=frac,
        val_dir=args.val_dir, output_dir=output, weights_path=None,
        device=args.device, batch_size=args.batch_size, workers=args.workers,
        checkpoint_every=args.checkpoint_every, limit=None, restart=False,
        calibration_manifest=args.calibration_manifest if calibration else None,
        collect_overflow=not calibration, prune_incumbent=incumbent if calibration else None)


def evaluate_and_release(args):
    try:
        return evaluate(args)
    finally:
        gc.collect()
        if torch.device(args.device).type == "cuda" and torch.cuda.is_initialized():
            torch.cuda.empty_cache()


def read_calibration(directory, bits, expected_samples, model, manifest_digest):
    rows = []
    for frac in range(bits):
        path = directory / ("w%d_f%d" % (bits, frac)) / "metrics.json"
        if not path.exists():
            raise ValueError("missing calibration result: " + str(path))
        row = json.loads(path.read_text())
        if (row.get("split") != "calibration" or row.get("model") != model
                or row.get("calibration_manifest_sha256") != manifest_digest
                or row.get("arithmetic") != ARITHMETIC):
            raise ValueError("candidate does not match the frozen training calibration and model")
        rows.append(row)
    return select_candidate(rows, bits, expected_samples)


def run(args):
    torch.set_num_threads(4)
    if len(set(args.models)) != len(args.models) or any(model not in MODEL_SPECS for model in args.models):
        raise ValueError("invalid or duplicate models")
    if len(set(args.widths)) != len(args.widths) or any(bits not in WIDTHS for bits in args.widths):
        raise ValueError("scan widths must be a subset of 16,20,24,28; 32/12 is a fixed anchor")
    dataset = ManifestDataset(args.calibration_manifest, validation_dir=args.val_dir)
    if len(dataset) != 2000 or dataset.manifest["per_class"] != 2:
        raise ValueError("this experiment requires 2000 training images, two per class")
    root = args.output_dir
    root.mkdir(parents=True, exist_ok=True)
    configuration = dict(schema="uniform-ring-width-scan-v2", arithmetic=ARITHMETIC,
        models=args.models, widths=args.widths, fractional_candidates="0..bits-1",
        calibration_manifest=str(args.calibration_manifest.resolve()),
        calibration_manifest_sha256=file_digest(args.calibration_manifest),
        calibration_samples=len(dataset), selection=SELECTION,
        calibration_pruning=dict(rule=PRUNING_RULE, order="increasing_frac", batch_size=args.batch_size),
        validation_dir=str(args.val_dir.resolve()), validation_samples=50000,
        fixed_anchor=dict(bits=32, frac=12, selection="previously fixed; not an optimized width-scan point"))
    frozen_json(root / "configuration.json", configuration)
    final_rows = []
    for model in args.models:
        model_dir = root / model
        model_dir.mkdir(exist_ok=True)
        calibration_dir = model_dir / "calibration"
        if args.phase in ("all", "calibrate"):
            for bits in args.widths:
                best = None
                for frac in range(bits):
                    candidate = evaluate_and_release(evaluation_args(args, model, bits, frac,
                        calibration_dir / ("w%d_f%d" % (bits, frac)), calibration=True,
                        incumbent=incumbent_record(best)))
                    if candidate["complete"] and (best is None or candidate["integer_top1_correct"] > best["integer_top1_correct"]):
                        best = candidate
        selected = []
        for bits in args.widths:
            best = read_calibration(calibration_dir, bits, len(dataset), model,
                                    configuration["calibration_manifest_sha256"])
            selected.append(dict(model=model, bits=bits, frac=best["frac"],
                                 calibration_samples=best["samples"],
                                 calibration_top1_pct=best["integer_top1_pct"],
                                 calibration_agreement_pct=best["top1_agreement_pct"]))
        selection = dict(selection=SELECTION, calibration_manifest_sha256=file_digest(args.calibration_manifest), rows=selected)
        frozen_json(model_dir / "selected.json", selection)
        if args.phase == "calibrate":
            continue
        for row in selected:
            output = model_dir / "validation" / ("w%d_f%d" % (row["bits"], row["frac"]))
            evaluate_and_release(evaluation_args(args, model, row["bits"], row["frac"], output))
            final = json.loads((output / "accuracy.json").read_text())
            final.update(role="calibration_selected", source=str(output / "accuracy.json"))
            final_rows.append(final)
            _write_json(root / "results.json", {"status": "running", "rows": final_rows})
        anchor_dir = args.anchor_dir / (model + "_w32_f12")
        anchor = json.loads((anchor_dir / "accuracy.json").read_text())
        metadata = json.loads((anchor_dir / "metrics.json").read_text())
        reference, transform = load_reference(model)
        if (anchor.get("status") != "complete" or anchor.get("samples") != 50000
                or anchor.get("evidence_scope") != "formal" or anchor.get("bits") != 32
                or anchor.get("frac") != 12 or anchor.get("model") != model
                or anchor.get("arithmetic") != ARITHMETIC
                or metadata.get("weights") != reference._weights_identity
                or metadata.get("input_transform") != repr(transform)
                or Path(metadata["val_dir"]).resolve() != args.val_dir.resolve()):
            raise ValueError("32/12 anchor does not match the frozen model, data and arithmetic")
        anchor.update(role="fixed_anchor", source=str(anchor_dir / "accuracy.json"),
                      overflow_counts_status="not measured in the historical anchor run")
        final_rows.append(anchor)
        del reference
    if args.phase != "calibrate":
        _write_json(root / "results.json", {"status": "complete", "rows": final_rows})
    return final_rows


def main():
    signal.signal(signal.SIGTERM, request_stop)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--calibration-manifest", type=Path, required=True)
    parser.add_argument("--val-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--anchor-dir", type=Path, required=True)
    parser.add_argument("--models", nargs="+", default=["resnet50", "resnet152"])
    parser.add_argument("--widths", type=int, nargs="+", default=list(WIDTHS))
    parser.add_argument("--phase", choices=("all", "calibrate", "validate"), default="all")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--checkpoint-every", type=int, default=256)
    args = parser.parse_args()
    if args.batch_size < 1 or args.workers < 0 or args.checkpoint_every < 1:
        parser.error("invalid worker, batch or checkpoint setting")
    run(args)


if __name__ == "__main__":
    main()
