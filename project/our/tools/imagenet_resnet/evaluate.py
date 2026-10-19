"""Resumable full ImageNet validation of the exact integer graph and FP32."""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Subset
from torchvision.datasets import ImageFolder

from model import ARITHMETIC, DEFAULT_FRAC, IntegerResNet, MODEL_SPECS, file_digest, load_reference
from width_stats import merge_counts


PRUNING_RULE = "exact_top1_upper_bound_ties_choose_smaller_frac_v1"
_STOP_REQUESTED = False


def request_stop(signum, frame):
    global _STOP_REQUESTED
    _STOP_REQUESTED = True


def check_stop():
    if _STOP_REQUESTED:
        raise SystemExit("width scan stop requested")


def pruning_certificate(processed, correct, requested, incumbent):
    remaining = requested - processed
    if incumbent is None or not 0 < processed < requested or correct + remaining > incumbent["correct"]:
        return None
    return dict(rule=PRUNING_RULE, processed=processed, correct=correct, remaining=remaining,
                upper_bound=correct + remaining, incumbent=incumbent)


def _write_json(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def _save_checkpoint(path, state):
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(state, str(temporary))
    temporary.replace(path)


def _initial_state(identity):
    return dict(identity=identity, processed=0, elapsed_seconds=0.0, status="running", reason=None,
                fp32_top1_correct=0, fp32_top5_correct=0,
                integer_top1_correct=0, integer_top5_correct=0,
                top1_agree=0, fp32_right_integer_wrong=0, fp32_wrong_integer_right=0,
                logit_abs_sum=0.0, logit_squared_sum=0.0, logit_abs_max=0.0,
                logit_values=0, fp32_predictions=[], integer_predictions=[])


def _summary(state, complete):
    count = state["processed"]
    result = dict(state["identity"])
    result.update(samples=count, complete=complete, elapsed_seconds=state["elapsed_seconds"],
                  arithmetic=ARITHMETIC, status=state["status"], integer_top1_correct=state["integer_top1_correct"])
    if "pruning_certificate" in state:
        result["pruning_certificate"] = state["pruning_certificate"]
    for source in ("fp32", "integer"):
        for k in (1, 5):
            result["%s_top%d_pct" % (source, k)] = (
                100.0 * state["%s_top%d_correct" % (source, k)] / count if count else None)
    result.update(top1_agreement_pct=100.0 * state["top1_agree"] / count if count else None,
                  fp32_right_integer_wrong=state["fp32_right_integer_wrong"],
                  fp32_wrong_integer_right=state["fp32_wrong_integer_right"])
    n = state["logit_values"]
    result.update(logit_mae=state["logit_abs_sum"] / n if n else None,
                  logit_rmse=(state["logit_squared_sum"] / n) ** 0.5 if n else None,
                  logit_max_abs_error=state["logit_abs_max"] if n else None)
    if "overflow_counts" in state:
        result["overflow_counts"] = state["overflow_counts"]
    return result


def accuracy_summary(state):
    count = state["processed"]
    complete = count == state["identity"]["requested_samples"]
    flat = _summary(state, complete)
    formal = (complete and count == 50000 and not state["identity"]["diagnostic"]
              and state.get("status") == "complete")
    result = dict(status=state.get("status", "complete" if complete else "running"),
                reason=state.get("reason"), model=flat["model"],
                bits=flat["bits"], frac=flat["frac"], samples=count, expected_samples=50000,
                evidence_scope="formal" if formal else "diagnostic",
                fp32={"top1_pct": flat["fp32_top1_pct"], "top5_pct": flat["fp32_top5_pct"]},
                fixed={"top1_pct": flat["integer_top1_pct"], "top5_pct": flat["integer_top5_pct"]},
                top1_agreement_pct=flat["top1_agreement_pct"],
                fp32_right_fixed_wrong=flat["fp32_right_integer_wrong"],
                fp32_wrong_fixed_right=flat["fp32_wrong_integer_right"],
                logit_mae=flat["logit_mae"], logit_rmse=flat["logit_rmse"],
                logit_max_abs_error=flat["logit_max_abs_error"],
                elapsed_seconds=flat["elapsed_seconds"], arithmetic=ARITHMETIC)
    if state["identity"].get("split") == "calibration":
        result.update(evidence_scope="calibration", expected_samples=state["identity"]["requested_samples"])
    if state["status"] == "pruned":
        result.update(evidence_scope="calibration_pruned", pruning_certificate=state["pruning_certificate"])
    if "overflow_counts" in state:
        result["overflow_counts"] = state["overflow_counts"]
    return result


@torch.inference_mode()
def evaluate(args):
    check_stop()
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    reference, transform = load_reference(args.model, args.weights_path)
    calibration_manifest = getattr(args, "calibration_manifest", None)
    if calibration_manifest:
        from calibration import ManifestDataset
        if args.limit is not None:
            raise ValueError("calibration uses the entire frozen manifest; do not use --limit")
        dataset = ManifestDataset(calibration_manifest, transform, args.val_dir)
    else:
        dataset = ImageFolder(str(args.val_dir), transform=transform)
    class_map = args.val_dir.parent / "class_to_idx.json"
    if class_map.exists() and json.loads(class_map.read_text()) != dataset.class_to_idx:
        raise ValueError("ImageFolder class mapping disagrees with data/class_to_idx.json")
    if not calibration_manifest and args.limit is None and (len(dataset) != 50000 or len(dataset.classes) != 1000):
        raise ValueError("full evaluation requires 50000 validation images in 1000 class directories")
    count = len(dataset) if args.limit is None else min(args.limit, len(dataset))
    if count < 1:
        raise ValueError("evaluation needs at least one image")
    frac = DEFAULT_FRAC[args.bits] if args.frac is None else args.frac
    output = args.output_dir
    output.mkdir(parents=True, exist_ok=True)
    collect_overflow = getattr(args, "collect_overflow", False)
    integer_kwargs = {"collect_overflow": True} if collect_overflow else {}
    integer = IntegerResNet(reference, args.model, args.bits, frac, args.batch_size, **integer_kwargs)
    graph_path, parameters_path = (None, None)
    if not calibration_manifest:
        graph_path, parameters_path = integer.export(output / "export")
    identity = dict(model=args.model, bits=args.bits, frac=frac,
                    weights=reference._weights_identity, graph_sha256=file_digest(graph_path) if graph_path else None,
                    parameters_sha256=file_digest(parameters_path) if parameters_path else None, arithmetic=ARITHMETIC,
                    numeric_contract={"dot": ARITHMETIC, "residual": "add_mod_2^bits",
                                      "relu": "max_signed_code_zero",
                                      "maxpool": "signed_max_padding_signed_min",
                                      "avgpool": "sum_mod_2^bits_then_signed_floor_div49",
                                      "encoding": "round_ties_even_wrap",
                                      "bn_folding": "float64_before_quantization"},
                    val_dir=str(args.val_dir.resolve()),
                    dataset_samples=len(dataset), dataset_classes=dataset.classes,
                    requested_samples=count, diagnostic=(args.limit is not None),
                    input_transform=repr(transform), encoding="round_ties_even_wrap",
                    topk_tie_break="class_index_ascending",
                    first_sample=dataset.samples[0][0], last_sample=dataset.samples[count - 1][0])
    if calibration_manifest:
        identity.update(split="calibration", calibration_manifest_sha256=file_digest(calibration_manifest),
                        calibration_manifest=str(Path(calibration_manifest).resolve()), diagnostic=True)
    incumbent = getattr(args, "prune_incumbent", None)
    if incumbent is not None:
        if (not calibration_manifest or incumbent.get("samples") != count
                or type(incumbent.get("correct")) is not int or not 0 <= incumbent["correct"] <= count
                or type(incumbent.get("frac")) is not int or not 0 <= incumbent["frac"] < frac):
            raise ValueError("pruning requires a complete earlier calibration incumbent")
        identity["calibration_pruning"] = dict(rule=PRUNING_RULE, batch_size=args.batch_size, incumbent=incumbent)
    if collect_overflow:
        identity["overflow_measurement"] = "exact_signed_boundaries_v1"
    checkpoint = output / "checkpoint.pt"
    state = _initial_state(identity)
    if collect_overflow:
        state["overflow_counts"] = integer.overflow.snapshot(reset=True)
    if checkpoint.exists() and not args.restart:
        state = torch.load(str(checkpoint), map_location="cpu", weights_only=False)
        if state["identity"] != identity:
            raise ValueError("checkpoint configuration/dataset differs; choose another output directory or --restart")
    if not 0 <= state["processed"] <= count:
        raise ValueError("checkpoint processed count is invalid")
    if state["status"] == "pruned":
        expected = pruning_certificate(state["processed"], state["integer_top1_correct"], count, incumbent)
        if expected is None or state.get("pruning_certificate") != expected:
            raise ValueError("checkpoint pruning certificate is invalid")
    if (state["processed"] == count and state["status"] == "complete") or state["status"] == "pruned":
        summary = _summary(state, state["processed"] == count)
        _write_json(output / "metrics.json", summary)
        _write_json(output / "accuracy.json", accuracy_summary(state))
        return summary
    state.update(status="running", reason=None)
    if device.type == "cuda":
        integer.release_weight_codes()
    reference, integer = reference.to(device), integer.to(device)
    loader = DataLoader(Subset(dataset, range(state["processed"], count)),
                        batch_size=args.batch_size, shuffle=False, num_workers=args.workers,
                        pin_memory=device.type == "cuda")
    last_saved = state["processed"]
    start = time.monotonic()
    previous_elapsed = state["elapsed_seconds"]
    try:
        for inputs, labels in loader:
            inputs, labels = inputs.to(device, non_blocking=True), labels.to(device, non_blocking=True)
            fp_logits = reference(inputs)
            codes = integer(inputs)
            fp_top5 = torch.argsort(fp_logits, dim=1, descending=True, stable=True)[:, :5]
            q_top5 = torch.argsort(codes, dim=1, descending=True, stable=True)[:, :5]
            fp_top1, q_top1 = fp_top5[:, 0], q_top5[:, 0]
            fp_correct, q_correct = fp_top1.eq(labels), q_top1.eq(labels)
            batch_stats = {}
            for source, correct, top5 in (("fp32", fp_correct, fp_top5), ("integer", q_correct, q_top5)):
                batch_stats[source + "_top1_correct"] = int(correct.sum().item())
                batch_stats[source + "_top5_correct"] = int(top5.eq(labels[:, None]).any(dim=1).sum().item())
            batch_stats["top1_agree"] = int(q_top1.eq(fp_top1).sum().item())
            batch_stats["fp32_right_integer_wrong"] = int((fp_correct & ~q_correct).sum().item())
            batch_stats["fp32_wrong_integer_right"] = int((~fp_correct & q_correct).sum().item())
            difference = codes.double() / (1 << frac) - fp_logits.double()
            batch_stats["logit_abs_sum"] = float(difference.abs().sum().item())
            batch_stats["logit_squared_sum"] = float(difference.square().sum().item())
            batch_abs_max = float(difference.abs().max().item())
            batch_stats["logit_values"] = difference.numel()
            fp_predictions, q_predictions = fp_top1.cpu().tolist(), q_top1.cpu().tolist()
            overflow_counts = integer.overflow.snapshot(reset=True) if collect_overflow else None
            # Commit only after the entire paired batch completed successfully.
            for key, value in batch_stats.items():
                state[key] += value
            state["logit_abs_max"] = max(state["logit_abs_max"], batch_abs_max)
            state["fp32_predictions"].extend(fp_predictions)
            state["integer_predictions"].extend(q_predictions)
            if overflow_counts is not None:
                merge_counts(state["overflow_counts"], overflow_counts)
            state["processed"] += labels.numel()
            state["status"] = "complete" if state["processed"] == count else "running"
            state["elapsed_seconds"] = previous_elapsed + time.monotonic() - start
            check_stop()
            certificate = pruning_certificate(state["processed"], state["integer_top1_correct"], count, incumbent)
            if certificate is not None:
                state.update(status="pruned", pruning_certificate=certificate)
                break
            if state["processed"] - last_saved >= args.checkpoint_every or state["processed"] == count:
                _save_checkpoint(checkpoint, state)
                _write_json(output / "metrics.json", _summary(state, state["processed"] == count))
                _write_json(output / "accuracy.json", accuracy_summary(state))
                last_saved = state["processed"]
                print(json.dumps({"processed": state["processed"], "total": count,
                                  "integer_top1_pct": 100.0 * state["integer_top1_correct"] / state["processed"]}), flush=True)
    except BaseException as error:
        state["status"] = "interrupted" if isinstance(error, (KeyboardInterrupt, SystemExit)) else "failed"
        state["reason"] = "%s: %s" % (type(error).__name__, error)
        raise
    finally:
        if state["processed"] == count and state["status"] == "running":
            state["status"] = "complete"
        state["elapsed_seconds"] = previous_elapsed + time.monotonic() - start
        _save_checkpoint(checkpoint, state)
        _write_json(output / "metrics.json", _summary(state, state["processed"] == count))
        _write_json(output / "accuracy.json", accuracy_summary(state))
    summary = _summary(state, state["processed"] == count)
    print(json.dumps({key: summary[key] for key in ("model", "bits", "frac", "samples", "complete", "status",
                                                   "elapsed_seconds", "integer_top1_pct")}), flush=True)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=sorted(MODEL_SPECS), default="resnet50")
    parser.add_argument("--bits", type=int, choices=sorted(DEFAULT_FRAC), default=32)
    parser.add_argument("--frac", type=int)
    parser.add_argument("--val-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--weights-path", type=Path)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--checkpoint-every", type=int, default=10)
    parser.add_argument("--limit", type=int, help="diagnostic subset only; omit for formal 50000-image validation")
    parser.add_argument("--restart", action="store_true")
    parser.add_argument("--calibration-manifest", type=Path)
    parser.add_argument("--collect-overflow", action="store_true")
    args = parser.parse_args()
    if args.batch_size < 1 or args.checkpoint_every < 1:
        parser.error("batch-size and checkpoint-every must be positive")
    evaluate(args)


if __name__ == "__main__":
    main()
