"""Measure one real-image diagnostic batch before launching the width scan."""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import torch

from evaluate import _write_json, evaluate


def run(args):
    if (args.output_dir / "evaluation" / "checkpoint.pt").exists():
        raise FileExistsError("choose a fresh diagnostic directory for an actual measured batch")
    if not torch.cuda.is_available():
        raise RuntimeError("this diagnostic requires CUDA")
    torch.set_num_threads(4)
    torch.cuda.reset_peak_memory_stats()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    evaluation = argparse.Namespace(model=args.model, bits=args.bits, frac=args.frac,
        val_dir=args.val_dir, output_dir=args.output_dir / "evaluation", weights_path=None,
        device="cuda", batch_size=args.batch_size, workers=args.workers,
        checkpoint_every=args.batch_size, limit=args.batch_size, restart=False,
        calibration_manifest=None, collect_overflow=True)
    start = time.monotonic()
    result = dict(status="running", evidence_scope="diagnostic", model=args.model,
                  bits=args.bits, frac=args.frac, batch_size=args.batch_size,
                  includes_model_construction=True, includes_exact_overflow_counts=True)
    try:
        accuracy = evaluate(evaluation)
        torch.cuda.synchronize()
        result.update(status="complete", samples=accuracy["samples"],
                      evaluation_seconds=accuracy["elapsed_seconds"],
                      fixed_top1_pct=accuracy["integer_top1_pct"])
    except BaseException as error:
        result.update(status="failed", reason="%s: %s" % (type(error).__name__, error))
        raise
    finally:
        result.update(total_seconds=time.monotonic() - start,
                      peak_allocated_bytes=torch.cuda.max_memory_allocated(),
                      peak_reserved_bytes=torch.cuda.max_memory_reserved(),
                      device=torch.cuda.get_device_name())
        _write_json(args.output_dir / "diagnostic.json", result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=("resnet50", "resnet152"), default="resnet152")
    parser.add_argument("--bits", type=int, choices=(16, 20, 24, 28, 32), default=28)
    parser.add_argument("--frac", type=int, default=27)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--val-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.batch_size < 1 or args.workers < 0 or not 0 <= args.frac < args.bits:
        parser.error("invalid batch, worker or fixed-point format")
    run(args)


if __name__ == "__main__":
    main()
