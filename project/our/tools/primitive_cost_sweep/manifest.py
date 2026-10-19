#!/usr/bin/env python3
"""Frozen SDCF-10 primitive sweep matrix and result-shape validation."""

from __future__ import annotations

import copy
import json
from pathlib import Path

MEMORY_BYTES = int(1.5 * (1 << 30))
TIMEOUT_SECONDS = 30
PRIMARY_NSEGS = (1, 2, 3, 4, 5, 8, 16)
NATIVE_BITS = (1, 3, 4, 5, 6, 7, 8, 9, 11, 14, 15, 23, 31)
REPRESENTATIVE_BITS = (16, 24, 32)


def _row(row_id, family, *, kind="dcf", backend="segmented", bits=None,
         comparison_bits=None, nseg=1, count=128, shift=None,
         operation=None, signed=None, numeric_mode=None, candidate_nseg=()):
    if bits is None:
        bits = comparison_bits
    return {
        "id": row_id,
        "family": family,
        "kind": kind,
        "backend": backend,
        "bits": bits,
        "comparison_bits": comparison_bits,
        "nseg": nseg,
        "count": count,
        "shift": shift,
        "operation": operation,
        "signed": signed,
        "numeric_mode": numeric_mode,
        "candidate_nseg": list(candidate_nseg),
        "memory_limit_bytes": MEMORY_BYTES,
        "timeout_seconds": TIMEOUT_SECONDS,
        "status": "pending",
        "reason": None,
        "measurements": None,
    }


def build_manifest():
    rows = []
    # A: primary DCF scan.  For 16 bits, nseg=bits duplicates nseg=16.
    row_id = 1
    for bits in (16, 20, 24, 32):
        candidates = PRIMARY_NSEGS + ((bits,) if bits not in PRIMARY_NSEGS else ())
        for nseg in dict.fromkeys(candidates):
            rows.append(_row(f"A-{row_id:03d}", "A",
                             backend="full_tree" if nseg == 1 else "segmented",
                             bits=bits, comparison_bits=bits, nseg=nseg,
                             candidate_nseg=candidates))
            row_id += 1

    # B: domains observed in the frozen native graph, excluding A's 16/24/32
    # rows.  nseg=1 and min(4,bits) are both attempted, with the one-bit
    # duplicate removed.
    row_id = 1
    for bits in NATIVE_BITS:
        candidates = (1, min(4, bits))
        for nseg in dict.fromkeys(candidates):
            rows.append(_row(f"B-{row_id:03d}", "B",
                             backend="full_tree" if nseg == 1 else "segmented",
                             bits=bits, comparison_bits=bits, nseg=nseg,
                             candidate_nseg=candidates))
            row_id += 1

    # C: conversion supplements.  Each operation is measured through the
    # same adapter with both full-tree and segmented policies.  nseg=1 is kept
    # explicitly for segmented so its cost remains comparable to full-tree.
    row_id = 1
    for input_bits, output_bits, shift, signed, operation, label in (
        (64, 32, 32, False, "truncate_and_reduce", "u64_to_u32_s32"),
        (32, 32, 14, True, "div_pow2", "s32_to_s32_s14"),
    ):
        candidates = (1, 2, 4, 8, 16)
        for backend, nseg in (("full_tree", 1), ("segmented", 1),
                              ("segmented", 2), ("segmented", 4),
                              ("segmented", 8), ("segmented", 16)):
            rows.append(_row(f"C-{row_id:03d}", "C", kind="conversion",
                             backend=backend, bits=input_bits,
                             comparison_bits=shift, nseg=nseg, count=128,
                             shift=shift, operation=operation, signed=signed,
                             numeric_mode="sci_compatible",
                             candidate_nseg=candidates))
            row_id += 1
        # Keep the labels available to evidence consumers without changing
        # the frozen row identity; the operation/rings are authoritative.
        for row in rows[-6:]:
            row["label"] = label

    # D: three larger batched DCF representatives.  These are the only rows
    # that exercise the 12,800-element batch used by prior regression evidence.
    for row_id, bits in enumerate(REPRESENTATIVE_BITS, 1):
        rows.append(_row(f"D-{row_id:03d}", "D", bits=bits,
                         comparison_bits=bits, nseg=4, count=12800,
                         candidate_nseg=(1, 2, 4, 8, 16)))
    return rows


def validate_manifest(rows):
    if not isinstance(rows, list):
        raise ValueError("manifest must be a list")
    ids = set()
    for row in rows:
        required = {"id", "family", "kind", "backend", "bits", "comparison_bits",
                    "nseg", "count", "candidate_nseg", "memory_limit_bytes",
                    "timeout_seconds", "status", "measurements"}
        missing = required - row.keys()
        if missing:
            raise ValueError("row missing fields: " + ",".join(sorted(missing)))
        if row["id"] in ids:
            raise ValueError("duplicate row id " + row["id"])
        ids.add(row["id"])
        if row["family"] not in {"A", "B", "C", "D"}:
            raise ValueError("unknown row family")
        if row["kind"] == "dcf":
            if not 1 <= row["nseg"] <= row["comparison_bits"]:
                raise ValueError("illegal DCF nseg")
        elif row["kind"] == "conversion":
            if row["operation"] not in {"truncate_and_reduce", "div_pow2"}:
                raise ValueError("unknown conversion operation")
            if row["backend"] == "full_tree" and row["nseg"] != 1:
                raise ValueError("full-tree conversion requires nseg=1")
        else:
            raise ValueError("unknown row kind")
        if row["memory_limit_bytes"] != MEMORY_BYTES or row["timeout_seconds"] != TIMEOUT_SECONDS:
            raise ValueError("resource budget drift")
        if row["status"] in {"unsupported", "timeout", "oom", "unavailable", "failed", "pending"}:
            if row["measurements"] is not None and row["status"] != "complete":
                raise ValueError("non-complete row must retain null measurements")
        elif row["status"] == "complete":
            if row["measurements"] is None:
                raise ValueError("complete row requires measurements")
        else:
            raise ValueError("unknown row status")
    return True


def write_manifest(path: Path, rows=None):
    rows = copy.deepcopy(build_manifest() if rows is None else rows)
    validate_manifest(rows)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(rows, indent=2, sort_keys=True) + "\n")
    return rows


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    write_manifest(args.output)
