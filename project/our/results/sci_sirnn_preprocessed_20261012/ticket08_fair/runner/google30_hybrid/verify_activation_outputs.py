#!/usr/bin/env python3
"""Verify existing Google-30 step diagnostics against the f12/q10 LUT oracle."""

import argparse
import json
import math
from pathlib import Path
import sys


OPERATORS = (
    ("Sigmoid", "tmp27", "tmp28", 14),
    ("TanH", "tmp29", "tmp30", 7),
)
DELTA = tuple(math.floor(4096 / (1 + math.exp(i / 64))) for i in range(1024))


def sigma_q12(x):
    # The half-open clipping interval ends at 16*4096 - 1.
    index = min(abs(x), 65535) >> 6
    delta = DELTA[index]
    return 4096 - delta if x > 0 else delta


def oracle(name, x_q7):
    x_q12 = x_q7 * 32
    if name == "Sigmoid":
        return sigma_q12(x_q12) * 4
    # Python // implements signed arithmetic floor, including negative outputs.
    return (2 * sigma_q12(2 * x_q12) - 4096) // 32


def read_outputs(path):
    data = json.loads(path.read_text())
    if not isinstance(data, dict) or not data:
        raise ValueError(f"{path}: expected nonempty fixture object")
    if any(not isinstance(key, str) or not key.isdecimal() for key in data):
        raise ValueError(f"{path}: fixture keys must be decimal indices")
    keys = sorted(data, key=int)
    if keys != [str(i) for i in range(len(keys))]:
        raise ValueError(f"{path}: fixture indices must be contiguous from zero")
    for case, tensors in data.items():
        if not isinstance(tensors, dict):
            raise ValueError(f"{path}: fixture {case} is not a tensor object")
        for _, input_name, output_name, _ in OPERATORS:
            for name in (input_name, output_name):
                values = tensors.get(name)
                if not isinstance(values, list) or len(values) != 100:
                    raise ValueError(f"{path}: fixture {case}/{name} must have 100 integers")
                if any(type(v) is not int or not -32768 <= v <= 32767 for v in values):
                    raise ValueError(f"{path}: fixture {case}/{name} is not signed int16")
    return data


def verify(directory):
    ours_path = directory / "SCI_ours_integer_outputs.json"
    sci_path = directory / "SCI_integer_outputs.json"
    ours, sci = read_outputs(ours_path), read_outputs(sci_path)
    if ours.keys() != sci.keys():
        raise ValueError("SCI and SCI + ours fixture indices differ")
    fixture_path = directory / "fixture" / "fixtures.json"
    fixture = json.loads(fixture_path.read_text())
    names = fixture.get("case_names")
    if fixture.get("batch") != 1 or not isinstance(names, list) or len(names) != len(ours):
        raise ValueError("expected Batch=1 metadata with one name per diagnostic fixture")
    report = {
        "passed": True,
        "scope": "public Batch=1 diagnostics; no secure performance rerun",
        "sources": {"ours": str(ours_path.resolve()), "sci": str(sci_path.resolve()),
                    "fixtures": str(fixture_path.resolve())},
        "contract": {
            "input_fractional_bits": 7, "internal_fractional_bits": 12,
            "lut_index_bits": 10, "lut_entries": 1024,
            "delta": "floor(4096 / (1 + exp(index / 64)))",
            "index": "min(abs(x_Q12), 65535) >> 6",
            "sigmoid": "sigma(x_Q7 * 32) * 4 (Q14 output)",
            "tanh": "floor((2 * sigma(2 * x_Q7 * 32) - 4096) / 32) (Q7 output)",
        },
        "fixtures": [],
        "summary": {"elements": 0, "oracle_equal": 0, "oracle_mismatches": 0,
                    "preactivation_mismatches": 0, "sci_equal": 0},
    }
    for case in sorted(ours, key=int):
        row = {"fixture": int(case), "name": names[int(case)], "operators": []}
        for name, input_name, output_name, frac in OPERATORS:
            inputs = ours[case][input_name]
            actual, baseline = ours[case][output_name], sci[case][output_name]
            expected = [oracle(name, x) for x in inputs]
            mismatches = [i for i, (a, b) in enumerate(zip(actual, expected)) if a != b]
            pre_mismatches = sum(a != b for a, b in zip(inputs, sci[case][input_name]))
            equal = sum(a == b for a, b in zip(actual, baseline))
            max_delta = max(abs(a - b) for a, b in zip(actual, baseline))
            row["operators"].append({
                "operator": name, "input_tensor": input_name, "output_tensor": output_name,
                "elements": len(actual), "oracle_equal": len(actual) - len(mismatches),
                "oracle_mismatches": len(mismatches), "preactivation_mismatches": pre_mismatches,
                "input_q7_range": [min(inputs), max(inputs)], "sci_equal": equal,
                "sci_max_integer_delta": max_delta, "output_fractional_bits": frac,
                "sci_max_real_delta": max_delta / (1 << frac),
                "oracle_mismatch_examples": [
                    {"index": i, "input_q7": inputs[i], "actual": actual[i], "oracle": expected[i]}
                    for i in mismatches[:8]],
            })
            totals = report["summary"]
            totals["elements"] += len(actual)
            totals["oracle_equal"] += len(actual) - len(mismatches)
            totals["oracle_mismatches"] += len(mismatches)
            totals["preactivation_mismatches"] += pre_mismatches
            totals["sci_equal"] += equal
        report["fixtures"].append(row)
    report["passed"] = (report["summary"]["oracle_mismatches"] == 0
                        and report["summary"]["preactivation_mismatches"] == 0)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--directory", required=True, type=Path,
                        help="existing Batch=1 result directory")
    parser.add_argument("--output", type=Path,
                        help="defaults to <directory>/oracle_check.json")
    args = parser.parse_args()
    try:
        report = verify(args.directory)
        output = args.output or args.directory / "oracle_check.json"
        output.write_text(json.dumps(report, indent=2) + "\n")
    except (ValueError, OSError, KeyError, TypeError) as error:
        print(f"verification failed: {error}", file=sys.stderr)
        return 2
    print(json.dumps({"passed": report["passed"], "output": str(output), **report["summary"]}))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    sys.exit(main())
