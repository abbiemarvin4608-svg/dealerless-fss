#!/usr/bin/env python3
"""Reconstruct diagnostic shares locally and compare every tensor to an oracle."""
import argparse
import json
import math
from pathlib import Path

import numpy as np


def load_pair(directory, oracle_directory):
    parties = [json.loads((directory / ("party%d.json" % role)).read_text()) for role in (1, 2)]
    oracle = json.loads((oracle_directory / "index.json").read_text())
    status = json.loads((directory / "STATUS.json").read_text()) if (directory / "STATUS.json").exists() else None
    expected_queries = status["queries"] if status else len(parties[0].get("queries", []))
    if expected_queries < 1 or any(party.get("status") != "complete" or party.get("role") != role
                                   for role, party in enumerate(parties, 1)):
        raise ValueError("integer audit requires completed, nonempty paired queries")
    if any(len(party["queries"]) != expected_queries or
           [query["query"] for query in party["queries"]] != list(range(expected_queries)) for party in parties):
        raise ValueError("integer audit requires complete matching query IDs from both parties")
    if any(party.get("model") != oracle["model"] or party.get("bits") != oracle["bits"] or
           party.get("frac") != oracle["frac"] for party in parties):
        raise ValueError("integer audit model/numeric identity differs from oracle")
    return parties, oracle


def audit_output(directory, oracle_directory, tensor="fc"):
    parties, oracle = load_pair(directory, oracle_directory)
    reference = oracle["tensors"][tensor]
    expected = np.fromfile(oracle_directory / reference["file"], dtype="<u8")
    if expected.size != reference["elements"] or expected.size != math.prod(reference["shape"]):
        raise ValueError("oracle tensor length differs from declared shape: " + tensor)
    rows = []
    for query in parties[1]["queries"]:
        actual = np.asarray(query["output_codes"], dtype=np.uint64)
        if actual.shape != expected.shape:
            raise ValueError("revealed output length differs from oracle")
        differences = np.flatnonzero(actual != expected)
        rows.append(dict(query=query["query"], tensor=tensor, elements=int(expected.size),
                         mismatches=int(differences.size)))
    result = dict(status="passed" if all(row["mismatches"] == 0 for row in rows) else "failed",
                  model=oracle["model"], bits=oracle["bits"], frac=oracle["frac"],
                  evidence_scope="revealed_output_only", values_checked=sum(row["elements"] for row in rows),
                  mismatches=sum(row["mismatches"] for row in rows), tensors=rows)
    (directory / "OUTPUT_AUDIT.json").write_text(json.dumps(result, indent=2) + "\n")
    return result


def audit(directory, oracle_directory):
    parties, oracle = load_pair(directory, oracle_directory)
    mask = np.uint64((1 << oracle["bits"]) - 1)
    rows = []
    expected_names = set(oracle["tensors"]) - {"input"}
    for query_index, queries in enumerate(zip(parties[0]["queries"], parties[1]["queries"])):
        inventories = [query["diagnostic_manifest"] for query in queries]
        if isinstance(inventories[0], list):
            inventories = [{record.get("name", record.get("tensor")): record for record in records} for records in inventories]
        if any(set(records) - {"input"} != expected_names for records in inventories):
            raise ValueError("diagnostic tensor coverage differs from oracle")
        for name in sorted(expected_names):
            reference = oracle["tensors"][name]
            arrays = []
            for records in inventories:
                record = records[name]
                if record["shape"] != reference["shape"] or record["bits"] != oracle["bits"]:
                    raise ValueError("diagnostic tensor format differs from oracle: " + name)
                if record.get("sharing") != "additive_share" or record.get("layout") != "NCHW":
                    raise ValueError("expected additive shares in canonical NCHW order")
                path = Path(record["path"])
                if not path.is_absolute():
                    path = directory / path
                values = np.fromfile(path, dtype="<u8")
                arrays.append(values)
            expected = np.fromfile(oracle_directory / reference["file"], dtype="<u8")
            if expected.size != reference["elements"] or expected.size != math.prod(reference["shape"]):
                raise ValueError("oracle tensor length differs from declared shape: " + name)
            if any(value.size != expected.size for value in arrays):
                raise ValueError("diagnostic tensor length differs from oracle")
            actual = (arrays[0] + arrays[1]) & mask
            differences = np.flatnonzero(actual != expected)
            rows.append(dict(query=query_index, tensor=name, elements=int(expected.size),
                             mismatches=int(differences.size),
                             first_mismatch=int(differences[0]) if differences.size else None))
    result = dict(status="passed" if rows and all(row["mismatches"] == 0 for row in rows) else "failed",
                  model=oracle["model"], bits=oracle["bits"], frac=oracle["frac"],
                  evidence_scope="diagnostic",
                  values_checked=sum(row["elements"] for row in rows),
                  mismatches=sum(row["mismatches"] for row in rows), tensors=rows)
    (directory / "INTEGER_AUDIT.json").write_text(json.dumps(result, indent=2) + "\n")
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--directory", type=Path, required=True)
    parser.add_argument("--oracle", type=Path, required=True)
    parser.add_argument("--final-only", action="store_true", help="compare client output, without writing intermediate shares")
    parser.add_argument("--tensor", default="fc")
    args = parser.parse_args()
    result = (audit_output(args.directory, args.oracle, args.tensor) if args.final_only
              else audit(args.directory, args.oracle))
    print(json.dumps({key: value for key, value in result.items() if key != "tensors"}))
    raise SystemExit(0 if result["status"] == "passed" else 1)
