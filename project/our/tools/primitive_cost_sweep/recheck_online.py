#!/usr/bin/env python3
"""Recheck local evaluation and scan costs with explicit cGGM worker limits."""

import argparse
import fcntl
import json
import os
import statistics
import subprocess
from pathlib import Path

from runner import run_pair


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--binary", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    rows = []
    for count, candidates in ((128, (1, 2, 4)), (12800, (4,))):
        for nseg in candidates:
            for threads in (1, 4):
                rows.append({
                    "id": f"b{count}-n{nseg}-t{threads}", "kind": "dcf",
                    "backend": "full_tree" if nseg == 1 else "segmented",
                    "bits": 16, "comparison_bits": 16, "count": count,
                    "nseg": nseg, "cggm_threads": threads, "runs": 10,
                    "timeout_seconds": 60,
                })
    repo = Path(__file__).resolve().parents[4]
    manifest = {
        "source_commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=repo, text=True).strip(),
        "cpu_affinity": sorted(os.sched_getaffinity(0)),
        "network": "localhost TCP, no injected delay or bandwidth limit",
        "warmup_runs": 0, "rows": rows,
    }
    (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    results = []
    with Path("/tmp/sdcf-measurement.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        for index, row in enumerate(rows):
            result = run_pair(row, args.output / row["id"], args.binary,
                              timeout=row["timeout_seconds"], port=33600 + index * 4)
            if result["measurements"]:
                parties = result["measurements"]["parties"]
                for field in ("tree_evaluation_seconds", "scan_seconds", "scan_layers"):
                    values = [max(a[field], b[field]) for a, b in
                              zip(parties[0]["samples"], parties[1]["samples"])]
                    result["measurements"][field] = statistics.median(values)
            results.append(result)
            (args.output / "results.json").write_text(json.dumps(results, indent=2) + "\n")
            print(row["id"], result["status"], flush=True)


if __name__ == "__main__":
    main()
