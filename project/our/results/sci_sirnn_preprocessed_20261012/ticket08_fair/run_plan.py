#!/usr/bin/env python3
"""Execute a declared, sequential set of existing two-party benchmark runners."""
import argparse
import datetime
import fcntl
import json
import os
from pathlib import Path
import statistics
import subprocess
import sys
import time


def write(path, value):
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def distribution(values):
    return dict(median=statistics.median(values), min=min(values), max=max(values),
                pstdev=statistics.pstdev(values), values=values)


def summarize_runtime(result):
    measured = result["measurements"]
    summary = {
        "setup_seconds": measured["setup"]["seconds"],
        "session_total_seconds": measured["total_seconds"],
        "first_query_seconds": measured.get("first_query", {}).get("seconds"),
        "peak_rss_kib_per_party": measured["peak_rss_kib_per_party"],
    }
    for phase in ("offline", "input", "ready", "online", "output", "query_total"):
        for field, suffix in (("seconds", "seconds"), ("sent_bytes", "bytes")):
            summary[phase + "_" + suffix] = distribution(
                [query[phase][field] for query in measured["queries"]])
    return summary


def run_plan(plan, output):
    output.mkdir(parents=True, exist_ok=False)
    ids = [row["id"] for row in plan["rows"]]
    if len(ids) != len(set(ids)) or any(Path(value).name != value for value in ids):
        raise ValueError("row identities must be unique directory names")
    rows = [dict(row, command=[part.replace("{output}", str(output / row["id"]))
                              for part in row["command"]]) for row in plan["rows"]]
    manifest = dict(plan, frozen_at=datetime.datetime.now(datetime.timezone.utc).isoformat(),
                    command_rows=rows)
    write(output / "manifest.json", manifest)
    summary = dict(manifest="manifest.json", rows=[])
    env = {key: value for key, value in os.environ.items()
           if not key.startswith(("SEGFSS_", "SDCF_"))
           and key not in ("GOOGLE30_PROXY_OFFSET", "SWEEP_CONNECT_OFFSET")}
    env["OMP_NUM_THREADS"] = "1"
    with Path("/tmp/sdcf-measurement.lock").open("a") as lock:
        for row in rows:
            fcntl.flock(lock, fcntl.LOCK_EX)
            with (output / (row["id"] + ".runner.log")).open("w") as log:
                child = subprocess.run(row["command"], stdout=log, stderr=subprocess.STDOUT,
                                       env=dict(env, **row.get("environment", {})))
            result_path = output / row["id"] / "result.json"
            result = json.loads(result_path.read_text()) if result_path.exists() else {
                "status": "failed", "reason": "runner exit %d; no result" % child.returncode}
            record = dict(id=row["id"], status=result["status"], reason=result.get("reason"),
                          result=str(result_path.relative_to(output)), measurements=None)
            if result["status"] == "complete":
                record["measurements"] = summarize_runtime(result)
            summary["rows"].append(record)
            write(output / "summary.json", summary)
            print(row["id"], result["status"], flush=True)
            fcntl.flock(lock, fcntl.LOCK_UN)
            time.sleep(0.05)
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    run_plan(json.loads(args.plan.read_text()), args.output.resolve())
