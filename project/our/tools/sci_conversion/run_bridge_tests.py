#!/usr/bin/env python3
"""Run spooled conversion API literals on both real protocol parties."""
import argparse
import json
import os
from pathlib import Path
import subprocess

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--binary", type=Path, required=True)
parser.add_argument("--output", type=Path, required=True)
args = parser.parse_args()
args.output.mkdir(parents=True, exist_ok=False)
rows = []
multiple = bool(os.environ.get("SEGFSS_TEST_MULTI_CONTEXT"))
for full in (0, 1):
    children, logs, commands = [], [], []
    try:
        for role in (1, 2):
            command = [str(args.binary.resolve()), str(role), str(37340 + full * 200), str(full)]
            log = (args.output / ("full%d-party%d.log" % (full, role))).open("w")
            logs.append(log); commands.append(command)
            children.append(subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT,
                env=dict(os.environ, OMP_NUM_THREADS="1")))
        exits = [child.wait(timeout=180) for child in children]
    finally:
        for child in children:
            if child.poll() is None: child.kill()
        for child in children: child.wait()
        for log in logs: log.close()
    rows.append({"backend": "full_tree" if full else "segmented", "commands": commands,
        "party_exit_codes": exits, "values": 144 if multiple else 48,
        "one_use_materials": 24 if multiple else 8,
        "independent_contexts": 2 if multiple else 1,
        "status": "passed" if exits == [0, 0] else "failed"})
report = {"status": "passed" if all(row["status"] == "passed" for row in rows) else "failed",
    "scope": "C ABI spool, literal sign extension/toward-zero/floor/reduction, fresh queries and exhaustion",
    "runs": rows}
(args.output / "summary.json").write_text(json.dumps(report, indent=2) + "\n")
print(json.dumps(report, indent=2))
raise SystemExit(0 if report["status"] == "passed" else 1)
