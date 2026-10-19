#!/usr/bin/env python3
"""Verify resident activation modes through the existing two-party timestep."""
import argparse
import json
from pathlib import Path
import subprocess
import sys

HERE = Path(__file__).resolve().parent
BACKENDS = ("SCI", "SCI_ours", "SCI_Beaver", "SCI_Beaver_segmented")


def tensors(path):
    return [line for line in path.read_text().splitlines() if line.startswith("TENSOR ")]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--binary", type=Path, required=True)
    parser.add_argument("--fixture", type=Path, required=True)
    parser.add_argument("--hidden", type=Path, required=True)
    parser.add_argument("--reference-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    reports = []
    for mode in ("default", "nonzero"):
        reference = tensors(args.reference_dir / mode / "SCI-n4-t4/party2.log")
        if len(reference) != 15:
            raise ValueError("expected the frozen native one-step diagnostic")
        for backend in BACKENDS:
            for workers in (1, 4):
                output = args.output / mode / (backend + "-a" + str(workers))
                output.parent.mkdir(exist_ok=True)
                command = [sys.executable, str(HERE / "run.py"), "--binary", str(args.binary),
                    "--fixture", str(args.fixture), "--output", str(output), "--backend", backend,
                    "--steps", "1", "--batch", "128", "--queries", "2", "--diagnostic",
                    "--activation-workers", str(workers), "--cggm-threads", "4",
                    "--server-cpus", "0,2,4,10", "--client-cpus", "6,8,12,16", "--timeout", "600"]
                if mode == "nonzero":
                    command += ["--hidden", str(args.hidden)]
                child = subprocess.run(command, text=True, capture_output=True)
                (output / "verification_command.json").write_text(json.dumps(command, indent=2) + "\n")
                if child.returncode:
                    raise RuntimeError(child.stdout + child.stderr)
                actual = tensors(output / "party2.log")
                if actual != reference * 2:
                    raise ValueError("native integer mismatch in " + str(output))
                report = {"backend": backend, "hidden": mode, "activation_workers": workers,
                    "queries": 2, "tensors_per_query": 15, "integers_per_query": 172928,
                    "status": "passed", "result": str(output / "result.json")}
                reports.append(report)
                (args.output / "summary.json").write_text(json.dumps({"rows": reports,
                    "scope": "diagnostic correctness only; no formal timing"}, indent=2) + "\n")
                print(json.dumps(report), flush=True)


if __name__ == "__main__":
    main()
