#!/usr/bin/env python3
"""Verify isolated and parallel single-step graphs against frozen SCI tensors."""
import argparse
import json
from pathlib import Path
import subprocess
import sys

HERE = Path(__file__).resolve().parent


def tensors(path):
    return [line for line in path.read_text().splitlines() if line.startswith("TENSOR ")]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--binary", type=Path, required=True)
    parser.add_argument("--fixture", type=Path, required=True)
    parser.add_argument("--hidden", type=Path, required=True)
    parser.add_argument("--reference-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--backends", nargs="+", default=["SCI", "SCI_ours", "SCI_Beaver", "SCI_Beaver_segmented"])
    parser.add_argument("--graph-modes", nargs="+", default=["sequential", "isolated", "parallel"])
    parser.add_argument("--activation-workers", type=int, choices=(1,2), default=2)
    parser.add_argument("--queries", type=int, default=2)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    reports = []
    for hidden in ("default", "nonzero"):
        reference = tensors(args.reference_dir / hidden / "SCI-n4-t4/party2.log")
        if len(reference) != 15 or sum(len(line.split()) - 7 for line in reference) != 172928:
            raise ValueError("expected the frozen 15-tensor, 172928-integer SCI diagnostic")
        for backend in args.backends:
            for mode in args.graph_modes:
                output = args.output / hidden / (backend + "-" + mode)
                output.parent.mkdir(exist_ok=True)
                command = [sys.executable, str(HERE / "run.py"), "--binary", str(args.binary),
                    "--fixture", str(args.fixture), "--output", str(output), "--backend", backend,
                    "--steps", "1", "--batch", "128", "--queries", str(args.queries), "--diagnostic",
                    "--activation-workers", str(args.activation_workers), "--cggm-threads", "2",
                    "--graph-mode", mode, "--server-cpus", "0,2,4,10", "--client-cpus", "6,8,12,16", "--timeout", "600"]
                if hidden == "nonzero":
                    command += ["--hidden", str(args.hidden)]
                child = subprocess.run(command, text=True, capture_output=True)
                (output / "verification_command.json").write_text(json.dumps(command, indent=2) + "\n")
                report = {"backend":backend, "hidden":hidden, "graph_mode":mode,
                    "activation_workers":args.activation_workers, "queries":args.queries,
                    "tensors_per_query":15, "integers_per_query":172928,
                    "status":"passed", "result":str(output / "result.json")}
                if child.returncode:
                    report.update(status="failed", reason=child.stdout + child.stderr)
                elif tensors(output / "party2.log") != reference * args.queries:
                    report.update(status="failed", reason="native integer tensors differ")
                reports.append(report)
                (args.output / "summary.json").write_text(json.dumps({"rows":reports,
                    "scope":"diagnostic correctness only; no formal timing"}, indent=2) + "\n")
                print(json.dumps(report), flush=True)
                if report["status"] != "passed":
                    return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
