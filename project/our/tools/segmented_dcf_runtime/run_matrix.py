#!/usr/bin/env python3
"""Freeze and run the four-backend Batch=128 measurement matrix sequentially."""
import argparse
import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys

from comparison import compare_predictions, markdown, summarize
from run import party_affinities

HERE = Path(__file__).resolve().parent
BACKENDS = ("SCI", "SCI_Beaver", "SCI_Beaver_full", "SCI_Beaver_segmented")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--binary", type=Path, required=True)
    parser.add_argument("--fixture", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--network", choices=("localhost", "sirnn_lan", "wan"), action="append")
    parser.add_argument("--steps", type=int, choices=(1, 99), action="append")
    parser.add_argument("--queries", type=int, default=3)
    parser.add_argument("--timeout", type=float, default=900)
    parser.add_argument("--memory-gib", type=float, default=1.5)
    parser.add_argument("--cpus", default="10,12,14")
    parser.add_argument("--cggm-threads", type=int, default=1)
    parser.add_argument("--server-cpus")
    parser.add_argument("--client-cpus")
    parser.add_argument("--max-segments", type=int, default=4)
    args = parser.parse_args()
    if args.queries < 1 or args.timeout <= 0 or args.memory_gib <= 0:
        parser.error("queries and resource budgets must be positive")
    try:
        party_affinities(argparse.Namespace(
            cpus=[int(value) for value in args.cpus.split(",")],
            cggm_threads=args.cggm_threads,
            server_cpus=args.server_cpus, client_cpus=args.client_cpus), os.sched_getaffinity(0))
    except ValueError as error:
        parser.error(str(error))
    args.output = args.output.resolve()
    args.output.mkdir(parents=True, exist_ok=False)
    commands = []
    for network in args.network or ("localhost", "sirnn_lan", "wan"):
        for steps in args.steps or (1, 99):
            for backend in BACKENDS:
                output = args.output / network / ("step" if steps == 1 else "full") / backend
                commands.append([sys.executable, str(HERE / "run.py"),
                                 "--binary", str(args.binary.resolve()), "--fixture", str(args.fixture.resolve()),
                                 "--output", str(output), "--backend", backend, "--steps", str(steps),
                                 "--batch", "128", "--queries", str(args.queries), "--warmup", "0",
                                 "--timeout", str(args.timeout), "--memory-gib", str(args.memory_gib),
                                 "--network", network, "--cpus", args.cpus,
                                 "--max-segments", str(args.max_segments),
                                 "--cggm-threads", str(args.cggm_threads)])
                for option, value in (("--server-cpus", args.server_cpus), ("--client-cpus", args.client_cpus)):
                    if value is not None:
                        commands[-1].extend([option, value])
    manifest = {"commands": commands, "resource_budget": vars(args),
                "quality_scope": "SDCF-09 deferred; fixed 128 records only",
                "scheduling": "sequential under shared measurement lock"}
    (args.output / "matrix.json").write_text(json.dumps(manifest, indent=2, default=str) + "\n")
    completed = []
    native_outputs = {}
    with Path("/tmp/sdcf-measurement.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        for command in commands:
            output = Path(command[command.index("--output") + 1])
            print("RUN " + str(output), flush=True)
            child = subprocess.run(command, capture_output=True, text=True)
            output.mkdir(parents=True, exist_ok=True)
            (output / "runner.log").write_text(child.stdout + child.stderr)
            path = output / "result.json"
            if not path.is_file():
                raise RuntimeError("runner did not preserve a result: " + str(output))
            result = json.loads(path.read_text())
            config = result["config"]
            if result["status"] == "complete" and config["steps"] == 99:
                log = (output / "party2.log").read_text()
                group = config["network"]["name"]
                if config["backend"] == "SCI":
                    native_outputs[group] = log
                if group in native_outputs:
                    result["fixed_batch_outputs"] = compare_predictions(
                        native_outputs[group], log, config["batch"], config["start"],
                        config["queries"] + config["warmup_queries"])
                    path.write_text(json.dumps(result, indent=2) + "\n")
            completed.append((str(path), result))
            rows = summarize(completed)
            (args.output / "summary.json").write_text(json.dumps({"rows": rows}, indent=2) + "\n")
            (args.output / "REPORT.md").write_text(markdown(rows))
            print(child.stdout.strip(), flush=True)


if __name__ == "__main__":
    main()
