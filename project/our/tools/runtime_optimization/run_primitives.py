#!/usr/bin/env python3
"""Run declared primitive rows with the established resource/measurement contract."""
import argparse
from contextlib import ExitStack
import datetime
import fcntl
import json
import os
from pathlib import Path
import selectors
import statistics
import subprocess
import sys

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "primitive_cost_sweep"))
from runner import run_pair, select_primitive_port


def write(path, value):
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def stop(child):
    if child.poll() is None:
        child.terminate()
    try:
        child.wait(timeout=3)
    except subprocess.TimeoutExpired:
        child.kill()
        child.wait()


def run(plan, output):
    output.mkdir(parents=True, exist_ok=False)
    write(output / "manifest.json", dict(plan, frozen_at=datetime.datetime.now(datetime.timezone.utc).isoformat()))
    summary = {"manifest": "manifest.json", "rows": []}
    env = {key: value for key, value in os.environ.items()
           if not key.startswith(("SEGFSS_", "SDCF_"))
           and key not in ("GOOGLE30_PROXY_OFFSET", "SWEEP_CONNECT_OFFSET")}
    env["OMP_NUM_THREADS"] = "1"
    with Path("/tmp/sdcf-measurement.lock").open("a") as lock:
        for row in plan["rows"]:
            fcntl.flock(lock, fcntl.LOCK_EX)
            folder = output / row["id"]
            folder.mkdir()
            row_env = dict(env, **row.get("environment", {}))
            port = select_primitive_port()
            with ExitStack() as stack:
                if row.get("network", "localhost") != "localhost":
                    profiles = {"sirnn_lan": (0.8, 3016), "wan": (2.0, 100)}
                    rtt, rate = profiles[row["network"]]
                    connector = output / "connect_offset.so"
                    if not connector.exists():
                        subprocess.run(["gcc", "-shared", "-fPIC", "-O2",
                                        str(HERE.parent / "segmented_dcf_runtime/connect_offset.c"),
                                        "-o", str(connector), "-ldl"], check=True)
                    mapping = ",".join(f"{port + 1000 + offset}:{port + offset}"
                                       for offset in (0, 3, 50, 100))
                    command = ["taskset", "-c", "14", sys.executable,
                               str(HERE.parent / "google30_benchmark/link_proxy.py"),
                               "--ports", mapping, "--rtt-ms", str(rtt), "--mbps", str(rate)]
                    log = stack.enter_context((folder / "relay.log").open("w"))
                    relay = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=log, text=True)
                    stack.callback(stop, relay)
                    with selectors.DefaultSelector() as selector:
                        selector.register(relay.stdout, selectors.EVENT_READ)
                        if not selector.select(10) or relay.stdout.readline().strip() != "READY":
                            raise RuntimeError("primitive relay readiness failed")
                    relay.stdout.close()
                    write(folder / "relay_command.json", command)
                    row_env.update(SWEEP_CONNECT_OFFSET="1000", LD_PRELOAD=str(connector))
                result = run_pair(row, folder, Path(row["binary"]),
                                  timeout=row["timeout_seconds"], port=port,
                                  party_cpus=([0, 2, 4, 10], [6, 8, 12, 16]), env=row_env)
            if result["status"] == "complete":
                measured = result["measurements"]
                samples = zip(*(party["samples"] for party in measured["parties"]))
                aligned = list(samples)
                for phase in ("offline", "online", "total"):
                    values = [max(left[phase + "_seconds"], right[phase + "_seconds"])
                              for left, right in aligned]
                    measured[phase + "_seconds_distribution"] = {
                        "median": statistics.median(values), "min": min(values), "max": max(values),
                        "pstdev": statistics.pstdev(values), "values": values}
                write(folder / "result.json", result)
            summary["rows"].append(result)
            write(output / "summary.json", summary)
            print(row["id"], result["status"], flush=True)
            fcntl.flock(lock, fcntl.LOCK_UN)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    run(json.loads(args.plan.read_text()), args.output.resolve())
