#!/usr/bin/env python3
"""Run matched, sequential before/after persistent-pool measurements."""

import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import platform
import statistics
import subprocess
import sys

from runner import command_for, run_pair

HERE = Path(__file__).resolve().parent
OUR = HERE.parents[1]
REPO = OUR.parents[1]
RUNTIME = HERE.parent / "segmented_dcf_runtime"
BACKENDS = ("SCI", "SCI_Beaver", "SCI_ours", "SCI_Beaver_segmented")


def digest(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            value.update(block)
    return value.hexdigest()


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def cpus(value):
    values = value.split(",")
    if any(not item.isascii() or not item.isdecimal() for item in values):
        raise argparse.ArgumentTypeError("CPUs must be comma-separated nonnegative integers")
    result = [int(item) for item in values]
    if len(set(result)) != len(result):
        raise argparse.ArgumentTypeError("duplicate CPU")
    return result


def topology(cpu_ids):
    result = []
    for cpu in cpu_ids:
        folder = Path("/sys/devices/system/cpu/cpu%d/topology" % cpu)
        result.append({"cpu": cpu, "core": int((folder / "core_id").read_text()),
                       "package": int((folder / "physical_package_id").read_text())})
    return result


def child_env():
    result = {key: value for key, value in os.environ.items()
              if not key.startswith(("SEGFSS_", "SDCF_"))
              and key not in ("GOOGLE30_PROXY_OFFSET", "SWEEP_CONNECT_OFFSET")}
    result["OMP_NUM_THREADS"] = "1"
    return result


def primitive_rows(args):
    label = "old" if args.phase == "before" else "new"
    rows = []
    shapes = [(128, segment, args.primitive_runs) for segment in (2, 4)]
    if args.large_batch:
        shapes.extend((12800, segment, args.large_runs) for segment in (2, 4))
    for count, nseg, runs in shapes:
        for workers in (1, 4):
            rows.append({"id": "%s-b%d-n%d-t%d" % (label, count, nseg, workers),
                         "kind": "dcf", "backend": "segmented", "bits": 16,
                         "comparison_bits": 16, "count": count, "nseg": nseg,
                         "cggm_threads": workers, "runs": runs,
                         "timeout_seconds": args.primitive_timeout})
    if args.phase == "after":
        for nseg in (2, 4):
            for workers in (1, 4):
                rows.append({"id": "new-conversion-b128-n%d-t%d" % (nseg, workers),
                             "kind": "conversion", "backend": "segmented",
                             "bits": 32, "comparison_bits": 14, "count": 128,
                             "nseg": nseg, "cggm_threads": workers,
                             "runs": args.conversion_runs,
                             "timeout_seconds": args.primitive_timeout,
                             "operation": "div_pow2", "shift": 14, "signed": True})
    return rows


def runtime_rows(args):
    if args.phase == "before":
        return []
    rows = []
    for backend in BACKENDS:
        segments = (4,) if args.profile == "runtime-opt-01" or backend in ("SCI", "SCI_Beaver") else (2, 4)
        workers = (4,) if backend in ("SCI", "SCI_Beaver") else (1, 4)
        for nseg in segments:
            for limit in workers:
                rows.append({"id": "%s-n%d-t%d" % (backend, nseg, limit),
                             "backend": backend, "max_segments": nseg,
                             "cggm_threads": limit, "queries": args.runtime_queries,
                             "timeout_seconds": args.runtime_timeout})
    return rows


def runtime_command(args, row, output, diagnostic=False, hidden=None):
    command = [sys.executable, str(RUNTIME / "run.py"),
               "--binary", str(args.runtime_binary), "--fixture", str(args.fixture),
               "--output", str(output), "--backend", row["backend"],
               "--batch", "128", "--steps", "1", "--start", "0",
               "--queries", "1" if diagnostic else str(row["queries"]),
               "--warmup", "0", "--network", "localhost",
               "--memory-gib", "1.5", "--timeout", str(row["timeout_seconds"]),
               "--cpus", "10,12,%d" % args.relay_cpu,
               "--server-cpus", ",".join(map(str, args.server_cpus)),
               "--client-cpus", ",".join(map(str, args.client_cpus)),
               "--cggm-threads", str(row["cggm_threads"]),
               "--max-segments", str(row["max_segments"])]
    if diagnostic:
        command += ["--diagnostic"]
        if hidden:
            command += ["--hidden", str(hidden)]
    return command


def distribution(values):
    return {"median": statistics.median(values), "min": min(values),
            "max": max(values), "pstdev": statistics.pstdev(values), "values": values}


def runtime_summary(row, result, relative):
    summary = dict(row, status=result["status"], reason=result.get("reason"),
                   result=relative, measurements=None)
    if result["status"] == "complete":
        measured = result["measurements"]
        queries = measured["queries"]
        summary["measurements"] = {
            "setup_seconds": measured["setup"]["seconds"],
            "session_total_seconds": measured["total_seconds"],
            "first_query_seconds": measured.get("first_query", {}).get("seconds"),
            "peak_rss_kib_per_party": measured["peak_rss_kib_per_party"],
        }
        for phase in ("offline", "online", "query_total"):
            summary["measurements"][phase + "_seconds"] = distribution(
                [query[phase]["seconds"] for query in queries])
        for phase in ("offline", "online"):
            summary["measurements"][phase + "_bytes"] = distribution(
                [query[phase]["sent_bytes"] for query in queries])
    return summary


def invoke(command, log, env):
    with Path(log).open("w") as stream:
        child = subprocess.run(command, stdout=stream, stderr=subprocess.STDOUT, env=env)
    return child.returncode


def run(args):
    available = os.sched_getaffinity(0)
    party_cpus = (args.server_cpus, args.client_cpus)
    all_cpus = args.server_cpus + args.client_cpus + [args.relay_cpu]
    if any(len(value) < 4 for value in party_cpus) or not set(all_cpus) <= available:
        raise ValueError("each party needs four available CPUs; relay CPU must also be available")
    physical = [(row["package"], row["core"]) for row in topology(all_cpus)]
    if len(set(physical)) != len(physical):
        raise ValueError("parties and relay must use disjoint physical cores")
    binary = args.old_binary if args.phase == "before" else args.new_binary
    if not binary:
        raise ValueError("selected phase requires --old-binary or --new-binary")
    binary = binary.resolve()
    if args.phase == "after":
        if not args.runtime_binary or not args.fixture:
            raise ValueError("after phase requires --runtime-binary and --fixture")
        args.runtime_binary, args.fixture = args.runtime_binary.resolve(), args.fixture.resolve()
    args.output = args.output.resolve()
    args.output.mkdir(parents=True, exist_ok=False)
    primitives, runtimes = primitive_rows(args), runtime_rows(args)
    environment = child_env()
    identity = {"path": str(binary), "sha256": digest(binary)}
    manifest = {
        "phase": args.phase, "profile": args.profile, "source_commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=REPO, text=True).strip(),
        "source_status": subprocess.check_output(
            ["git", "status", "--porcelain"], cwd=REPO, text=True),
        "machine": platform.uname()._asdict(), "cpu_topology": topology(all_cpus),
        "parent_affinity": sorted(available), "party_cpu_affinities": party_cpus,
        "relay_cpu": args.relay_cpu, "primitive_binary": identity,
        "primitive_source": args.old_source if args.phase == "before" else args.new_source,
        "sci_protocol_threads_per_party": 1, "OMP_NUM_THREADS": "1",
        "network": "unshaped localhost TCP", "memory_limit_bytes_per_party": int(1.5 * (1 << 30)),
        "warmup_queries": 0, "runtime_steps": 1, "batch": 128,
        "scope": "one recurrent step; SDCF-09 remains deferred",
        "primitive_rows": primitives, "runtime_rows": runtimes,
        "primitive_commands": [[
            ["taskset", "-c", ",".join(map(str, party_cpus[role - 1]))]
            + command_for(binary, row, 22000 + index * 137, role)
            for role in (1, 2)] for index, row in enumerate(primitives)],
        "runtime_commands": [runtime_command(args, row, args.output / "runtime" / row["id"])
                             for row in runtimes],
        "diagnostic_commands": [runtime_command(
            args, row, args.output / "diagnostic" / state / row["id"], diagnostic=True,
            hidden=args.output / "diagnostic/nonzero_hidden.i16" if state == "nonzero" else None)
            for state in ("default", "nonzero") for row in runtimes if row["cggm_threads"] == 4],
        "environment_policy": "remove inherited SEGFSS_*/SDCF_* and connector offsets; OMP_NUM_THREADS=1",
        "primitive_port_policy": "planned low ports; probe offsets 0,3,50,100 before each row; actual commands in execution.json",
        "diagnostic_policy": "separate default and nonzero-hidden query per backend/segment at workers=4; skip failed secure rows",
        "build_profile": "Release -O3; actual compiler commands and source identities in runtime_build",
        "measurement_rule": "align parties by query, maximum latency, sum sends; direct totals; nested work is not wall time",
        "network_supplements": {"sirnn_lan": {"rtt_ms": 0.8, "mbps": 3016}, "wan": {"rtt_ms": 2.0, "mbps": 100}},
    }
    if runtimes:
        manifest["runtime_binary"] = {"path": str(args.runtime_binary), "sha256": digest(args.runtime_binary)}
        manifest["runtime_build"] = json.loads((args.runtime_binary.parent / "runtime_manifest.json").read_text())
        manifest["fixture"] = {"path": str(args.fixture), "files": {
            name: digest(args.fixture / name) for name in ("server_model.i16", "client_input.i16", "contract.json")}}
    write_json(args.output / "manifest.json", manifest)
    if args.plan_only:
        print(str(args.output / "manifest.json"), flush=True)
        return
    summary = {"manifest": "manifest.json", "primitives": [], "runtime": [], "diagnostics": []}
    with Path("/tmp/sdcf-measurement.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        for index, row in enumerate(primitives):
            output = args.output / "primitive" / row["id"]
            result = run_pair(row, output, binary, timeout=row["timeout_seconds"],
                              port=22000 + index * 137, party_cpus=party_cpus, env=environment)
            if result["measurements"]:
                parties = result["measurements"]["parties"]
                for field in ("tree_evaluation_seconds", "scan_seconds", "scan_layers"):
                    values = [max(a[field], b[field]) for a, b in
                              zip(parties[0]["samples"], parties[1]["samples"])]
                    result["measurements"][field] = distribution(values)
            summary["primitives"].append(result)
            write_json(args.output / "summary.json", summary)
            print(row["id"], result["status"], flush=True)
        for row in runtimes:
            output = args.output / "runtime" / row["id"]
            output.parent.mkdir(exist_ok=True)
            command = runtime_command(args, row, output)
            code = invoke(command, output.parent / (row["id"] + ".runner.log"), environment)
            path = output / "result.json"
            result = json.loads(path.read_text()) if path.exists() else {
                "status": "failed", "reason": "runtime runner exit %d; no result" % code}
            summary["runtime"].append(runtime_summary(row, result, str(path.relative_to(args.output))))
            write_json(args.output / "summary.json", summary)
            print(row["id"], result["status"], flush=True)
        if args.phase == "after" and not args.no_diagnostics:
            diagnostic_root = args.output / "diagnostic"
            diagnostic_root.mkdir()
            hidden = diagnostic_root / "nonzero_hidden.i16"
            subprocess.run([sys.executable, str(RUNTIME / "prepare_hidden.py"),
                            "--batch", "128", "--output", str(hidden)], check=True, env=environment)
            write_json(diagnostic_root / "input.json", {"path": str(hidden), "sha256": digest(hidden)})
            native_logs = {}
            diagnostic_rows = [(state, row) for state in ("default", "nonzero") for row in summary["runtime"]]
            for state, row in diagnostic_rows:
                if row["cggm_threads"] != 4:
                    continue
                record = {"id": row["id"], "hidden_state": state, "status": "skipped", "reason": row["status"]}
                if row["status"] == "complete":
                    output = diagnostic_root / state / row["id"]
                    output.parent.mkdir(exist_ok=True)
                    command = runtime_command(args, row, output, diagnostic=True, hidden=hidden if state == "nonzero" else None)
                    code = invoke(command, output.parent / (row["id"] + ".runner.log"), environment)
                    path = output / "result.json"
                    result = json.loads(path.read_text()) if path.exists() else {
                        "status": "failed", "reason": "diagnostic runner exit %d; no result" % code}
                    record.update(status=result["status"], reason=result.get("reason"),
                                  result=str(path.relative_to(args.output)))
                    if result["status"] == "complete":
                        if row["backend"] == "SCI":
                            native_logs[state] = output / "party2.log"
                            record["comparison"] = "native SCI reference"
                        elif state in native_logs:
                            compared = output / "comparison.json"
                            code = invoke([sys.executable, str(HERE.parent / "sci_beaver/compare_diagnostics.py"),
                                           "--native", str(native_logs[state]), "--beaver", str(output / "party2.log"),
                                           "--output", str(compared)], output / "comparison.log", environment)
                            record["comparison"] = json.loads(compared.read_text()) if compared.exists() else {
                                "status": "failed", "reason": "comparison exit %d" % code}
                        else:
                            record["comparison"] = {"status": "unavailable", "reason": "native diagnostic failed"}
                summary["diagnostics"].append(record)
                write_json(args.output / "summary.json", summary)
                print("diagnostic", row["id"], record["status"], flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", choices=("pool-recheck", "runtime-opt-01"), default="pool-recheck",
                        help="runtime-opt-01 freezes the approved new ten-query baseline")
    parser.add_argument("--phase", choices=("before", "after"), required=True)
    parser.add_argument("--old-binary", type=Path)
    parser.add_argument("--new-binary", type=Path)
    parser.add_argument("--old-source", default="96581cd")
    parser.add_argument("--new-source", default="current source recorded by source_commit")
    parser.add_argument("--runtime-binary", type=Path)
    parser.add_argument("--fixture", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--server-cpus", type=cpus, default=cpus("0,2,4,10"))
    parser.add_argument("--client-cpus", type=cpus, default=cpus("6,8,12,16"))
    parser.add_argument("--relay-cpu", type=int, default=14)
    parser.add_argument("--primitive-runs", type=int, default=30)
    parser.add_argument("--large-batch", action="store_true")
    parser.add_argument("--large-runs", type=int, default=10)
    parser.add_argument("--conversion-runs", type=int, default=5)
    parser.add_argument("--runtime-queries", type=int, default=5)
    parser.add_argument("--primitive-timeout", type=int, default=120)
    parser.add_argument("--runtime-timeout", type=int, default=600)
    parser.add_argument("--plan-only", action="store_true")
    parser.add_argument("--no-diagnostics", action="store_true")
    args = parser.parse_args()
    if args.profile == "runtime-opt-01":
        if args.phase != "after":
            parser.error("runtime-opt-01 requires --phase after with the selected frozen binaries")
        args.server_cpus, args.client_cpus, args.relay_cpu = [0, 2, 4, 10], [6, 8, 12, 16], 14
        args.primitive_runs, args.large_runs, args.conversion_runs, args.runtime_queries = 30, 10, 5, 10
        args.primitive_timeout, args.runtime_timeout, args.large_batch = 120, 600, True
    if min(args.primitive_runs, args.large_runs, args.conversion_runs, args.runtime_queries,
           args.primitive_timeout, args.runtime_timeout) < 1:
        parser.error("query counts and timeouts must be positive")
    try:
        run(args)
    except (ValueError, OSError, subprocess.SubprocessError) as error:
        parser.error(str(error))


if __name__ == "__main__":
    main()
