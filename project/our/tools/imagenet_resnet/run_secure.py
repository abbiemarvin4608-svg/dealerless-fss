#!/usr/bin/env python3
"""Launch a paired ResNet protocol run and retain measured phase evidence."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import signal
import socket
import subprocess
import sys
import time

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "segmented_dcf_runtime"))
from summarize import query_phases, validate_paired_measurements

RUNTIME_PARAMETERS = {
    "SEGFSS_CGGM_THREADS": "1", "SEGFSS_GENERATION_TILE_KEYS": "256",
    "SEGFSS_GENERATION_BUDGET_BYTES": "268435456", "SEGFSS_GENERATION_RESERVE_BYTES": "67108864",
    "SEGFSS_PREFETCH_BYTES": "8388608",
}


def party_affinities(cpus_per_party):
    available = sorted(os.sched_getaffinity(0))
    if len(available) < cpus_per_party * 2:
        raise ValueError("not enough allowed CPUs for two disjoint parties")
    return [available[i * cpus_per_party:(i + 1) * cpus_per_party] for i in range(2)]


def runtime_environment(parent):
    environment = {key: value for key, value in parent.items() if not key.startswith(("SEGFSS_", "SDCF_"))}
    environment.update(RUNTIME_PARAMETERS)
    environment.update(MALLOC_ARENA_MAX="2", MALLOC_TRIM_THRESHOLD_="1", OMP_NUM_THREADS="1")
    return environment


def digest(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            result.update(block)
    return result.hexdigest()


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")


def free_port():
    for port in range(27000, 50000, 96):
        sockets = []
        try:
            for offset in range(80):
                current = socket.socket()
                sockets.append(current)
                current.bind(("127.0.0.1", port + offset))
            return port
        except OSError:
            pass
        finally:
            for current in sockets:
                current.close()
    raise RuntimeError("no free protocol port range")


def stop(processes):
    for process in processes:
        if process.poll() is None:
            try:
                os.killpg(process.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
    for process in processes:
        try:
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            os.killpg(process.pid, signal.SIGKILL)
            process.wait()


def run(args):
    output = args.output.resolve()
    if (output / "STATUS.json").exists():
        raise ValueError("use a fresh output directory to preserve previous evidence")
    graph = json.loads(args.graph.read_text())
    diagnostic = args.diagnostic or graph.get("evidence_scope") == "diagnostic"
    if not diagnostic:
        counts = {kind: sum(node["op"] == kind for node in graph["nodes"])
                  for kind in ("conv2d", "relu", "add", "maxpool", "avgpool", "linear")}
        expected = {
            "resnet50": dict(conv2d=53, relu=49, add=16, maxpool=1, avgpool=1, linear=1),
            "resnet152": dict(conv2d=155, relu=151, add=50, maxpool=1, avgpool=1, linear=1),
        }
        if graph.get("model") not in expected or counts != expected[graph["model"]]:
            raise ValueError("formal run requires a complete exported ResNet-50 or ResNet-152 graph")
        from run_performance import require_memory_guard
        require_memory_guard()
    output.mkdir(parents=True, exist_ok=True)
    round_trace = getattr(args, "round_trace", "on")
    optimization_profile = getattr(args, "optimization_profile", "optimized")
    native_spool_budget = getattr(args, "native_spool_budget", 0)
    plan = {
        "status": "running", "kind": "secure-two-party-inference",
        "model": graph["model"], "batch": graph["batch"], "backend": args.backend,
        "bits": graph["bits"], "frac": graph["frac"], "queries": args.queries,
        "diagnostic": diagnostic, "network": "unshaped-localhost",
        "round_trace": round_trace,
        "optimization_profile": optimization_profile,
        "runtime_parameters": RUNTIME_PARAMETERS,
        "measurement_mode": "time_and_bytes" if round_trace == "off" else "time_bytes_and_rounds",
        "binary": str(args.binary.resolve()), "binary_sha256": digest(args.binary),
        "graph": str(args.graph.resolve()), "graph_sha256": digest(args.graph),
        "parameters": str(args.parameters.resolve()), "parameters_sha256": digest(args.parameters),
        "input": str(args.input.resolve()) if args.input else None,
        "input_sha256": digest(args.input) if args.input else None,
        "timeout_seconds": args.timeout, "started_epoch": time.time(),
        "tile": args.tile, "spool_budget_bytes_per_cnn_or_beaver_store": args.spool_budget,
        "native_spool_budget_bytes": native_spool_budget,
        "truncation_spool_budgeted": False,
        "offline_budget_seconds": args.offline_budget_seconds,
        "cpu_affinity": [], "commands": [],
    }
    bridge_library = args.binary.resolve().parent / "libcnn_ours.so"
    if args.backend in ("SCI_ours", "SCI_ours_beaver"):
        if not bridge_library.is_file():
            raise ValueError("ours requires libcnn_ours.so alongside the executable")
        plan["bridge_library"] = str(bridge_library)
        plan["bridge_library_sha256"] = digest(bridge_library)
    guard_report = os.environ.get("IMAGENET_MEMORY_GUARD_REPORT")
    if guard_report:
        guard = json.loads(Path(guard_report).read_text())
        membership = dict(line.split(":", 2)[1:] for line in Path("/proc/self/cgroup").read_text().splitlines())
        if membership.get("memory") != "/" + Path(guard["cgroup"]).name:
            raise ValueError("memory guard metadata does not match the runner's actual cgroup")
        plan["resource_guard"] = {key: guard[key] for key in
            ("cgroup", "memory_limit_bytes", "memory_and_swap_limit_bytes", "started_epoch", "boot_id")}
        plan["resource_guard"]["report"] = str(Path(guard_report).resolve())
    cpus = party_affinities(args.cpus_per_party)
    plan["cpu_affinity"] = cpus
    port = args.port or free_port()
    env = runtime_environment(os.environ)
    processes, logs = [], []
    status = "failed"
    reason = None
    protocol_started = time.monotonic()
    protocol_finished = None
    try:
        for index in range(2):
            role = index + 1
            command = [str(args.binary.resolve()), "--role", str(role), "--port", str(port),
                       "--backend", args.backend, "--graph", str(args.graph.resolve()),
                       "--parameters", str(args.parameters.resolve()), "--queries", str(args.queries),
                       "--tile", str(args.tile), "--spool-budget", str(args.spool_budget),
                       "--native-spool-budget", str(native_spool_budget),
                       "--optimization-profile", optimization_profile,
                       "--offline-budget-seconds", str(args.offline_budget_seconds),
                       "--output", str(output / ("party%d.json" % role))]
            if args.input:
                command += ["--input", str(args.input.resolve())]
            if diagnostic:
                command += ["--diagnostic"]
            plan["commands"].append(command)
            role_env = child_environment(env, output, role, round_trace)
            log = (output / ("party%d.log" % role)).open("wb")
            logs.append(log)
            affinity = cpus[index]
            process = subprocess.Popen(
                command, stdout=log, stderr=subprocess.STDOUT, env=role_env,
                start_new_session=True,
                preexec_fn=lambda selected=affinity: os.sched_setaffinity(0, selected),
            )
            processes.append(process)
        write_json(output / "PLAN.json", plan)
        write_json(output / "STATUS.json", plan)
        deadline = time.monotonic() + args.timeout
        last_status = time.monotonic()
        while any(process.poll() is None for process in processes):
            if any(process.poll() not in (None, 0) for process in processes):
                reason = "one party exited with an error"
                break
            if time.monotonic() >= deadline:
                status, reason = "timeout", "full inference exceeded the recorded wall-time budget"
                break
            if time.monotonic() - last_status >= 30:
                print(json.dumps({"status": "running", "model": graph["model"], "backend": args.backend,
                                  "elapsed_seconds": round(time.time() - plan["started_epoch"], 1)}), flush=True)
                last_status = time.monotonic()
            time.sleep(.2)
        else:
            if all(process.returncode == 0 for process in processes):
                status = "complete"
        if status != "complete":
            stop(processes)
        protocol_finished = time.monotonic()
        if status == "complete":
            parties = [json.loads((output / ("party%d.json" % role)).read_text()) for role in (1, 2)]
            validate_paired_measurements(parties, plan, graph)
            if round_trace == "on":
                expected_phases = [[dict(query=query, phase=phase, sent_bytes=value["sent_bytes"],
                                         received_bytes=value["received_bytes"])
                                    for (query, phase), value in query_phases(party).items()] for party in parties]
                from round_trace_stream import analyze_pair
                rounds = analyze_pair(output / "party1.trace.jsonl", output / "party2.trace.jsonl", expected_phases)
                write_json(output / "ROUNDS.json", rounds)
                if rounds["status"] != "passed":
                    raise ValueError("causal trace did not qualify: " + str(rounds.get("reason")))
            write_json(output / "PARTIES.json", parties)
    except BaseException as error:
        stop(processes)
        if protocol_finished is None:
            protocol_finished = time.monotonic()
        reason = "%s: %s" % (type(error).__name__, error)
        if isinstance(error, (KeyboardInterrupt, SystemExit)):
            status = "interrupted"
        else:
            status = "failed"
    finally:
        for log in logs:
            log.close()
        final = dict(plan, status=status, reason=reason, finished_epoch=time.time(),
                     protocol_process_wall_seconds=(protocol_finished or time.monotonic()) - protocol_started,
                     launcher_wall_seconds=time.monotonic() - protocol_started,
                     exit_codes=[process.returncode for process in processes])
        write_json(output / "STATUS.json", final)
    print(json.dumps({"status": status, "reason": reason, "output": str(output)}, ensure_ascii=False), flush=True)
    return 0 if status == "complete" else 1


def child_environment(env, output, role, round_trace):
    """Disable callbacks completely, including when the parent inherited a trace path."""
    role_env = dict(env)
    if round_trace == "on":
        role_env["SDCF_ROUND_TRACE"] = str(output / ("party%d.trace.jsonl" % role))
    elif round_trace == "off":
        role_env.pop("SDCF_ROUND_TRACE", None)
    else:
        raise ValueError("round_trace must be on or off")
    return role_env


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--binary", type=Path, required=True)
    parser.add_argument("--graph", type=Path, required=True)
    parser.add_argument("--parameters", type=Path, required=True)
    parser.add_argument("--input", type=Path)
    parser.add_argument("--backend", choices=("SCI", "SCI_preprocessed", "SCI_ours", "SCI_ours_beaver"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--queries", type=int, default=3)
    parser.add_argument("--cpus-per-party", type=int, default=4)
    parser.add_argument("--timeout", type=float, default=3600)
    parser.add_argument("--port", type=int)
    parser.add_argument("--tile", type=int, default=64)
    parser.add_argument("--spool-budget", type=int, default=0,
                        help="per-party CNN/Beaver store cap in bytes; excludes truncation; zero is unlimited")
    parser.add_argument("--native-spool-budget", type=int, default=0,
                        help="per-party SCI preprocessing disk cap; zero inherits --spool-budget")
    parser.add_argument("--optimization-profile", choices=("optimized", "baseline"), default="optimized")
    parser.add_argument("--offline-budget-seconds", type=float, default=0,
                        help="checked between Beaver preparation calls; --timeout covers the entire protocol")
    parser.add_argument("--diagnostic", action="store_true")
    parser.add_argument("--round-trace", choices=("on", "off"), default="on",
                        help="off measures only time/bytes and skips causal trace recording/replay")
    args = parser.parse_args()
    if min(args.queries, args.cpus_per_party, args.timeout) <= 0:
        parser.error("queries, CPUs and timeout must be positive")
    if not 1 <= args.tile <= 4096 or min(args.spool_budget, args.native_spool_budget, args.offline_budget_seconds) < 0:
        parser.error("tile must be 1..4096 and material budgets nonnegative")
    return run(args)


if __name__ == "__main__":
    raise SystemExit(main())
