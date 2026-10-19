#!/usr/bin/env python3
"""Sequential two-party localhost runner for the frozen SDCF-10 matrix."""

from __future__ import annotations

import argparse
import json
import os
import resource
import signal
import socket
import statistics
import subprocess
import time
from pathlib import Path

from manifest import MEMORY_BYTES, TIMEOUT_SECONDS, build_manifest, validate_manifest, write_manifest

PRIMITIVE_CHANNEL_OFFSETS = (0, 3, 50, 100)


def select_primitive_port(preferred=None):
    """Probe Peer and SCI channels outside Linux's outgoing ephemeral range."""
    ephemeral_start = int(Path("/proc/sys/net/ipv4/ip_local_port_range").read_text().split()[0])
    last_base = ephemeral_start - max(PRIMITIVE_CHANNEL_OFFSETS)
    candidates = list(range(20000, last_base, 137))
    if preferred is not None and 1024 <= preferred < last_base:
        candidates.insert(0, preferred)
    for base in candidates:
        sockets = []
        try:
            for offset in PRIMITIVE_CHANNEL_OFFSETS:
                channel = socket.socket()
                sockets.append(channel)
                channel.bind(("0.0.0.0", base + offset))
            return base
        except OSError:
            continue
        finally:
            for channel in sockets:
                channel.close()
    raise RuntimeError("no free primitive Peer/SCI channel set below ephemeral range")


def command_for(executable: Path, row: dict, port: int, role: int):
    command = [str(executable), f"role={role}", f"port={port}",
               f"kind={row['kind']}", f"backend={row['backend']}",
               f"bits={row['bits']}", f"comparison-bits={row['comparison_bits']}",
               f"nseg={row['nseg']}", f"count={row['count']}",
               f"runs={row.get('runs', 3)}", f"timeout={row['timeout_seconds']}"]
    if "cggm_threads" in row:
        command.append(f"cggm-threads={row['cggm_threads']}")
    for field in ("segment_policy", "generation_tile_keys", "generation_budget_bytes", "generation_reserve_bytes"):
        if field in row:
            command.append(f"{field.replace('_', '-')}={row[field]}")
    if row["kind"] == "conversion":
        command += [f"shift={row['shift']}", f"operation={row['operation']}",
                    f"signed={1 if row['signed'] else 0}"]
    return command


def classify_status(returncode, stdout, stderr, timed_out=False):
    if timed_out:
        return "timeout"
    if "OOM" in stderr or "OOM" in stdout:
        return "oom"
    if "UNSUPPORTED" in stderr or "UNSUPPORTED" in stdout:
        return "unsupported"
    if returncode == 0 and "SWEEP_ERROR" not in stderr and "Usage:" not in stderr:
        return "complete"
    if ("bad_alloc" in stderr.lower() or "out of memory" in stderr.lower() or
            "cannot allocate" in stderr.lower() or "resource temporarily unavailable" in stderr.lower()):
        return "oom"
    return "failed"


def aggregate_parties(parties):
    """Align fresh runs before aggregation; communication counts both senders."""
    if len(parties) != 2 or len(parties[0]["samples"]) != len(parties[1]["samples"]):
        raise ValueError("two aligned party sample lists are required")
    aligned = list(zip(parties[0]["samples"], parties[1]["samples"]))
    result = {"parties": parties, "repetitions": len(aligned),
              "latency_aggregation": "median of per-run maximum across parties",
              "communication_aggregation": "sum of both parties sent bytes",
              "setup_seconds": max(p["setup_seconds"] for p in parties)}
    for field in ("offline_seconds", "online_seconds", "total_seconds"):
        values = [max(a[field], b[field]) for a, b in aligned]
        result[field] = statistics.median(values)
        result[field + "_min"] = min(values)
        result[field + "_max"] = max(values)
    for phase in ("offline", "online"):
        result[phase + "_bytes"] = statistics.median(
            a[phase + "_sent_bytes"] + b[phase + "_sent_bytes"] for a, b in aligned)
    return result


def _limits(memory_bytes):
    resource.setrlimit(resource.RLIMIT_AS, (memory_bytes, memory_bytes))


def _kill(process):
    if process.poll() is None:
        try:
            os.killpg(process.pid, signal.SIGTERM)
        except ProcessLookupError:
            pass
        try:
            process.wait(timeout=2)
        except subprocess.TimeoutExpired:
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            process.wait()


def _spawn(command, stdout_path, stderr_path, memory_bytes, cpu_affinity=None, env=None):
    out = stdout_path.open("w")
    err = stderr_path.open("w")
    try:
        def limits():
            _limits(memory_bytes)
            if cpu_affinity is not None:
                os.sched_setaffinity(0, cpu_affinity)
        process = subprocess.Popen(command, stdout=out, stderr=err, text=True,
                                   start_new_session=True, preexec_fn=limits, env=env)
    except Exception:
        out.close()
        err.close()
        raise
    return process, out, err


def run_pair(row, output, executable, timeout=TIMEOUT_SECONDS, executable_args=None,
             port=None, party_cpus=None, env=None):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    executable = Path(executable)
    requested_port = port
    if executable_args is None:
        port = select_primitive_port(port)
        commands = [command_for(executable, row, port, role) for role in (1, 2)]
    else:
        commands = [[str(executable)] + list(executable_args) for _ in (1, 2)]
    if party_cpus is not None:
        available = os.sched_getaffinity(0)
        if (len(party_cpus) != 2 or
                any(not cpus or len(set(cpus)) != len(cpus) or
                    not set(cpus) <= available for cpus in party_cpus)):
            raise ValueError("party_cpus must contain two nonempty available CPU lists")
        commands = [["taskset", "-c", ",".join(map(str, cpus))] + command
                    for cpus, command in zip(party_cpus, commands)]
    started = time.monotonic()
    processes = []
    streams = []
    timed_out = False
    peer_failed = False
    unavailable = False
    spawn_error = ""
    observed_affinities = []
    try:
        # The server waits for the client; both parties are part of one row and
        # are started together.  Rows themselves are always processed serially.
        for index, command in enumerate(commands, 1):
            try:
                process, out, err = _spawn(command, output / f"party{index}.stdout.log",
                                           output / f"party{index}.stderr.log", MEMORY_BYTES,
                                           None if party_cpus is None else party_cpus[index - 1], env)
            except FileNotFoundError as error:
                unavailable = True
                spawn_error = str(error)
                break
            processes.append(process)
            streams.append((out, err))
            if party_cpus is not None:
                observed_affinities.append(sorted(os.sched_getaffinity(process.pid)))
        deadline = started + timeout
        while processes and any(process.poll() is None for process in processes):
            if any(process.poll() not in (None, 0) for process in processes):
                peer_failed = True
                break
            if time.monotonic() >= deadline:
                timed_out = True
                break
            time.sleep(0.02)
    finally:
        if timed_out or peer_failed or unavailable:
            for process in processes:
                _kill(process)
        else:
            for process in processes:
                process.wait()
        for out, err in streams:
            out.close()
            err.close()
    elapsed = time.monotonic() - started
    logs = []
    returncodes = []
    for index in (1, 2):
        stdout_path, stderr_path = output / f"party{index}.stdout.log", output / f"party{index}.stderr.log"
        stdout = stdout_path.read_text() if stdout_path.exists() else ""
        stderr = stderr_path.read_text() if stderr_path.exists() else ""
        logs.append((stdout, stderr))
    for process in processes:
        returncodes.append(process.returncode)
    if unavailable:
        status = "unavailable"
    elif timed_out:
        status = "timeout"
    else:
        statuses = [classify_status(code, stdout, stderr) for code, (stdout, stderr) in zip(returncodes, logs)]
        status = "complete" if len(statuses) == 2 and all(value == "complete" for value in statuses) else next(
            (value for value in statuses if value in {"unsupported", "oom"}), "failed")
    measurements = None
    validation_error = ""
    if status == "complete":
        parties = []
        try:
            for stdout, _ in logs:
                records = [json.loads(line[len("RESULT "):]) for line in stdout.splitlines()
                           if line.startswith("RESULT ")]
                if len(records) != 1 or not records[0].get("correctness_passed"):
                    raise ValueError("missing verified party result")
                parties.append(records[0])
            measurements = aggregate_parties(parties)
        except (ValueError, KeyError, TypeError) as error:
            status = "failed"
            measurements = None
            validation_error = str(error)
    result = dict(row)
    failure_log = next((stderr.strip() for _, stderr in logs if
                        "OOM" in stderr or "UNSUPPORTED" in stderr or "SWEEP_ERROR" in stderr), "")
    if status != "complete" and not failure_log:
        failure_log = next((stderr.strip() for code, (_, stderr) in zip(returncodes, logs)
                            if code and stderr.strip()), "")
    reason = spawn_error or ("wall timeout" if timed_out else validation_error or failure_log)
    if status != "complete" and not reason:
        reason = "party exit codes: " + str(returncodes)
    result.update({"status": status, "reason": reason or None,
                   "measurements": measurements})
    execution = {"row_id": row["id"], "commands": commands, "returncodes": returncodes,
                 "elapsed_seconds": elapsed, "timeout_seconds": timeout,
                 "memory_limit_bytes": MEMORY_BYTES, "status": status}
    if executable_args is None:
        execution.update(requested_port=requested_port, selected_port=port,
                         channel_offsets=list(PRIMITIVE_CHANNEL_OFFSETS))
    if party_cpus is not None:
        execution.update(requested_cpu_affinity_per_party=party_cpus,
                         observed_cpu_affinity_per_party=observed_affinities)
    (output / "execution.json").write_text(json.dumps(execution, indent=2, sort_keys=True) + "\n")
    (output / "commands.json").write_text(json.dumps(commands, indent=2) + "\n")
    (output / "result.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    return result


def run_manifest(executable, output, limit=None, timeout=TIMEOUT_SECONDS):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    rows = build_manifest()
    if limit is not None:
        rows = rows[:limit]
    validate_manifest(rows)
    write_manifest(output / "manifest.json", rows)
    results = []
    for index, row in enumerate(rows):
        row_output = output / row["id"]
        results.append(run_pair(row, row_output, executable, timeout=timeout,
                                port=22000 + index * 137))
        print(row["id"], results[-1]["status"], flush=True)
        (output / "results.json").write_text(json.dumps(results, indent=2, sort_keys=True) + "\n")
    validate_manifest(results)
    (output / "results.json").write_text(json.dumps(results, indent=2, sort_keys=True) + "\n")
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--binary", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--timeout", type=float, default=TIMEOUT_SECONDS)
    args = parser.parse_args()
    run_manifest(args.binary, args.output, args.limit, args.timeout)


if __name__ == "__main__":
    main()
