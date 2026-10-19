#!/usr/bin/env python3
"""Compare physically separated Google-30 phases using one frozen binary.

Both backends run sequentially on the same batch.  SCI + ours prepares fresh
FSS material before every query.  Online wall time comes directly from the
driver's query observations; offline wall time comes from OFFLINE_QUERY log
records.  No phase duration is inferred by subtracting it from total time.

The sirnn_lan profile models 377 MB/s (3016 Mbit/s), 0.8 ms RTT through the
existing all-channel TCP relay.  It is a local network simulation, not a run
on the machines used in the SiRNN paper.  localhost uses direct local sockets.
"""
import argparse
from contextlib import ExitStack
import hashlib
import json
import math
import os
from pathlib import Path
import re
import selectors
import socket
import statistics
import subprocess
import sys
import time


HERE = Path(__file__).resolve().parent
CHANNEL_OFFSETS = (0, 3, 20, 23, 50, 70, 100, 120, 200)
FLOAT = r"[+-]?(?:[0-9]+(?:\.[0-9]*)?|\.[0-9]+)(?:[eE][+-]?[0-9]+)?"


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def digest(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


def stop(child):
    if child.poll() is None:
        child.terminate()
    try:
        child.wait(timeout=3)
    except subprocess.TimeoutExpired:
        child.kill()
        child.wait()


def reserve_ports(use_relay):
    offsets = CHANNEL_OFFSETS
    if use_relay:
        offsets += tuple(1000 + value for value in CHANNEL_OFFSETS)
    for base in range(18000, 48000, 137):
        sockets = []
        try:
            for offset in offsets:
                sock = socket.socket()
                sockets.append(sock)
                sock.bind(("127.0.0.1", base + offset))
            return base
        except OSError:
            pass
        finally:
            for sock in sockets:
                sock.close()
    raise RuntimeError("no free SCI/bridge port range")


def records(log, tag):
    """Parse scalar key=value records without relying on field ordering."""
    rows = []
    for match in re.finditer(r"^" + re.escape(tag) + r"\s+([^\r\n]*)$", log,
                             re.MULTILINE):
        row = {}
        for key, value in re.findall(r"([A-Za-z][A-Za-z0-9_]*)=(" + FLOAT + r")", match[1]):
            row[key] = float(value) if any(c in value for c in ".eE") else int(value)
        rows.append(row)
    return rows


def record_seconds(row):
    for key in ("wall_seconds", "offline_wall_seconds", "seconds"):
        if key in row:
            return float(row[key])
    for key in ("wall_ms", "offline_wall_ms"):
        if key in row:
            return float(row[key]) / 1000.0
    raise ValueError("OFFLINE_QUERY lacks a physical wall-time field")


def check_pair_bytes(left, right, sent_key, received_key, label):
    values = [left.get(sent_key), left.get(received_key),
              right.get(sent_key), right.get(received_key)]
    if any(value is None for value in values):
        raise ValueError(label + " lacks sent/received counters")
    if values[0] != values[3] or values[2] != values[1]:
        raise ValueError(label + " has unbalanced two-party communication")
    return values[0] + values[2]


def phase_pair(metrics, name, required=True):
    pair = [metric["phases"].get(name) for metric in metrics]
    if any(value is None for value in pair):
        if not required and all(value is None for value in pair):
            return None
        raise ValueError("missing phase on one or both parties: " + name)
    return {
        "seconds": max(value["seconds"] for value in pair),
        "seconds_per_party": [value["seconds"] for value in pair],
        "sci_link_bytes": check_pair_bytes(*pair, "sent_bytes", "received_bytes", name),
        "per_party": pair,
    }


def indexed(rows, count, tag):
    if len(rows) != count:
        raise ValueError("expected %d %s records, found %d" % (count, tag, len(rows)))
    observed = [row.get("index", row.get("query", row.get("query_index"))) for row in rows]
    if observed != list(range(count)):
        raise ValueError(tag + " records are missing, duplicated, or out of order")
    return rows


def aggregate(metrics, logs, backend, batch, queries):
    if [metric.get("party") for metric in metrics] != [1, 2]:
        raise ValueError("metrics must contain distinct ordered parties 1 and 2")
    for metric in metrics:
        if len(metric.get("queries", [])) != queries:
            raise ValueError("query metric count differs from requested queries")
    bridge_queries = [indexed(records(log, "BRIDGE_QUERY"), queries, "BRIDGE_QUERY")
                      for log in logs]
    bridge_details = ([indexed(records(log, "BRIDGE_DETAIL"), queries, "BRIDGE_DETAIL")
                       for log in logs] if backend == "SCI_ours" else [[], []])
    offline = ([indexed(records(log, "OFFLINE_QUERY"), queries, "OFFLINE_QUERY")
                for log in logs] if backend == "SCI_ours" else [[], []])
    bridge_totals = [records(log, "BRIDGE") for log in logs]
    if any(len(rows) != 1 for rows in bridge_totals):
        raise ValueError("missing or duplicate final BRIDGE record")
    bridge_totals = [rows[0] for rows in bridge_totals]
    result = {
        "setup": {
            "sci_base_ot": phase_pair(metrics, "initialization_base_ot"),
            "fss_bridge": phase_pair(metrics, "fss_bridge_setup", required=backend == "SCI_ours"),
        },
        "application_io": phase_pair(metrics, "application_io", required=False),
        "queries": [],
        "total_seconds": max(metric["total_seconds"] for metric in metrics),
        "total_seconds_per_party": [metric["total_seconds"] for metric in metrics],
        "peak_rss_kib_per_party": [metric["peak_rss_kib"] for metric in metrics],
        "bridge_totals_per_party": bridge_totals,
        "dependent_rounds": None,
        "dependent_rounds_reason": "socket send/receive counts are not protocol dependency rounds",
        "phase_rule": "party maximum wall time; offline from OFFLINE_QUERY; online directly from query metrics",
        "communication_rule": "sum each party's sends once, SCI and bridge channels separately; never add receives",
    }
    for index in range(queries):
        pair = [metric["queries"][index] for metric in metrics]
        bridge_pair = [rows[index] for rows in bridge_queries]
        detail_pair = [rows[index] for rows in bridge_details] if bridge_details[0] else []
        sci_bytes = check_pair_bytes(*pair, "sent_bytes", "received_bytes", "query SCI")
        online_seconds = max(value["seconds"] for value in pair)
        if any(value.get("offline_ms", 0) != 0 for value in bridge_pair):
            raise ValueError("FSS offline work occurred inside the online query")
        if backend == "SCI_ours":
            for value in bridge_pair:
                if (value.get("calls") != 297 or value.get("elements") != 297 * batch * 100 or
                        value.get("segment_keys") != 297 * batch * 100 * 4):
                    raise ValueError("FSS replacement coverage differs from the fixed graph")
            offline_pair = [rows[index] for rows in offline]
            offline_seconds = max(record_seconds(value) for value in offline_pair)
            offline_bytes = check_pair_bytes(*offline_pair, "sent_bytes", "received_bytes", "offline FSS")
            # The new driver's per-query FSS counters exclude offline and setup.
            # Do not substitute cumulative bridge bytes if this field is absent.
            if all("online_sent_bytes" in value for value in bridge_pair):
                fss_bytes = check_pair_bytes(*bridge_pair, "online_sent_bytes", "online_received_bytes", "query FSS")
            elif all("sent_bytes" in value for value in bridge_pair):
                fss_bytes = check_pair_bytes(*bridge_pair, "sent_bytes", "received_bytes", "query FSS")
            else:
                fss_bytes = None
        else:
            if any(value.get("calls", 0) != 0 for value in bridge_pair):
                raise ValueError("SCI baseline unexpectedly invoked the FSS bridge")
            offline_pair, offline_seconds, offline_bytes, fss_bytes = [], 0.0, 0, 0
        result["queries"].append({
            "index": index, "batch": batch,
            "offline_seconds": offline_seconds,
            "offline_seconds_per_party": [record_seconds(value) for value in offline_pair] if offline_pair else [0.0, 0.0],
            "online_seconds": online_seconds,
            "online_seconds_per_party": [value["seconds"] for value in pair],
            "online_seconds_per_sample": online_seconds / batch,
            "offline_link_bytes": offline_bytes,
            "online_sci_link_bytes": sci_bytes,
            "online_fss_link_bytes": fss_bytes,
            "online_link_bytes": sci_bytes + fss_bytes if fss_bytes is not None else None,
            "offline_per_party": offline_pair,
            "bridge_online_per_party": bridge_pair,
            "bridge_detail_per_party": detail_pair,
            "sci_query_per_party": pair,
            "fss_online_seconds": max(value.get("online_ms", 0) for value in bridge_pair) / 1000.0,
        })
    result["online_seconds_median"] = statistics.median(row["online_seconds"] for row in result["queries"])
    result["offline_seconds_median"] = statistics.median(row["offline_seconds"] for row in result["queries"])
    result["fss_offline_phase"] = phase_pair(metrics, "fss_offline_precompute", required=backend == "SCI_ours")
    result["sci_link_bytes"] = sum(check_pair_bytes(metrics[0]["phases"][name], metrics[1]["phases"][name],
                                  "sent_bytes", "received_bytes", "phase " + name)
                                  for name in metrics[0]["phases"])
    result["fss_link_bytes"] = check_pair_bytes(*bridge_totals, "sent_bytes", "received_bytes", "bridge total")
    result["total_link_bytes"] = result["sci_link_bytes"] + result["fss_link_bytes"]
    return result


def predictions(log, batch, queries, start, oracle):
    rows = [(int(index), int(value)) for index, value in
            re.findall(r"^PRED\s+(\d+)\s+(-?\d+)$", log, re.MULTILINE)]
    expected_indices = list(range(start, start + batch)) * queries
    if [index for index, _ in rows] != expected_indices:
        raise ValueError("predictions do not contain one complete ordered batch per query")
    if any(not 0 <= value < 30 for _, value in rows):
        raise ValueError("prediction is outside the 30 classes")
    blocks = [[value for _, value in rows[q * batch:(q + 1) * batch]] for q in range(queries)]
    if any(block != blocks[0] for block in blocks[1:]):
        raise ValueError("fresh-material queries produced different predictions")
    verification = {"samples_per_query": batch, "queries": queries,
                    "ordered_indices": list(range(start, start + batch)),
                    "predictions_per_query": blocks, "fresh_query_agreement": True,
                    "oracle_available": oracle is not None, "oracle_match": None}
    if oracle is not None:
        import numpy as np
        expected = np.load(oracle, allow_pickle=False).reshape(-1)
        if len(expected) < start + batch:
            raise ValueError("oracle does not cover requested sample range")
        expected = [int(value) for value in expected[start:start + batch]]
        verification.update(oracle_path=str(oracle), oracle_sha256=digest(oracle),
                            oracle_predictions=expected, oracle_match=all(block == expected for block in blocks))
        if not verification["oracle_match"]:
            raise ValueError("secure predictions differ from the frozen integer oracle")
    return verification


def build_connector(output):
    source = HERE.parent / "sci_sweep" / "connect_offset.c"
    binary = output / "connect_offset.so"
    command = ["gcc", "-shared", "-fPIC", "-O2", str(source), "-o", str(binary), "-ldl"]
    env = dict(os.environ, TMPDIR=str(output / "tmp"))
    with (output / "connect_offset_build.log").open("w") as log:
        result = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, env=env, cwd=output)
    write_json(output / "connect_offset_build.json", {"command": command, "exit_code": result.returncode})
    result.check_returncode()
    return binary


def relay_ready(child, timeout=10):
    with selectors.DefaultSelector() as selector:
        selector.register(child.stdout, selectors.EVENT_READ)
        if not selector.select(timeout):
            raise RuntimeError("relay readiness timed out")
        if child.stdout.readline().strip() != "READY":
            raise RuntimeError("relay did not announce READY")
    child.stdout.close()


def run_backend(args, backend, output, connector):
    output.mkdir()
    network = {"name": args.network, "simulated": args.network == "sirnn_lan",
               "description": ("local relay model of SiRNN LAN; not original paper hardware" if args.network == "sirnn_lan"
                               else "unshaped localhost sockets, both parties on one machine"),
               "rtt_ms": 0.8 if args.network == "sirnn_lan" else None,
               "mbps": 3016 if args.network == "sirnn_lan" else None}
    base = reserve_ports(connector is not None)
    commands, environment, children = [], [], []
    relay_command = None
    status, reason = "complete", None
    started = time.monotonic()
    result = {"backend": backend, "batch": args.batch, "queries_requested": args.queries,
              "start": args.start, "network": network, "binary": str(args.binary),
              "binary_sha256": digest(args.binary), "cpus": args.cpus,
              "timeout_seconds": args.timeout, "address_space_cap": None,
              "application_threads_per_party": 1, "nseg": 4 if backend == "SCI_ours" else None}
    try:
        with ExitStack() as stack:
            if connector is not None:
                mapping = ",".join("%d:%d" % (base + 1000 + offset, base + offset) for offset in CHANNEL_OFFSETS)
                relay_command = [sys.executable, str(HERE.parent / "google30_benchmark" / "link_proxy.py"),
                                 "--ports", mapping, "--rtt-ms", "0.8", "--mbps", "3016"]
                if args.cpus[2] is not None:
                    relay_command = ["taskset", "-c", str(args.cpus[2])] + relay_command
                relay_log = stack.enter_context((output / "relay.log").open("w"))
                relay = subprocess.Popen(relay_command, stdout=subprocess.PIPE, stderr=relay_log, text=True, cwd=output)
                stack.callback(stop, relay)
                relay_ready(relay)
            for role in (1, 2):
                owned = args.fixture / ("server_model.i16" if role == 1 else "client_input.i16")
                command = [str(args.binary), str(role), str(base), str(owned), str(args.start),
                           str(args.batch), "secure", str(args.queries)]
                if args.cpus[role - 1] is not None:
                    command = ["taskset", "-c", str(args.cpus[role - 1])] + command
                commands.append(command)
                env = dict(os.environ)
                for key in list(env):
                    if key.startswith("SEGFSS_SCI_") or key in ("GOOGLE30_METRICS", "SWEEP_CONNECT_OFFSET", "GOOGLE30_PROXY_OFFSET"):
                        del env[key]
                updates = {"OMP_NUM_THREADS": "1", "SEGFSS_SCI_ENABLE": "1" if backend == "SCI_ours" else "0",
                           "SEGFSS_SCI_WIDTH": "16", "SEGFSS_SCI_NSEG": "4",
                           "SEGFSS_SCI_PORT": str(base + 20), "SEGFSS_SCI_HOST": "127.0.0.1",
                           "SEGFSS_SCI_TRACE": str(output / ("party%d_trace.jsonl" % role)),
                           "GOOGLE30_METRICS": str(output / ("party%d_metrics.json" % role)),
                           "TMPDIR": str(args.output / "tmp")}
                if backend == "SCI_ours":
                    updates["SEGFSS_SCI_PRECOMPUTE_FIXED"] = "1"
                if connector is not None:
                    updates.update(SWEEP_CONNECT_OFFSET="1000", GOOGLE30_PROXY_OFFSET="1000",
                                   LD_PRELOAD=str(connector) + (":" + env["LD_PRELOAD"] if env.get("LD_PRELOAD") else ""))
                env.update(updates)
                environment.append(updates)
                log = stack.enter_context((output / ("party%d.log" % role)).open("w"))
                child = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT, env=env, cwd=output)
                children.append(child)
                stack.callback(stop, child)
            result.update(commands=commands, environment_overrides=environment, relay_command=relay_command)
            write_json(output / "run.json", result)
            deadline = started + args.timeout
            while any(child.poll() is None for child in children):
                exits = [child.poll() for child in children]
                if any(code is not None and code != 0 for code in exits):
                    status, reason = "failed", "party exited unsuccessfully"
                    break
                if connector is not None and relay.poll() is not None:
                    status, reason = "failed", "relay exited before both parties completed"
                    break
                if time.monotonic() >= deadline:
                    status, reason = "censored", "predeclared wall timeout"
                    break
                time.sleep(0.05)
    except Exception as error:
        status, reason = "failed", str(error)
    result.update(commands=commands, environment_overrides=environment, relay_command=relay_command,
                  party_exit_codes=[child.returncode for child in children],
                  observed_wall_seconds=time.monotonic() - started, status=status, reason=reason)
    if status == "complete" and result["party_exit_codes"] != [0, 0]:
        result.update(status="failed", reason="party exit codes are not both zero")
    if result["status"] == "complete":
        try:
            metrics = [json.loads((output / ("party%d_metrics.json" % role)).read_text()) for role in (1, 2)]
            logs = [(output / ("party%d.log" % role)).read_text() for role in (1, 2)]
            if any("TENSOR " in log for log in logs):
                raise ValueError("secure benchmark exposed diagnostic intermediate tensors")
            result["prediction_verification"] = predictions(logs[1], args.batch, args.queries, args.start, args.oracle)
            result["physical_phases"] = aggregate(metrics, logs, backend, args.batch, args.queries)
            write_json(output / "predictions.json", result["prediction_verification"])
        except Exception as error:
            result.update(status="failed", reason="result validation: " + str(error))
    write_json(output / "result.json", result)
    return result


def parse_cpus(value):
    items = value.split(",")
    if len(items) != 3:
        raise argparse.ArgumentTypeError("cpus must be party1,party2,relay (for example 10,12,14)")
    try:
        values = [None if item.lower() in ("none", "-") else int(item) for item in items]
    except ValueError as error:
        raise argparse.ArgumentTypeError(str(error))
    if any(cpu is not None and cpu < 0 for cpu in values):
        raise argparse.ArgumentTypeError("CPU indices must be nonnegative")
    return values


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--binary", type=Path, required=True)
    parser.add_argument("--fixture", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batch", type=int, default=128)
    parser.add_argument("--queries", type=int, default=2)
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--network", choices=("localhost", "sirnn_lan"), default="localhost")
    parser.add_argument("--cpus", type=parse_cpus, default=[10, 12, 14])
    parser.add_argument("--timeout", type=float, default=2400, help="wall timeout per backend in seconds")
    parser.add_argument("--oracle", type=Path, help="frozen predictions .npy; checked for every query")
    args = parser.parse_args(argv)
    if not 1 <= args.batch <= 256 or not 1 <= args.queries <= 2 or args.start < 0 or not math.isfinite(args.timeout) or args.timeout <= 0:
        parser.error("batch must be 1..256, queries 1..2, start nonnegative, timeout positive")
    args.binary, args.fixture, args.output = [path.resolve() for path in (args.binary, args.fixture, args.output)]
    if args.oracle:
        args.oracle = args.oracle.resolve()
    if not args.binary.is_file():
        parser.error("binary does not exist")
    for name in ("server_model.i16", "client_input.i16"):
        if not (args.fixture / name).is_file():
            parser.error("fixture lacks " + name)
    if (args.fixture / "client_input.i16").stat().st_size < (args.start + args.batch) * 3168 * 2:
        parser.error("fixture has fewer samples than requested")
    if args.oracle is None:
        args.oracle = next((path for path in (args.fixture / "oracle_predictions.npy", args.fixture / "oracle" / "predictions.npy") if path.is_file()), None)
    if args.oracle is not None and not args.oracle.is_file():
        parser.error("oracle does not exist")
    args.output.mkdir(parents=True, exist_ok=False)
    (args.output / "tmp").mkdir()
    configuration = {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()}
    configuration.update(backend_order=["SCI", "SCI_ours"], address_space_cap=None,
                         same_batch_each_query=True, full_test_set_evaluation=False)
    manifest = args.binary.parent / "runtime_manifest.json"
    if manifest.is_file():
        configuration["runtime_manifest"] = json.loads(manifest.read_text())
    contract = args.fixture / "contract.json"
    if contract.is_file():
        data = json.loads(contract.read_text())
        configuration["fixture_identity"] = data.get("identity")
        configuration["sample_ids"] = data.get("sample_ids", [])[args.start:args.start + args.batch]
        configuration["labels"] = data.get("labels", [])[args.start:args.start + args.batch]
    write_json(args.output / "config.json", configuration)
    connector = build_connector(args.output) if args.network == "sirnn_lan" else None
    results = []
    for backend in ("SCI", "SCI_ours"):
        result = run_backend(args, backend, args.output / backend, connector)
        results.append(result)
        print(json.dumps({"backend": backend, "status": result["status"], "reason": result["reason"],
                          "observed_wall_seconds": result["observed_wall_seconds"]}), flush=True)
        write_json(args.output / "manifest.json", {"config": configuration, "results": results})
    agreement = None
    if all(result["status"] == "complete" for result in results):
        agreement = results[0]["prediction_verification"]["predictions_per_query"] == results[1]["prediction_verification"]["predictions_per_query"]
    summary = {"schema_version": "google30-physical-phase-pair-v1", "config": configuration,
               "results": results, "cross_backend_prediction_agreement": agreement,
               "complete": all(result["status"] == "complete" for result in results) and agreement is True}
    write_json(args.output / "manifest.json", summary)
    return 0 if summary["complete"] else 1


if __name__ == "__main__":
    sys.exit(main())
