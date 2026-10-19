#!/usr/bin/env python3
"""Matched four-mode conversion experiment with fresh resident material."""
import argparse
import gzip
import importlib.util
import json
import os
from pathlib import Path
import resource
import statistics
import subprocess


def records(path, name):
    prefix = name + " "
    return [json.loads(line[len(prefix):]) for line in path.read_text().splitlines()
            if line.startswith(prefix)]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--binary", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--count", type=int, default=12800)
    parser.add_argument("--samples", type=int, default=3)
    parser.add_argument("--port", type=int, default=34000)
    args = parser.parse_args()
    if args.count < 1 or args.samples < 1:
        parser.error("count and samples must be positive")
    cpus = sorted(os.sched_getaffinity(0))
    if len(cpus) < 8:
        parser.error("eight logical CPUs required for disjoint four-CPU parties")
    spec = importlib.util.spec_from_file_location("pruning_round_trace",
        Path(__file__).with_name("round_trace.py"))
    round_trace = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(round_trace)
    args.output.mkdir(parents=True, exist_ok=True)
    binary = str(args.binary.resolve())
    budget = 1536 << 20
    evidence = []
    modes = [(0, 0), (1, 0), (0, 1), (1, 1)]
    for sample in range(args.samples):
        for equality, leaf in (modes if sample % 2 == 0 else modes[::-1]):
            case = args.output / ("e%d-l%d-s%d" % (equality, leaf, sample))
            case.mkdir()
            children = []
            for index, role in enumerate((2, 3)):
                affinity = cpus[index * 4:(index + 1) * 4]

                def limits(affinity=affinity):
                    resource.setrlimit(resource.RLIMIT_AS, (budget, budget))
                    os.sched_setaffinity(0, affinity)

                stream = (case / ("party%d.log" % role)).open("w")
                env = dict(os.environ, SDCF_ROUND_TRACE=str((case / ("trace%d.jsonl" % role)).resolve()))
                child = subprocess.Popen([binary, str(role), str(args.port), str(args.count),
                    str(equality), str(leaf)], stdout=stream, stderr=subprocess.STDOUT,
                    env=env, preexec_fn=limits)
                children.append((child, stream))
            try:
                statuses = [child.wait(timeout=600) for child, _ in children]
                if statuses != [0, 0]:
                    raise RuntimeError("two-party benchmark failed: %r" % statuses)
            finally:
                for child, stream in children:
                    if child.poll() is None:
                        child.terminate()
                    stream.close()
            logs = [case / ("party%d.log" % role) for role in (2, 3)]
            phases = [records(log, "METRIC") for log in logs]
            ready = [records(log, "PRUNING_READY") for log in logs]
            detail = [records(log, "PRUNING_DETAIL") for log in logs]
            rounds = round_trace.analyze_pair(case / "trace2.jsonl", case / "trace3.jsonl")
            if rounds["status"] != "passed":
                raise RuntimeError("causal trace failed: " + rounds.get("reason", ""))
            measured = {row["phase"]: row for row in rounds["phases"]}
            report = {"equality": bool(equality), "leaf": bool(leaf), "sample": sample,
                "count": args.count, "memory_budget_bytes_per_party": budget,
                "cpu_affinities": [cpus[:4], cpus[4:8]], "parties": [], "phases": {}}
            for index in range(2):
                if (len(ready[index]) != 1 or not ready[index][0]["before_inputs"] or
                        ready[index][0]["material_load_online_ms"] != 0 or len(detail[index]) != 1):
                    raise RuntimeError("material residency evidence missing")
                row = detail[index][0]
                if (row["generated_scan_triples"] != row["consumed_scan_triples"] or
                        row["scan_and_gates"] != row["generated_scan_triples"] or
                        row["max_rss_bytes"] > budget or row["resident_bytes"] > budget or
                        row["eval_participants"] > 4):
                    raise RuntimeError("inventory or resource contract failed")
                report["parties"].append({"ready": ready[index][0], "detail": row, "phases": phases[index]})
            for phase in ("offline", "online"):
                rows = [next(row for row in side if row["phase"] == phase) for side in phases]
                trace = measured[phase]
                if ([row["sent_bytes"] for row in rows] != trace["party_sent_bytes"] or
                        [row["received_bytes"] for row in rows] != trace["party_received_bytes"]):
                    raise RuntimeError("phase metric differs from actual traced wire payload")
                wire = sum(row["sent_bytes"] for row in rows)
                depth = trace["logical_rounds"]
                report["phases"][phase] = {"seconds": max(row["seconds"] for row in rows),
                    "sent_bytes_both_parties": wire, "causal_rounds": depth,
                    "lan_model_seconds": wire * 8 / 1e9 + depth * 0.0005,
                    "wan_model_seconds": wire * 8 / 4e8 + depth * 0.004}
            (case / "result.json").write_text(json.dumps(report, indent=2) + "\n")
            (case / "rounds.json").write_text(json.dumps(rounds, indent=2) + "\n")
            for role in (2, 3):
                path = case / ("trace%d.jsonl" % role)
                with path.open("rb") as source, gzip.open(str(path) + ".gz", "wb") as target:
                    target.write(source.read())
                path.unlink()
            evidence.append(report)
            print(case.name, "PASS", flush=True)
    summary = {"definition": "isolated resident signed div_pow2 conversion, not full FastGRNN",
        "count": args.count, "samples_per_mode": args.samples, "trace_definition": round_trace.DEFINITION,
        "network": {"lan_bps": 1000000000, "lan_rtt_seconds": 0.0005,
                    "wan_bps": 400000000, "wan_rtt_seconds": 0.004}, "modes": []}
    for equality, leaf in modes:
        rows = [row for row in evidence if row["equality"] == bool(equality) and row["leaf"] == bool(leaf)]
        summary["modes"].append({"equality": bool(equality), "leaf": bool(leaf),
            "phases": {phase: {field: statistics.median(row["phases"][phase][field] for row in rows)
                for field in rows[0]["phases"][phase]} for phase in ("offline", "online")},
            "party_details": [row["parties"][0]["detail"] for row in rows]})
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")


if __name__ == "__main__":
    main()
