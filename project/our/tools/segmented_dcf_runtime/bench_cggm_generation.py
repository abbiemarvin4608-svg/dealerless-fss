#!/usr/bin/env python3
"""Matched isolated cGGM offline-generation and resident local-evaluation ablations."""
import argparse
import gzip
import importlib.util
import json
import os
from pathlib import Path
import resource
import statistics
import subprocess
import time


def records(path, name):
    prefix = name + " "
    return [json.loads(line[len(prefix):]) for line in path.read_text().splitlines()
            if line.startswith(prefix)]


def sample_resources(pid, peak):
    try:
        rows = Path("/proc/%d/status" % pid).read_text().splitlines()
    except FileNotFoundError:
        return
    for line in rows:
        name = line.split(":", 1)[0]
        if name in ("VmRSS", "VmSize", "Threads"):
            value = int(line.split()[1]) * (1024 if name != "Threads" else 1)
            peak[name] = max(peak.get(name, 0), value)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--binary", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--count", type=int, default=12800)
    parser.add_argument("--depth", type=int, default=8)
    parser.add_argument("--tile", type=int, default=1024)
    parser.add_argument("--samples", type=int, default=3)
    parser.add_argument("--port", type=int, default=35000)
    parser.add_argument("--aes-offline", type=int, choices=(0, 1), default=0)
    args = parser.parse_args()
    if args.count < 1 or args.tile < 1 or args.samples < 1 or not 1 <= args.depth <= 16:
        parser.error("positive count/tile/samples and depth in [1,16] required")
    cpus = sorted(os.sched_getaffinity(0))
    if len(cpus) < 8:
        parser.error("eight logical CPUs required for disjoint four-CPU parties")
    spec = importlib.util.spec_from_file_location("cggm_round_trace", Path(__file__).with_name("round_trace.py"))
    round_trace = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(round_trace)
    args.output.mkdir(parents=True, exist_ok=True)
    binary = str(args.binary.resolve())
    budget = 1536 << 20
    modes = [("legacy", 0, 0), ("block_ot", 0, 0), ("direct_ot_legacy", 0, 0),
             ("direct_ot", 0, 0), ("direct_ot", 1, 0), ("direct_ot", 0, 1), ("direct_ot", 1, 1)]
    evidence = []
    for sample in range(args.samples):
        for mode, merge, reuse in (modes if sample % 2 == 0 else modes[::-1]):
            case = args.output / ("%s-c%d-w%d-s%d" % (mode, merge, reuse, sample))
            case.mkdir()
            children = []
            for index, role in enumerate((2, 3)):
                affinity = cpus[index * 4:(index + 1) * 4]

                def limits(affinity=affinity):
                    resource.setrlimit(resource.RLIMIT_AS, (budget, budget))
                    os.sched_setaffinity(0, affinity)

                stream = (case / ("party%d.log" % role)).open("w")
                env = dict(os.environ, SDCF_ROUND_TRACE=str((case / ("trace%d.jsonl" % role)).resolve()),
                           OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1")
                child = subprocess.Popen([binary, str(role), str(args.port), str(args.count),
                    str(args.depth), str(args.tile), mode, str(merge), str(reuse),
                    str(args.aes_offline)],
                    stdout=stream, stderr=subprocess.STDOUT, env=env, preexec_fn=limits)
                children.append((child, stream, {}))
            try:
                deadline = time.monotonic() + 600
                while any(child.poll() is None for child, _, _ in children):
                    for child, _, peak in children:
                        sample_resources(child.pid, peak)
                    if time.monotonic() > deadline:
                        raise RuntimeError("two-party benchmark exceeded 600 seconds")
                    time.sleep(0.05)
                statuses = [child.returncode for child, _, _ in children]
                if statuses != [0, 0]:
                    raise RuntimeError("two-party benchmark failed: %r" % statuses)
            finally:
                for child, stream, _ in children:
                    if child.poll() is None:
                        child.terminate()
                    stream.close()
            logs = [case / ("party%d.log" % role) for role in (2, 3)]
            phases = [records(log, "METRIC") for log in logs]
            ready = [records(log, "CGGM_READY") for log in logs]
            detail = [records(log, "CGGM_DETAIL") for log in logs]
            rounds = round_trace.analyze_pair(case / "trace2.jsonl", case / "trace3.jsonl")
            if rounds["status"] != "passed":
                raise RuntimeError("causal trace failed: " + rounds.get("reason", ""))
            measured = {row["phase"]: row for row in rounds["phases"]}
            report = {"mode": mode, "merge_leaf_cw": bool(merge), "reuse_workspace": bool(reuse),
                "aes_offline": bool(args.aes_offline),
                "sample": sample, "count": args.count, "depth": args.depth, "tile": args.tile,
                "memory_budget_bytes_per_party": budget, "cpu_affinities": [cpus[:4], cpus[4:8]],
                "parties": [], "phases": {}}
            for index in range(2):
                if (len(ready[index]) != 1 or not ready[index][0]["before_inputs"] or
                        not ready[index][0]["workspace_destroyed"] or
                        ready[index][0]["material_load_online_ms"] != 0 or len(detail[index]) != 1):
                    raise RuntimeError("material residency evidence missing")
                row = detail[index][0]
                products = args.count * args.depth
                direct = mode.startswith("direct_ot")
                packed = mode in ("block_ot", "direct_ot")
                if (row["products"] != products or row["temporary_triples"] != (0 if direct else products) or
                        row["consumed_temporary_triples"] != row["temporary_triples"] or
                        row["general_ot_64"] != (0 if packed else 4 * products) or
                        row["general_ot_128"] != (2 * products if packed else 0) or
                        row["max_rss_bytes"] > budget or row["vm_peak_bytes"] > budget or
                        row["resident_bytes"] > budget or row["keygen_participants"] > 4 or
                        row["eval_participants"] > 4 or
                        (reuse and row["workspace_growths_after_first_tile"] != 0)):
                    raise RuntimeError("inventory or resource contract failed")
                report["parties"].append({"ready": ready[index][0], "detail": row,
                    "phases": phases[index], "sampled_resources": children[index][2]})
            for phase in ("setup", "offline", "online"):
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
            if report["phases"]["online"]["sent_bytes_both_parties"] or report["phases"]["online"]["causal_rounds"]:
                raise RuntimeError("local cGGM evaluation unexpectedly communicated")
            (case / "result.json").write_text(json.dumps(report, indent=2) + "\n")
            (case / "rounds.json").write_text(json.dumps(rounds, indent=2) + "\n")
            for role in (2, 3):
                path = case / ("trace%d.jsonl" % role)
                with path.open("rb") as source, gzip.open(str(path) + ".gz", "wb") as target:
                    target.write(source.read())
                path.unlink()
            evidence.append(report)
            print(case.name, "PASS", flush=True)
    summary = {"definition": "isolated real two-party cGGM generation and resident local evaluation; not full FastGRNN",
        "count": args.count, "depth": args.depth, "tile": args.tile, "samples_per_mode": args.samples,
        "aes_offline": bool(args.aes_offline),
        "schedule": "fresh sessions, forward/reverse/forward seven-mode order; no priming or warmup",
        "base_ot_accounting": "Peer construction is setup; lazily first-used OT/base cost remains offline",
        "query_fixture": "%d independent depth-%d segment keys; fixed XOR-shared targets, queries created after ready" %
                         (args.count, args.depth),
        "trace_definition": round_trace.DEFINITION,
        "network": {"lan_bps": 1000000000, "lan_rtt_seconds": 0.0005,
                    "wan_bps": 400000000, "wan_rtt_seconds": 0.004}, "modes": []}
    for mode, merge, reuse in modes:
        rows = [row for row in evidence if (row["mode"], row["merge_leaf_cw"], row["reuse_workspace"]) ==
                (mode, bool(merge), bool(reuse))]
        summary["modes"].append({"mode": mode, "merge_leaf_cw": bool(merge), "reuse_workspace": bool(reuse),
            "phases": {phase: {field: statistics.median(row["phases"][phase][field] for row in rows)
                for field in rows[0]["phases"][phase]} for phase in ("setup", "offline", "online")},
            "samples": [{"sample": row["sample"], "phases": row["phases"]} for row in rows],
            "party_details": [row["parties"][0]["detail"] for row in rows]})
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")


if __name__ == "__main__":
    main()
