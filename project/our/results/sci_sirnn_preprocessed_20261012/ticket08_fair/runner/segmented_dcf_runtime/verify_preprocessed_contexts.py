#!/usr/bin/env python3
"""Freeze and verify the complete batch-128 context diagnostic matrix."""
import argparse
import json
from pathlib import Path
import sys

from build import digest
from run import collect, write_json
from runtime import dependency_schedule
from verify_native_workers import tensors

HERE = Path(__file__).resolve().parent
BACKENDS = ("SCI", "SCI_preprocessed", "SCI_ours", "SCI_Beaver_segmented")
COHORTS = ("a4-sequential", "a2-parallel")


def diagnostic_rows(binary, fixture, hidden):
    shared = [sys.executable, str(HERE / "run.py"), "--binary", str(binary),
        "--fixture", str(fixture), "--output", "{output}", "--batch", "128", "--steps", "1",
        "--start", "0", "--queries", "2", "--warmup", "0", "--diagnostic",
        "--network", "localhost", "--memory-gib", "1.5", "--timeout", "600", "--cpus", "10,12,14",
        "--server-cpus", "0,2,4,10", "--client-cpus", "6,8,12,16", "--graph-openings", "independent",
        "--max-segments", "4", "--material-layout", "contiguous", "--segment-policy", "fixed-4",
        "--generation-tile-keys", "0", "--generation-budget-bytes", "1610612736",
        "--generation-reserve-bytes", "536870912", "--fuse-domains", "--pair-extensions",
        "--pack-bits", "--aes-online", "--beaver-algebra", "--beaver-tile-rows", "32"]
    rows = []
    for state in ("default", "nonzero"):
        for cohort in COHORTS:
            workers, graph = ("4", "sequential") if cohort == "a4-sequential" else ("2", "parallel")
            for backend in BACKENDS:
                command = shared + ["--cggm-threads", workers, "--activation-workers", workers,
                                    "--graph-mode", graph, "--backend", backend]
                if backend != "SCI":
                    command += ["--native-preprocessing"]
                if state == "nonzero":
                    command += ["--hidden", str(hidden)]
                rows.append(dict(id=state + "-" + cohort + "-" + backend, hidden=state,
                                 cohort=cohort, backend=backend, command=command))
    return rows


def make_plan(binary, fixture, hidden, bridge):
    return dict(scope="diagnostic correctness only; excluded from formal timing",
                binary=str(binary), binary_sha256=digest(binary), bridge=str(bridge),
                bridge_sha256=digest(bridge), queries=2, batch=128, steps=1,
                tensors_per_query=15, integers_per_query=172928,
                rows=[dict(row, environment={"LD_LIBRARY_PATH": str(bridge.parent)})
                      for row in diagnostic_rows(binary, fixture, hidden)])


def verify(directory, reference_dir):
    manifest = json.loads((directory / "manifest.json").read_text())
    if digest(manifest["binary"]) != manifest["binary_sha256"]:
        raise ValueError("frozen diagnostic binary changed")
    if digest(manifest["bridge"]) != manifest["bridge_sha256"]:
        raise ValueError("frozen diagnostic bridge changed")
    if len(manifest["rows"]) != 16:
        raise ValueError("the final diagnostic matrix must contain all sixteen rows")
    reports = []
    for row in manifest["rows"]:
        output = directory / row["id"]
        result = json.loads((output / "result.json").read_text())
        if result["status"] != "complete":
            raise ValueError("diagnostic row incomplete: " + row["id"])
        config = result["config"]
        strict = row["backend"] != "SCI"
        if (config["binary_sha256"] != manifest["binary_sha256"] or
                config["native_preprocessing_requested"] is not strict or
                config["native_preprocessing_effective"] is not strict or
                config["memory_limit_bytes_per_party"] != 1610612736 or
                config["cpu_affinity_per_party"] != [[0, 2, 4, 10], [6, 8, 12, 16]]):
            raise ValueError("diagnostic row differs from frozen resources or native mode: " + row["id"])
        actual = tensors(output / "party2.log")
        control = directory / (row["hidden"] + "-" + row["cohort"] + "-SCI") / "party2.log"
        if (len(actual) != 30 or sum(len(line.split()) - 7 for line in actual) != 345856 or
                actual != tensors(control)):
            raise ValueError("fresh SCI integer mismatch: " + row["id"])
        reference = tensors(reference_dir / row["hidden"] / "SCI-n4-t4/party2.log")
        if len(reference) != 15 or actual != reference * 2:
            raise ValueError("frozen native integer mismatch: " + row["id"])
        logs = [(output / ("party%d.log" % role)).read_text() for role in (1, 2)]
        schema = json.loads((Path(config["binary"]).parent / "schema.json").read_text())
        parties, _ = collect(logs, 2, dependency_schedule(schema, 128, 1, classifier=False),
                             True, row["backend"], native_preprocessing=strict)
        branch_limit = 4 if row["cohort"] == "a4-sequential" else 2
        for party in parties:
            if (len(party["native_worker_config"]) != 1 or
                    party["native_worker_config"][0]["protocol_contexts"] != 4 or
                    party["peak_rss_kib"] * 1024 >= 1610612736):
                raise ValueError("diagnostic context or memory allocation differs: " + row["id"])
            if any(worker["ot_participants"] != 1 for worker in party["activation_workers"]):
                raise ValueError("activation OT teams exceeded one participant: " + row["id"])
            for detail in party["activations"]:
                slots = {worker["slot"] for worker in party["activation_workers"]
                         if (worker["query"], worker["index"]) == (detail["query"], detail["index"])}
                expected_slots = (set(range(4)) if branch_limit == 4 else
                                  {0, 1} if detail["function"] == "Sigmoid" else {2, 3})
                if slots != expected_slots:
                    raise ValueError("activation context ownership differs: " + row["id"])
            if any(detail["compute_participants"] > branch_limit for detail in party["activations"]):
                raise ValueError("activation exceeded the branch allocation: " + row["id"])
            if any(task["observed_compute_participants"] > task["branch_compute_limit"]
                   for task in party["graph_tasks"]):
                raise ValueError("graph task exceeded the branch allocation: " + row["id"])
            if strict and any({item["context"] for item in party["native_public_plan"]
                               if item["query"] == query} != set(range(4)) for query in range(2)):
                raise ValueError("prepared inventory omits a live protocol context: " + row["id"])
        reports.append(dict(id=row["id"], backend=row["backend"], hidden=row["hidden"],
            cohort=row["cohort"], status="passed", queries=2, tensors_per_query=15,
            integers_per_query=172928, fresh_sci_equal=True, frozen_native_equal=True,
            native_preprocessing=strict,
            peak_rss_kib_per_party=[party["peak_rss_kib"] for party in parties],
            planned_ot_calls_per_party=[len(party["native_public_plan"]) for party in parties],
            consumed_ot_per_party=[sum(item["consumed"] for item in party["native_material_balances"]) for party in parties],
            consumed_triples_per_party=[sum(item["consumed"] for item in party["native_triple_balances"]) for party in parties]))
    report = dict(scope=manifest["scope"], binary_sha256=manifest["binary_sha256"],
                  bridge_sha256=manifest["bridge_sha256"], rows=reports)
    write_json(directory / "integer_verification.json", report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    plan = commands.add_parser("plan")
    for name in ("binary", "fixture", "hidden", "bridge", "output"):
        plan.add_argument("--" + name, type=Path, required=True)
    check = commands.add_parser("verify")
    check.add_argument("--directory", type=Path, required=True)
    check.add_argument("--reference-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "plan":
        write_json(args.output.resolve(), make_plan(args.binary.resolve(), args.fixture.resolve(),
                   args.hidden.resolve(), args.bridge.resolve()))
    else:
        report = verify(args.directory.resolve(), args.reference_dir.resolve())
        print(json.dumps(dict(status="passed", rows=len(report["rows"]))))


if __name__ == "__main__":
    main()
