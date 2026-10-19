#!/usr/bin/env python3
"""Qualify the repaired historical DPF control before freezing new timings."""
import argparse
import json
from pathlib import Path
import sys

from benchmark import MEMORY, REPO, RUNTIME, command, declarations, digest, read, write

sys.path.insert(0, str(REPO / "project/our/tools/runtime_optimization"))
from run_plan import run_plan
sys.path.insert(0, str(RUNTIME))
import run as runtime
from audit_folded_step import audit


def qualify(directory, binary, fixture):
    result = read(directory / "result.json")
    config = result["config"]
    logs = [(directory / ("party%d.log" % role)).read_text() for role in (1, 2)]
    parties, _ = runtime.collect(logs, 2, read(directory / "schedule.json"), True,
        config["backend"], config["native_preprocessing_effective"])
    integers = audit(directory, REPO / "project/our/results/paper_folded_activation_20261013/ticket02",
                     fixture, "sci")
    integer_path = directory / "integer-audit.json"
    write(integer_path, integers)
    checks = dict(
        complete=result["status"] == "complete" and result["party_exit_codes"] == [0, 0],
        current_binary=config["binary_sha256"] == digest(binary),
        workload=config["batch"] == 128 and config["steps"] == 1 and config["queries"] == 2
            and config["warmup_queries"] == 0 and config["diagnostic"],
        integer=integers["status"] == "passed" and len(integers["rows"]) == 2
            and all(row["integer_values"] == 172928 and row["mismatches"] == 0 for row in integers["rows"]),
        actual_limit=result["address_space_hard_limit_bytes_per_party"] == MEMORY,
        affinity=config["cpu_affinity_per_party"] == [[0, 2, 4, 10], [6, 8, 12, 16]],
        causal_trace=result["round_measurements"]["status"] == "passed",
        material_residency=True, joint_prefix_budget=True, resources=True,
        online_prefix_ready=True, participant_limit=True)
    evidence = []
    for index, (log, party) in enumerate(zip(logs, parties)):
        reserves = runtime.records(log, "DPF_LUT_RESERVE")
        prepared = runtime.records(log, "DPF_LUT_PREPARED")
        balanced = runtime.records(log, "DPF_LUT_BALANCE")
        reserve = reserves[0]["resident_budget_bytes"] if len(reserves) == 1 else 0
        resident = [sum(row["resident_bytes"] for row in prepared if row["query"] == query)
                    for query in range(2)]
        checks["material_residency"] &= reserve > 0 and len(prepared) == 8 and all(
            0 < size <= reserve for size in resident) and all(
            row["pre_input_eval_all"] and row["no_online_reload"] for row in prepared) and len(balanced) == 2
        budgets = party["conversion_prefetch_budgets"]
        checks["joint_prefix_budget"] &= len(budgets) == 2 and all(
            row["configured_party_budget_bytes"] == 536870912 and row["dpf_reserved_bytes"] == reserve
            and row["party_budget_bytes"] == 536870912 - reserve for row in budgets)
        peaks = result["process_resource_peaks"][index]
        vm = max(peaks["vm_peak_bytes"], party.get("vm_peak_bytes", 0))
        rss = max(peaks["rss_peak_bytes"], party["peak_rss_kib"] * 1024)
        checks["resources"] &= 0 < vm <= MEMORY and 0 < rss <= MEMORY
        details = party["conversion_details"]
        checks["online_prefix_ready"] &= bool(details) and all(
            row["prefetched"] == 1 and row["material_load_ms"] == row["prefetch_wait_ms"] == 0
            for row in details)
        checks["participant_limit"] &= bool(party["graph_groups"]) and all(
            row["observed_compute_upper_bound"] <= 4 for row in party["graph_groups"]) and all(
            row["compute_participants"] <= 4 for row in party["activations"])
        evidence.append(dict(party=index + 1, reserve_bytes=reserve, resident_bytes_per_query=resident,
                             vm_peak_bytes=vm, rss_peak_bytes=rss))
    return dict(status="passed" if all(checks.values()) else "failed", checks=checks,
        result_path=str(directory / "result.json"), integer_audit_path=str(integer_path),
        integer_values_per_query=[row["integer_values"] for row in integers["rows"]], parties=evidence)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("binary", "bridge", "fixture", "hidden", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args()
    binary, fixture, output = args.binary.resolve(), args.fixture.resolve(), args.output.resolve()
    declaration = next(row for row in declarations() if row["id"] == "parallel-a2-SCI_Beaver_segmented_DPF_LUT")
    rows = []
    for state in ("default", "nonzero"):
        invocation = command(declaration, binary, fixture)
        invocation[invocation.index("--queries") + 1] = "2"
        invocation += ["--diagnostic"]
        if state == "nonzero":
            invocation += ["--hidden", str(args.hidden.resolve())]
        rows.append(dict(id=state, command=["/usr/bin/prlimit", "--as=%d" % MEMORY, "--"] + invocation))
    run_plan(dict(scope="DPF budget repair correctness and residency; not formal performance", rows=rows), output)
    proofs = []
    for row in rows:
        try:
            proof = qualify(output / row["id"], binary, fixture)
        except (OSError, ValueError, KeyError, TypeError) as error:
            proof = dict(status="failed", reason=str(error))
        proofs.append(dict(id=row["id"], **proof))
    report = dict(status="passed" if all(row["status"] == "passed" for row in proofs) else "failed",
        binary_sha256=digest(binary), bridge_sha256=digest(args.bridge), rows=proofs)
    write(output / "qualification.json", report)
    print(json.dumps(report))
    return 0 if report["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
