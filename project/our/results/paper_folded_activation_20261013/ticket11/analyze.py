#!/usr/bin/env python3
"""Qualify all frozen rows and report measured costs without dropping failures."""
from __future__ import annotations

import argparse
from functools import lru_cache
import json
import math
from pathlib import Path
import re
import statistics
import sys

from benchmark import BASELINE, FAMILIES, HERE, MEMORY, NETWORKS, PAPER, PREFETCH, RUNTIME, digest, read, verify_manifest, write

sys.path.insert(0, str(BASELINE))
from analyze_samples import compare_groups, distribution
from network_model import phase_time
sys.path.insert(0, str(RUNTIME))
import run as runtime
from audit_folded_parallel import audit as audit_folded
from scope_rounds import replay

PHASES = ("offline", "input", "ready", "online", "output", "query_total")
KINDS = ("sigmoid", "tanh")
LUT_STAGES = runtime.FOLDED_STAGE_NAMES[8:16]


def exactly(rows, **identity):
    matches = [row for row in rows if all(row.get(key) == value for key, value in identity.items())]
    if len(matches) != 1:
        raise ValueError("missing or duplicated measured boundary: " + str(identity))
    return matches[0]


def link_cost(seconds, sent_bytes, rounds):
    return dict(wall_seconds=seconds, sent_bytes=sent_bytes, causal_rounds=rounds,
        network={name: phase_time(sent_bytes, rounds, **spec) for name, spec in NETWORKS.items()}
            if rounds is not None else None)


def union_rounds(scopes, query, phase, kind):
    matches = [row for row in scopes if row.get("label") == "ticket11_lut_union" and
               (row["query"], row["phase"], row["kind"]) == (query, phase, kind)]
    return exactly(matches, query=query)["logical_rounds"] if matches else None


def window(parties, query, kind, stage_indices):
    spans = []
    for party in parties:
        stages = [row["online"][index] for row in party["folded_activation_consumed"]
                  if (row["query"], row["kind"]) == (query, kind) for index in stage_indices]
        if not stages or any("begin_us" not in row or "end_us" not in row for row in stages):
            return None
        if any(not 0 <= row["begin_us"] <= row["end_us"] for row in stages):
            raise ValueError("invalid monotonic LUT stage bounds")
        spans.append((max(row["end_us"] for row in stages) - min(row["begin_us"] for row in stages)) / 1e6)
    return dict(seconds=max(spans), per_party_seconds=spans,
                boundary="earliest selected stage begin to latest end on each party, including scheduling gaps")


def extract(result, parties):
    measured, rounds = result["measurements"], result["round_measurements"]
    scopes = rounds["scopes"]
    queries = []
    for query, phases in enumerate(measured["queries"]):
        step = {}
        for phase in PHASES:
            row = exactly(rounds["phases"], query=query, phase=phase) if phase != "query_total" else None
            step[phase] = link_cost(phases[phase]["seconds"], phases[phase]["sent_bytes"],
                                    row["logical_rounds"] if row else None)
        activations = {}
        for kind, name in enumerate(KINDS):
            observed = [exactly(party["activations"], query=query, function="Sigmoid" if kind == 0 else "TanH")
                        for party in parties]
            online = exactly(scopes, query=query, phase="online", label="activation", kind=kind)
            value = dict(online=link_cost(max(row["wall_seconds"] for row in observed),
                sum(online["party_sent_bytes"]), online["logical_rounds"]),
                actual_workers_per_party=[row["actual_workers"] for row in observed],
                compute_participants_per_party=[row["compute_participants"] for row in observed])
            if result["config"]["backend"] == PAPER:
                attribution = measured["folded_activations"][query][name]
                value["offline_wall_seconds"] = None
                value["offline_wall_reason"] = "per-kind offline timers are work attribution; no isolated-kind elapsed boundary"
                value["stages"], value["lut"] = {}, {}
                for phase in ("offline", "online"):
                    stages = attribution[phase]["stages"]
                    value["stages"][phase] = {stage: dict(
                        work_seconds_per_party=[item / 1e6 for item in row["work_us_per_party"]],
                        work_seconds=max(row["work_us_per_party"]) / 1e6,
                        sent_bytes=row["sent_bytes"],
                        causal_scopes=[scope for scope in scopes if scope["label"] == "activation_stage"
                            and (scope["query"], scope["phase"], scope["kind"], scope["stage"]) ==
                            (query, phase, kind, runtime.FOLDED_STAGE_NAMES.index(stage))])
                        for stage, row in stages.items()}
                    work = [sum(stages[stage]["work_us_per_party"][party] for stage in LUT_STAGES) / 1e6
                            for party in (0, 1)]
                    sent = sum(stages[stage]["sent_bytes"] for stage in LUT_STAGES)
                    causal = union_rounds(scopes, query, phase, kind)
                    value["lut"][phase] = dict(link_cost(None, sent, causal),
                        work_seconds=max(work), work_seconds_per_party=work,
                        work_rule="sum selected LUT stages per party; maximum party work, not elapsed time")
                    union = [scope for scope in scopes if scope.get("label") == "ticket11_lut_union" and
                             (scope["query"], scope["phase"], scope["kind"]) == (query, phase, kind)]
                    if union and sum(union[0]["party_sent_bytes"]) != sent:
                        raise ValueError("LUT union traffic differs from stage attribution")
                value["lut"]["online"]["event_window"] = window(parties, query, kind, (14, 15))
                value["lut"]["online"]["opening_window"] = window(parties, query, kind, (14,))
                value["lut"]["online"]["product_window"] = window(parties, query, kind, (15,))
                value["calls"] = attribution["calls"]
            activations[name] = value
        row = dict(query=query, step=step, activations=activations)
        if result["config"]["backend"] == PAPER:
            prepare = [exactly(party["folded_prepare_groups"], query=query) for party in parties]
            scope = exactly(scopes, query=query, phase="offline", label="folded_prepare")
            row["folded_prepare"] = dict(link_cost(max(item["wall_seconds"] for item in prepare),
                sum(item["sent_bytes"] for item in prepare), scope["logical_rounds"]),
                schedule="serial_slot_major")
        queries.append(row)
    table = [party.get("folded_public_table_preparation", []) for party in parties]
    public_table = None
    if any(table):
        if not all(table):
            raise ValueError("public-table preparation evidence is absent for one party")
        work = [sum(row["elapsed_us"] for row in rows) / 1e6 for rows in table]
        public_table = dict(work_seconds_per_party=work, work_seconds=max(work), records=table,
            sent_bytes=sum(row["sent_bytes"] for rows in table for row in rows),
            accounting="subset of setup; do not add it again to setup or first request")
    return dict(queries=queries, setup=measured["setup"], model_prepare=measured.get("model_prepare"),
        public_table_prepare=public_table, first_query=measured.get("first_query"),
        session_total_seconds=measured["total_seconds"],
        process_resource_peaks=result.get("process_resource_peaks"),
        conservative_process_resource_peaks=[dict(
            vm_peak_bytes=max(result["process_resource_peaks"][index]["vm_peak_bytes"], party["vm_peak_bytes"]),
            rss_peak_bytes=max(result["process_resource_peaks"][index]["rss_peak_bytes"], party["peak_rss_kib"] * 1024))
            for index, party in enumerate(parties)],
        peak_rss_kib_per_party=measured["peak_rss_kib_per_party"],
        activation_algorithm=result["config"]["activation_algorithm"],
        actual_worker_config=[party["native_worker_config"] for party in parties],
        actual_graph_groups=[party["graph_groups"] for party in parties])


def qualify(path, declaration, plan, replay_rounds=False):
    result = read(path / "result.json")
    config = result["config"]
    checks = dict(complete=result.get("status") == "complete" and result.get("party_exit_codes") == [0, 0],
        fresh_queries=config["queries"] == 10 and config["warmup_queries"] == 0 and not config["diagnostic"],
        backend=config["backend"] == declaration["backend"],
        graph=config["graph_mode"] == declaration["graph_mode"] and
              config["activation_workers_per_party"] == declaration["activation_workers"],
        workload=config["batch"] == 128 and config["steps"] == 1 and config["start"] == 0 and not config["hidden_file"],
        frozen_binary=config["binary_sha256"] == plan["binary_sha256"],
        frozen_fixture=config["fixture_files"] == {name: plan["file_identities"][str(Path(plan["fixture"]) / name)]
                         for name in ("server_model.i16", "client_input.i16")},
        resources=config["memory_limit_bytes_per_party"] == MEMORY and
                  result.get("address_space_hard_limit_bytes_per_party") == MEMORY and
                  config["cpu_affinity_per_party"] == [[0, 2, 4, 10], [6, 8, 12, 16]],
        public_policy=config["segment_policy"] == "fixed-4" and config["prefetch_bytes"] == PREFETCH and
                      config["model_bias_precomputed"] and config["cggm_worker_limit_per_party"] == 2,
        unshaped_network=config["network"]["name"] == "localhost", trace_requested=config["round_trace"] is True,
        no_exclusion=not any((ancestor / "PERFORMANCE_EXCLUDED.json").exists()
                             for ancestor in (path, path.parent, path.parent.parent)))
    if declaration["backend"] == PAPER:
        checks["folded_identity"] = config["lut_key_family"] == FAMILIES[declaration["family"]] and \
            config["folded_sign_backend"] == declaration["sign_backend"] and config["folded_activations"] == "both"
    if not all(checks.values()):
        raise ValueError("formal contract failed: " + ", ".join(key for key, value in checks.items() if not value))
    logs = [(path / ("party%d.log" % role)).read_text() for role in (1, 2)]
    parties, recomputed = runtime.collect(logs, 10, read(path / "schedule.json"), False,
        config["backend"], config["native_preprocessing_effective"],
        folded_activations=config["folded_activations"] if declaration["backend"] == PAPER else None,
        lut_key_family=config["lut_key_family"], folded_sign_backend=config.get("folded_sign_backend"),
        folded_sign_segments=config.get("folded_sign_segments"))
    recomputed["warmup_queries"] = []
    if "folded_activations" in recomputed:
        recomputed["folded_activation_warmup_queries"] = []
    checks["raw_measurements_recomputed"] = recomputed == result["measurements"]
    checks["all_queries_present"] = len(recomputed["queries"]) == 10
    rounds = result["round_measurements"]
    checks["saved_rounds_qualified"] = rounds["status"] == "passed" and rounds == read(path / "rounds.json")
    runtime.validate_scope_measurements(rounds, parties)
    for query, phases in enumerate(recomputed["queries"]):
        for phase in PHASES:
            pair = phases[phase]["per_party"]
            if phases[phase]["seconds"] != max(row["seconds"] for row in pair) or \
                    phases[phase]["sent_bytes"] != sum(row["sent_bytes"] for row in pair):
                raise ValueError("phase wall or bytes differ from same-query party aggregation")
            if phase != "query_total":
                scope = exactly(rounds["phases"], query=query, phase=phase)
                if sum(scope["party_sent_bytes"]) != phases[phase]["sent_bytes"]:
                    raise ValueError("phase trace bytes differ from raw phase metrics")
    checks["online_prefix_ready"] = all(row["prefetched"] == 1 and
        row["material_load_ms"] == row["prefetch_wait_ms"] == 0
        for party in parties for row in party["conversion_details"])
    checks["participant_limit"] = all(row["compute_participants"] <= 4
        for party in parties for row in party["activations"]) and all(
        row["observed_compute_upper_bound"] <= 4 for party in parties for row in party["graph_groups"])
    checks["memory_peaks"] = all(0 < max(result["process_resource_peaks"][index]["vm_peak_bytes"],
        party["vm_peak_bytes"]) <= MEMORY and 0 < max(result["process_resource_peaks"][index]["rss_peak_bytes"],
        party["peak_rss_kib"] * 1024) <= MEMORY for index, party in enumerate(parties))
    folded = audit_folded(path, expected_queries=10) if declaration["backend"] == PAPER else None
    if folded:
        checks["folded_material_resource_audit"] = folded["status"] == "passed"
    traces = []
    for role in (1, 2):
        raw = path / ("party%d.roundtrace.jsonl" % role)
        traces.append(raw if raw.exists() else raw.with_suffix(".jsonl.gz"))
    if replay_rounds:
        replayed, unions = replay(*traces, parties)
        saved = dict(rounds, scopes=[row for row in rounds["scopes"] if row["label"] != "ticket11_lut_union"])
        checks["rounds_replayed"] = replayed == saved
        previous = [row for row in rounds["scopes"] if row["label"] == "ticket11_lut_union"]
        if previous and unions != previous:
            raise ValueError("LUT causal union replay differs from saved scope evidence")
        result["round_measurements"] = dict(replayed, scopes=replayed["scopes"] + unions)
    if not all(checks.values()):
        raise ValueError("raw evidence failed: " + ", ".join(key for key, value in checks.items() if not value))
    data = extract(result, parties)
    return dict(status="qualified", checks=checks, data=data, folded_audit=folded,
        round_replay=replay_rounds, trace_files=[dict(path=str(path), sha256=digest(path)) for path in traces])


def metric_samples(data):
    samples = {}
    for row in data["queries"]:
        for phase, cost in row["step"].items():
            samples.setdefault("step/" + phase, []).append(cost["wall_seconds"])
        if "folded_prepare" in row:
            samples.setdefault("folded_prepare/offline", []).append(row["folded_prepare"]["wall_seconds"])
        for name, value in row["activations"].items():
            samples.setdefault(name + "/online", []).append(value["online"]["wall_seconds"])
            for phase, cost in value.get("lut", {}).items():
                samples.setdefault(name + "/lut/" + phase + "/work", []).append(cost["work_seconds"])
                if phase == "online":
                    for key in ("event_window", "opening_window", "product_window"):
                        if cost.get(key):
                            samples.setdefault(name + "/lut/online/" + key, []).append(cost[key]["seconds"])
            for phase, stages in value.get("stages", {}).items():
                for stage, cost in stages.items():
                    samples.setdefault(name + "/stage/" + phase + "/" + stage + "/work", []).append(cost["work_seconds"])
    return samples


def network_samples(data):
    samples = {}
    for row in data["queries"]:
        scopes = {"step/" + phase: value for phase, value in row["step"].items() if phase != "query_total"}
        if "folded_prepare" in row:
            scopes["folded_prepare/offline"] = row["folded_prepare"]
        for name, value in row["activations"].items():
            scopes[name + "/online"] = value["online"]
            scopes.update({name + "/lut/" + phase: cost for phase, cost in value.get("lut", {}).items()})
        for scope, cost in scopes.items():
            for network, projection in (cost["network"] or {}).items():
                samples.setdefault(scope + "/" + network, []).append(projection["total_seconds"])
    return samples


def cost_distributions(data):
    samples = {}
    for query in data["queries"]:
        scopes = {"step/" + phase: cost for phase, cost in query["step"].items()}
        if "folded_prepare" in query:
            scopes["folded_prepare/offline"] = query["folded_prepare"]
        for name, activation in query["activations"].items():
            scopes[name + "/online"] = activation["online"]
            scopes.update({name + "/lut/" + phase: cost for phase, cost in activation.get("lut", {}).items()})
        for scope, cost in scopes.items():
            fields = samples.setdefault(scope, {})
            for field in ("sent_bytes", "causal_rounds", "wall_seconds", "work_seconds"):
                if cost.get(field) is not None:
                    fields.setdefault(field, []).append(cost[field])
    return {scope: {field: distribution(values) for field, values in fields.items()} for scope, fields in samples.items()}


@lru_cache(maxsize=None)
def ratio(left, right):
    return compare_groups([list(group) for group in left], [list(group) for group in right])


def sample_groups(row, extractor):
    groups = row.get("groups")
    if groups is None:
        groups = [dict(data=row["data"])]
    extracted = [extractor(group["data"]) for group in groups]
    common = set.intersection(*(set(group) for group in extracted))
    return {key: [group[key] for group in extracted] for key in common}


def network_ratio(left, right):
    before = statistics.median([value for group in left for value in group])
    after = statistics.median([value for group in right for value in group])
    return dict(speedup=before / after,
        verdict="regressed" if after > before else "improved" if after < before else "unchanged",
        baseline_group_sizes=list(map(len, left)), candidate_group_sizes=list(map(len, right)),
        method="deterministic ratio of pooled communication-model medians")


def compare(pair, rows):
    before, after = (rows[pair[key]] for key in ("baseline", "candidate"))
    if before["status"] != "qualified" or after["status"] != "qualified":
        return dict(pair, status="unavailable", reason="both complete qualified fresh rows are required")
    wall = [sample_groups(row, metric_samples) for row in (before, after)]
    costs = [sample_groups(row, network_samples) for row in (before, after)]
    wall_comparisons = {}
    for key in sorted(set(wall[0]) & set(wall[1])):
        if "/stage/" in key:
            continue
        if any(value <= 0 for groups in (wall[0][key], wall[1][key]) for values in groups for value in values):
            continue
        wall_comparisons[key] = ratio(*(tuple(tuple(group) for group in sample[key]) for sample in wall))
    network = {}
    for key in sorted(set(costs[0]) & set(costs[1])):
        if any(value <= 0 for groups in (costs[0][key], costs[1][key]) for group in groups for value in group):
            continue
        network[key] = network_ratio(costs[0][key], costs[1][key])
    required = [wall_comparisons.get("step/" + phase) for phase in ("offline", "online")]
    link = [network.get("step/" + phase + "/" + name) for phase in ("offline", "online") for name in NETWORKS]
    evidence = required + link
    verdicts = [item["verdict"] for item in evidence if item]
    if len(verdicts) != len(evidence):
        admission = "unavailable"
    elif "regressed" in verdicts:
        admission = "experimental_regression"
    elif "unresolved" in verdicts:
        admission = "unresolved"
    elif "improved" in verdicts:
        admission = "eligible_default"
    else:
        admission = "unchanged"
    reasons = []
    if admission == "unresolved":
        reasons.append("step_admission")
    if pair["kind"] == "lut_construction":
        for key in ("sigmoid/online", "tanh/online", "sigmoid/lut/online/event_window", "tanh/lut/online/event_window"):
            if wall_comparisons.get(key, {}).get("verdict") == "unresolved":
                reasons.append(key)
    budget = max(len(row.get("groups", [row])) for row in (before, after)) >= 3
    extra = "blocked_by_group_budget" if reasons and budget else "required" if reasons else "not_needed"
    return dict(pair, status="complete", wall=wall_comparisons, network=network,
        default_verdict=admission, additional_groups_status=extra, additional_groups_needed=extra == "required",
        extra_group_reasons=reasons,
        interpretation="work metrics are attribution; LUT event windows include scheduling gaps; complete activation and step metrics are elapsed")


def pooled_data(groups):
    first = groups[0]["data"]
    queries = [dict(query, group=group["group"]) for group in groups for query in group["data"]["queries"]]
    return dict(first, queries=queries,
        session_metadata_rule="scalar session fields describe initial group; complete per-group data is retained in groups",
        sessions=[dict(group=group["group"], path=group["path"],
                       **{key: value for key, value in group["data"].items() if key != "queries"}) for group in groups])


def accounting(groups):
    sessions = [group["data"] for group in groups]
    summary = {}
    for field in ("setup", "model_prepare", "first_query"):
        if all(session.get(field) is not None for session in sessions):
            summary[field] = dict(wall_seconds=distribution([session[field]["seconds"] for session in sessions]),
                sent_bytes=distribution([session[field]["sent_bytes"] for session in sessions]))
    if all(session.get("public_table_prepare") is not None for session in sessions):
        summary["public_table_prepare"] = dict(work_seconds=distribution(
            [session["public_table_prepare"]["work_seconds"] for session in sessions]),
            sent_bytes=distribution([session["public_table_prepare"]["sent_bytes"] for session in sessions]),
            accounting="included in setup; never add twice")
    summary["session_total_seconds"] = distribution([session["session_total_seconds"] for session in sessions])
    summary["peak_rss_bytes_per_party"] = [max(max(session["peak_rss_kib_per_party"][party] * 1024,
        session.get("conservative_process_resource_peaks", session["process_resource_peaks"])[party]["rss_peak_bytes"])
        for session in sessions) for party in (0, 1)]
    summary["process_peak_bytes_per_party"] = [dict(
        vm=max(session.get("conservative_process_resource_peaks", session["process_resource_peaks"])[party]["vm_peak_bytes"]
               for session in sessions),
        rss=max(session.get("conservative_process_resource_peaks", session["process_resource_peaks"])[party]["rss_peak_bytes"]
                for session in sessions)) for party in (0, 1)]
    return summary


def execution_groups(directory, summary, plan):
    """Reconstruct every declaration from the frozen pairs and saved decisions."""
    expected = {row["id"]: [dict(group=1, path=row["id"], trigger_comparison=None)] for row in plan["rows"]}
    records = summary["rows"]
    executions = {row["id"]: row for row in records}
    if len(executions) != len(records) or set(executions) != set(expected):
        raise ValueError("fresh-group inventory differs from the frozen row inventory")
    pairs = {pair["id"]: pair for pair in plan["comparisons"]}
    for decision in summary.get("extra_group_decisions", []):
        pair = pairs.get(decision["comparison"])
        if pair is None:
            raise ValueError("fresh-group inventory names an unfrozen comparison")
        order = []
        for key in ("candidate", "baseline"):
            identity = pair[key]
            number = len(expected[identity]) + 1
            if number > plan.get("maximum_groups_per_row", 3):
                raise ValueError("fresh-group inventory exceeds the frozen group budget")
            path = identity + "/group%d" % number
            expected[identity].append(dict(group=number, path=path, trigger_comparison=pair["id"]))
            order.append(path)
        if decision["order"] != order:
            raise ValueError("fresh-group inventory violates candidate then baseline declaration order")
    for identity, groups in expected.items():
        declared = executions[identity].get("groups", [])
        inventory = [{key: group.get(key) for key in ("group", "path", "trigger_comparison")} for group in declared]
        if inventory != groups:
            raise ValueError("fresh-group inventory differs from saved decisions: " + identity)
    expected_numbers = {identity: {group["group"] for group in groups}
                        for identity, groups in expected.items()}
    # A deleted summary entry must not hide a group directory or runner artifact.
    for identity, numbers in expected_numbers.items():
        row_path = directory / identity
        if not row_path.is_dir():
            continue
        nested_numbers = numbers - {1}
        for child in row_path.iterdir():
            if not child.name.startswith("group"):
                continue
            match = re.fullmatch(r"group([0-9]+)", child.name)
            if not child.is_dir() or match is None or int(match.group(1)) not in nested_numbers:
                raise ValueError("fresh-group inventory contains an undeclared group directory: " + str(child))
    artifact_pattern = re.compile(r"(.+)\.group([0-9]+)\.(runner\.log|controller\.resources\.txt)")
    known_identities = set(expected_numbers)
    for artifact in directory.iterdir():
        if not artifact.is_file():
            continue
        match = artifact_pattern.fullmatch(artifact.name)
        if match is None:
            continue
        identity, number = match.group(1), int(match.group(2))
        if identity not in known_identities or number not in expected_numbers[identity]:
            raise ValueError("fresh-group inventory contains an undeclared runner artifact: " + str(artifact))
    for row_path in directory.iterdir():
        if not row_path.is_dir() or row_path.name in known_identities:
            continue
        if any(child.is_dir() and child.name.startswith("group") for child in row_path.iterdir()):
            raise ValueError("fresh-group inventory contains an undeclared row directory: " + str(row_path))
    return executions


def analyze(directory, replay_rounds=False, frozen_plan=None):
    directory = Path(directory).resolve()
    plan = verify_manifest(directory, frozen_plan)
    summary_path = directory / "summary.json"
    executions = execution_groups(directory, read(summary_path), plan) if summary_path.exists() else {}
    rows = {}
    for declaration in plan["rows"]:
        identity = declaration["id"]
        path = directory / identity
        execution = executions.get(identity, dict(status="not_started", reason="no controller summary record"))
        record = dict(id=identity, declaration={key: value for key, value in declaration.items() if key != "command"},
            path=str(path), execution=execution, status=execution["status"], reason=execution.get("reason"), groups=[])
        try:
            declared_groups = execution.get("groups", [dict(execution, group=1, path=identity)])
            if not 1 <= len(declared_groups) <= plan.get("maximum_groups_per_row", 3):
                raise ValueError("invalid number of declared fresh groups")
            record["groups"] = [dict(group=number, path=str(directory / item["path"]), execution=item,
                status=item["status"], reason=item.get("reason")) for number, item in enumerate(declared_groups, 1)]
            for number, (item, group) in enumerate(zip(declared_groups, record["groups"]), 1):
                expected = identity if number == 1 else identity + "/group%d" % number
                if item["group"] != number or item["path"] != expected:
                    raise ValueError("fresh-group identity or order differs from the declared contract")
                group_path = directory / item["path"]
                if item["status"] != "complete":
                    if (group_path / "result.json").exists():
                        partial = read(group_path / "result.json")
                        group["partial_result"] = dict(status=partial.get("status"), reason=partial.get("reason"),
                            party_exit_codes=partial.get("party_exit_codes"), measurements=partial.get("measurements"),
                            process_resource_peaks=partial.get("process_resource_peaks"))
                    raise ValueError(item.get("reason") or "formal controller group is not complete")
                try:
                    group.update(qualify(group_path, declaration, plan, replay_rounds))
                except (OSError, KeyError, ValueError, TypeError, json.JSONDecodeError) as error:
                    group.update(status="unqualified", reason=str(error))
                    raise
            record.update(status="qualified", reason=None, data=pooled_data(record["groups"]),
                accounting=accounting(record["groups"]))
            record["distributions"] = {key: distribution(values) for key, values in metric_samples(record["data"]).items()}
            record["network_distributions"] = {key: distribution(values) for key, values in network_samples(record["data"]).items()}
            record["cost_distributions"] = cost_distributions(record["data"])
        except (OSError, KeyError, ValueError, TypeError, json.JSONDecodeError) as error:
            if record["status"] in ("complete", "qualified"):
                record["status"] = "unqualified"
            record["reason"] = str(error)
        rows[identity] = record
    pairs = [compare(pair, rows) for pair in plan["comparisons"]]
    gaps = []
    for identity, row in rows.items():
        if row["status"] == "qualified" and row["declaration"]["backend"] == PAPER:
            if any(group["data"]["public_table_prepare"] is None for group in row["groups"]):
                gaps.append(dict(id=identity, field="public_table_prepare", reason="separate setup subset is uninstrumented"))
            for name in KINDS:
                if name + "/lut/online/event_window" not in row["distributions"]:
                    gaps.append(dict(id=identity, field=name + "/lut/online/event_window", reason="monotonic LUT stage bounds are absent"))
                if name + "/lut/online/LAN" not in row["network_distributions"]:
                    gaps.append(dict(id=identity, field=name + "/lut/online/causal_rounds", reason="LUT union scopes were not captured; replay raw traces to recover"))
    pending = any(pair.get("additional_groups_needed") for pair in pairs)
    blocked = any(pair.get("additional_groups_status") == "blocked_by_group_budget" for pair in pairs)
    sampling_status = "pending_extra_groups" if pending else "exhausted_budget" if blocked else "settled"
    return dict(schema=1, status="complete" if all(row["status"] == "qualified" for row in rows.values()) and not gaps and not pending else "incomplete",
        manifest=str(directory / "manifest.json"), rows=rows, comparisons=pairs, measurement_gaps=gaps,
        sampling_status=sampling_status, maximum_groups_per_row=plan.get("maximum_groups_per_row", 3),
        networks=NETWORKS, statistical_method=plan["bootstrap"], encoding=plan["encoding"],
        communication_formula=plan["communication_formula"], interpretation=plan["interpretation"],
        resource_accounting="party RLIMIT_AS and process peaks separately from external controller.resources.txt",
        recommendation="per-parent stage qualification only; final method/quality recommendation belongs to ticket12")


def fmt(value, places=6):
    return "unavailable" if value is None else ("%.*f" % (places, value))


def markdown(report):
    lines = ["# 工单 11：完整激活及单步实测", "",
        "状态：`" + report["status"] + "`；补测状态：`" + report["sampling_status"] + "`。每组包含 10 次 fresh 查询，warmup=0；全部已声明的组及失败记录均保留。数值为中位数，内存为各组最大峰值。", "",
        "## 完整查询边界", "",
        "| 冻结行 | 状态 | 组数 | Offline (s) | Online (s) | Query total (s) | Sigmoid online (s) | TanH online (s) |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for identity, row in report["rows"].items():
        values = row.get("distributions", {})
        lines.append("| " + " | ".join([identity, row["status"], str(len(row["groups"]))] + [fmt(values.get(key, {}).get("median")) for key in
            ("step/offline", "step/online", "step/query_total", "sigmoid/online", "tanh/online")]) + " |")
    lines += ["", "## 会话准备及资源", "",
        "| 冻结行 | Setup (s) | 公开表准备 work (s) | Owner model (s) | First request (s) | Session total (s) | 两方 RSS 峰值 (MiB) | 两方 VM 峰值 (MiB) |",
        "| --- | ---: | ---: | ---: | ---: | ---: | --- | --- |"]
    for identity, row in report["rows"].items():
        values = row.get("accounting", {})
        cells = [fmt(values.get(key, {}).get(field, {}).get("median")) for key, field in
                 (("setup", "wall_seconds"), ("public_table_prepare", "work_seconds"),
                  ("model_prepare", "wall_seconds"), ("first_query", "wall_seconds"))]
        cells.append(fmt(values.get("session_total_seconds", {}).get("median")))
        cells.append(", ".join(fmt(value / (1 << 20), 2) for value in values.get("peak_rss_bytes_per_party", [])) or "unavailable")
        cells.append(", ".join(fmt(value["vm"] / (1 << 20), 2) for value in values.get("process_peak_bytes_per_party", [])) or "unavailable")
        lines.append("| " + " | ".join([identity] + cells) + " |")
    lines += ["", "公开表准备已计入 setup，不能重复加到 setup 或 first request。每组的 setup、owner model、first request 和 session total 是独立直接边界；完整原值在 `groups[].data`。每方及外部解析控制进程的地址空间上限均为 1.5 GiB；控制进程的资源与归档耗时单独记录，不进入协议延迟。", "",
        "## 配对比较", "",
        "| 比较 | Step online 加速比 [95% BCa] | Online 判定 | Offline 判定 | 两阶段准入 | 补测状态 |",
        "| --- | --- | --- | --- | --- | --- |"]
    for pair in report["comparisons"]:
        value = pair.get("wall", {}).get("step/online")
        ratio = "unavailable" if not value else "%s [%s, %s]" % tuple(fmt(item, 4) for item in
            (value["speedup"], *value["interval95"]))
        lines.append("| " + " | ".join([pair["id"], ratio, value["verdict"] if value else "unavailable",
            pair.get("wall", {}).get("step/offline", {}).get("verdict", "unavailable"), pair.get("default_verdict", "unavailable"),
            pair.get("additional_groups_status", "unavailable")]) + " |")
    lines += ["", "加速比为 baseline/candidate；LUT 构造为 segmented/full。Wall 使用冻结的 10000 次、seed20261013 BCa，在每个测量组内独立抽样，再计算合并中位数之比。网络模型沿用确定性中位数判定，相等为 unchanged。准入与 LUT online 的预定相关指标触发 candidate→baseline 逆序补测，每行最多 3 组；预算用尽后的 unresolved 保留为实验性结论。", "",
        "## LUT online 边界", "",
        "| 冻结行 | 激活 | LUT work (s) | 开启窗口 (s) | 本地乘积窗口 (s) | LUT 联合窗口 (s) | 完整激活 (s) |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: |"]
    for identity, row in report["rows"].items():
        if row["status"] != "qualified" or row["declaration"]["backend"] != PAPER:
            continue
        for name in KINDS:
            acts = [query["activations"][name] for query in row["data"]["queries"]]
            cells = [fmt(statistics.median([act["lut"]["online"]["work_seconds"] for act in acts]))]
            for key in ("opening_window", "product_window", "event_window"):
                values = [act["lut"]["online"].get(key) for act in acts]
                cells.append(fmt(statistics.median([value["seconds"] for value in values])) if all(values) else "unavailable")
            cells.append(fmt(row["distributions"][name + "/online"]["median"]))
            lines.append("| " + " | ".join([identity, name] + cells) + " |")
    lines += ["", "LUT 联合窗口使用每方最早开启到最晚本地乘积结束，再取两方最大值，包含调度间隙。阶段 work 是各 worker 的工作量归因，不能作为并行 elapsed。每种激活独立的 offline wall 未单独计时，全部 folded preparation 的直接计时保留在 JSON。", "",
        "## 通信与因果轮次", "",
        "| 冻结行 | 边界 | 双方 sent bytes | 因果轮次 | LAN 模型 (s) | WAN 模型 (s) |",
        "| --- | --- | ---: | ---: | ---: | ---: |"]
    for identity, row in report["rows"].items():
        for scope, cost in row.get("cost_distributions", {}).items():
            if scope not in ("step/offline", "step/online", "sigmoid/online", "tanh/online", "sigmoid/lut/online", "tanh/lut/online"):
                continue
            cells = [fmt(cost.get(field, {}).get("median"), 0) for field in ("sent_bytes", "causal_rounds")]
            cells += [fmt(row["network_distributions"].get(scope + "/" + net, {}).get("median")) for net in NETWORKS]
            lines.append("| " + " | ".join([identity, scope] + cells) + " |")
    lines += ["", "模型仅计算通信：`8 × 双方 sent bytes / bandwidth + RTT × causal rounds`。LAN 为 1 Gbit/s、0.5 ms RTT；WAN 为 400 Mbit/s、4 ms RTT。不添加 localhost compute。LUT 因果轮次在完整 phase 依赖 DAG 上计算所选阶段联合发送深度，保留外部依赖，不能累加 worker 轮次。", "",
        "## 每阶段归因", "",
        "| 冻结行 | 激活 | 阶段 | Offline work (s) | Online work (s) | Offline sent bytes | Online sent bytes |",
        "| --- | --- | --- | ---: | ---: | ---: | ---: |"]
    for identity, row in report["rows"].items():
        if row["status"] != "qualified" or row["declaration"]["backend"] != PAPER:
            continue
        for name in KINDS:
            acts = [query["activations"][name] for query in row["data"]["queries"]]
            for stage in runtime.FOLDED_STAGE_NAMES:
                cells = [fmt(row["distributions"].get(name + "/stage/" + phase + "/" + stage + "/work", {}).get("median"))
                         for phase in ("offline", "online")]
                cells += [fmt(statistics.median([act["stages"][phase][stage]["sent_bytes"] for act in acts]), 0)
                          for phase in ("offline", "online")]
                lines.append("| " + " | ".join([identity, name, stage] + cells) + " |")
    lines += ["", "q10 索引当前以 2 字节编码，两个激活的双方 mask-index payload 共 102400 字节；64000 字节的紧凑 bit 编码尚未实现，不计为收益。LUT 构造同时改变树分段和 arithmetic-payload/B2A；SCI 对照还改变数值激活算法。最终方法及质量建议由工单 12 汇总。"]
    failures = [(identity, row.get("reason")) for identity, row in report["rows"].items() if row["status"] != "qualified"]
    if failures or report["measurement_gaps"]:
        lines += ["", "## 缺失或失败证据", ""]
        lines += ["- `%s`: %s" % (identity, reason) for identity, reason in failures]
        lines += ["- `%s` / `%s`: %s" % (gap["id"], gap["field"], gap["reason"]) for gap in report["measurement_gaps"]]
    return "\n".join(lines) + "\n"


def save_report(directory, report):
    directory = Path(directory)
    write(directory / "analysis.json", report)
    (directory / "REPORT.md").write_text(markdown(report))
    write(directory / "ELIGIBILITY.json", dict(eligible=report["status"] == "complete",
        rows=[dict(id=row["id"], path=row["path"], eligible=row["status"] == "qualified", reason=row.get("reason"))
              for row in report["rows"].values()]))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--directory", type=Path, required=True)
    parser.add_argument("--plan", type=Path,
        help="original frozen plan; defaults to the runner's frozen-plan.json snapshot")
    parser.add_argument("--replay-rounds", action="store_true", help="independently replay saved traces, including LUT unions; runs after party exit")
    args = parser.parse_args()
    report = analyze(args.directory, args.replay_rounds, args.plan)
    save_report(args.directory, report)
    print(json.dumps(dict(status=report["status"], rows={key: row["status"] for key, row in report["rows"].items()},
                         measurement_gaps=report["measurement_gaps"])))
    return 0 if report["status"] == "complete" else 1


if __name__ == "__main__":
    raise SystemExit(main())
