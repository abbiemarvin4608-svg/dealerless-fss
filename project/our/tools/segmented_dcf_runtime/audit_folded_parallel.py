#!/usr/bin/env python3
"""Qualify complete folded schedules with public context and batch ownership."""
from collections import Counter, defaultdict
import json
from pathlib import Path

from run import records, validate_native_plan_summary, validate_scope_measurements

MEMORY_BYTES = 1536 << 20
PREFETCH_BYTES = 512 << 20


def audit(directory, expected_queries=2):
    directory = Path(directory)
    result = json.loads((directory / "result.json").read_text())
    config = result["config"]
    parties = json.loads((directory / "parties.json").read_text())
    queries = config["queries"] + config.get("warmup_queries", 0)
    parallel = config["graph_mode"] == "parallel"
    branches = 2 if parallel else 1
    workers = config["activation_workers_per_party"]
    sign_backend = config.get("folded_sign_backend")
    sign_segments = config.get("folded_sign_segments", 0)
    sign_valid = ((sign_backend == "bitdec" and sign_segments == 0) or
                  (sign_backend == "segmented_fss" and sign_segments ==
                   (2 if config.get("segment_policy") == "fixed-2" else 4)))
    per_element_reserve = 8192 + (1024 + 256 * sign_segments if sign_backend == "segmented_fss" else 0)
    reserve = config["batch"] * 200 * per_element_reserve + (8 << 20)
    checks = dict(runtime_complete=result.get("status") == "complete" and result.get("party_exit_codes") == [0, 0],
        explicit_schedule=config.get("backend") == "SCI_Beaver_segmented_PaperLUT" and
            config.get("folded_activations") == "both" and sign_valid and
            (config["graph_mode"], workers) in (("sequential", 1), ("parallel", 1), ("parallel", 2)) and
            config["steps"] == 1 and queries == expected_queries and config["warmup_queries"] == 0 and config["batch"] > 0,
        address_space_limit_enforced=result.get("address_space_hard_limit_bytes_per_party") == MEMORY_BYTES,
        four_cpu_affinity=config.get("cpu_affinity_per_party") == [[0, 2, 4, 10], [6, 8, 12, 16]],
        both_parties_present=len(parties) == 2,
        scope_traffic_qualified=True)
    try:
        validate_scope_measurements(result["round_measurements"], parties)
        checks["scope_traffic_qualified"] = result["round_measurements"]["status"] == "passed"
    except (KeyError, ValueError):
        checks["scope_traffic_qualified"] = False
    reports = []
    for index, party in enumerate(parties):
        log = (directory / ("party%d.log" % (index + 1))).read_text()
        local = dict(complete_outer_prefixes=True, prefix_before_input=True, zero_online_load_or_wait=True,
                     joint_budget_accounted=True, complete_folded_inventory=True,
                     zero_postprepare_generation=True, complete_sci_activation_replacement=True,
                     compute_budget_bounded=True, joint_process_resources_bounded=True)
        budgets = party["conversion_prefetch_budgets"]
        allocations = {row["context"]: row["context_budget_bytes"] for row in budgets}
        declarations = records(log, "FOLDED_ACTIVATION_RESERVE")
        local["joint_budget_accounted"] = config.get("prefetch_bytes") == PREFETCH_BYTES and config.get("prefetch_mode") == "prepared_prefix" and len(declarations) == 1 and declarations[0]["resident_budget_bytes"] == reserve and len(budgets) == len(allocations) == branches and set(allocations) == set(range(branches)) and sum(allocations.values()) == PREFETCH_BYTES - reserve and all(row["contexts"] == branches and row["configured_party_budget_bytes"] == PREFETCH_BYTES and row["party_budget_bytes"] == PREFETCH_BYTES - reserve and row["dpf_reserved_bytes"] == reserve for row in budgets)
        prefix_rows, ready_rows, preprocessed = defaultdict(list), defaultdict(list), defaultdict(list)
        query = 0
        for position, line in enumerate(log.splitlines()):
            prefix, _, payload = line.partition(" ")
            if prefix not in ("CONVERSION_PREFETCH", "CONVERSION_PREPROCESS_DETAIL", "MATERIAL_READY", "METRIC"):
                continue
            row = json.loads(payload)
            if prefix == "CONVERSION_PREFETCH": prefix_rows[query].append((position, row))
            elif prefix == "CONVERSION_PREPROCESS_DETAIL": preprocessed[query].append((position, row))
            elif prefix == "MATERIAL_READY": ready_rows[row["query"]].append(position)
            elif row["phase"] == "query_total": query = row["query"] + 1
        prefix_evidence, folded_evidence = [], []
        for query in range(queries):
            calls = [row for row in party["conversion_calls"] if row["query"] == query]
            details = [row for row in party["conversion_details"] if row["query"] == query]
            generated = preprocessed[query]
            ready = ready_rows[query]
            prefixes = prefix_rows[query]
            grouped = defaultdict(list)
            call_context = {(row["index"], row["slot"]): row["context"] for row in calls}
            for row in details: grouped[call_context[(row["index"], row["slot"])]].append(row)
            expected = Counter((len(rows), allocations.get(context), rows[0]["resident_bytes"]) for context, rows in grouped.items())
            actual = Counter((row["calls"], row["budget_bytes"], row["resident_bytes"]) for _, row in prefixes)
            expected_ordinals = [ordinal for context in sorted(grouped) for ordinal in range(len(grouped[context]))]
            local["complete_outer_prefixes"] &= set(grouped) == set(allocations) and len(prefixes) == branches and expected == actual and len(calls) == len(details) == len(generated) and [row["ordinal"] for _, row in generated] == expected_ordinals and {(row["index"], row["slot"]) for row in calls} == {(row["index"], row["slot"]) for row in details} and all(row["fresh"] and row["dealerless"] for _, row in generated)
            local["prefix_before_input"] &= len(ready) == 1 and all(position < ready[0] for position, _ in prefixes + generated)
            local["zero_online_load_or_wait"] &= all(row["prefetched"] == 1 and row["material_load_ms"] == row["prefetch_wait_ms"] == 0 and row["material_preload_ms"] >= 0 for row in details)
            for context, rows in grouped.items():
                live = [row["resident_bytes"] for row in rows]
                local["complete_outer_prefixes"] &= all(0 < size <= allocations[context] for size in live) and all(a > b for a, b in zip(live, live[1:])) and all(row["prefetch_budget_bytes"] == allocations[context] for row in rows)
            local["complete_outer_prefixes"] &= all(row["charged_phase"] == "offline" and row["mode"] == "prepared_prefix" and 0 < row["address_space_bytes"] <= MEMORY_BYTES and row["workspace_reserve_bytes"] > 0 for _, row in prefixes)
            material = [row for row in party["folded_activation_prepared"] if row["query"] == query]
            counts = [sum(row["count"] for row in material if row["kind"] == kind) for kind in (0, 1)]
            local["complete_folded_inventory"] &= party["folded_activation_verified"] and counts == [config["batch"] * 100] * 2 and 0 < sum(row["resident_bytes"] for row in material) <= reserve
            prefix_evidence.append(dict(query=query, actual_conversion_calls=len(calls), contexts={context: len(rows) for context, rows in grouped.items()}, prefixes=[row for _, row in prefixes]))
            folded_evidence.append(dict(query=query, prepared_counts=counts, owned_calls=len(material), folded_capacity_bytes=sum(row["resident_bytes"] for row in material)))
        local["zero_postprepare_generation"] = all(row.get(field) == 0 for row in party["phases"] if row["phase"] in ("input", "ready", "online", "output") for field in ("native_generation_epoch_delta", "bridge_generation_epoch_delta", "dpf_generation_epoch_delta", "folded_generation_epoch_delta"))
        local["explicit_empty_native_inventory"] = validate_native_plan_summary(party["native_public_plan_summary"], party["native_public_plan"], party["native_triple_plan"], queries) and not any(party[name] for name in ("native_material_prepared", "native_material_consumed", "native_triple_prepared", "native_triple_consumed"))
        local["complete_sci_activation_replacement"] = not any(party[name] for name in ("dpf_lut_plan", "dpf_lut_consumed", "conversion_remaining_sci")) and all(row["conversion"] == "folded_sigma_f12_q10" for row in party["activations"]) and all(row["sent_bytes"] == row["native_conversion_calls"] == 0 for row in party["activation_workers"])
        worker = party["native_worker_config"]
        local["compute_budget_bounded"] = len(worker) == 1 and worker[0]["worker_local_cggm_cap"] and worker[0]["eigen_threads"] == 1 and worker[0]["requested_activation_workers"] == workers and worker[0]["protocol_contexts"] == (4 if parallel else 1) and worker[0]["max_compute_participants"] <= 4 and all(row["observed_compute_upper_bound"] <= 4 for row in party["graph_groups"]) and all(row["compute_participants"] <= row["requested_compute_participants"] <= 4 for row in party["activations"]) and all(row["compute_participants"] <= row["requested_ot_participants"] and row["observed_cggm_participants"] <= row["cggm_participant_limit"] <= row["requested_ot_participants"] and row["observed_transport_participants"] <= row["requested_ot_participants"] for row in party["activation_workers"])
        sampled = result["process_resource_peaks"][index]
        peak_vm = max(sampled["vm_peak_bytes"], party["vm_peak_bytes"])
        peak_rss = max(sampled["rss_peak_bytes"], party["peak_rss_kib"] * 1024)
        local["joint_process_resources_bounded"] = all(0 < value <= MEMORY_BYTES for value in (peak_vm, peak_rss))
        reports.append(dict(party=index + 1, checks=local, prefix=prefix_evidence, folded=folded_evidence,
                            resources=dict(conservative_vm_peak_bytes=peak_vm, conservative_rss_peak_bytes=peak_rss,
                                           sampled=sampled)))
    passed = all(checks.values()) and all(all(row["checks"].values()) for row in reports)
    return dict(status="passed" if passed else "failed", checks=checks, parties=reports,
                scope="complete single step, fresh queries, public batch and owned parallel schedule",
                expected_queries=expected_queries)
