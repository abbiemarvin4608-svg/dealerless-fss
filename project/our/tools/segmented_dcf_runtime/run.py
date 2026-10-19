#!/usr/bin/env python3
"""Run one frozen fixed-batch workload with physical offline/online phases."""
import argparse
from contextlib import ExitStack
import json
import math
import os
from pathlib import Path
import platform
import resource
import selectors
import socket
import statistics
import subprocess
import sys
import time

from build import digest
from runtime import BACKENDS, aggregate_phases, call_arguments, dependency_schedule, summarize_queries, validate_fixture
from round_trace import ATOMIC_PHASES

HERE = Path(__file__).resolve().parent
CHANNEL_OFFSETS = (0, 3, 20, 23, 50, 70, 100, 120, 200)
# The relay names retain compatibility with prior plans.  The modeled links
# follow the current experiment contract: 1 Gbit/s with 0.5 ms RTT for LAN,
# and 400 Mbit/s with 4 ms RTT for WAN.
NETWORKS = {"localhost": (None, None), "sirnn_lan": (0.5, 1000), "wan": (4.0, 400)}
NETWORK_PROJECTION_OPTIONS = {
    "lan": {"LAN": NETWORKS["sirnn_lan"]},
    "wan": {"WAN": NETWORKS["wan"]},
    "both": {"LAN": NETWORKS["sirnn_lan"], "WAN": NETWORKS["wan"]},
}
FOLDED_BACKEND = "SCI_Beaver_segmented_PaperLUT"
FOLDED_STAGE_NAMES = (
    "input_adapter", "shared_sign_prepare", "shared_bitxa_prepare", "sign", "abs_bitxa",
    "clip_compare", "clip_bitxa", "index_truncation", "lut_mask", "lut_keygen",
    "lut_materialize", "lut_tensor_and", "lut_leaf_b2a", "lut_payload", "lut_open",
    "lut_product", "restore_bitxa", "output_adapter")
FOLDED_IDENTITY = ("query", "layer", "slot", "call")
FOLDED_SHAPE = ("kind", "count", "input_bits", "output_bits", "input_scale", "output_scale",
                "lut_family", "sign_backend", "sign_segments", "table_version")


def channel_offsets(activation_workers, graph_mode="sequential", dpf_activations=False):
    contexts = 4 if graph_mode != "sequential" else activation_workers
    return tuple(sorted(set(CHANNEL_OFFSETS) | ({820, 823, 870, 920} if graph_mode != "sequential" else set()) | {
        300 + 150 * index + offset
        for index in range(contexts - 1) for offset in (0, 50, 100)} | ({
        2000 + 150 * index + offset
        for index in range(contexts) for offset in (0, 3, 50, 53, 100, 103)
    } if dpf_activations else set())))


def process_resources(pid):
    """Kernel address-space and RSS high-water marks from one live process."""
    try:
        fields = dict(line.split(":", 1) for line in Path("/proc/%d/status" % pid).read_text().splitlines() if ":" in line)
    except (FileNotFoundError, ProcessLookupError):
        return {}
    result = {}
    for source, target in (("VmPeak", "vm_peak_bytes"), ("VmSize", "vm_size_bytes"), ("VmHWM", "rss_peak_bytes")):
        if source in fields:
            result[target] = int(fields[source].split()[0]) * 1024
    if "Threads" in fields:
        result["threads_peak"] = int(fields["Threads"].strip())
    return result


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def project_network_costs(rounds, network, query_start=0, query_count=None):
    """Attach the declared link model to qualified phase byte/round records.

    The runtime's wall clock remains a separate measurement.  This projection
    uses the two-party send total and the causal critical-path round count so
    that link bandwidth and RTT can be changed without rerunning the protocol.
    """
    rtt_ms, mbps = network
    projection = {
        "eligible": False,
        "network": {"rtt_ms": rtt_ms, "mbps": mbps},
        "transmission_formula": "8 * sum(party_sent_bytes) / (mbps * 1000000)",
        "propagation_formula": "logical_rounds * (rtt_ms / 1000)",
        "aggregation": "per-query phase components; medians over fresh queries",
        "phases": {},
    }
    if rounds is None or rounds.get("status") != "passed":
        projection["reason"] = "qualified round trace is required"
        return projection
    if rtt_ms is None or mbps is None:
        projection["reason"] = "localhost has no declared bandwidth or RTT"
        return projection
    phase_rows = {(row.get("query"), row.get("phase")): row
                  for row in rounds.get("phases", [])}
    phase_names = ("offline", "online")
    queries = sorted({query for query, phase in phase_rows
                      if isinstance(query, int) and query >= query_start and phase in phase_names})
    if query_count is not None:
        queries = [query for query in queries if query < query_start + query_count]
    if not queries or any((query, phase) not in phase_rows for query in queries for phase in phase_names):
        projection["reason"] = "round trace lacks complete offline/online query coverage"
        return projection
    phase_values = {phase: [] for phase in phase_names}
    for query in queries:
        for phase in phase_names:
            row = phase_rows[(query, phase)]
            sent = row.get("party_sent_bytes")
            rounds_count = row.get("logical_rounds")
            if (not isinstance(sent, list) or len(sent) != 2 or
                    any(type(value) is not int or value < 0 for value in sent) or
                    type(rounds_count) is not int or rounds_count < 0):
                projection["reason"] = "invalid qualified round record"
                return projection
            communication_bytes = sum(sent)
            transmission = 8 * communication_bytes / (mbps * 1000000)
            propagation = rounds_count * (rtt_ms / 1000)
            phase_values[phase].append({
                "query": query,
                "bytes": communication_bytes,
                "rounds": rounds_count,
                "sent_bytes": sent,
                "transmission_seconds": transmission,
                "propagation_seconds": propagation,
                "communication_seconds": transmission + propagation,
            })
    projection["eligible"] = True
    projection["phases"] = {phase: rows for phase, rows in phase_values.items()}
    projection["queries"] = [{
        "query": query,
        "phases": {phase: next(row for row in phase_values[phase] if row["query"] == query)
                   for phase in phase_names},
        "offline_plus_online_seconds": sum(
            next(row for row in phase_values[phase] if row["query"] == query)["communication_seconds"]
            for phase in phase_names),
    } for query in queries]
    projection["summary"] = {}
    for phase in phase_names:
        rows = phase_values[phase]
        projection["summary"][phase] = {
            "bytes_median": statistics.median(row["bytes"] for row in rows),
            "rounds_median": statistics.median(row["rounds"] for row in rows),
            "transmission_seconds_median": statistics.median(row["transmission_seconds"] for row in rows),
            "propagation_seconds_median": statistics.median(row["propagation_seconds"] for row in rows),
            "communication_seconds_median": statistics.median(row["communication_seconds"] for row in rows),
        }
    projection["summary"]["offline_plus_online_seconds_median"] = statistics.median(
        row["offline_plus_online_seconds"] for row in projection["queries"])
    return projection


def project_network_costs_set(rounds, projection_mode, query_start=0, query_count=None):
    """Project one qualified trace onto the requested named links."""
    if projection_mode not in NETWORK_PROJECTION_OPTIONS:
        raise ValueError("unknown network projection mode: " + str(projection_mode))
    return {
        "schema_version": 1,
        "mode": projection_mode,
        "aggregation": "per-query phase components; medians over fresh queries",
        "networks": {
            name: project_network_costs(rounds, network, query_start=query_start,
                                        query_count=query_count)
            for name, network in NETWORK_PROJECTION_OPTIONS[projection_mode].items()
        },
    }


def attach_round_measurements(measurements, rounds):
    """Copy qualified causal-round counters into the phase measurements.

    The protocol collector records byte counters and wall time, while the
    round-trace analyzer owns causal flight depth.  Keeping both on the same
    query phase makes the interface record self-contained without replacing
    the existing per-party IO counters.
    """
    if not isinstance(rounds, dict) or rounds.get("status") != "passed":
        return
    rows = {(row.get("query"), row.get("phase")): row
            for row in rounds.get("phases", [])}
    for query_index, query in enumerate(measurements.get("queries", [])):
        query_id = query_index
        for candidate in query.values():
            per_party = candidate.get("per_party") if isinstance(candidate, dict) else None
            if (isinstance(per_party, list) and per_party and isinstance(per_party[0], dict)
                    and isinstance(per_party[0].get("query"), int)):
                query_id = per_party[0]["query"]
                break
        for phase, value in query.items():
            if phase not in ATOMIC_PHASES or not isinstance(value, dict):
                continue
            row = rows.get((query_id, phase))
            if row is None:
                continue
            value["logical_rounds"] = row.get("logical_rounds")
            value["send_flight_depths"] = row.get("send_flight_depths")
            value["party_sent_bytes"] = row.get("party_sent_bytes")
            value["party_received_bytes"] = row.get("party_received_bytes")
            per_party = value.get("per_party")
            if isinstance(per_party, list) and len(per_party) == 2:
                for index, party in enumerate(per_party):
                    if not isinstance(party, dict):
                        continue
                    party["logical_rounds"] = row.get("logical_rounds")
                    party["round_send_flight_depth"] = row.get("send_flight_depths", [None, None])[index]
                    party["round_sent_bytes"] = row.get("party_sent_bytes", [None, None])[index]
                    party["round_received_bytes"] = row.get("party_received_bytes", [None, None])[index]


def attach_network_measurements(measurements, projection):
    """Expose per-query LAN/WAN components beside each offline/online phase."""
    if not isinstance(projection, dict):
        return
    networks = projection.get("networks")
    if not isinstance(networks, dict):
        return
    by_network = {}
    for name, network in networks.items():
        if not isinstance(network, dict) or not network.get("eligible"):
            continue
        by_network[name] = {(row.get("query"), phase): row
                            for phase, phase_rows in network.get("phases", {}).items()
                            for row in phase_rows}
    if not by_network:
        return
    for query_index, query in enumerate(measurements.get("queries", [])):
        query_id = query_index
        for candidate in query.values():
            per_party = candidate.get("per_party") if isinstance(candidate, dict) else None
            if (isinstance(per_party, list) and per_party and isinstance(per_party[0], dict)
                    and isinstance(per_party[0].get("query"), int)):
                query_id = per_party[0]["query"]
                break
        for phase in ("offline", "online"):
            value = query.get(phase)
            if not isinstance(value, dict):
                continue
            components = {}
            for name, rows in by_network.items():
                row = rows.get((query_id, phase))
                if row is not None:
                    components[name] = {
                        "transmission_seconds": row["transmission_seconds"],
                        "propagation_seconds": row["propagation_seconds"],
                        "communication_seconds": row["communication_seconds"],
                    }
            if components:
                value["network_projection"] = components


def party_affinities(args, available):
    activation = getattr(args, "activation_workers", 1)
    if not 1 <= activation <= 4:
        raise ValueError("activation workers must be in 1..4")
    if args.cggm_threads < 1:
        raise ValueError("cggm threads must be positive")
    graph_mode = getattr(args, "graph_mode", "sequential")
    overlap = getattr(args, "offline_overlap", False)
    if overlap and args.cggm_threads > 2:
        raise ValueError("offline preparation cGGM limit must be at most two")
    if graph_mode != "sequential" and max(args.cggm_threads, activation) > 2:
        raise ValueError("graph branch activation and cGGM limits must be at most two")
    required = 4 if graph_mode == "parallel" or overlap else max(args.cggm_threads, activation)
    if required > 4:
        raise ValueError("runtime supports at most four compute participants")
    if len(args.cpus) != 3 or not set(args.cpus) <= available:
        raise ValueError("cpus must specify three available CPUs")
    explicit = (args.server_cpus, args.client_cpus)
    if required > 1 and any(value is None for value in explicit):
        raise ValueError("multiple workers require explicit server-cpus and client-cpus")
    affinities = []
    for index, value in enumerate(explicit):
        if value is None:
            affinity = [args.cpus[index]]
        else:
            parts = value.split(",")
            if any(not part.isascii() or not part.isdecimal() for part in parts):
                raise ValueError("party CPUs must be comma-separated nonnegative CPU IDs")
            affinity = [int(part) for part in parts]
        if len(set(affinity)) != len(affinity) or not set(affinity) <= available:
            raise ValueError("party CPUs must be distinct available CPU IDs")
        if len(affinity) < required:
            raise ValueError("each party needs at least the requested worker count of available CPUs")
        affinities.append(affinity)
    return affinities


def child_environment(base, updates):
    env = {key: value for key, value in base.items()
           if not key.startswith(("SEGFSS_", "SDCF_"))
           and key not in ("GOOGLE30_PROXY_OFFSET", "SWEEP_CONNECT_OFFSET", "OMP_THREAD_LIMIT", "OMP_DYNAMIC")}
    env.update(updates)
    return env


def native_preprocessing_mode(backend, requested, steps):
    if backend == "SCI" and requested:
        raise ValueError("SCI control cannot request native preprocessing; use SCI_preprocessed")
    effective = backend in ("SCI_preprocessed", "SCI_Beaver_segmented_DPF_LUT", FOLDED_BACKEND) or requested
    if effective and steps != 1:
        raise ValueError("native preprocessing supports only the frozen single-step interface")
    return effective


def stop(child):
    if child.poll() is None:
        child.terminate()
    try:
        child.wait(timeout=3)
    except subprocess.TimeoutExpired:
        child.kill()
        child.wait()


def reserve_ports(relay, channels=CHANNEL_OFFSETS):
    offsets = channels + (tuple(1000 + value for value in channels) if relay else ())
    for base in range(21000, 49000, 137):
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


def records(log, prefix):
    return [json.loads(line[len(prefix) + 1:]) for line in log.splitlines() if line.startswith(prefix + " ")]


def validate_native_plan_summary(summary, ot_plan, triple_plan, queries):
    if (len(summary) != queries or {row.get("query") for row in summary} != set(range(queries)) or
            any(type(row.get(field)) is not int or row[field] < 1 for row in summary
                for field in ("registered_ot_streams", "registered_triple_contexts"))):
        raise ValueError("native public plan summary is incomplete")
    for row in summary:
        for field, plan in (("ot_calls", ot_plan), ("triple_calls", triple_plan)):
            if (type(row.get(field)) is not int or row[field] !=
                    sum(call.get("query") == row["query"] for call in plan)):
                raise ValueError("native public plan summary differs from actual requests")
    return True


def validate_native_materials(ready, prepared, consumed, balances, phases, queries, empty_verified=False):
    if not (ready or prepared or consumed or balances):
        return False, None
    ready_by_query = {row.get("query"): row for row in ready}
    if len(ready_by_query) != len(ready) or set(ready_by_query) != set(range(queries)):
        raise ValueError("native material-ready query coverage differs")
    strict_queries = {query for query, row in ready_by_query.items() if row.get("strict") is True}
    if not strict_queries:
        if prepared or consumed or balances:
            raise ValueError("native material evidence without strict preprocessing")
        return False, None
    if strict_queries != set(range(queries)):
        raise ValueError("native preprocessing mode changed between queries")
    for query in strict_queries:
        for phase in ("input", "ready", "online", "output"):
            rows = [row for row in phases if row.get("query") == query and row.get("phase") == phase]
            if len(rows) != 1 or rows[0].get("native_generation_epoch_delta") != 0:
                raise ValueError("native material generated or unmetered after prepare")

    def key(row):
        value = (row.get("query"), row.get("context"), row.get("direction"))
        if (type(value[0]) is not int or value[0] not in strict_queries or
                type(value[1]) is not int or value[1] < 0 or not isinstance(value[2], str) or not value[2]):
            raise ValueError("invalid native material identity")
        return value

    def index(rows):
        result = {}
        for row in rows:
            identity = key(row)
            if identity in result:
                raise ValueError("duplicate native material inventory")
            result[identity] = row
        return result

    generated, finished = index(prepared), index(balances)
    if empty_verified and not (prepared or consumed or balances):
        return True, []
    if not generated or set(generated) != set(finished):
        raise ValueError("native material generation and consumption inventories differ")
    consumed_totals, call_ids, contracts = {}, set(), []
    for row in consumed:
        identity = key(row)
        call = row.get("call")
        count = row.get("count")
        if (identity not in generated or type(call) is not int or call < 0 or
                identity + (call,) in call_ids or type(count) is not int or count < 1):
            raise ValueError("invalid or duplicate native material consumption")
        call_ids.add(identity + (call,))
        consumed_totals[identity] = consumed_totals.get(identity, 0) + count
        contracts.append(identity + (call, row.get("protocol"), count,
                         row.get("message_bits"), row.get("messages_per_ot", 1)))
    for identity, before in generated.items():
        after = finished[identity]
        requested, actual = before.get("requested"), before.get("generated")
        if (type(requested) is not int or type(actual) is not int or requested < 1 or actual < requested or
                after.get("requested") != requested or after.get("generated") != actual or
                after.get("consumed") != requested or consumed_totals.get(identity) != requested or
                after.get("padding_discarded") != actual - requested or after.get("remaining") != 0):
            raise ValueError("native material inventory does not balance")
    return True, sorted(contracts)


def validate_native_public_plan(planned, consumed, queries, empty_verified=False):
    def contracts(rows):
        indexed = {}
        for row in rows:
            identity = tuple(row.get(field) for field in ("query", "context", "direction", "call"))
            query, context, direction, call = identity
            count, bits, groups = (row.get(field) for field in ("count", "message_bits", "messages_per_ot"))
            protocol, widths = row.get("protocol"), row.get("widths")
            if (type(query) is not int or query not in range(queries) or
                    type(context) is not int or context < 0 or not isinstance(direction, str) or not direction or
                    type(call) is not int or call < 0 or identity in indexed or
                    type(count) is not int or count < 1 or type(bits) is not int or not 1 <= bits <= 64 or
                    type(groups) is not int or groups < 1 or
                    protocol not in ("iknp_got", "iknp_batched_got", "kkot_got", "iknp_cot", "iknp_batched_cot") or
                    not isinstance(widths, list) or any(type(width) is not int or not 1 <= width <= 64 for width in widths)):
                raise ValueError("native public OT plan has an invalid or duplicate contract")
            cot = protocol in ("iknp_cot", "iknp_batched_cot")
            if ((cot and (not widths or max(widths) != bits or count % len(widths))) or
                    (protocol == "iknp_cot" and (len(widths) != 1 or groups != 1)) or
                    (not cot and widths)):
                raise ValueError("native public OT plan has an invalid full-width contract")
            indexed[identity] = (protocol, count, bits, groups, tuple(widths))
        return indexed

    expected, actual = contracts(planned), contracts(consumed)
    if empty_verified and not expected and not actual:
        return []
    if not expected or expected != actual:
        raise ValueError("native public OT plan differs from complete one-use consumption")
    if {identity[0] for identity in expected} != set(range(queries)):
        raise ValueError("native public OT plan has incomplete query coverage")
    streams = {identity[:3] for identity in expected}
    for stream in streams:
        calls = {identity[3] for identity in expected if identity[:3] == stream}
        if calls != set(range(len(calls))):
            raise ValueError("native public OT plan has an incomplete public call sequence")
    return sorted(expected.items())


def validate_native_planning_purity(log, purity, queries):
    if (len(purity) != queries or {row.get("query") for row in purity} != set(range(queries)) or
            any(row.get("separate_public_buffers") is not True or
                any(type(row.get(field)) is not int or row[field] != 0 for field in
                    ("sent_bytes", "received_bytes", "generation_delta", "bridge_generation_delta", "prg_blocks_delta"))
                for row in purity)):
        raise ValueError("native planning purity is missing or nonzero")
    positions = {prefix: {} for prefix in ("NATIVE_PLAN_PURITY", "MATERIAL_READY")}
    for line_index, line in enumerate(log.splitlines()):
        prefix, _, payload = line.partition(" ")
        if prefix in positions:
            positions[prefix][json.loads(payload).get("query")] = line_index
    if any(positions["NATIVE_PLAN_PURITY"][query] >= positions["MATERIAL_READY"].get(query, -1)
           for query in range(queries)):
        raise ValueError("native planning purity was reported after material-ready")
    return sorted(purity, key=lambda row: row["query"])


def validate_activations(details, workers, schedule, queries):
    expected = {(query, layer["index"]): layer["output_elements"]
                for query in range(queries) for layer in schedule
                if layer["function"] in ("Sigmoid", "TanH")}
    observed = {(row["query"], row["index"]): row for row in details}
    if len(observed) != len(details) or set(observed) != set(expected):
        raise ValueError("activation coverage differs from the public schedule")
    for key, elements in expected.items():
        detail = observed[key]
        chunks = sorted((row for row in workers if (row["query"], row["index"]) == key),
                        key=lambda row: row["begin"])
        if len(chunks) != detail["actual_workers"] or len({row["slot"] for row in chunks}) != len(chunks):
            raise ValueError("activation workers have duplicate or missing ownership")
        offset = 0
        for chunk in chunks:
            if chunk["begin"] != offset or chunk["count"] < 1:
                raise ValueError("activation chunks overlap or omit input elements")
            offset += chunk["count"]
            if "compute_participants" in chunk:
                cap = chunk.get("requested_ot_participants")
                if (type(cap) is not int or not 1 <= chunk["compute_participants"] <= cap or
                        not 0 <= chunk.get("observed_cggm_participants", -1) <= chunk.get("cggm_participant_limit", -1) <= cap):
                    raise ValueError("activation worker exceeded its owned compute budget")
                if not 0 <= chunk.get("observed_transport_participants", -1) <= cap:
                    raise ValueError("activation transport exceeded its owned compute budget")
        if offset != elements or detail["count"] != elements:
            raise ValueError("activation workers did not cover the complete vector")
        if not 1 <= detail["compute_participants"] <= 4:
            raise ValueError("activation exceeded the four-participant allocation")
        if all("compute_participants" in row for row in chunks) and detail["compute_participants"] != sum(
                row["compute_participants"] for row in chunks):
            raise ValueError("activation compute measurement omits a worker")


def validate_scope_measurements(rounds, parties):
    if not any(row.get("worker_local_cggm_cap") for party in parties for row in party["native_worker_config"]):
        return
    scopes = rounds.get("scopes", [])
    layers = {(row["query"], row["index"]) for row in parties[0]["activations"]}
    complete = [row for row in scopes if row["label"] == "activation" and row["phase"] == "online"]
    if len(complete) != len(layers) or {(row["query"], row["layer"]) for row in complete} != layers:
        raise ValueError("complete activation scope coverage differs from native records")
    for row in complete:
        actual = []
        for party in parties:
            sent = sum(item["sent_bytes"] for item in party["activation_workers"]
                       if (item["query"], item["index"]) == (row["query"], row["layer"]))
            sent += sum(item["sent_bytes"] for field in ("folded_activation_consumed", "dpf_lut_consumed")
                        for item in party[field] if (item["query"], item["layer"]) == (row["query"], row["layer"]))
            actual.append(sent)
        if row["party_sent_bytes"] != actual or row["party_received_bytes"] != actual[::-1]:
            raise ValueError("complete activation scope traffic differs from native records")
    for phase, field in (("offline", "folded_activation_prepared"), ("online", "folded_activation_consumed")):
        calls = [{tuple(row[name] for name in FOLDED_IDENTITY): row for row in party[field]} for party in parties]
        boundaries = [row for row in scopes if row["label"] == "folded_call" and row["phase"] == phase]
        if len(boundaries) != len(calls[0]) or {tuple(row[name] for name in FOLDED_IDENTITY) for row in boundaries} != set(calls[0]):
            raise ValueError("folded call scope coverage differs from native records")
        stage_totals = {}
        for row in scopes:
            if row["label"] != "activation_stage" or row["phase"] != phase:
                continue
            key = tuple(row[name] for name in FOLDED_IDENTITY) + (row["stage"],)
            if key[:-1] not in calls[0] or not 0 <= row["stage"] < len(FOLDED_STAGE_NAMES):
                raise ValueError("activation stage scope is not owned by a folded call")
            total = stage_totals.setdefault(key, [[0, 0], [0, 0]])
            for direction, field_name in enumerate(("party_sent_bytes", "party_received_bytes")):
                for party_index in (0, 1):
                    total[direction][party_index] += row[field_name][party_index]
        for row in boundaries:
            key = tuple(row[name] for name in FOLDED_IDENTITY)
            pair = [party[key] for party in calls]
            if (row["party_sent_bytes"] != [item["sent_bytes"] for item in pair] or
                    row["party_received_bytes"] != [item["received_bytes"] for item in pair]):
                raise ValueError("folded call scope traffic differs from native records")
            if phase == "online":
                for party_index, party in enumerate(parties):
                    workers = [item for item in party["activation_workers"] if
                               (item["query"], item["index"], item.get("slot")) == key[:3]]
                    if workers and workers[0].get("requested_ot_participants") == 1 and row.get("actors_per_party", [0, 0])[party_index] != 1:
                        raise ValueError("cap-one folded call used an unowned communication helper")
            for stage in range(len(FOLDED_STAGE_NAMES)):
                expected = [[item[phase][stage][direction] for item in pair]
                            for direction in ("sent_bytes", "received_bytes")]
                if stage_totals.get(key + (stage,), [[0, 0], [0, 0]]) != expected:
                    raise ValueError("activation stage traffic differs from native records")
    groups = [row for row in scopes if row["label"] == "folded_prepare"]
    expected_groups = [{row["query"]: row for row in party["folded_prepare_groups"]} for party in parties]
    if len(groups) != len(expected_groups[0]) or {row["query"] for row in groups} != set(expected_groups[0]):
        raise ValueError("folded prepare group scope coverage differs from native records")
    for row in groups:
        pair = [index[row["query"]] for index in expected_groups]
        if (row["party_sent_bytes"] != [item["sent_bytes"] for item in pair] or
                row["party_received_bytes"] != [item["received_bytes"] for item in pair]):
            raise ValueError("folded prepare group scope traffic differs from native records")


def validate_native_triples(planned, prepared, consumed, balances, ready, queries, empty_verified=False):
    if not (planned or prepared or consumed or balances):
        if empty_verified and {row.get("query") for row in ready if row.get("strict") is True} == set(range(queries)):
            return True, []
        return False, None
    strict_queries = {row.get("query") for row in ready if row.get("strict") is True}
    if strict_queries != set(range(queries)):
        raise ValueError("native bit-triple inventory without strict query coverage")

    def identity(row, call=False):
        key = tuple(row.get(field) for field in (("query", "context", "call") if call else ("query", "context")))
        if (any(type(value) is not int or value < 0 for value in key) or key[0] not in strict_queries):
            raise ValueError("native bit-triple identity invalid")
        return key

    def calls(rows):
        indexed = {}
        for row in rows:
            key = identity(row, True)
            count, method, packed, offset = (row.get(field) for field in ("count", "method", "packed", "offset"))
            if (key in indexed or type(count) is not int or count < 1 or count % 2 or
                    type(method) is not int or method not in (2, 3) or type(packed) is not bool or
                    type(offset) is not int or offset < 1 or count % offset or
                    (packed and count % 8) or (method == 3 and count % (2 * offset))):
                raise ValueError("native bit-triple contract unsafe, invalid or repeated")
            indexed[key] = (count, method, packed, offset)
        return indexed

    plan, used = calls(planned), calls(consumed)
    if not plan or plan != used:
        raise ValueError("native bit-triple consumption differs from complete public plan")

    def inventory(rows):
        indexed = {}
        for row in rows:
            key = identity(row)
            if key in indexed:
                raise ValueError("native bit-triple inventory repeated")
            indexed[key] = row
        return indexed

    before, after = inventory(prepared), inventory(balances)
    contexts = {key[:2] for key in plan}
    if set(before) != contexts or set(after) != contexts:
        raise ValueError("native bit-triple inventory coverage differs")
    for key in contexts:
        entries = {call[2]: value for call, value in plan.items() if call[:2] == key}
        if set(entries) != set(range(len(entries))):
            raise ValueError("native bit-triple public call sequence incomplete")
        requested = sum(value[0] for value in entries.values())
        generated, finished = before[key], after[key]
        if (any(type(row.get(field)) is not int or row[field] != requested
                    for row, fields in ((generated, ("requested", "generated")),
                                        (finished, ("requested", "generated", "consumed"))) for field in fields) or
                generated.get("standard_calls") != sum(value[1] == 2 for value in entries.values()) or
                generated.get("correlated_calls") != sum(value[1] == 3 for value in entries.values()) or
                finished.get("remaining") != 0 or finished.get("padding_discarded") != 0):
            raise ValueError("native bit-triple inventory does not balance")
    return True, sorted((key, value) for key, value in plan.items())


def planned_conversion_calls(layer, backend, paired):
    arguments = call_arguments(layer)
    numeric = []
    for value in arguments:
        try:
            numeric.append(int(value))
        except ValueError:
            break
    a = numeric
    a[0] = layer["shape"][0]
    calls = []

    def log2(value):
        if value < 1 or value & (value - 1):
            raise ValueError("conversion schedule has a non-power-of-two scale")
        return value.bit_length() - 1

    def append(count, input_bits, output_bits, shift, call):
        if shift or output_bits > input_bits:
            calls.append(dict(count=count, input_bits=input_bits, output_bits=output_bits,
                              shift=shift, call=call))

    beaver = backend.startswith("SCI_Beaver")
    function = layer["function"]
    if function == "MatMul":
        depth = (a[1] - 1).bit_length()
        product, count = a[8] + a[9], a[0] * a[2]
        append(a[0] * a[1], a[8], product + depth if beaver else a[8] + depth, 0, 3)
        append(count, product + depth, product, depth, 2)
        shift = log2(a[3]) + log2(a[4]) + log2(a[7]) + a[5] - depth
        if shift > 0:
            append(count, product, product, shift, 0)
    elif function == "MulCir":
        count, product = a[0] * a[1], a[5] + a[6]
        if beaver:
            if paired:
                if a[5] != a[6]:
                    raise ValueError("conversion schedule cannot group heterogeneous extension")
                append(2 * count, a[5], product, 0, 3)
            else:
                append(count, a[5], product, 0, 3)
                append(count, a[6], product, 0, 3)
        append(count, product, product, log2(a[2]) + log2(a[3]) + log2(a[4]), 0)
    elif function == "ScalarMul":
        count = a[0] * a[1]
        append(count, a[6], a[7], 0, 3)
        append(count, a[7], a[7], log2(a[2]) + log2(a[3]) + log2(a[4]), 0)
    elif function in ("MatAdd", "MatSubBroadCastA", "MatAddBroadCastB"):
        count, temp = a[0] * a[1], a[8]
        scalar_a, scalar_b = function == "MatSubBroadCastA", function == "MatAddBroadCastB"
        append(count, temp if scalar_a else a[6], temp, 0, 3)
        append(count, temp if scalar_b else a[7], temp, 0, 3)
        append(count, temp, temp, 0 if scalar_a else log2(a[2]) + log2(a[4]), 0)
        append(count, temp, temp, 0 if scalar_b else log2(a[3]) + log2(a[4]), 0)
        append(count, temp, temp, log2(a[5]), 0)
    elif function not in ("Sigmoid", "TanH"):
        raise ValueError("conversion schedule has an unsupported public operation")
    return calls


def validate_conversions(offline, online, preprocessing, groups, schedule, queries, backend,
                         batched_openings=False):
    layers = {layer["index"]: layer for layer in schedule}
    enabled = backend in ("SCI_ours", "SCI_Beaver_full", "SCI_Beaver_segmented", "SCI_Beaver_segmented_DPF_LUT", FOLDED_BACKEND)
    if backend is None:
        enabled = bool(offline or online)
    if enabled and (not offline or not online):
        raise ValueError("enabled conversion backend has missing offline/online evidence")
    if not enabled and (offline or online or preprocessing or groups):
        raise ValueError("native conversion backend has unexpected FSS evidence")
    required = ("count", "input_bits", "output_bits", "shift", "call", "numeric_mode", "comparison_backend")
    fields = required + ("signed_arithmetic", "low_comparison_bits", "sign_comparison_bits",
                         "wrap_comparison_bits", "low_nseg", "sign_nseg", "wrap_nseg",
                         "low_widths", "sign_widths", "wrap_widths", "segment_policy", "sample_groups",
                         "fused_sign_wrap")

    def identity(row):
        key = tuple(row.get(field) for field in ("query", "index", "slot"))
        if (any(type(value) is not int for value in key) or
                not 0 <= key[0] < queries or key[1] not in layers or key[2] < 0 or
                layers[key[1]]["function"] in ("Sigmoid", "TanH")):
            raise ValueError("conversion identity differs from the public schedule")
        return key

    def index_calls(rows):
        indexed = {}
        slots = {}
        for row in rows:
            key = identity(row)
            if key in indexed:
                raise ValueError("duplicate conversion material or consumption")
            if (any(field not in row for field in required) or
                    any(type(row[field]) is not int for field in required[:5]) or
                    row["count"] < 1 or not 1 <= row["input_bits"] <= 64 or
                    not 1 <= row["output_bits"] <= 64 or not 0 <= row["shift"] < row["input_bits"] or
                    row["call"] not in (0, 1, 2, 3) or row["numeric_mode"] != "sci_compatible" or
                    row["comparison_backend"] not in ("full_tree", "segmented_dcf") or
                    ("signed_arithmetic" in row and row["signed_arithmetic"] not in (True, 1)) or
                    row.get("sample_groups", 1) != 1):
                raise ValueError("invalid conversion numerical contract")
            if backend is not None and row["comparison_backend"] != (
                    "full_tree" if backend == "SCI_Beaver_full" else "segmented_dcf"):
                raise ValueError("conversion comparison backend differs from requested backend")
            indexed[key] = {field: row.get(field) for field in fields}
            slots.setdefault(key[:2], set()).add(key[2])
        if enabled and {key[0] for key in indexed} != set(range(queries)):
            raise ValueError("conversion queries have incomplete material coverage")
        if any(values != set(range(len(values))) for values in slots.values()):
            raise ValueError("conversion slots have incomplete material coverage")
        return indexed

    prepared, consumed = index_calls(offline), index_calls(online)
    if prepared != consumed:
        raise ValueError("conversion offline/online identities or numerical contracts differ")
    if preprocessing and (len(preprocessing) != len(prepared) or any(
            row.get("fresh") is not True or row.get("dealerless") is not True for row in preprocessing)):
        raise ValueError("conversion preprocessing has incomplete or nonfresh material coverage")
    grouped = {}
    for row in groups:
        key = identity(row)
        call = consumed.get(key)
        if (key in grouped or call is None or layers[key[1]]["function"] != "MulCir" or
                row.get("input_groups") != 2 or type(row.get("group_count")) is not int or
                row["group_count"] < 1 or 2 * row["group_count"] != call["count"] or
                row["group_count"] != layers[key[1]]["output_elements"] or
                row.get("mapping") != "lhs_then_rhs" or row.get("contract") != "homogeneous_signed_extension" or
                call["call"] != 3 or call["shift"] != 0 or call["output_bits"] <= call["input_bits"]):
            raise ValueError("conversion group differs from its consumed numerical contract")
        grouped[key] = {field: row.get(field) for field in
                        ("input_groups", "group_count", "mapping", "contract")}
    paired = {key for key, call in consumed.items()
              if layers[key[1]]["function"] == "MulCir" and call["call"] == 3 and
              call["count"] == 2 * layers[key[1]]["output_elements"]}
    if set(grouped) != paired:
        raise ValueError("conversion group coverage differs from paired extension calls")
    if enabled and backend is not None and all("source_call" in layer and "shape" in layer for layer in schedule):
        expected = {}
        paired_layers = {key[1] for key in grouped}
        for query in range(queries):
            for layer in schedule:
                calls = planned_conversion_calls(layer, backend, layer["index"] in paired_layers)
                if batched_openings and layer["index"] % 15 == 7:
                    calls = []
                elif batched_openings and layer["index"] % 15 == 5:
                    calls = [dict(call, count=2 * call["count"]) for call in calls]
                for slot, call in enumerate(calls):
                    expected[(query, layer["index"], slot)] = call
        observed = {key: {field: value[field] for field in required[:5]} for key, value in consumed.items()}
        if observed != expected:
            raise ValueError("conversion coverage differs from the complete public call plan")
    return prepared, grouped


def validate_graph_openings(groups, schedule, queries, batched):
    expected = {(query, layer["index"]) for query in range(queries) for layer in schedule
                if layer["index"] % 15 == 5} if batched else set()
    indexed = {}
    layers = {layer["index"]: layer for layer in schedule}
    for row in groups:
        key = (row.get("query"), row.get("index"))
        index = row.get("index")
        if (key in indexed or key not in expected or row.get("indices") != [index, index + 2] or
                row.get("count_per_group") != layers[index]["output_elements"] or
                layers[index]["function"] != "MatAdd" or layers.get(index + 2, {}).get("function") != "MatAdd" or
                row.get("input_groups") != 2 or row.get("mapping") != "gate_bias_then_candidate_bias" or
                row.get("contract") != "homogeneous_signed_truncation"):
            raise ValueError("graph opening group differs from the frozen MatAdd mapping")
        indexed[key] = row
    if set(indexed) != expected:
        raise ValueError("graph opening groups have incomplete query coverage")


def validate_prefetch_budgets(budgets, prefixes, contexts, queries):
    if not budgets:
        return
    if (len(budgets) != contexts or {row.get("context") for row in budgets} != set(range(contexts)) or
            any(row.get("contexts") != contexts or type(row.get("context_budget_bytes")) is not int or
                row["context_budget_bytes"] < 0 for row in budgets) or
            len({row.get("party_budget_bytes") for row in budgets}) != 1 or
            sum(row["context_budget_bytes"] for row in budgets) != budgets[0]["party_budget_bytes"]):
        raise ValueError("conversion prefetch context budgets exceed the party allocation")
    allocations = [row["context_budget_bytes"] for row in budgets]
    if (len(prefixes) > queries * sum(value > 0 for value in allocations) or any(
            row.get("budget_bytes") not in allocations or type(row.get("resident_bytes")) is not int or
            not 0 <= row["resident_bytes"] <= row["budget_bytes"] for row in prefixes)):
        raise ValueError("conversion prefetch resident bytes exceed the context allocation")


def normalize_beaver_offline(rows):
    result = []
    for row in rows:
        row = dict(row)
        if "sent_bytes" in row and row.get("receive_instrumented") is not True:
            row.update(received_bytes=None, receive_instrumented=False)
        result.append(row)
    return result


def validate_beaver_traffic(parties):
    indexed = []
    for party in parties:
        rows = {}
        for row in party.get("beaver_offline_calls", []):
            if row.get("receive_instrumented") is not True:
                continue
            key = (row.get("query"), row.get("index"))
            if (key in rows or any(type(row.get(field)) is not int or row[field] < 0
                    for field in ("sent_bytes", "received_bytes")) or
                    row.get("cross_term_batches") != 2 * row.get("tiles", 0)):
                raise ValueError("Beaver per-owner traffic has invalid instrumentation")
            rows[key] = row
        indexed.append(rows)
    if set(indexed[0]) != set(indexed[1]) or any(
            indexed[0][key]["sent_bytes"] != indexed[1][key]["received_bytes"] or
            indexed[1][key]["sent_bytes"] != indexed[0][key]["received_bytes"] for key in indexed[0]):
        raise ValueError("Beaver per-owner send/receive traffic differs between parties")


def validate_dpf_inventory(planned, prepared, consumed, balances, phases, queries, enabled):
    if not enabled:
        if planned or prepared or consumed or balances:
            raise ValueError("control backend unexpectedly used DPF activation materials")
        return False
    identity=("query","layer","slot","call")
    shape=("family","count","index_bits","tables","output_bits","grouped_tables")
    def index(rows):
        result={}
        for row in rows:
            key=tuple(row.get(field) for field in identity)
            if key in result or any(type(value) is not int for value in key):
                raise ValueError("DPF activation request identity invalid or repeated")
            result[key]=tuple(tuple(row[field]) if isinstance(row[field],list) else row[field] for field in shape)
        return result
    plan=index(planned)
    if not plan or plan!=index(consumed):
        raise ValueError("DPF activation prepared and consumed shapes differ")
    if any(row.get("online_full_domain_eval") is not False or row.get("online_loaded_bytes")!=0 or
           row.get("online_generated")!=0 for row in consumed):
        raise ValueError("DPF activation material generated or loaded online")
    for query in range(queries):
        expected=sum(row["count"] for row in planned if row["query"]==query)
        prep=[row for row in prepared if row.get("query")==query]
        balance=[row for row in balances if row.get("query")==query]
        planned_slots={row["slot"] for row in planned if row["query"]==query}
        if (len({row.get("slot") for row in prep})!=len(prep) or
            not planned_slots <= {row.get("slot") for row in prep} or
            sum(row.get("requests",0) for row in prep)!=expected or
            any(row.get("pre_input_eval_all") is not True or row.get("no_online_reload") is not True for row in prep) or
            len(balance)!=1 or balance[0].get("generated")!=expected or balance[0].get("consumed")!=expected or
            balance[0].get("remaining")!=0):
            raise ValueError("DPF activation inventory is incomplete")
        for phase in ("input","ready","online","output"):
            rows=[row for row in phases if row.get("query")==query and row.get("phase")==phase]
            if len(rows)!=1 or rows[0].get("dpf_generation_epoch_delta")!=0:
                raise ValueError("DPF activation generation after material readiness")
    return True


def validate_folded_inventory(planned, prepared, consumed, balances, phases, queries, enabled, log=None):
    if not enabled:
        if planned or prepared or consumed or balances:
            raise ValueError("control backend unexpectedly used folded activation materials")
        return False

    def index(rows):
        result = {}
        for row in rows:
            key = tuple(row.get(field) for field in FOLDED_IDENTITY)
            shape = tuple(row.get(field) for field in FOLDED_SHAPE)
            if (key in result or any(type(value) is not int or value < 0 for value in key) or
                    key[0] not in range(queries) or any(type(value) is not int for value in shape) or
                    row["kind"] not in (0, 1) or row["count"] < 1 or
                    row["input_bits"] != 16 or row["output_bits"] != 16 or row["input_scale"] != 7 or
                    row["output_scale"] != (14 if row["kind"] == 0 else 7) or
                    row["lut_family"] not in (0, 1) or row["sign_backend"] not in (0, 1) or
                    (row["sign_backend"] == 0 and row["sign_segments"] != 0) or
                    (row["sign_backend"] == 1 and not 1 <= row["sign_segments"] <= 31) or
                    row["table_version"] != 1):
                raise ValueError("folded activation request has an invalid or repeated native contract")
            result[key] = shape
        return result

    plan = index(planned)
    if not plan or plan != index(prepared) or plan != index(consumed):
        raise ValueError("folded activation planned, prepared and consumed contracts differ")
    if {key[0] for key in plan} != set(range(queries)):
        raise ValueError("folded activation query coverage is incomplete")
    for stream in {key[:1] + key[2:3] for key in plan}:
        calls = {key[3] for key in plan if key[:1] + key[2:3] == stream}
        if calls != set(range(len(calls))):
            raise ValueError("folded activation call sequence is incomplete")
    for row in prepared + consumed:
        for field in ("offline_wall_us", "online_wall_us", "resident_bytes", "peak_working_bytes"):
            if type(row.get(field)) is not int or row[field] < 0:
                raise ValueError("folded activation actual metrics are missing or invalid")
        for phase in ("offline", "online"):
            stages = row.get(phase)
            if (not isinstance(stages, list) or len(stages) != len(FOLDED_STAGE_NAMES) or
                    any(not isinstance(stage, dict) or any(type(stage.get(field)) is not int or stage[field] < 0
                        for field in ("elapsed_us", "sent_bytes", "received_bytes")) for stage in stages)):
                raise ValueError("folded activation stage metrics are missing or invalid")
        if row.get("online_generated") != 0 or row.get("online_loaded_bytes") != 0:
            raise ValueError("folded activation material was generated or loaded online")
    if any(row.get("native_ot_consumed") != 0 or row.get("bit_triples_consumed") != 0
           for row in consumed):
        raise ValueError("complete folded activation consumed native SCI inventory or lacks attribution")
    slot_calls = {}
    for row in planned:
        key = (row["query"], row["slot"])
        slot_calls[key] = slot_calls.get(key, 0) + 1
    if any(row.get("pre_input") is not True or
           row.get("requests") != slot_calls[(row["query"], row["slot"])] or
           row.get("calls") != slot_calls[(row["query"], row["slot"])] or
           type(row.get("generation_epoch")) is not int or
           row["generation_epoch"] < 1 for row in prepared):
        raise ValueError("folded activation was not fully prepared before input")
    epochs = {}
    for row in prepared:
        key = (row["slot"], row["query"])
        epochs.setdefault(key, set()).add(row["generation_epoch"])
    previous = {}
    for (slot, query), values in sorted(epochs.items()):
        if len(values) != 1 or min(values) <= previous.get(slot, 0):
            raise ValueError("folded activation query did not generate fresh resident material")
        previous[slot] = min(values)
    for query in range(queries):
        expected = sum(row["count"] for row in planned if row["query"] == query)
        calls = sum(row["query"] == query for row in planned)
        balance = [row for row in balances if row.get("query") == query]
        if (len(balance) != 1 or balance[0].get("generated") != calls or
                balance[0].get("consumed") != calls or balance[0].get("remaining") != 0 or
                balance[0].get("generated_elements") != expected or
                balance[0].get("consumed_elements") != expected or
                any(balance[0].get(field) != 0 for field in
                    ("online_generation_epoch_delta", "online_generated", "online_loaded_bytes"))):
            raise ValueError("folded activation complete one-use inventory does not balance")
        for phase in ("input", "ready", "online", "output"):
            rows = [row for row in phases if row.get("query") == query and row.get("phase") == phase]
            if len(rows) != 1 or rows[0].get("folded_generation_epoch_delta") != 0:
                raise ValueError("folded activation generation after material readiness")
    if len(balances) != queries:
        raise ValueError("folded activation balance query coverage differs")
    if log is not None:
        positions = {prefix: {} for prefix in ("FOLDED_ACTIVATION_PLAN", "FOLDED_ACTIVATION_PREPARED",
                                               "FOLDED_ACTIVATION_CONSUMED", "MATERIAL_READY")}
        for line_index, line in enumerate(log.splitlines()):
            prefix, _, payload = line.partition(" ")
            if prefix in positions:
                row = json.loads(payload)
                key = row.get("query") if prefix == "MATERIAL_READY" else tuple(row.get(field) for field in FOLDED_IDENTITY)
                if key in positions[prefix]:
                    raise ValueError("folded activation readiness ordering evidence is duplicated")
                positions[prefix][key] = line_index
        if set(positions["MATERIAL_READY"]) != set(range(queries)):
            raise ValueError("folded activation material-ready boundary is missing")
        for key in plan:
            if not (positions["FOLDED_ACTIVATION_PLAN"].get(key, -1) <
                    positions["FOLDED_ACTIVATION_PREPARED"].get(key, -1) <
                    positions["MATERIAL_READY"][key[0]] <
                    positions["FOLDED_ACTIVATION_CONSUMED"].get(key, -1)):
                raise ValueError("folded activation preparation or consumption crossed the material-ready boundary")
    return True


def validate_folded_config(configs, planned, schedule, queries, enabled,
                           expected_selection=None, expected_family=None,
                           expected_sign_backend=None, expected_sign_segments=None):
    if not enabled:
        if configs:
            raise ValueError("control backend unexpectedly configured folded activations")
        return None
    fields = ("sigmoid", "tanh", "lut_family", "sign_backend", "sign_segments", "index_bits", "table_entries",
              "table_version", "selection_storage")
    contracts = {tuple(row.get(field, 0) if field == "sign_segments" else row.get(field)
                       for field in fields) for row in configs}
    slots = [row.get("slot") for row in configs]
    if (not configs or len(contracts) != 1 or any(type(slot) is not int or slot < 0 for slot in slots) or
            len(set(slots)) != len(slots) or set(slots) != set(range(len(slots))) or
            any(row.get("port_offset") != 2000 + 150 * row["slot"] for row in configs)):
        raise ValueError("folded activation owned contexts are incomplete or inconsistent")
    reference = configs[0]
    selected = {kind for kind, name in enumerate(("sigmoid", "tanh"))
                if reference.get(name) == "folded_sigma_f12_q10"}
    if (not selected or any(reference.get(name) not in ("folded_sigma_f12_q10", "native_sci")
            for name in ("sigmoid", "tanh")) or reference.get("lut_family") not in (0, 1) or
            type(reference.get("sign_backend")) is not int or reference.get("sign_backend") not in (0, 1) or
            type(reference.get("sign_segments", 0)) is not int or
            (reference["sign_backend"] == 0 and reference.get("sign_segments", 0) != 0) or
            (reference["sign_backend"] == 1 and not 1 <= reference.get("sign_segments", 0) <= 31) or
            reference.get("index_bits") != 10 or
            reference.get("table_entries") != 1024 or reference.get("table_version") != 1 or
            reference.get("selection_storage") != "u32_ring_shares"):
        raise ValueError("folded activation configuration differs from the frozen algorithm")
    if expected_selection is not None and selected != {"both": {0, 1}, "sigmoid": {0}, "tanh": {1}}[expected_selection]:
        raise ValueError("folded activation selection differs from the requested ablation")
    if expected_family is not None and reference["lut_family"] != (
            0 if expected_family == "full_q_arithmetic" else 1):
        raise ValueError("folded LUT family differs from the requested construction")
    if expected_sign_backend is not None and reference["sign_backend"] != (
            0 if expected_sign_backend == "bitdec" else 1):
        raise ValueError("folded sign backend differs from the requested candidate")
    if expected_sign_segments is not None and reference.get("sign_segments", 0) != expected_sign_segments:
        raise ValueError("folded sign segmentation differs from the public outer policy")
    lut_widths = [10] if reference["lut_family"] == 0 else [5, 5]
    if any(type(row.get("lut_segments")) is not int or row["lut_segments"] != len(lut_widths) or
           row.get("lut_segment_widths") != lut_widths or
           not isinstance(row.get("lut_segment_widths"), list) or
           any(type(width) is not int for width in row["lut_segment_widths"]) for row in configs):
        raise ValueError("folded LUT segmentation differs from its selected construction")
    expected = {(query, layer["index"]): (0 if layer["function"] == "Sigmoid" else 1, layer["output_elements"])
                for query in range(queries) for layer in schedule
                if layer["function"] in ("Sigmoid", "TanH") and
                (0 if layer["function"] == "Sigmoid" else 1) in selected}
    actual = {}
    for row in planned:
        key = (row["query"], row["layer"])
        if (row["slot"] not in slots or row["kind"] not in selected or
                row["lut_family"] != reference["lut_family"] or row["sign_backend"] != reference["sign_backend"] or
                row["sign_segments"] != reference.get("sign_segments", 0)):
            raise ValueError("folded activation plan differs from its owned configuration")
        kind, count = actual.get(key, (row["kind"], 0))
        if kind != row["kind"]:
            raise ValueError("folded activation layer mixes different numeric algorithms")
        actual[key] = (kind, count + row["count"])
    if actual != expected:
        raise ValueError("complete folded activations do not cover the selected public layers")
    return dict(zip(fields, next(iter(contracts))), lut_segments=len(lut_widths),
                lut_segment_widths=lut_widths)


def aggregate_folded_activations(parties, queries):
    indexes = []
    for party in parties:
        indexes.append({phase: {tuple(row[field] for field in FOLDED_IDENTITY): row for row in party[field]}
                        for phase, field in (("offline", "folded_activation_prepared"),
                                             ("online", "folded_activation_consumed"))})
    if len(indexes) != 2 or any(set(indexes[0][phase]) != set(indexes[1][phase]) for phase in ("offline", "online")):
        raise ValueError("folded activation call identities differ between parties")
    result = []
    for query in range(queries):
        summary = {"query": query,
                   "time_rule": "stage work sums per party; overlapping workers are not query wall time"}
        for kind, name in enumerate(("sigmoid", "tanh")):
            keys = sorted(key for key, row in indexes[0]["online"].items()
                          if row["query"] == query and row["kind"] == kind)
            if not keys:
                continue
            value = {"count": sum(indexes[0]["online"][key]["count"] for key in keys), "calls": []}
            for phase in ("offline", "online"):
                stages = {name: dict(work_us_per_party=[0, 0], sent_bytes=0) for name in FOLDED_STAGE_NAMES}
                actual_sent = 0
                for key in keys:
                    left, right = (index[phase][key] for index in indexes)
                    if any(left[field] != right[field] for field in FOLDED_SHAPE):
                        raise ValueError("folded activation numeric contracts differ between parties")
                    for stage_index, stage_name in enumerate(FOLDED_STAGE_NAMES):
                        pair = [row[phase][stage_index] for row in (left, right)]
                        if pair[0]["sent_bytes"] != pair[1]["received_bytes"] or pair[1]["sent_bytes"] != pair[0]["received_bytes"]:
                            raise ValueError("folded activation stage traffic differs between parties")
                        for party_index in (0, 1):
                            stages[stage_name]["work_us_per_party"][party_index] += pair[party_index]["elapsed_us"]
                            stages[stage_name]["sent_bytes"] += pair[party_index]["sent_bytes"]
                    if phase == "online" or any(field in row for row in (left,right)
                                                 for field in ("sent_bytes","received_bytes")):
                        if (left.get("sent_bytes") != right.get("received_bytes") or
                                right.get("sent_bytes") != left.get("received_bytes") or
                                any(type(row.get(field)) is not int or row[field] < 0
                                    for row in (left, right) for field in ("sent_bytes", "received_bytes")) or
                                any(sum(stage[field] for stage in row[phase]) != row[field]
                                    for row in (left, right) for field in ("sent_bytes", "received_bytes"))):
                            raise ValueError("folded activation complete traffic is unbalanced or missing stage attribution")
                        actual_sent += left["sent_bytes"] + right["sent_bytes"]
                value[phase] = {"stages": stages, "sent_bytes": actual_sent if phase == "online" else
                                sum(stage["sent_bytes"] for stage in stages.values())}
            for key in keys:
                calls = [index["online"][key] for index in indexes]
                value["calls"].append(dict(zip(FOLDED_IDENTITY, key),
                    count=calls[0]["count"], offline_wall_us=max(row["offline_wall_us"] for row in calls),
                    online_wall_us=max(row["online_wall_us"] for row in calls),
                    resident_bytes_per_party=[row["resident_bytes"] for row in calls],
                    peak_working_bytes_per_party=[row["peak_working_bytes"] for row in calls]))
            summary[name] = value
        result.append(summary)
    return result


def collect(logs, queries, schedule, diagnostic, backend=None, native_preprocessing=None,
            folded_activations=None, lut_key_family=None, folded_sign_backend=None, folded_sign_segments=None):
    parties = []
    conversion_contracts = []
    for role, log in enumerate(logs, 1):
        finished = records(log, "FINISHED")
        if len(finished) != 1 or finished[0].get("party") != role:
            raise ValueError("missing completed party record")
        if not diagnostic and any(line.startswith("TENSOR ") for line in log.splitlines()):
            raise ValueError("secure run exposed diagnostic tensors")
        coverage = records(log, "COVERAGE")
        expected = [(query, layer["index"], layer["output_elements"]) for query in range(queries) for layer in schedule]
        actual = [(row["query"], row["index"], row["elements"]) for row in coverage]
        if len(set(actual)) != len(actual) or set(actual) != set(expected) or any(row["sample_groups"] != 1 for row in coverage):
            raise ValueError("dependency layer coverage does not match the fixed batch schedule")
        native_config = records(log, "NATIVE_WORKER_CONFIG")
        batched_openings = bool(native_config and native_config[0].get("graph_openings") == "batched")
        opening_groups = records(log, "GRAPH_OPENING_GROUP")
        validate_graph_openings(opening_groups, schedule, queries, batched_openings)
        graph_tasks = records(log, "GRAPH_TASK")
        graph_groups = records(log, "GRAPH_GROUP")
        if graph_tasks:
            identities = [(row["query"], row["index"]) for row in graph_tasks]
            if len(set(identities)) != len(identities) or set(identities) != {(query, layer["index"]) for query in range(queries) for layer in schedule}:
                raise ValueError("graph task identity coverage differs from the public schedule")
            if any(not 1 <= row["observed_ot_participants"] <= row["branch_compute_limit"] for row in graph_tasks):
                raise ValueError("observed graph task OT team exceeds its branch allocation")
            if any(not 1 <= row.get("observed_compute_participants",1) <= row["branch_compute_limit"] for row in graph_tasks):
                raise ValueError("observed graph task compute participants exceed its branch allocation")
        if native_config and native_config[0].get("graph_mode") == "parallel":
            expected_groups = {(query,index) for query in range(queries) for index in ((0,5,9) if native_config[0].get("tail_parallel",False) else (0,5))}
            identities = [(row["query"],row["index"]) for row in graph_groups]
            if len(set(identities)) != len(identities) or set(identities) != expected_groups:
                raise ValueError("graph branch join coverage differs from the single-step schedule")
            if any(row["observed_compute_upper_bound"] > 4 for row in graph_groups):
                raise ValueError("graph branches exceeded the total four-participant allocation")
        activations, activation_workers = records(log, "ACTIVATION_DETAIL"), records(log, "ACTIVATION_WORKER")
        if native_config:
            if len(native_config) != 1 or native_config[0]["max_compute_participants"] > 4:
                raise ValueError("invalid native worker configuration")
            validate_activations(activations, activation_workers, schedule, queries)
        conversion_offline = records(log, "CONVERSION_OFFLINE")
        conversion_online = records(log, "CONVERSION_ONLINE")
        preprocessing = records(log, "CONVERSION_PREPROCESS_DETAIL")
        conversion_groups = records(log, "CONVERSION_GROUP")
        conversion_contracts.append(validate_conversions(conversion_offline, conversion_online,
            preprocessing, conversion_groups, schedule, queries, backend, batched_openings))
        prefetch_budgets, prefetch = records(log, "CONVERSION_PREFETCH_BUDGET"), records(log, "CONVERSION_PREFETCH")
        validate_prefetch_budgets(prefetch_budgets, prefetch,
            native_config[0].get("graph_branches", 1) if native_config else 1, queries)
        phases = records(log, "METRIC")
        model_bias_preparation = records(log, "MODEL_BIAS_PREPARATION")
        if model_bias_preparation:
            model_rows = [row for row in phases if row.get("phase") == "model_prepare" and row.get("query") == -1]
            if len(model_bias_preparation) != 1 or len(model_rows) != 1:
                raise ValueError("model bias preparation is missing or duplicated")
            bias = model_bias_preparation[0]
            if (bias.get("owner") != 1 or bias.get("divisor") != 8 or bias.get("input_bits") != 16 or
                    bias.get("output_bits") != 16 or bias.get("rounding") != "toward_zero" or
                    bias.get("query_randomness_cached") is not False or
                    bias.get("cached_ring_bytes") != (400 if bias.get("enabled") and role == 1 else 0) or
                    model_rows[0]["sent_bytes"] != 0 or model_rows[0]["received_bytes"] != 0):
                raise ValueError("model bias preparation differs from the owner-only frozen contract")
        native_ready = records(log, "MATERIAL_READY")
        if native_preprocessing is not None and (
                len(native_ready) != queries or
                {row.get("query") for row in native_ready} != set(range(queries)) or
                any(row.get("strict") is not native_preprocessing for row in native_ready)):
            raise ValueError("requested native preprocessing differs from material-ready evidence")
        native_prepared = records(log, "NATIVE_MATERIAL_PREPARED")
        native_consumed = records(log, "NATIVE_MATERIAL_CONSUMED")
        native_balances = records(log, "NATIVE_MATERIAL_BALANCE")
        native_plan = records(log, "NATIVE_PUBLIC_PLAN")
        triple_plan = records(log, "NATIVE_PUBLIC_TRIPLE_PLAN")
        native_summary = records(log, "NATIVE_PUBLIC_PLAN_SUMMARY")
        empty_verified = False
        if native_preprocessing is True and (native_summary or not (native_plan or triple_plan)):
            empty_verified = validate_native_plan_summary(native_summary, native_plan, triple_plan, queries)
        native_verified, native_contract = validate_native_materials(native_ready,
            native_prepared, native_consumed, native_balances, phases, queries, empty_verified)
        native_plan_contract = (validate_native_public_plan(native_plan, native_consumed, queries, empty_verified)
                                if native_preprocessing is True else None)
        native_purity = records(log, "NATIVE_PLAN_PURITY")
        native_purity_contract = (validate_native_planning_purity(log, native_purity, queries)
                                  if native_preprocessing is True else None)
        triple_prepared = records(log, "NATIVE_TRIPLE_PREPARED")
        triple_consumed = records(log, "NATIVE_TRIPLE_CONSUMED")
        triple_balances = records(log, "NATIVE_TRIPLE_BALANCE")
        triple_verified, triple_contract = validate_native_triples(triple_plan, triple_prepared,
            triple_consumed, triple_balances, native_ready, queries, empty_verified)
        if native_preprocessing is True:
            if not (native_verified and triple_verified):
                raise ValueError("complete native preprocessing requires verified OT and bit-triple inventories")
            for query in range(queries):
                for phase in ("input", "ready", "online", "output"):
                    rows = [row for row in phases if row.get("query") == query and row.get("phase") == phase]
                    if (len(rows) != 1 or type(rows[0].get("bridge_generation_epoch_delta")) is not int or
                            rows[0]["bridge_generation_epoch_delta"] != 0):
                        raise ValueError("bridge material generated or unmetered after prepare")
        dpf_plan=records(log,"DPF_LUT_PLAN")
        dpf_prepared=records(log,"DPF_LUT_PREPARED")
        dpf_consumed=records(log,"DPF_LUT_CONSUMED")
        dpf_balances=records(log,"DPF_LUT_BALANCE")
        dpf_verified=validate_dpf_inventory(dpf_plan,dpf_prepared,dpf_consumed,dpf_balances,phases,queries,
            backend=="SCI_Beaver_segmented_DPF_LUT")
        folded_plan = records(log, "FOLDED_ACTIVATION_PLAN")
        folded_prepared = records(log, "FOLDED_ACTIVATION_PREPARED")
        folded_consumed = records(log, "FOLDED_ACTIVATION_CONSUMED")
        folded_balances = records(log, "FOLDED_ACTIVATION_BALANCE")
        folded_config = records(log, "FOLDED_ACTIVATION_CONFIG")
        folded_groups = records(log, "FOLDED_PREPARE_GROUP")
        if native_config and native_config[0].get("worker_local_cggm_cap") and backend == FOLDED_BACKEND:
            if (len(folded_groups) != queries or {row["query"] for row in folded_groups} != set(range(queries)) or
                    any(row.get("wall_seconds", -1) < 0 or row.get("schedule") != "serial_slot_major" or
                        row.get("kind_timers") != "work_attribution" for row in folded_groups)):
                raise ValueError("complete folded offline preparation wall is missing or misclassified")
        folded_verified = validate_folded_inventory(folded_plan, folded_prepared, folded_consumed,
            folded_balances, phases, queries, backend == FOLDED_BACKEND, log=log)
        folded_contract = validate_folded_config(folded_config, folded_plan, schedule, queries,
            backend == FOLDED_BACKEND, expected_selection=folded_activations, expected_family=lut_key_family,
            expected_sign_backend=folded_sign_backend, expected_sign_segments=folded_sign_segments)
        remaining_sci = records(log, "CONVERSION_REMAINING_SCI")
        if folded_verified:
            if not (native_verified and triple_verified and all(row.get("strict") is True for row in native_ready)):
                raise ValueError("folded activation requires complete strict native preprocessing")
            selected_layers = {row["layer"] for row in folded_plan}
            if any(row.get("index") in selected_layers for row in remaining_sci):
                raise ValueError("selected folded activation retained the native SCI body")
        parties.append(dict(finished[0], phases=phases,
                            folded_activation_plan=folded_plan,
                            folded_activation_prepared=folded_prepared,
                            folded_activation_consumed=folded_consumed,
                            folded_activation_balances=folded_balances,
                            folded_activation_config=folded_config,
                            folded_prepare_groups=folded_groups,
                            folded_public_table_preparation=records(log, "FOLDED_PUBLIC_TABLE_PREPARATION"),
                            folded_activation_contract=folded_contract,
                            folded_activation_verified=folded_verified,
                            conversion_remaining_sci=remaining_sci,
                            dpf_lut_plan=dpf_plan,dpf_lut_prepared=dpf_prepared,
                            dpf_lut_consumed=dpf_consumed,dpf_lut_balances=dpf_balances,
                            dpf_lut_verified=dpf_verified,
                            dpf_activation_config=records(log,"DPF_ACTIVATION_CONFIG"),
                            model_bias_preparation=model_bias_preparation,
                            native_material_ready=native_ready,
                            native_material_prepared=native_prepared,
                            native_material_consumed=native_consumed,
                            native_material_balances=native_balances,
                            native_preprocessing_verified=native_verified,
                            native_material_contract=native_contract,
                            native_public_plan=native_plan,
                            native_public_plan_summary=native_summary,
                            native_public_plan_verified=native_plan_contract is not None,
                            native_public_plan_contract=native_plan_contract,
                            native_planning_purity=native_purity,
                            native_planning_verified=native_purity_contract is not None,
                            native_triple_plan=triple_plan,
                            native_triple_prepared=triple_prepared,
                            native_triple_consumed=triple_consumed,
                            native_triple_balances=triple_balances,
                            native_triples_verified=triple_verified,
                            native_triple_contract=triple_contract,
                            conversion_calls=conversion_online,
                            conversion_offline_calls=conversion_offline,
                            conversion_preprocess_details=preprocessing,
                            dcf_generation=records(log, "DCF_GENERATION"),
                            conversion_groups=conversion_groups,
                            conversion_prefetch=prefetch,
                            conversion_prefetch_budgets=prefetch_budgets,
                            conversion_details=records(log, "CONVERSION_DETAIL"),
                            beaver_calls=records(log, "BEAVER_ONLINE"),
                            beaver_offline_calls=normalize_beaver_offline(records(log, "BEAVER_OFFLINE")),
                            offline_preparation=records(log, "OFFLINE_PREPARATION"),
                            public_scalar_calls=records(log, "PUBLIC_SCALARMUL"),
                            native_worker_config=native_config,
                            graph_tasks=graph_tasks,
                            graph_groups=graph_groups,
                            graph_opening_groups=opening_groups,
                            activations=activations,
                            activation_workers=activation_workers))
    if conversion_contracts[0] != conversion_contracts[1]:
        raise ValueError("conversion identities or numerical contracts differ between parties")
    bias_contracts = [{key: value for key, value in rows[0].items() if key != "cached_ring_bytes"}
                     if rows else None for rows in (party["model_bias_preparation"] for party in parties)]
    if bias_contracts[0] != bias_contracts[1]:
        raise ValueError("model bias preparation permissions differ between parties")
    if parties[0]["native_material_contract"] != parties[1]["native_material_contract"]:
        raise ValueError("native material contracts differ between parties")
    if parties[0]["native_triple_contract"] != parties[1]["native_triple_contract"]:
        raise ValueError("native bit-triple contracts differ between parties")
    if parties[0]["native_public_plan_contract"] != parties[1]["native_public_plan_contract"]:
        raise ValueError("native public OT plans differ between parties")
    if native_preprocessing is True and parties[0]["native_planning_purity"] != parties[1]["native_planning_purity"]:
        raise ValueError("native planning purity differs between parties")
    validate_beaver_traffic(parties)
    measurements = aggregate_phases(parties, queries)
    if backend == FOLDED_BACKEND:
        if parties[0]["folded_activation_contract"] != parties[1]["folded_activation_contract"]:
            raise ValueError("folded activation configurations differ between parties")
        measurements["folded_activations"] = aggregate_folded_activations(parties, queries)
    return parties, measurements


def run(args):
    args.activation_workers = getattr(args, "activation_workers", 1)
    args.aes_online = getattr(args, "aes_online", False)
    args.aes_offline = getattr(args, "aes_offline", False)
    args.graph_mode = getattr(args, "graph_mode", "sequential")
    args.graph_openings = getattr(args, "graph_openings", "independent")
    args.tail_parallel = getattr(args, "tail_parallel", False)
    args.pack_ring = getattr(args, "pack_ring", False)
    args.prune_equality = getattr(args, "prune_equality", False)
    args.prune_leaf_hashes = getattr(args, "prune_leaf_hashes", False)
    args.bit_block_mode = getattr(args, "bit_block_mode", "legacy")
    args.merge_leaf_cw = getattr(args, "merge_leaf_cw", False)
    args.reuse_generation_workspace = getattr(args, "reuse_generation_workspace", False)
    args.fuse_sign_wrap = getattr(args, "fuse_sign_wrap", False)
    args.native_ring_cot = getattr(args, "native_ring_cot", False)
    args.cache_lut_leaves = getattr(args, "cache_lut_leaves", False)
    args.folded_reuse_input_sign = getattr(args, "folded_reuse_input_sign", False)
    args.folded_clip_bits = getattr(args, "folded_clip_bits", 32)
    args.lut_skip_zero = getattr(args, "lut_skip_zero", False)
    args.lut_simd = getattr(args, "lut_simd", False)
    if args.bit_block_mode not in ("legacy", "block_ot", "direct_ot_legacy", "direct_ot", "direct_cot"):
        raise ValueError("invalid instantaneous bit-block product mode")
    if args.tail_parallel and args.graph_mode != "parallel":
        raise ValueError("tail parallel requires parallel graph contexts")
    args.offline_overlap = getattr(args, "offline_overlap", False)
    args.beaver_tile_rows = getattr(args, "beaver_tile_rows", 32)
    args.native_preprocessing = getattr(args, "native_preprocessing", False)
    args.round_trace = getattr(args, "round_trace", False)
    args.network_projection = getattr(args, "network_projection", "none")
    args.dpf_activations = getattr(args, "dpf_activations", "both")
    args.folded_activations = getattr(args, "folded_activations", "both")
    args.lut_key_family = getattr(args, "lut_key_family", "full_q_arithmetic")
    args.folded_sign_backend = getattr(args, "folded_sign_backend", "bitdec")
    args.folded_sign_segments = (2 if args.segment_policy == "fixed-2" else 4) if args.folded_sign_backend == "segmented_fss" else 0
    if args.backend != "SCI_Beaver_segmented_DPF_LUT" and args.dpf_activations != "both":
        raise ValueError("DPF activation ablations require the DPF-LUT backend")
    if args.backend != FOLDED_BACKEND and (args.folded_activations != "both" or
            args.lut_key_family != "full_q_arithmetic" or args.folded_sign_backend != "bitdec" or
            args.folded_reuse_input_sign or args.folded_clip_bits != 32 or
            args.lut_skip_zero or args.lut_simd):
        raise ValueError("folded activation and LUT family selectors require the PaperLUT backend")
    if args.folded_clip_bits not in (22, 32):
        raise ValueError("native folded clip comparison supports 22 or 32 bits")
    if args.folded_activations not in ("both", "sigmoid", "tanh") or args.lut_key_family not in (
            "full_q_arithmetic", "segmented_boolean_tensor_b2a") or args.folded_sign_backend not in ("bitdec", "segmented_fss"):
        raise ValueError("unsupported folded activation or LUT key family")
    args.model_bias_precompute = getattr(args, "model_bias_precompute", False)
    effective_native_preprocessing = native_preprocessing_mode(args.backend, args.native_preprocessing, args.steps)
    if args.offline_overlap and args.backend not in ("SCI_Beaver_full", "SCI_Beaver_segmented", "SCI_Beaver_segmented_DPF_LUT", FOLDED_BACKEND):
        raise ValueError("offline overlap requires Beaver and FSS conversion")
    if args.offline_overlap and not args.generation_budget_bytes:
        args.generation_budget_bytes = int(args.memory_gib * (1 << 30))
    if args.network_projection not in ("none", "lan", "wan", "both"):
        raise ValueError("unsupported network projection mode")
    affinities = party_affinities(args, os.sched_getaffinity(0))
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    (output / "tmp").mkdir()
    args.binary, args.fixture = args.binary.resolve(), args.fixture.resolve()
    description = json.loads(subprocess.check_output([str(args.binary), "--describe"], text=True))
    rtt, rate = NETWORKS[args.network]
    config = {"binary": str(args.binary), "binary_sha256": digest(args.binary), "backend": args.backend,
              "dpf_activations": args.dpf_activations if args.backend == "SCI_Beaver_segmented_DPF_LUT" else "none",
              "folded_activations": args.folded_activations if args.backend == FOLDED_BACKEND else "none",
              "activation_algorithm": "folded_sigma_f12_q10" if args.backend == FOLDED_BACKEND else "native_sci",
              "lut_key_family": args.lut_key_family if args.backend == FOLDED_BACKEND else None,
              "lut_segments": (1 if args.lut_key_family == "full_q_arithmetic" else 2)
                  if args.backend == FOLDED_BACKEND else None,
              "lut_segment_widths": ([10] if args.lut_key_family == "full_q_arithmetic" else [5, 5])
                  if args.backend == FOLDED_BACKEND else None,
              "folded_sign_backend": args.folded_sign_backend if args.backend == FOLDED_BACKEND else None,
              "folded_sign_segments": args.folded_sign_segments if args.backend == FOLDED_BACKEND else None,
              "native_preprocessing_requested": args.native_preprocessing,
              "native_preprocessing_effective": effective_native_preprocessing,
              "model_bias_precomputed": args.model_bias_precompute,
              "batch": args.batch, "steps": args.steps, "start": args.start,
              "queries": args.queries, "warmup_queries": args.warmup, "diagnostic": args.diagnostic,
              "fixture": str(args.fixture), "hidden_file": str(args.hidden) if args.hidden else None,
              "network": {"name": args.network, "rtt_ms": rtt, "mbps": rate,
                          "simulated": args.network != "localhost",
                          "description": "unshaped localhost" if rtt is None else "local application-payload relay; not original paper hardware"},
              "network_projection": {
                  "mode": args.network_projection,
                  "links": list(NETWORK_PROJECTION_OPTIONS.get(args.network_projection, {})),
                  "protocol_network": args.network,
                  "description": "deterministic bandwidth/RTT projection from qualified round trace"
                      if args.network_projection != "none" else "disabled",
              },
              "cpus": args.cpus, "application_threads_per_party": 1,
              "sci_protocol_threads_per_party": 1,
              "activation_workers_per_party": args.activation_workers,
              "conversion_optimizations": {"fuse_domains": args.fuse_domains,
                                           "pair_extensions": args.pair_extensions,
                                           "pack_bits": args.pack_bits},
              "graph_mode": args.graph_mode,
              "graph_openings": args.graph_openings,
              "tail_parallel": args.tail_parallel, "pack_ring": args.pack_ring,
              "prune_equality": args.prune_equality, "prune_leaf_hashes": args.prune_leaf_hashes,
              "bit_block_mode": args.bit_block_mode,
              "merge_leaf_cw": args.merge_leaf_cw,
              "reuse_generation_workspace": args.reuse_generation_workspace,
              "fuse_sign_wrap": args.fuse_sign_wrap,
              "native_ring_cot": args.native_ring_cot,
              "cache_lut_leaves": args.cache_lut_leaves,
              "folded_reuse_input_sign": args.folded_reuse_input_sign,
              "folded_clip_bits": args.folded_clip_bits,
              "lut_skip_zero": args.lut_skip_zero,
              "lut_simd": args.lut_simd,
              "graph_branches": 2 if args.graph_mode != "sequential" else 1,
              "branch_compute_limit": 2 if args.graph_mode != "sequential" else 4,
              "native_protocol_contexts_per_party": 4 if args.graph_mode != "sequential" else args.activation_workers,
              "native_eigen_threads": 1,
              "native_default_ot_participants": 2,
              "native_activation_ot_participants": 2 if args.activation_workers == 1 else 1,
              "native_ot_direction_schedule": "both_directions_static_loop",
              "channel_offsets": channel_offsets(args.activation_workers, args.graph_mode,
                  args.backend in ("SCI_Beaver_segmented_DPF_LUT", FOLDED_BACKEND)),
              "cggm_worker_limit_per_party": args.cggm_threads,
              "beaver_algebra": args.beaver_algebra,
              "beaver_mapped_layout": args.beaver_mapped_layout,
              "beaver_tile_rows": args.beaver_tile_rows, "offline_overlap": args.offline_overlap,
              "offline_task_count": 2 if args.offline_overlap else 1,
              "offline_task_compute_limit": 2 if args.offline_overlap else max(2, args.cggm_threads),
              "requested_cpu_affinity_per_party": affinities,
              "cpu_affinity_per_party": None,
              "memory_limit_bytes_per_party": int(args.memory_gib * (1 << 30)), "timeout_seconds": args.timeout,
              "max_segments": args.max_segments, "machine": platform.uname()._asdict(),
              "material_layout": args.material_layout, "prefetch_bytes": args.prefetch_bytes,
              "segment_policy": args.segment_policy, "generation_tile_keys": args.generation_tile_keys,
              "generation_budget_bytes": args.generation_budget_bytes,
              "generation_reserve_bytes": args.generation_reserve_bytes,
              "prefetch_mode": "prepared_prefix" if args.prefetch_bytes else "off",
              "aes_online": args.aes_online, "aes_offline": args.aes_offline,
              "cpu_model": next((line.split(":", 1)[1].strip() for line in Path("/proc/cpuinfo").read_text().splitlines() if line.startswith("model name")), None),
              "full_test_set_evaluation": False, "quality_validation": "SDCF-09 deferred by user",
              "setup_amortization": "one persistent setup; fresh one-use material per query",
              "measurement_boundaries": {
                  "model_prepare": "owner-only exact model constants; rebuilt per immutable model process, separately metered",
                  "offline": "fresh material generation and material-ready work before input",
                  "online": "after input/ready through all protocol work, loads, waits, and releases",
                  "query_total": "direct wall measurement from fresh preprocessing through output",
                  "session_total": "direct wall measurement including application I/O, setup, all queries, teardown",
                  "conversion_detail": "nested subcall intervals; absent fields are uninstrumented, not zero",
                  "parallel_work": "child elapsed intervals may overlap; never sum them as query wall time"},
              "backend_execution": "sequential separate invocations", "binary_description": description,
              "round_trace": args.round_trace,
              "round_convention": "two-party exchange; causal send-flight clocks" if args.round_trace else None,
              "output_policy": "client receives final hidden state for steps=1; prediction index for steps=99"}
    write_json(output / "config.json", config)
    result = {"config": config, "status": "unavailable", "reason": None, "measurements": None, "party_exit_codes": []}
    if args.backend not in description.get("backends", []):
        result["reason"] = "compiled binary does not implement requested backend; no fallback"
        write_json(output / "result.json", result)
        return result
    if args.model_bias_precompute and description.get("supports_model_bias_precomputation") is not True:
        result["reason"] = "compiled binary does not implement exact model bias preparation"
        write_json(output / "result.json", result)
        return result
    if args.round_trace and description.get("round_trace_api") != 1:
        result["reason"] = "compiled binary does not implement complete passive round tracing"
        write_json(output / "result.json", result)
        return result
    if effective_native_preprocessing and (description.get("supports_native_preprocessing") is not True or
            args.steps not in description.get("native_preprocessing_steps", [])):
        result["reason"] = "compiled binary does not implement requested native preprocessing"
        write_json(output / "result.json", result)
        return result
    if args.activation_workers != 1 and args.activation_workers not in description.get("activation_workers", []):
        result["reason"] = "compiled binary does not implement requested activation workers"
        write_json(output / "result.json", result)
        return result
    if args.graph_mode != "sequential" and args.graph_mode not in description.get("graph_modes", []):
        result["reason"] = "compiled binary does not implement requested graph mode"
        write_json(output / "result.json", result)
        return result
    if args.graph_mode != "sequential" and args.steps != 1:
        result["reason"] = "isolated graph modes support only the frozen single-step entry"
        write_json(output / "result.json", result)
        return result
    if args.graph_openings != "independent" and (args.steps != 1 or
            args.graph_openings not in description.get("graph_openings", [])):
        result["reason"] = "compiled binary does not implement batched single-step graph openings"
        write_json(output / "result.json", result)
        return result
    if (args.aes_online or args.aes_offline) and not description.get("batched_aes", False):
        result["reason"] = "compiled binary does not implement requested batched AES"
        write_json(output / "result.json", result)
        return result
    for name in ("server_model.i16", "client_input.i16"):
        path = args.fixture / name
        if not path.is_file():
            raise ValueError("fixture lacks " + name)
    if (args.fixture / "client_input.i16").stat().st_size < (args.start + args.batch) * 3168 * 2:
        raise ValueError("client fixture is smaller than the selected batch")
    schema = json.loads((args.binary.parent / "schema.json").read_text())
    schedule = dependency_schedule(schema, args.batch, args.steps, classifier=args.steps == 99,
                                   model_bias_precomputed=args.model_bias_precompute)
    write_json(output / "schedule.json", schedule)
    config["fixture_files"] = {name: digest(args.fixture / name) for name in ("server_model.i16", "client_input.i16")}
    if args.hidden:
        config["hidden_sha256"] = digest(args.hidden)
    manifest = args.binary.parent / "runtime_manifest.json"
    config["runtime_manifest"] = json.loads(manifest.read_text())
    contract = json.loads((args.fixture / "contract.json").read_text())
    validate_fixture(contract, config["runtime_manifest"], config["fixture_files"])
    config["fixture_identity"] = contract.get("identity")
    config["sample_ids"] = contract.get("sample_ids", [])[args.start:args.start + args.batch]
    write_json(output / "config.json", config)
    connector = None
    if rtt is not None:
        connector = output / "connect_offset.so"
        command = ["gcc", "-shared", "-fPIC", "-O2", str(HERE / "connect_offset.c"), "-o", str(connector), "-ldl"]
        child = subprocess.run(command, capture_output=True, text=True)
        (output / "connector_build.log").write_text(child.stdout + child.stderr)
        child.check_returncode()
    channels = channel_offsets(args.activation_workers, args.graph_mode,
        args.backend in ("SCI_Beaver_segmented_DPF_LUT", FOLDED_BACKEND))
    base = reserve_ports(connector is not None, channels)
    children, commands, environments = [], [], []
    status, reason = "complete", None
    start = time.monotonic()

    def limits(affinity):
        cap = config["memory_limit_bytes_per_party"]
        resource.setrlimit(resource.RLIMIT_AS, (cap, cap))
        os.sched_setaffinity(0, affinity)

    observed_affinities = []
    resource_peaks = [{"vm_peak_bytes": 0, "vm_size_bytes": 0, "rss_peak_bytes": 0, "threads_peak": 0} for _ in (1,2)]
    try:
        with ExitStack() as stack:
            relay = None
            if connector:
                mapping = ",".join("%d:%d" % (base + 1000 + offset, base + offset) for offset in channels)
                relay_command = ["taskset", "-c", str(args.cpus[2]), sys.executable,
                                 str(HERE.parent / "google30_benchmark/link_proxy.py"),
                                 "--ports", mapping, "--rtt-ms", str(rtt), "--mbps", str(rate)]
                relay_log = stack.enter_context((output / "relay.log").open("w"))
                relay = subprocess.Popen(relay_command, stdout=subprocess.PIPE, stderr=relay_log, text=True)
                stack.callback(stop, relay)
                with selectors.DefaultSelector() as selector:
                    selector.register(relay.stdout, selectors.EVENT_READ)
                    if not selector.select(10) or relay.stdout.readline().strip() != "READY":
                        raise RuntimeError("relay readiness failed")
                relay.stdout.close()
                write_json(output / "relay_command.json", relay_command)
            for role in (1, 2):
                affinity = affinities[role - 1]
                owned = args.fixture / ("server_model.i16" if role == 1 else "client_input.i16")
                command = ["taskset", "-c", ",".join(map(str, affinity)), str(args.binary), str(role), str(base), str(owned),
                           str(args.start), str(args.batch), str(args.steps), "diagnostic" if args.diagnostic else "secure",
                           str(args.queries + args.warmup), args.backend, str(args.hidden.resolve()) if args.hidden else "none"]
                updates = {"OMP_NUM_THREADS": "1", "SDCF_BACKEND": args.backend,
                           "SDCF_DPF_SIGMOID": "1" if args.backend == "SCI_Beaver_segmented_DPF_LUT" and args.dpf_activations in ("sigmoid","both") else "0",
                           "SDCF_DPF_TANH": "1" if args.backend == "SCI_Beaver_segmented_DPF_LUT" and args.dpf_activations in ("tanh","both") else "0",
                           "SDCF_FOLDED_SIGMOID": "1" if args.backend == FOLDED_BACKEND and args.folded_activations in ("sigmoid", "both") else "0",
                           "SDCF_FOLDED_TANH": "1" if args.backend == FOLDED_BACKEND and args.folded_activations in ("tanh", "both") else "0",
                           "SDCF_FOLDED_LUT_FAMILY": "1" if args.lut_key_family == "segmented_boolean_tensor_b2a" else "0",
                           "SDCF_FOLDED_SIGN_BACKEND": "1" if args.folded_sign_backend == "segmented_fss" else "0",
                           "SDCF_FOLDED_SIGN_SEGMENTS": str(args.folded_sign_segments),
                           "SDCF_MODEL_BIAS_PRECOMPUTE": "1" if args.model_bias_precompute else "0",
                           "SDCF_NATIVE_PREPROCESSING": "1" if effective_native_preprocessing else "0",
                           "SEGFSS_FUSE_DOMAINS": "1" if args.fuse_domains else "0",
                           "SEGFSS_PACK_BITS": "1" if args.pack_bits else "0",
                           "SDCF_PAIR_EXTENSIONS": "1" if args.pair_extensions else "0",
                           "SEGFSS_MATERIAL_LAYOUT": args.material_layout,
                           "SEGFSS_PREFETCH_BYTES": str(args.prefetch_bytes),
                           "SEGFSS_AES_ONLINE": "1" if args.aes_online else "0",
                           "SEGFSS_AES_OFFLINE": "1" if args.aes_offline else "0",
                           "SEGFSS_SEGMENT_POLICY": args.segment_policy,
                           "SEGFSS_GENERATION_TILE_KEYS": str(args.generation_tile_keys),
                           "SEGFSS_GENERATION_BUDGET_BYTES": str(args.generation_budget_bytes),
                           "SEGFSS_GENERATION_RESERVE_BYTES": str(args.generation_reserve_bytes),
                           "OMP_DYNAMIC": "FALSE", "OMP_MAX_ACTIVE_LEVELS": "1",
                           "SDCF_ACTIVATION_WORKERS": str(args.activation_workers),
                           "SDCF_GRAPH_MODE": args.graph_mode,
                           "SDCF_GRAPH_OPENINGS": args.graph_openings,
                           "SDCF_TAIL_PARALLEL": str(int(args.tail_parallel)),
                           "SEGFSS_PACK_RING": str(int(args.pack_ring)),
                           "SEGFSS_PRUNE_EQUALITY": str(int(args.prune_equality)),
                           "SEGFSS_PRUNE_LEAF_HASHES": str(int(args.prune_leaf_hashes)),
                           "SEGFSS_BIT_BLOCK_MODE": args.bit_block_mode,
                           "SEGFSS_MERGE_LEAF_CW": str(int(args.merge_leaf_cw)),
                           "SEGFSS_REUSE_GENERATION_WORKSPACE": str(int(args.reuse_generation_workspace)),
                           "SEGFSS_FUSE_SIGN_WRAP": str(int(args.fuse_sign_wrap)),
                           "SEGFSS_NATIVE_RING_COT": str(int(args.native_ring_cot)),
                           "SEGFSS_CACHE_LUT_LEAVES": str(int(args.cache_lut_leaves)),
                           "SEGFSS_FOLDED_REUSE_SIGN": str(int(args.folded_reuse_input_sign)),
                           "SEGFSS_FOLDED_CLIP_BITS": str(args.folded_clip_bits),
                           "SEGFSS_LUT_SKIP_ZERO": str(int(args.lut_skip_zero)),
                           "SEGFSS_LUT_SIMD": str(int(args.lut_simd)),
                           "SEGFSS_CGGM_THREADS": str(args.cggm_threads),
                           "SDCF_BEAVER_ALGEBRA": str(int(args.beaver_algebra)),
                           "SDCF_BEAVER_MAPPED_LAYOUT": str(int(args.beaver_mapped_layout)),
                           "SDCF_BEAVER_TILE_ROWS": str(args.beaver_tile_rows),
                           "SDCF_OFFLINE_OVERLAP": str(int(args.offline_overlap)),
                           "SDCF_PORT": str(base + 20), "SDCF_HOST": "127.0.0.1",
                           "SDCF_MAX_SEGMENTS": str(args.max_segments), "TMPDIR": str(output / "tmp")}
                if args.round_trace:
                    updates["SDCF_ROUND_TRACE"] = str(output / ("party%d.roundtrace.jsonl" % role))
                if connector:
                    updates.update(SWEEP_CONNECT_OFFSET="1000", LD_PRELOAD=str(connector))
                env = child_environment(os.environ, updates)
                log = stack.enter_context((output / ("party%d.log" % role)).open("w"))
                child = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT, env=env,
                                         preexec_fn=lambda: limits(affinity), cwd=output)
                children.append(child); commands.append(command); environments.append(updates)
                stack.callback(stop, child)
                observed_affinities.append(sorted(os.sched_getaffinity(child.pid)))
                if observed_affinities[-1] != sorted(affinity):
                    raise RuntimeError("party CPU affinity differs from requested configuration")
            config["cpu_affinity_per_party"] = observed_affinities
            write_json(output / "config.json", config)
            write_json(output / "commands.json", {"commands": commands, "environment": environments})
            while any(child.poll() is None for child in children):
                for index, child in enumerate(children):
                    sample = process_resources(child.pid)
                    for field, value in sample.items():
                        resource_peaks[index][field] = max(resource_peaks[index][field], value)
                if any(child.poll() not in (None, 0) for child in children):
                    status, reason = "failed", "party exited unsuccessfully"
                    break
                if relay and relay.poll() is not None:
                    status, reason = "failed", "relay exited before parties"
                    break
                if time.monotonic() - start >= args.timeout:
                    status, reason = "timeout", "predeclared wall-time budget exceeded"
                    break
                time.sleep(0.05)
    except Exception as error:
        status, reason = "failed", str(error)
    config["cpu_affinity_per_party"] = observed_affinities or None
    write_json(output / "config.json", config)
    result["process_resource_peaks"] = resource_peaks
    result["process_resource_sampling_ms"] = 50
    result["address_space_hard_limit_bytes_per_party"] = config["memory_limit_bytes_per_party"]
    exits = [child.returncode for child in children]
    logs = [(output / ("party%d.log" % role)).read_text() if (output / ("party%d.log" % role)).is_file() else "" for role in (1, 2)]
    if status == "complete" and exits != [0, 0]:
        status, reason = "failed", "both parties must exit zero"
    if status == "failed" and any("bad_alloc" in log or "Cannot allocate memory" in log for log in logs):
        status, reason = "oom", "allocation failure under declared address-space cap"
    if status == "complete":
        try:
            parties, measurements = collect(logs, args.queries + args.warmup, schedule, args.diagnostic,
                args.backend, native_preprocessing=effective_native_preprocessing,
                folded_activations=args.folded_activations if args.backend == FOLDED_BACKEND else None,
                lut_key_family=args.lut_key_family if args.backend == FOLDED_BACKEND else None,
                folded_sign_backend=args.folded_sign_backend if args.backend == FOLDED_BACKEND else None,
                folded_sign_segments=args.folded_sign_segments if args.backend == FOLDED_BACKEND else None)
            if args.round_trace:
                from round_trace import analyze_pair
                rounds = analyze_pair(output / "party1.roundtrace.jsonl", output / "party2.roundtrace.jsonl",
                                      expected_phases=parties)
                write_json(output / "rounds.json", rounds)
                if rounds.get("status") != "passed":
                    raise ValueError("communication trace qualification: " + str(rounds.get("reason")))
                validate_scope_measurements(rounds, parties)
                result["round_measurements"] = rounds
            else:
                rounds = None
            attach_round_measurements(measurements, rounds)
            measurements["warmup_queries"] = measurements["queries"][:args.warmup]
            measurements["queries"] = measurements["queries"][args.warmup:]
            if "folded_activations" in measurements:
                measurements["folded_activation_warmup_queries"] = measurements["folded_activations"][:args.warmup]
                measurements["folded_activations"] = measurements["folded_activations"][args.warmup:]
            summarize_queries(measurements)
            if args.network_projection == "none":
                network_projection = project_network_costs(
                    rounds, NETWORKS[args.network], query_start=args.warmup, query_count=args.queries)
            else:
                network_projection = project_network_costs_set(
                    rounds, args.network_projection, query_start=args.warmup, query_count=args.queries)
            measurements["network_projection"] = network_projection
            attach_network_measurements(measurements, network_projection)
            result["measurements"] = measurements
            write_json(output / "parties.json", parties)
        except Exception as error:
            status, reason = "failed", "result validation: " + str(error)
    result.update(status=status, reason=reason, party_exit_codes=exits,
                  observed_process_wall_seconds=time.monotonic() - start)
    write_json(output / "result.json", result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--binary", type=Path, required=True)
    parser.add_argument("--fixture", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--backend", choices=BACKENDS, default="SCI")
    parser.add_argument("--dpf-activations", choices=("sigmoid", "tanh", "both"), default="both",
                        help="DPF-LUT activation selection; single choices are explicit ablations")
    parser.add_argument("--folded-activations", choices=("sigmoid", "tanh", "both"), default="both",
                        help="complete paper activation selection; single choices are explicit ablations")
    parser.add_argument("--lut-key-family", choices=("full_q_arithmetic", "segmented_boolean_tensor_b2a"),
                        default="full_q_arithmetic", help="resident 10-bit selector construction for PaperLUT")
    parser.add_argument("--folded-sign-backend", choices=("bitdec", "segmented_fss"), default="bitdec",
                        help="complete folded sign and clip comparisons; FSS uses the public outer segment policy")
    parser.add_argument("--model-bias-precompute", action="store_true",
                        help="prepare owner-held Bg/Bh exact signed /8 once before query material generation; same permission for all backends")
    parser.add_argument("--native-preprocessing", action="store_true",
                        help="prepare remaining native random OT and bit-triples before query input")
    parser.add_argument("--round-trace", action="store_true",
                        help="trace all communication and qualify causal two-party exchange rounds")
    parser.add_argument("--batch", type=int, default=128)
    parser.add_argument("--steps", type=int, choices=(1, 99), default=99)
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--queries", type=int, default=3)
    parser.add_argument("--warmup", type=int, default=0)
    parser.add_argument("--network", choices=NETWORKS, default="localhost")
    parser.add_argument("--network-projection", choices=("none", "lan", "wan", "both"), default="none",
                        help="write deterministic LAN/WAN projections from the qualified round trace; does not shape localhost")
    parser.add_argument("--cpus", default="10,12,14")
    parser.add_argument("--cggm-threads", type=int, default=1,
                        help="local cGGM worker limit per party; SCI protocol threads stay at one")
    parser.add_argument("--activation-workers", type=int, choices=(1, 2, 3, 4), default=1,
                        help="resident independent native SCI activation contexts")
    parser.add_argument("--graph-mode", choices=("sequential", "isolated", "parallel"), default="sequential")
    parser.add_argument("--tail-parallel", action="store_true")
    parser.add_argument("--pack-ring", action="store_true")
    parser.add_argument("--bit-block-mode", choices=("legacy", "block_ot", "direct_ot_legacy", "direct_ot", "direct_cot"), default="legacy",
                        help="independent block-OT and direct-product ablations for instantaneous FSS products")
    parser.add_argument("--merge-leaf-cw", action="store_true",
                        help="open independent leaf high/low correction words in one byte-exact exchange")
    parser.add_argument("--reuse-generation-workspace", action="store_true",
                        help="retain bounded tree/hash capacity across FSS generation tiles before input")
    parser.add_argument("--graph-openings", choices=("independent", "batched"), default="independent")
    parser.add_argument("--server-cpus", help="explicit comma-separated CPU IDs for the server")
    parser.add_argument("--client-cpus", help="explicit comma-separated CPU IDs for the client")
    parser.add_argument("--timeout", type=float, default=3600)
    parser.add_argument("--memory-gib", type=float, default=1.5)
    parser.add_argument("--max-segments", type=int, default=4)
    parser.add_argument("--beaver-algebra", action="store_true",
                        help="merge the server's distributive local matrix products")
    parser.add_argument("--beaver-mapped-layout", action="store_true",
                        help="use single-participant row-major integer Eigen maps")
    parser.add_argument("--beaver-tile-rows", type=int, choices=(1, 8, 32, 128), default=32)
    parser.add_argument("--offline-overlap", action="store_true")
    parser.add_argument("--fuse-domains", action="store_true")
    parser.add_argument("--pair-extensions", action="store_true")
    parser.add_argument("--pack-bits", action="store_true")
    parser.add_argument("--prune-equality", action="store_true")
    parser.add_argument("--fuse-sign-wrap", action="store_true",
                        help="derive signed wrap correction from the shared lower-bit comparison")
    parser.add_argument("--native-ring-cot", action="store_true",
                        help="use arithmetic correlated OT for dFSS ring correlations")
    parser.add_argument("--cache-lut-leaves", action="store_true",
                        help="reuse full arithmetic LUT leaves within budgeted generation tiles")
    parser.add_argument("--folded-reuse-input-sign", action="store_true",
                        help="reuse the native signed16 input extension's secret sign share")
    parser.add_argument("--folded-clip-bits", type=int, choices=(22, 32), default=32,
                        help="comparison width for native folded clipping; arithmetic stays ring32")
    parser.add_argument("--lut-skip-zero", action="store_true",
                        help="skip publicly zero coefficients in the rotated LUT inner product")
    parser.add_argument("--lut-simd", action="store_true",
                        help="use runtime-dispatched AVX2 for the ring32 LUT inner product")
    parser.add_argument("--prune-leaf-hashes", action="store_true")
    parser.add_argument("--material-layout", choices=("legacy", "contiguous"), default="legacy")
    parser.add_argument("--segment-policy", choices=("legacy", "fixed-2", "fixed-4", "depth-cap-8"), default="legacy")
    parser.add_argument("--generation-tile-keys", type=int, default=0)
    parser.add_argument("--generation-budget-bytes", type=int, default=0)
    parser.add_argument("--generation-reserve-bytes", type=int, default=512 << 20)
    parser.add_argument("--prefetch-bytes", type=int, default=0,
                        help="bounded same-query material prefix; charged to offline")
    parser.add_argument("--aes-online", action="store_true")
    parser.add_argument("--aes-offline", action="store_true")
    parser.add_argument("--diagnostic", action="store_true")
    parser.add_argument("--hidden", type=Path)
    args = parser.parse_args()
    try:
        args.cpus = [int(value) for value in args.cpus.split(",")]
        if min(args.generation_tile_keys, args.generation_budget_bytes, args.generation_reserve_bytes) < 0:
            raise ValueError("generation limits must be nonnegative")
        if args.generation_budget_bytes > int(args.memory_gib * (1 << 30)):
            raise ValueError("generation budget must fit the declared process budget")
        if not math.isfinite(args.memory_gib) or args.memory_gib <= 0 or args.prefetch_bytes < 0 or args.prefetch_bytes > int(args.memory_gib * (1 << 30)):
            raise ValueError("prefetch bytes must fit the declared process budget")
        if args.prefetch_bytes and args.material_layout != "contiguous":
            raise ValueError("prefetch requires contiguous material layout")
        if len(args.cpus) != 3 or not set(args.cpus) <= os.sched_getaffinity(0):
            raise ValueError("cpus must specify three available CPUs")
        if not 1 <= args.batch <= 256 or args.queries < 1 or args.warmup < 0 or args.start < 0 or not math.isfinite(args.timeout) or args.timeout <= 0 or not math.isfinite(args.memory_gib) or args.memory_gib <= 0 or not 1 <= args.max_segments <= 64:
            raise ValueError("invalid batch, query or resource budget")
        result = run(args)
    except (ValueError, OSError, subprocess.SubprocessError) as error:
        parser.error(str(error))
    print(json.dumps({"status": result["status"], "reason": result["reason"],
                      "result": str(args.output.resolve() / "result.json")}))
    return 0 if result["status"] == "complete" else 3 if result["status"] == "unavailable" else 1


if __name__ == "__main__":
    sys.exit(main())
