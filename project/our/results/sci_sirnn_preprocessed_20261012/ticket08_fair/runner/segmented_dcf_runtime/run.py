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

HERE = Path(__file__).resolve().parent
CHANNEL_OFFSETS = (0, 3, 20, 23, 50, 70, 100, 120, 200)
NETWORKS = {"localhost": (None, None), "sirnn_lan": (0.8, 3016), "wan": (2.0, 100)}


def channel_offsets(activation_workers, graph_mode="sequential"):
    contexts = 4 if graph_mode != "sequential" else activation_workers
    return tuple(sorted(set(CHANNEL_OFFSETS) | ({820, 823, 870, 920} if graph_mode != "sequential" else set()) | {
        300 + 150 * index + offset
        for index in range(contexts - 1) for offset in (0, 50, 100)}))


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


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
    effective = backend == "SCI_preprocessed" or requested
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


def validate_native_materials(ready, prepared, consumed, balances, phases, queries):
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


def validate_native_public_plan(planned, consumed, queries):
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
        if offset != elements or detail["count"] != elements:
            raise ValueError("activation workers did not cover the complete vector")
        if not 1 <= detail["compute_participants"] <= 4:
            raise ValueError("activation exceeded the four-participant allocation")


def validate_native_triples(planned, prepared, consumed, balances, ready, queries):
    if not (planned or prepared or consumed or balances):
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
    enabled = backend in ("SCI_ours", "SCI_Beaver_full", "SCI_Beaver_segmented")
    if backend is None:
        enabled = bool(offline or online)
    if enabled and (not offline or not online):
        raise ValueError("enabled conversion backend has missing offline/online evidence")
    if not enabled and (offline or online or preprocessing or groups):
        raise ValueError("native conversion backend has unexpected FSS evidence")
    required = ("count", "input_bits", "output_bits", "shift", "call", "numeric_mode", "comparison_backend")
    fields = required + ("signed_arithmetic", "low_comparison_bits", "sign_comparison_bits",
                         "wrap_comparison_bits", "low_nseg", "sign_nseg", "wrap_nseg",
                         "low_widths", "sign_widths", "wrap_widths", "segment_policy", "sample_groups")

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


def collect(logs, queries, schedule, diagnostic, backend=None, native_preprocessing=None):
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
            expected_groups = {(query,index) for query in range(queries) for index in (0,5)}
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
        native_ready = records(log, "MATERIAL_READY")
        if native_preprocessing is not None and (
                len(native_ready) != queries or
                {row.get("query") for row in native_ready} != set(range(queries)) or
                any(row.get("strict") is not native_preprocessing for row in native_ready)):
            raise ValueError("requested native preprocessing differs from material-ready evidence")
        native_prepared = records(log, "NATIVE_MATERIAL_PREPARED")
        native_consumed = records(log, "NATIVE_MATERIAL_CONSUMED")
        native_balances = records(log, "NATIVE_MATERIAL_BALANCE")
        native_verified, native_contract = validate_native_materials(native_ready,
            native_prepared, native_consumed, native_balances, phases, queries)
        native_plan = records(log, "NATIVE_PUBLIC_PLAN")
        native_plan_contract = (validate_native_public_plan(native_plan, native_consumed, queries)
                                if native_preprocessing is True else None)
        native_purity = records(log, "NATIVE_PLAN_PURITY")
        native_purity_contract = (validate_native_planning_purity(log, native_purity, queries)
                                  if native_preprocessing is True else None)
        triple_plan = records(log, "NATIVE_PUBLIC_TRIPLE_PLAN")
        triple_prepared = records(log, "NATIVE_TRIPLE_PREPARED")
        triple_consumed = records(log, "NATIVE_TRIPLE_CONSUMED")
        triple_balances = records(log, "NATIVE_TRIPLE_BALANCE")
        triple_verified, triple_contract = validate_native_triples(triple_plan, triple_prepared,
            triple_consumed, triple_balances, native_ready, queries)
        if native_preprocessing is True:
            if not (native_verified and triple_verified):
                raise ValueError("complete native preprocessing requires verified OT and bit-triple inventories")
            for query in range(queries):
                for phase in ("input", "ready", "online", "output"):
                    rows = [row for row in phases if row.get("query") == query and row.get("phase") == phase]
                    if (len(rows) != 1 or type(rows[0].get("bridge_generation_epoch_delta")) is not int or
                            rows[0]["bridge_generation_epoch_delta"] != 0):
                        raise ValueError("bridge material generated or unmetered after prepare")
        parties.append(dict(finished[0], phases=phases,
                            native_material_ready=native_ready,
                            native_material_prepared=native_prepared,
                            native_material_consumed=native_consumed,
                            native_material_balances=native_balances,
                            native_preprocessing_verified=native_verified,
                            native_material_contract=native_contract,
                            native_public_plan=native_plan,
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
    if parties[0]["native_material_contract"] != parties[1]["native_material_contract"]:
        raise ValueError("native material contracts differ between parties")
    if parties[0]["native_triple_contract"] != parties[1]["native_triple_contract"]:
        raise ValueError("native bit-triple contracts differ between parties")
    if parties[0]["native_public_plan_contract"] != parties[1]["native_public_plan_contract"]:
        raise ValueError("native public OT plans differ between parties")
    if native_preprocessing is True and parties[0]["native_planning_purity"] != parties[1]["native_planning_purity"]:
        raise ValueError("native planning purity differs between parties")
    validate_beaver_traffic(parties)
    return parties, aggregate_phases(parties, queries)


def run(args):
    args.activation_workers = getattr(args, "activation_workers", 1)
    args.aes_online = getattr(args, "aes_online", False)
    args.aes_offline = getattr(args, "aes_offline", False)
    args.graph_mode = getattr(args, "graph_mode", "sequential")
    args.graph_openings = getattr(args, "graph_openings", "independent")
    args.offline_overlap = getattr(args, "offline_overlap", False)
    args.beaver_tile_rows = getattr(args, "beaver_tile_rows", 32)
    args.native_preprocessing = getattr(args, "native_preprocessing", False)
    effective_native_preprocessing = native_preprocessing_mode(args.backend, args.native_preprocessing, args.steps)
    if args.offline_overlap and args.backend not in ("SCI_Beaver_full", "SCI_Beaver_segmented"):
        raise ValueError("offline overlap requires Beaver and FSS conversion")
    if args.offline_overlap and not args.generation_budget_bytes:
        args.generation_budget_bytes = int(args.memory_gib * (1 << 30))
    affinities = party_affinities(args, os.sched_getaffinity(0))
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    (output / "tmp").mkdir()
    args.binary, args.fixture = args.binary.resolve(), args.fixture.resolve()
    description = json.loads(subprocess.check_output([str(args.binary), "--describe"], text=True))
    rtt, rate = NETWORKS[args.network]
    config = {"binary": str(args.binary), "binary_sha256": digest(args.binary), "backend": args.backend,
              "native_preprocessing_requested": args.native_preprocessing,
              "native_preprocessing_effective": effective_native_preprocessing,
              "batch": args.batch, "steps": args.steps, "start": args.start,
              "queries": args.queries, "warmup_queries": args.warmup, "diagnostic": args.diagnostic,
              "fixture": str(args.fixture), "hidden_file": str(args.hidden) if args.hidden else None,
              "network": {"name": args.network, "rtt_ms": rtt, "mbps": rate,
                          "simulated": args.network != "localhost",
                          "description": "unshaped localhost" if rtt is None else "local application-payload relay; not original paper hardware"},
              "cpus": args.cpus, "application_threads_per_party": 1,
              "sci_protocol_threads_per_party": 1,
              "activation_workers_per_party": args.activation_workers,
              "conversion_optimizations": {"fuse_domains": args.fuse_domains,
                                           "pair_extensions": args.pair_extensions,
                                           "pack_bits": args.pack_bits},
              "graph_mode": args.graph_mode,
              "graph_openings": args.graph_openings,
              "graph_branches": 2 if args.graph_mode != "sequential" else 1,
              "branch_compute_limit": 2 if args.graph_mode != "sequential" else 4,
              "native_protocol_contexts_per_party": 4 if args.graph_mode != "sequential" else args.activation_workers,
              "native_eigen_threads": 1,
              "native_default_ot_participants": 2,
              "native_activation_ot_participants": 2 if args.activation_workers == 1 else 1,
              "native_ot_direction_schedule": "both_directions_static_loop",
              "channel_offsets": channel_offsets(args.activation_workers, args.graph_mode),
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
                  "offline": "fresh material generation and material-ready work before input",
                  "online": "after input/ready through all protocol work, loads, waits, and releases",
                  "query_total": "direct wall measurement from fresh preprocessing through output",
                  "session_total": "direct wall measurement including application I/O, setup, all queries, teardown",
                  "conversion_detail": "nested subcall intervals; absent fields are uninstrumented, not zero",
                  "parallel_work": "child elapsed intervals may overlap; never sum them as query wall time"},
              "backend_execution": "sequential separate invocations", "binary_description": description,
              "output_policy": "client receives final hidden state for steps=1; prediction index for steps=99"}
    write_json(output / "config.json", config)
    result = {"config": config, "status": "unavailable", "reason": None, "measurements": None, "party_exit_codes": []}
    if args.backend not in description.get("backends", []):
        result["reason"] = "compiled binary does not implement requested backend; no fallback"
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
    schedule = dependency_schedule(schema, args.batch, args.steps, classifier=args.steps == 99)
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
    channels = channel_offsets(args.activation_workers, args.graph_mode)
    base = reserve_ports(connector is not None, channels)
    children, commands, environments = [], [], []
    status, reason = "complete", None
    start = time.monotonic()

    def limits(affinity):
        cap = config["memory_limit_bytes_per_party"]
        resource.setrlimit(resource.RLIMIT_AS, (cap, cap))
        os.sched_setaffinity(0, affinity)

    observed_affinities = []
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
                           "SEGFSS_CGGM_THREADS": str(args.cggm_threads),
                           "SDCF_BEAVER_ALGEBRA": str(int(args.beaver_algebra)),
                           "SDCF_BEAVER_MAPPED_LAYOUT": str(int(args.beaver_mapped_layout)),
                           "SDCF_BEAVER_TILE_ROWS": str(args.beaver_tile_rows),
                           "SDCF_OFFLINE_OVERLAP": str(int(args.offline_overlap)),
                           "SDCF_PORT": str(base + 20), "SDCF_HOST": "127.0.0.1",
                           "SDCF_MAX_SEGMENTS": str(args.max_segments), "TMPDIR": str(output / "tmp")}
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
    exits = [child.returncode for child in children]
    logs = [(output / ("party%d.log" % role)).read_text() if (output / ("party%d.log" % role)).is_file() else "" for role in (1, 2)]
    if status == "complete" and exits != [0, 0]:
        status, reason = "failed", "both parties must exit zero"
    if status == "failed" and any("bad_alloc" in log or "Cannot allocate memory" in log for log in logs):
        status, reason = "oom", "allocation failure under declared address-space cap"
    if status == "complete":
        try:
            parties, measurements = collect(logs, args.queries + args.warmup, schedule, args.diagnostic,
                                             args.backend, native_preprocessing=effective_native_preprocessing)
            measurements["warmup_queries"] = measurements["queries"][:args.warmup]
            measurements["queries"] = measurements["queries"][args.warmup:]
            summarize_queries(measurements)
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
    parser.add_argument("--native-preprocessing", action="store_true",
                        help="prepare remaining native random OT and bit-triples before query input")
    parser.add_argument("--batch", type=int, default=128)
    parser.add_argument("--steps", type=int, choices=(1, 99), default=99)
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--queries", type=int, default=3)
    parser.add_argument("--warmup", type=int, default=0)
    parser.add_argument("--network", choices=NETWORKS, default="localhost")
    parser.add_argument("--cpus", default="10,12,14")
    parser.add_argument("--cggm-threads", type=int, default=1,
                        help="local cGGM worker limit per party; SCI protocol threads stay at one")
    parser.add_argument("--activation-workers", type=int, choices=(1, 2, 3, 4), default=1,
                        help="resident independent native SCI activation contexts")
    parser.add_argument("--graph-mode", choices=("sequential", "isolated", "parallel"), default="sequential")
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
