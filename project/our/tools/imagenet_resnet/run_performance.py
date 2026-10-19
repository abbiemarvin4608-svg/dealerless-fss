#!/usr/bin/env python3
"""Run full CNN comparisons with native and preprocessed SCI baselines."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time

from summarize import extract_run, query_phases

HERE = Path(__file__).resolve().parent
ROWS = (("resnet50", "SCI", 7200), ("resnet50", "SCI_preprocessed", 10800),
        ("resnet50", "SCI_ours_beaver", 10800), ("resnet152", "SCI", 10800),
        ("resnet152", "SCI_preprocessed", 21600), ("resnet152", "SCI_ours_beaver", 21600))


def write_json(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    temporary.replace(path)


def read_json(path):
    return json.loads(path.read_text())


def digest(path):
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            value.update(block)
    return value.hexdigest()


def expected_identity(binary, model_directory, backend, round_trace="on", optimization_profile=None,
                      spool_budget=0, native_spool_budget=0):
    graph = read_json(model_directory / "graph.json")
    value = dict(model=graph["model"], batch=graph["batch"], backend=backend,
                 bits=graph["bits"], frac=graph["frac"], queries=1, diagnostic=False,
                 network="unshaped-localhost", tile=64,
                 spool_budget_bytes_per_cnn_or_beaver_store=spool_budget)
    if optimization_profile is not None:
        from run_secure import RUNTIME_PARAMETERS, party_affinities
        value.update(optimization_profile=optimization_profile, native_spool_budget_bytes=native_spool_budget,
                     runtime_parameters=RUNTIME_PARAMETERS, cpu_affinity=party_affinities(4))
    if round_trace == "off":
        value["round_trace"] = "off"
    for name, path in (("binary", binary), ("graph", model_directory / "graph.json"),
                       ("parameters", model_directory / "parameters.bin"),
                       ("input", model_directory / "audit/input.bin")):
        value[name + "_sha256"] = digest(path)
    if backend in ("SCI_ours", "SCI_ours_beaver"):
        value["bridge_library_sha256"] = digest(binary.parent / "libcnn_ours.so")
    return value


def native_resource_preflight(graph, identity, disk=None):
    from estimate_runtime_resources import estimate

    shape = estimate(graph, identity["tile"])
    fields = ("status", "model", "batch", "bits", "frac", "tile", "abi", "native_scope",
              "scalar_macs", "linear_cot_instances", "linear_cot_calls")
    lower_bound = {key: value for key, value in shape.items()
                   if key in fields or key.startswith("native_linear_spool_bytes_")}
    if disk is None:
        # C++ std::tmpfile uses libc's /tmp, independently of Python TMPDIR.
        path = Path("/tmp").resolve()
        filesystem = os.statvfs(path)
        disk = dict(path=str(path), free_bytes=filesystem.f_bavail * filesystem.f_frsize,
                    total_bytes=filesystem.f_blocks * filesystem.f_frsize,
                    device=path.stat().st_dev)
    if disk.get("path") != str(Path("/tmp").resolve()) or \
            any(isinstance(disk.get(key), bool) or not isinstance(disk.get(key), int) or disk[key] < 0
                for key in ("free_bytes", "total_bytes", "device")):
        raise ValueError("native preflight requires the recorded libc /tmp filesystem capacity")
    budget = identity.get("native_spool_budget_bytes", 0) or identity["spool_budget_bytes_per_cnn_or_beaver_store"]
    violations = []
    required = lower_bound["native_linear_spool_bytes_both_parties_lower_bound"]
    if required > disk["free_bytes"]:
        violations.append(dict(constraint="both_party_spools_exceed_available_disk",
                               required_bytes=required, available_bytes=disk["free_bytes"]))
    if budget:
        for role in (1, 2):
            required = lower_bound["native_linear_spool_bytes_party%d_lower_bound" % role]
            if required > budget:
                violations.append(dict(constraint="party_spool_exceeds_explicit_budget", party=role,
                                       required_bytes=required, available_bytes=budget))
    return dict(status="resource_unavailable" if violations else "lower_bound_fits",
                evidence_scope="public_shape_resource_preflight_not_measurement",
                estimate=lower_bound, disk=disk, native_spool_budget_bytes_per_party=budget,
                violations=violations,
                qualification="Linear COT is a lower bound; fitting does not guarantee total inventory fits.")


def record_resource_unavailable(directory, identity, preflight, context, binary, model_directory):
    directory.mkdir(parents=True, exist_ok=True)
    record = dict(identity, status="resource_unavailable", protocol_started=False,
                  kind="secure-two-party-inference", evidence_scope=preflight["evidence_scope"],
                  reason="; ".join(value["constraint"] for value in preflight["violations"]),
                  resource_preflight=preflight, resource_guard=context,
                  binary=str(binary.resolve()), graph=str(model_directory / "graph.json"),
                  parameters=str(model_directory / "parameters.bin"), input=str(model_directory / "audit/input.bin"),
                  round_trace=identity.get("round_trace", "on"),
                  measurement_mode="time_and_bytes" if identity.get("round_trace") == "off" else "time_bytes_and_rounds",
                  commands=[], started_epoch=time.time(), finished_epoch=time.time())
    write_json(directory / "RESOURCE_PREFLIGHT.json", preflight)
    write_json(directory / "PLAN.json", record)
    write_json(directory / "STATUS.json", record)
    return record


def validate_resource_unavailable(directory, identity, graph):
    plan, status = [read_json(directory / name) for name in ("PLAN.json", "STATUS.json")]
    for record in (plan, status):
        if any(record.get(key) != value for key, value in identity.items()):
            raise ValueError("existing resource preflight identity differs from the requested run")
        if record.get("status") != "resource_unavailable" or record.get("protocol_started") is not False or \
                record.get("evidence_scope") != "public_shape_resource_preflight_not_measurement":
            raise ValueError("resource unavailability requires an unstarted static preflight")
    if identity["backend"] != "SCI_preprocessed" or list(directory.glob("party*")) or \
            any((directory / name).exists() for name in ("PARTIES.json", "ROUNDS.json", "OUTPUT_AUDIT.json")):
        raise ValueError("resource preflight cannot replace started protocol evidence")
    preflight = read_json(directory / "RESOURCE_PREFLIGHT.json")
    expected = native_resource_preflight(graph, identity, preflight["disk"])
    if expected["status"] != "resource_unavailable" or preflight != expected or \
            any(record.get("resource_preflight") != expected for record in (plan, status)):
        raise ValueError("recorded resource preflight does not substantiate unavailability")
    return status


def validate_existing(directory, identity, graph):
    plan, status = [read_json(directory / name) for name in ("PLAN.json", "STATUS.json")]
    for record in (plan, status):
        if any(record.get(key) != value for key, value in identity.items()):
            raise ValueError("existing run identity differs from the requested run: " + str(directory))
    parties = [read_json(directory / ("party%d.json" % role)) for role in (1, 2)]
    required = {(-1, "setup"), *((0, phase) for phase in ("offline", "input", "online", "output"))}
    for role, party in enumerate(parties, 1):
        if party.get("status") != "complete" or party.get("role") != role or any(
                party.get(key) != identity[key] for key in ("model", "backend", "bits", "frac")):
            raise ValueError("both parties must confirm matching completed inference")
        if [query.get("query") for query in party.get("queries", [])] != [0]:
            raise ValueError("complete single-query coverage is required")
        if party.get("method", {}).get("diagnostic") is not False:
            raise ValueError("formal recovery requires nondiagnostic protocol evidence")
        if not required.issubset(query_phases(party)):
            raise ValueError("complete phase measurements are required")
        layers = party.get("layers", [])
        if len(layers) != len(graph["nodes"]) or any(
                (layer.get("query"), layer.get("layer"), layer.get("name"), layer.get("op")) !=
                (0, index, node["name"], node["op"])
                for index, (layer, node) in enumerate(zip(layers, graph["nodes"]))):
            raise ValueError("complete ordered full-graph layer measurements are required")
        if identity.get("round_trace", "on") != "off" and not (directory / ("party%d.trace.jsonl" % role)).is_file():
            raise ValueError("both complete protocol traces are required")
    return plan, status, parties


def guard_context():
    report = os.environ.get("IMAGENET_MEMORY_GUARD_REPORT")
    if not report:
        return None
    value = read_json(Path(report))
    context = {key: value[key] for key in ("cgroup", "memory_limit_bytes", "memory_and_swap_limit_bytes",
                                          "started_epoch", "boot_id")}
    context["report"] = str(Path(report).resolve())
    return context


def require_memory_guard():
    context = guard_context()
    membership = dict(line.split(":", 2)[1:] for line in Path("/proc/self/cgroup").read_text().splitlines())
    if context is None or membership.get("memory") != "/" + Path(context["cgroup"]).name:
        raise ValueError("full-network measurement requires a real enclosing memory guard; use memory_guard.py run")
    report = read_json(Path(context["report"]))
    if report.get("status") != "running" or context["boot_id"] != Path("/proc/sys/kernel/random/boot_id").read_text().strip():
        raise ValueError("memory guard must be active in the current boot")
    group = Path(context["cgroup"])
    limits = (int((group / "memory.limit_in_bytes").read_text()),
              int((group / "memory.memsw.limit_in_bytes").read_text()))
    if limits != (context["memory_limit_bytes"], context["memory_and_swap_limit_bytes"]) or \
            not 0 < limits[0] <= 2048 << 20 or not limits[0] <= limits[1] <= 2560 << 20:
        raise ValueError("memory guard limits must match the active cgroup and be at most 2 GiB RAM plus 512 MiB swap")
    return context


def guarded_baseline_rows(refresh, round_trace="on"):
    suffix = "_cost_q1" if round_trace == "off" else "_q1"
    rows = [(model, backend, timeout, model + "_" + backend + suffix)
            for model, backend, timeout in ROWS]
    if refresh:
        if round_trace == "off":
            raise ValueError("cost-only rows already include a fresh SCI baseline")
        context = guard_context()
        membership = dict(line.split(":", 2)[1:] for line in Path("/proc/self/cgroup").read_text().splitlines())
        if context is None or membership.get("memory") != "/" + Path(context["cgroup"]).name:
            raise ValueError("baseline refresh requires a real enclosing memory guard")
        rows.append(("resnet50", "SCI", 7200, "resnet50_SCI_guarded_q1"))
    return rows


def preserve_json(path):
    if path.exists():
        history = path.parent / "recovery_history"
        history.mkdir(exist_ok=True)
        destination = history / (path.stem + ".%d.json" % time.time_ns())
        destination.write_bytes(path.read_bytes())


def qualify_existing(directory, identity, graph, oracle, analyze=None):
    if read_json(directory / "STATUS.json").get("status") == "resource_unavailable":
        validate_resource_unavailable(directory, identity, graph)
        return "skipped_resource_unavailable"
    plan, status, parties = validate_existing(directory, identity, graph)
    if identity.get("round_trace") == "off":
        from audit_secure import audit_output
        audit = audit_output(directory, oracle)
        if audit["status"] != "passed" or audit["values_checked"] != 1000:
            raise ValueError("cost-only output differs from the 1000-logit integer oracle")
        if status.get("status") == "complete":
            extract_run(directory)
            return "skipped_qualified_cost_measurement"
        preserve_json(directory / "STATUS.json")
        write_json(directory / "PARTIES.json", parties)
        completed = dict(status, status="complete", reason=None,
                         recovery="validated_completed_cost_measurement_without_protocol_rerun",
                         recovered_epoch=time.time())
        write_json(directory / "STATUS.json", completed)
        try:
            extract_run(directory)
        except BaseException:
            write_json(directory / "STATUS.json", status)
            raise
        return "recovered_completed_cost_measurement"
    if status.get("status") == "complete":
        try:
            value = extract_run(directory)
            rounds = read_json(directory / "ROUNDS.json")
            qualified = read_json(HERE / ".round_trace_native/qualified.json")
            if value["status"] == "complete" and rounds.get("analyzer") == "native" and \
                    rounds.get("analyzer_binary_sha256") == qualified["binary_sha256"] and \
                    read_json(directory / "PARTIES.json") == parties:
                from audit_secure import audit_output
                audit = audit_output(directory, oracle)
                if audit["status"] == "passed" and audit["values_checked"] == 1000:
                    return "skipped_qualified"
        except (OSError, ValueError, KeyError, TypeError):
            pass
    if analyze is None:
        from round_trace_stream import analyze_pair
        analyze = analyze_pair
    for name in ("STATUS.json", "ROUNDS.json", "PARTIES.json", "OUTPUT_AUDIT.json", "RECOVERY.json"):
        preserve_json(directory / name)
    recovery = dict(status="running", started_epoch=time.time(),
                    operation="reanalyze_completed_protocol_without_protocol_rerun",
                    previous_status=status.get("status"), resource_guard=guard_context())
    write_json(directory / "RECOVERY.json", recovery)
    expected_phases = [[dict(query=query, phase=phase, sent_bytes=value["sent_bytes"],
                             received_bytes=value["received_bytes"])
                        for (query, phase), value in query_phases(party).items()] for party in parties]
    print(json.dumps(dict(status="recovering_trace", directory=str(directory))), flush=True)
    try:
        rounds = analyze(directory / "party1.trace.jsonl", directory / "party2.trace.jsonl", expected_phases)
        if rounds.get("status") != "passed":
            raise ValueError("recovered causal trace did not qualify: " + str(rounds.get("reason")))
        write_json(directory / "ROUNDS.json", rounds)
        write_json(directory / "PARTIES.json", parties)
        from audit_secure import audit_output
        audit = audit_output(directory, oracle)
        if audit["status"] != "passed" or audit["values_checked"] != 1000:
            raise ValueError("recovered output differs from the 1000-logit integer oracle")
        final = dict(status, status="complete", reason=None, finished_epoch=time.time(), recovered_epoch=time.time(),
                     recovery="reanalyzed_completed_protocol_without_protocol_rerun",
                     recovery_resource_guard=recovery["resource_guard"])
        write_json(directory / "STATUS.json", final)
        extract_run(directory)
    except BaseException as error:
        recovery.update(status="failed", reason="%s: %s" % (type(error).__name__, error),
                        finished_epoch=time.time())
        write_json(directory / "RECOVERY.json", recovery)
        write_json(directory / "STATUS.json", dict(status, status="recovery_failed",
                   reason=recovery["reason"], recovery_resource_guard=recovery["resource_guard"]))
        raise
    recovery.update(status="complete", finished_epoch=time.time())
    write_json(directory / "RECOVERY.json", recovery)
    return "recovered_completed_protocol"


def run(args):
    context = require_memory_guard()
    round_trace = getattr(args, "round_trace", "on")
    optimization_profile = getattr(args, "optimization_profile", "optimized")
    spool_budget = getattr(args, "spool_budget", 0)
    native_spool_budget = getattr(args, "native_spool_budget", 0)
    if min(spool_budget, native_spool_budget) < 0:
        raise ValueError("material budgets must be nonnegative")
    targets = guarded_baseline_rows(args.refresh_guarded_baseline, round_trace)
    results = args.results.resolve()
    output = results / "performance"
    output.mkdir(parents=True, exist_ok=True)
    batch_path = output / ("COST_BATCH_STATUS.json" if round_trace == "off" else "BATCH_STATUS.json")
    if batch_path.exists():
        if not args.resume:
            raise ValueError("existing batch evidence requires --resume")
        preserve_json(batch_path)
    batch = {"status": "running", "evidence_scope": "measured_full_graph_single_query_localhost",
             "started_epoch": time.time(), "resume": args.resume, "resource_guard": context,
             "round_trace": round_trace, "rows": []}
    write_json(batch_path, batch)
    for model, backend, timeout, directory_name in targets:
        models_root = getattr(args, "models_root", None) or results / "models"
        model_directory = models_root.resolve() / (model + "_w32_f12")
        directory = output / directory_name
        row = dict(model=model, backend=backend, status="running", timeout_seconds=timeout,
                   directory=str(directory), started_epoch=time.time())
        batch["rows"].append(row)
        write_json(batch_path, batch)
        graph = read_json(model_directory / "graph.json")
        identity = expected_identity(args.binary.resolve(), model_directory, backend, round_trace,
                                     optimization_profile, spool_budget, native_spool_budget)
        if directory.exists() and any(directory.iterdir()):
            if not args.resume:
                raise ValueError("existing run evidence requires --resume: " + str(directory))
            try:
                recovery = qualify_existing(directory, identity, graph, model_directory / "audit")
            except BaseException as error:
                row.update(status="failed", failed_stage="resume_validation_or_qualification",
                           reason="%s: %s" % (type(error).__name__, error), finished_epoch=time.time())
                batch["status"] = "failed"
                write_json(batch_path, batch)
                raise
            row.update(status="resource_unavailable" if recovery == "skipped_resource_unavailable" else "qualified",
                       resume_action=recovery, finished_epoch=time.time())
            if recovery == "skipped_resource_unavailable":
                previous = read_json(directory / "STATUS.json")
                row.update(reason=previous["reason"], protocol_started=False,
                           resource_preflight=previous["resource_preflight"])
            write_json(batch_path, batch)
            print(json.dumps(row), flush=True)
            continue
        row["resource_guard"] = batch["resource_guard"]
        if backend == "SCI_preprocessed":
            preflight = native_resource_preflight(graph, identity)
            row["resource_preflight"] = preflight
            if preflight["status"] == "resource_unavailable":
                unavailable = record_resource_unavailable(directory, identity, preflight, context,
                                                         args.binary, model_directory)
                row.update(status="resource_unavailable", reason=unavailable["reason"], protocol_started=False,
                           finished_epoch=time.time())
                write_json(batch_path, batch)
                print(json.dumps(row), flush=True)
                continue
        resource_name = directory_name if round_trace == "off" or directory_name.endswith("_guarded_q1") else model + "_" + backend
        command = ["/usr/bin/time", "-v", "-o", str(output / (resource_name + ".resource.txt")),
            "python3", str(HERE / "run_secure.py"), "--binary", str(args.binary.resolve()),
            "--graph", str(model_directory / "graph.json"), "--parameters", str(model_directory / "parameters.bin"),
            "--input", str(model_directory / "audit/input.bin"), "--backend", backend, "--queries", "1",
            "--tile", "64", "--spool-budget", str(spool_budget), "--native-spool-budget", str(native_spool_budget),
            "--timeout", str(timeout), "--port", str(args.port),
            "--optimization-profile", optimization_profile,
            "--output", str(directory), "--round-trace", round_trace]
        row["command"] = command
        write_json(batch_path, batch)
        protocol = subprocess.run(command)
        if row["resource_guard"]:
            for name in ("PLAN.json", "STATUS.json"):
                path = directory / name
                if path.exists():
                    write_json(path, dict(read_json(path), resource_guard=row["resource_guard"]))
        if protocol.returncode:
            row.update(status="failed", failed_stage="protocol_or_trace", finished_epoch=time.time())
            batch["status"] = "failed"
            write_json(batch_path, batch)
            return protocol.returncode
        audit = subprocess.run(["python3", str(HERE / "audit_secure.py"), "--directory", str(directory),
            "--oracle", str(model_directory / "audit"), "--final-only"])
        if audit.returncode:
            row.update(status="failed", failed_stage="final_output_audit", finished_epoch=time.time())
            batch["status"] = "failed"
            write_json(batch_path, batch)
            return audit.returncode
        row.update(status="qualified", finished_epoch=time.time())
        validate_existing(directory, identity, read_json(model_directory / "graph.json"))
        extract_run(directory)
        write_json(batch_path, batch)
        print(json.dumps({"status": "qualified", "model": model, "backend": backend,
                          "directory": str(directory)}), flush=True)
    batch.update(status="complete_with_resource_failures" if any(row["status"] == "resource_unavailable"
                 for row in batch["rows"]) else "complete", finished_epoch=time.time())
    write_json(batch_path, batch)
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--binary", type=Path, required=True)
    parser.add_argument("--models-root", type=Path, help="reuse the existing fixed 32/12 exported models and integer oracles")
    parser.add_argument("--port", type=int, default=42000)
    parser.add_argument("--resume", action="store_true", help="qualify completed evidence and resume only missing rows")
    parser.add_argument("--round-trace", choices=("on", "off"), default="off",
                        help="off measures time/bytes without detailed traces, causal rounds, or RTT projections")
    parser.add_argument("--refresh-guarded-baseline", action="store_true",
                        help="append a fresh guarded ResNet-50 SCI measurement after the planned rows")
    parser.add_argument("--optimization-profile", choices=("optimized", "baseline"), default="optimized")
    parser.add_argument("--spool-budget", type=int, default=0,
                        help="per-party CNN/Beaver store cap in bytes; zero is unlimited")
    parser.add_argument("--native-spool-budget", type=int, default=0,
                        help="per-party SCI preprocessing disk cap; zero inherits --spool-budget")
    raise SystemExit(run(parser.parse_args()))
