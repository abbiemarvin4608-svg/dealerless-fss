#!/usr/bin/env python3
"""Freeze and execute the declared ten-row folded-activation comparison."""
from __future__ import annotations

import argparse
import datetime
import fcntl
import gzip
import hashlib
import json
import os
from pathlib import Path
import signal
import shutil
import subprocess
import sys
import time

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[4]
RUNTIME = REPO / "project/our/tools/segmented_dcf_runtime"
BASELINE = REPO / "project/our/results/sci_ours_joint_optimization_baseline_20261013"
MEMORY = 1610612736
PREFETCH = 536870912
PAPER = "SCI_Beaver_segmented_PaperLUT"
FAMILIES = {"full": "full_q_arithmetic", "segmented": "segmented_boolean_tensor_b2a"}
NETWORKS = {
    "LAN": dict(bandwidth_bits_per_second=1000000000, rtt_seconds=0.0005),
    "WAN": dict(bandwidth_bits_per_second=400000000, rtt_seconds=0.004),
}
SUPPLEMENTARY_SAMPLING = dict(order="candidate then baseline for each matched fresh pair",
    selection="first frozen comparison with unresolved non-regressed step admission or LUT-construction online activation/event-window evidence",
    lut_online_metrics=["sigmoid/online", "tanh/online", "sigmoid/lut/online/event_window", "tanh/lut/online/event_window"],
    stop="resolved, failed qualification, or either row reaches three groups; retain every declared group")
ALLOCATOR_ENV = {
    "MALLOC_ARENA_MAX": "2",
    "MALLOC_TRIM_THRESHOLD_": "1",
    "MALLOC_MMAP_THRESHOLD_": "131072",
}
BOOTSTRAP = dict(replicates=10000, seed=20261013,
    method="stratified independent BCa ratio of pooled medians", interval=0.95,
    orientation="baseline / candidate; segmented / full for LUT comparisons")


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    Path(path).write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def digest(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            value.update(block)
    return value.hexdigest()


def declarations():
    # Interleave LUT constructions before moving to the next public schedule.
    rows = []
    for graph, workers in (("sequential", 1), ("parallel", 2)):
        for family in ("segmented", "full"):
            rows.append(dict(id="%s-a%d-%s-bitdec" % (graph, workers, family),
                backend=PAPER, graph_mode=graph, activation_workers=workers,
                family=family, sign_backend="bitdec", role="lut_construction"))
    rows.append(dict(id="parallel-a1-full-bitdec", backend=PAPER,
        graph_mode="parallel", activation_workers=1, family="full",
        sign_backend="bitdec", role="branch_attribution"))
    for backend in ("SCI_preprocessed", "SCI_Beaver_segmented", "SCI_Beaver_segmented_DPF_LUT"):
        rows.append(dict(id="parallel-a2-" + backend, backend=backend,
            graph_mode="parallel", activation_workers=2, family=None,
            sign_backend=None, role="named_method_control"))
    for graph, workers in (("sequential", 1), ("parallel", 2)):
        rows.append(dict(id="%s-a%d-full-segmented-sign" % (graph, workers),
            backend=PAPER, graph_mode=graph, activation_workers=workers,
            family="full", sign_backend="segmented_fss", role="sign_candidate"))
    return rows


def comparisons():
    pairs = []
    for graph, workers in (("sequential", 1), ("parallel", 2)):
        prefix = "%s-a%d-" % (graph, workers)
        pairs.append(dict(id=prefix + "lut-construction", kind="lut_construction",
            baseline=prefix + "segmented-bitdec", candidate=prefix + "full-bitdec"))
        pairs.append(dict(id=prefix + "sign", kind="sign_candidate",
            baseline=prefix + "full-bitdec", candidate=prefix + "full-segmented-sign"))
    for family in ("segmented", "full"):
        pairs.append(dict(id=family + "-schedule", kind="schedule",
            baseline="sequential-a1-" + family + "-bitdec",
            candidate="parallel-a2-" + family + "-bitdec"))
    pairs += [dict(id="full-branches", kind="schedule",
                  baseline="sequential-a1-full-bitdec", candidate="parallel-a1-full-bitdec"),
              dict(id="full-sharding", kind="schedule",
                  baseline="parallel-a1-full-bitdec", candidate="parallel-a2-full-bitdec")]
    for backend in ("SCI_preprocessed", "SCI_Beaver_segmented", "SCI_Beaver_segmented_DPF_LUT"):
        pairs.append(dict(id="method-vs-" + backend, kind="method_comparison",
            baseline="parallel-a2-" + backend, candidate="parallel-a2-full-bitdec"))
    return pairs


def command(row, binary, fixture):
    result = [sys.executable, str(HERE / "instrumented_run.py"), "--binary", str(binary),
        "--fixture", str(fixture), "--output", "{output}", "--backend", row["backend"],
        "--batch", "128", "--steps", "1", "--start", "0", "--queries", "10", "--warmup", "0",
        "--timeout", "600", "--network", "localhost", "--memory-gib", "1.5",
        "--cpus", "10,12,14", "--server-cpus", "0,2,4,10", "--client-cpus", "6,8,12,16",
        "--cggm-threads", "2", "--activation-workers", str(row["activation_workers"]),
        "--graph-mode", row["graph_mode"], "--native-preprocessing", "--model-bias-precompute",
        "--material-layout", "contiguous", "--segment-policy", "fixed-4", "--max-segments", "4",
        "--prefetch-bytes", str(PREFETCH), "--generation-budget-bytes", str(MEMORY),
        "--generation-reserve-bytes", str(PREFETCH), "--fuse-domains", "--pair-extensions",
        "--pack-bits", "--aes-online", "--beaver-algebra", "--beaver-tile-rows", "32",
        "--bit-block-mode", "direct_ot", "--merge-leaf-cw", "--reuse-generation-workspace",
        "--prune-equality", "--prune-leaf-hashes", "--pack-ring", "--round-trace"]
    if row["graph_mode"] == "parallel":
        result += ["--tail-parallel"]
    if row["backend"] == PAPER:
        result += ["--folded-activations", "both", "--lut-key-family", FAMILIES[row["family"]],
                   "--folded-sign-backend", row["sign_backend"]]
    return result


def frozen_inputs(binary, bridge, fixture):
    binary, bridge, fixture = (Path(path) for path in (binary, bridge, fixture))
    files = [binary, bridge, binary.parent / "runtime_manifest.json", binary.parent / "schema.json",
             fixture / "contract.json", fixture / "server_model.i16", fixture / "client_input.i16"]
    files += list(RUNTIME.glob("*.py"))
    files += [BASELINE / "analyze_samples.py", BASELINE / "network_model.py"]
    files += list(HERE.glob("*.py"))
    return files


def freeze(binary, bridge, fixture, qualifications):
    binary, bridge, fixture = (Path(path).resolve() for path in (binary, bridge, fixture))
    if not qualifications:
        raise ValueError("passed final diagnostic qualification artifacts are required")
    proofs = []
    for path in qualifications:
        path = Path(path).resolve()
        data = read(path)
        if not (data.get("status") == "passed" or data.get("eligible") is True):
            raise ValueError("diagnostic qualification did not pass: " + str(path))
        proofs.append(dict(path=str(path), sha256=digest(path)))
    description = json.loads(subprocess.check_output([str(binary), "--describe"], text=True))
    if not {row["backend"] for row in declarations()} <= set(description.get("backends", [])):
        raise ValueError("the common binary must implement every named backend")
    identity = {str(path): digest(path) for path in frozen_inputs(binary, bridge, fixture)}
    rows = [dict(row, command=command(row, binary, fixture)) for row in declarations()]
    return dict(schema=1, frozen_at=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        purpose="ten fresh queries per frozen row; complete single-step folded activation costs",
        binary=str(binary), binary_sha256=digest(binary), bridge=str(bridge),
        bridge_sha256=digest(bridge), fixture=str(fixture), qualification_artifacts=proofs,
        file_identities=identity, binary_description=description, rows=rows, comparisons=comparisons(),
        networks={name: dict(spec) for name, spec in NETWORKS.items()},
        query_count=10, warmup_queries=0, timeout_seconds=600,
        initial_groups=1, maximum_groups_per_row=3,
        supplementary_sampling=SUPPLEMENTARY_SAMPLING,
        allocator_environment=dict(ALLOCATOR_ENV),
        controller_address_space_limit_bytes=MEMORY,
        trace_archive="lossless gzip after each group; verify exact content and complete footer before removing raw trace",
        bootstrap=dict(BOOTSTRAP),
        communication_formula="8 * sum(both parties sent bytes) / bandwidth_bits_per_second + RTT * causal rounds",
        network_scope="communication-only projection; excludes localhost compute and relay measurements",
        input_state="default zero hidden state; default/nonzero correctness covered by qualification artifacts",
        limits=dict(memory_bytes_per_party=MEMORY, inclusive_prefetch_bytes=PREFETCH,
            server_cpus=[0, 2, 4, 10], client_cpus=[6, 8, 12, 16], segment_policy="fixed-4"),
        encoding=dict(index_bits=10, bytes_per_index=2, selection_ring_bits=32,
            online_index_bytes_for_both_activations=102400, unimplemented_tightly_packed_bytes=64000),
        interpretation=dict(lut="tree segmentation and arithmetic-payload/B2A constructions both differ",
            methods="SCI controls use a different numeric activation algorithm",
            wall="same-query maximum of parties at the complete measured boundary",
            work="stage or worker sums describe work attribution, never parallel elapsed time"))


def verify(plan):
    if [row["id"] for row in plan["rows"]] != [row["id"] for row in declarations()]:
        raise ValueError("the frozen ten-row order was modified")
    if plan["networks"] != NETWORKS or plan["bootstrap"] != BOOTSTRAP:
        raise ValueError("the frozen network or statistical contract was modified")
    if (plan.get("query_count"), plan.get("warmup_queries"), plan.get("timeout_seconds")) != (10, 0, 600):
        raise ValueError("the frozen query contract was modified")
    if plan.get("initial_groups") != 1 or plan.get("maximum_groups_per_row") != 3:
        raise ValueError("the frozen measurement-group budget was modified")
    if plan.get("supplementary_sampling") != SUPPLEMENTARY_SAMPLING or \
            plan.get("controller_address_space_limit_bytes") != MEMORY:
        raise ValueError("the frozen supplementary-sampling or controller-memory contract was modified")
    if plan.get("allocator_environment") != ALLOCATOR_ENV:
        raise ValueError("the frozen allocator environment was modified")
    expected_inputs = {str(path) for path in frozen_inputs(plan["binary"], plan["bridge"], plan["fixture"])}
    if set(plan["file_identities"]) != expected_inputs:
        raise ValueError("the frozen input inventory was modified")
    if plan["binary_sha256"] != plan["file_identities"][plan["binary"]] or \
            plan["bridge_sha256"] != plan["file_identities"][plan["bridge"]]:
        raise ValueError("the frozen binary identity differs from its file identity")
    for name, expected in plan["file_identities"].items():
        if not Path(name).is_file():
            raise ValueError("a frozen input is missing: " + name)
        if digest(name) != expected:
            raise ValueError("a frozen input changed: " + name)
    if not plan["qualification_artifacts"]:
        raise ValueError("passed diagnostic qualification artifacts are required")
    for proof in plan["qualification_artifacts"]:
        if not Path(proof["path"]).is_file():
            raise ValueError("a diagnostic qualification artifact is missing: " + proof["path"])
        if digest(proof["path"]) != proof["sha256"]:
            raise ValueError("a diagnostic qualification artifact changed")
    expected = [dict(row, command=command(row, Path(plan["binary"]), Path(plan["fixture"])))
                for row in declarations()]
    if plan["rows"] != expected or plan["comparisons"] != comparisons():
        raise ValueError("a frozen row or comparison was modified")


def verify_manifest(directory, frozen_plan=None):
    directory = Path(directory)
    reference = Path(frozen_plan) if frozen_plan is not None else directory / "frozen-plan.json"
    if not reference.is_file():
        raise ValueError("the original frozen plan is missing: " + str(reference))
    plan = read(reference)
    if read(directory / "manifest.json") != plan:
        raise ValueError("the manifest differs from the original frozen plan: " + str(reference))
    verify(plan)
    return plan


def group_record(identity, number, comparison=None):
    path = identity if number == 1 else identity + "/group%d" % number
    return dict(group=number, status="not_started", reason="declared fresh group has not run",
        path=path, result=path + "/result.json", trigger_comparison=comparison)


def next_extra_pair(report, summary, plan):
    counts = {row["id"]: len(row["groups"]) for row in summary["rows"]}
    for pair in report["comparisons"]:
        if pair.get("additional_groups_needed") and all(counts[pair[key]] < plan["maximum_groups_per_row"]
                                                        for key in ("baseline", "candidate")):
            return pair
    return None


def declare_extra_pair(pair, summary):
    records = {row["id"]: row for row in summary["rows"]}
    if any(len(records[pair[key]]["groups"]) >= 3 for key in ("baseline", "candidate")):
        raise ValueError("matched extra groups exceed a row budget")
    order = []
    for key in ("candidate", "baseline"):
        row = records[pair[key]]
        group = group_record(row["id"], len(row["groups"]) + 1, pair["id"])
        row["groups"].append(group)
        order.append((row, group))
    summary["extra_group_decisions"].append(dict(comparison=pair["id"],
        reasons=pair["extra_group_reasons"], order=[group["path"] for _, group in order]))
    return order


def archive_trace(path, require_complete):
    compressed = path.with_suffix(path.suffix + ".gz")
    if compressed.exists():
        raise ValueError("trace archive already exists: " + str(compressed))
    with path.open("rb") as source, gzip.open(compressed, "wb", compresslevel=1) as target:
        shutil.copyfileobj(source, target, 1 << 20)
    with path.open("rb") as source, gzip.open(compressed, "rb") as restored:
        while True:
            original, copy = source.read(1 << 20), restored.read(1 << 20)
            if original != copy:
                raise ValueError("compressed trace differs from original: " + str(path))
            if not original:
                break
    first, last, count = None, None, 0
    with gzip.open(compressed, "rt") as stream:
        for line in stream:
            if line.strip():
                first = line if first is None else first
                last, count = line, count + 1
    try:
        metadata, footer = json.loads(first), json.loads(last)
        complete = metadata.get("type") == "meta" and metadata.get("version") == 1 and \
            footer.get("type") == "complete" and footer.get("events") == count - 2
    except (TypeError, json.JSONDecodeError):
        footer, complete = {}, False
    if require_complete and not complete:
        raise ValueError("completed row trace lacks a valid complete footer: " + str(path))
    record = dict(path=str(compressed), raw_bytes=path.stat().st_size, gzip_bytes=compressed.stat().st_size,
        content_equal=True, complete_footer=complete, events=footer.get("events") if complete else None)
    path.unlink()
    return record


def run_plan(plan, output):
    verify(plan)
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    write(output / "frozen-plan.json", plan)
    write(output / "manifest.json", plan)
    summary = dict(status="running", extra_group_decisions=[], rows=[dict(id=row["id"], status="not_started",
        reason="frozen row has not run", result=row["id"] + "/result.json",
        groups=[group_record(row["id"], 1)]) for row in plan["rows"]])
    write(output / "summary.json", summary)
    env = {key: value for key, value in os.environ.items()
           if not key.startswith(("SEGFSS_", "SDCF_", "MALLOC_")) and key not in
           ("GOOGLE30_PROXY_OFFSET", "SWEEP_CONNECT_OFFSET", "OMP_THREAD_LIMIT", "OMP_DYNAMIC", "GLIBC_TUNABLES")}
    env["OMP_NUM_THREADS"] = "1"
    env.update(ALLOCATOR_ENV)
    declarations_by_id = {row["id"]: row for row in plan["rows"]}
    with Path("/tmp/sdcf-measurement.lock").open("a") as lock:
        def execute(row, group):
            declaration = declarations_by_id[row["id"]]
            fcntl.flock(lock, fcntl.LOCK_EX)
            try:
                if verify_manifest(output) != plan:
                    raise ValueError("the original frozen plan changed during measurement")
                command_line = [part.replace("{output}", str(output / group["path"]))
                                for part in declaration["command"]]
                group.update(status="running", reason=None, command=command_line,
                    controller_address_space_limit_bytes=MEMORY)
                row.update(status="running", reason=None)
                write(output / "summary.json", summary)
                started = time.monotonic()
                prefix = row["id"] + ".group%d" % group["group"]
                resource_file = output / (prefix + ".controller.resources.txt")
                with (output / (prefix + ".runner.log")).open("w") as log:
                    child = subprocess.Popen(["/usr/bin/time", "-v", "-o", str(resource_file),
                        "/usr/bin/prlimit", "--as=%d" % MEMORY, "--"] + command_line,
                        stdout=log, stderr=subprocess.STDOUT, env=env, start_new_session=True)
                    try:
                        group["runner_exit_code"] = child.wait()
                    except BaseException:
                        os.killpg(child.pid, signal.SIGTERM)
                        try:
                            child.wait(timeout=5)
                        except subprocess.TimeoutExpired:
                            os.killpg(child.pid, signal.SIGKILL)
                            child.wait()
                        group.update(status="interrupted", reason="controller interrupted; raw evidence retained")
                        raise
                group["controller_elapsed_seconds"] = time.monotonic() - started
                group["controller_resources"] = resource_file.name
                path = output / group["result"]
                result = read(path) if path.exists() else dict(status="failed", reason="runner produced no result.json")
                group.update(status=result["status"], reason=result.get("reason"))
                if group["runner_exit_code"] and group["status"] == "complete":
                    group.update(status="failed", reason="controller exited nonzero after party measurements")
                archive_started = time.monotonic()
                try:
                    group["trace_archive"] = [archive_trace(raw, group["status"] == "complete") for raw in
                        sorted((output / group["path"]).glob("*.roundtrace.jsonl"))]
                except (OSError, ValueError) as error:
                    group.update(status="failed", reason="trace archival failed: " + str(error))
                group["archive_elapsed_seconds"] = time.monotonic() - archive_started
                row.update(status="complete" if all(item["status"] == "complete" for item in row["groups"])
                           else group["status"], reason=group.get("reason"))
                write(output / "summary.json", summary)
                print(row["id"], "group%d" % group["group"], group["status"], flush=True)
            except BaseException:
                if group["status"] == "running":
                    group.update(status="failed", reason="controller failed before completing the group")
                row.update(status=group["status"], reason=group.get("reason"))
                summary["status"] = "interrupted"
                write(output / "summary.json", summary)
                raise
            finally:
                fcntl.flock(lock, fcntl.LOCK_UN)
        for row in summary["rows"]:
            execute(row, row["groups"][0])
        from analyze import analyze, save_report
        report = analyze(output)
        save_report(output, report)
        while all(row["status"] == "qualified" for row in report["rows"].values()):
            pair = next_extra_pair(report, summary, plan)
            if pair is None:
                break
            order = declare_extra_pair(pair, summary)
            write(output / "summary.json", summary)
            for row, group in order:
                execute(row, group)
            report = analyze(output)
            save_report(output, report)
    summary["status"] = "complete" if report["status"] == "complete" else "incomplete"
    summary["sampling_status"] = report["sampling_status"]
    write(output / "summary.json", summary)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    actions = parser.add_subparsers(dest="action", required=True)
    prepare = actions.add_parser("freeze", help="freeze qualified common build and commands; does not benchmark")
    for name in ("binary", "bridge", "fixture", "output"):
        prepare.add_argument("--" + name, type=Path, required=True)
    prepare.add_argument("--qualification", type=Path, action="append", required=True)
    execute = actions.add_parser("run", help="run every frozen row sequentially without rebuilding")
    execute.add_argument("--plan", type=Path, required=True)
    execute.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.action == "freeze":
        if args.output.exists():
            raise ValueError("a frozen plan must not overwrite an existing artifact")
        plan = freeze(args.binary, args.bridge, args.fixture, args.qualification)
        write(args.output, plan)
        print(json.dumps(dict(status="frozen", rows=len(plan["rows"]), output=str(args.output))))
        return 0
    summary = run_plan(read(args.plan), args.output)
    return 0 if summary["status"] == "complete" else 1


if __name__ == "__main__":
    raise SystemExit(main())
