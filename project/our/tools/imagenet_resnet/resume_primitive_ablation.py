#!/usr/bin/env python3
"""Resume only the interrupted WAN/input28/nseg8 formal measurements."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import time

from analyze_rnn_widths import read_json, write_json
from prepare_primitive_ablation import DEFAULT_OUTPUT, aggregate_missing, build_identity, load_runtime


NETWORK = "low_latency_wan_model"
POINT = "ours_w28_n8"
TAGS = ["repeat%d" % index for index in range(1, 6)]


def file_identity(path):
    path = Path(path).resolve()
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return dict(path=str(path), bytes=path.stat().st_size, sha256=digest.hexdigest())


def input_identity(directory, started, first_result_time):
    directory = Path(directory).resolve()
    if directory.stat().st_uid != os.getuid() or directory.stat().st_mode & 0o077:
        raise ValueError("preserved input directory must remain private to the experiment owner")
    records = []
    for role in (1, 2):
        path = directory / ("party%d.bin" % role)
        stat = path.stat()
        if stat.st_uid != os.getuid() or stat.st_mode & 0o077 or stat.st_size != 128 * 8:
            raise ValueError("preserved input shape or access differs from the original private shares")
        if not started <= stat.st_mtime <= first_result_time or not started <= stat.st_ctime <= first_result_time:
            raise ValueError("preserved input timestamps fall outside the original input creation interval")
        records.append(dict(file_identity(path), role=role, inode=stat.st_ino, device=stat.st_dev,
                            mtime_ns=stat.st_mtime_ns, ctime_ns=stat.st_ctime_ns))
    return records


def original_state(output):
    root = output / "missing_width_runs"
    config = read_json(output / "missing_width_config.json")
    plan = read_json(root / "plan.json")
    if any(plan.get(key) != value for key, value in config.items()):
        raise ValueError("original plan differs from the frozen configuration")
    if config["widths"] != [28] or config["batch"] != 128 or config["repetitions"] != 5:
        raise ValueError("resume supports only the frozen input28/batch128/five-repeat plan")
    if build_identity(Path(config["build"])) != plan["build_identity"]:
        raise ValueError("original build identity changed")
    return root, config, plan


def validate_completed(path, row):
    if row["status"] != "complete":
        return
    parties = [read_json(path.parent / ("party%d.json" % role)) for role in (1, 2)]
    for role, party in enumerate(parties, 1):
        if any(party[key] != value for key, value in
               (("party", role), ("backend", row["backend"]), ("input_bits", 28), ("count", 128))):
            raise ValueError("party identity differs from the original result")
        if party["sent_bytes"] != parties[2 - role]["received_bytes"]:
            raise ValueError("original party communication counts differ")
    if row["total_ms"] != max(party["total_ms"] for party in parties):
        raise ValueError("original result timing differs from its party records")
    if row["link_bytes"] != sum(party["sent_bytes"] for party in parties):
        raise ValueError("original result bytes differ from its party records")
    if row["mismatches"] != 0 or row["opened_outputs"] != (256 if row["mode"] == "check" else 0):
        raise ValueError("original output evidence differs from its mode")


def original_records(root, config):
    records, missing = [], []
    for network in config["networks"]:
        for backend, nseg in [("sci", 1)] + [("ours", count) for count in config["nseg"]]:
            point = "%s_w28_n%d" % (backend, nseg)
            for tag in ["warmup"] + TAGS:
                path = root / "runs" / network["name"] / point / tag / "result.json"
                if not path.exists():
                    missing.append((network["name"], point, tag))
                    continue
                row = read_json(path)
                if (row["network"], row["backend"], row["nseg"], row["tag"], row["input_bits"], row["count"]) != (
                        network, backend, nseg if backend == "ours" else None, tag, 28, 128):
                    raise ValueError("original result identity differs from its frozen point")
                validate_completed(path, row)
                records.append(dict(row, artifact=str(path.relative_to(root))))
    expected = [(NETWORK, POINT, tag) for tag in TAGS]
    if missing != expected:
        raise ValueError("only the five missing WAN/nseg8 formal repeats may be resumed")
    for path in sorted((root / "boundary").glob("*/result.json")):
        row = read_json(path)
        validate_completed(path, row)
        records.append(dict(row, artifact=str(path.relative_to(root))))
    if len(records) != 60:
        raise ValueError("the original run must retain all 55 results and five boundary records")
    return records


def pending_rows(config):
    network = next(value for value in config["networks"] if value["name"] == NETWORK)
    return [dict(status="not_run", reason="resource_interruption_pending_resume", backend="ours",
                 input_bits=28, comparison_bits=27, nseg=8, count=128, mode="perf", network=network,
                 tag=tag, total_ms=None, completed_calls=0) for tag in TAGS]


def prepare_resume(output, shares_directory):
    root, config, plan = original_state(output)
    resume = root / "resume_wan_n8"
    if resume.exists():
        raise FileExistsError("resume plan already exists; it will not be replaced")
    report_path = Path(plan["resource_guard"]["report"])
    guard = read_json(report_path)
    if guard["status"] != "resource_stopped" or guard["reason"] != "sustained low system memory":
        raise ValueError("narrow recovery requires the recorded whole-run resource interruption")
    records = original_records(root, config)
    first_result_time = (root / "runs/lan_model/sci_w28_n1/warmup/result.json").stat().st_mtime
    shares = input_identity(shares_directory, guard["started_epoch"], first_result_time)
    sources = [output / "missing_width_config.json", root / "plan.json", report_path]
    sources += sorted(path for name in ("runs", "boundary") for path in (root / name).rglob("*") if path.is_file())
    frozen = dict(status="prepared_not_executed", operation="resume_only_five_missing_wan_nseg8_repeats",
                  prepared_epoch=time.time(), source_files=[file_identity(path) for path in sources],
                  build_identity=plan["build_identity"], input_identity=shares,
                  input_provenance="preserved_original_private_files; hashes first recorded after interruption",
                  interrupted_attempt="runs/low_latency_wan_model/ours_w28_n8/repeat1",
                  recovered_input_directory=str(shares_directory.resolve()), tags=TAGS)
    resume.mkdir()
    write_json(resume / "plan.json", frozen)
    (resume / "summary_before_resume.json").write_bytes((root / "summary.json").read_bytes())
    write_json(root / "RESOURCE_INTERRUPTION.json", dict(status="resource_interrupted",
        reason=guard["reason"], guard_report=str(report_path), original_exit_code=guard["exit_code"],
        original_finished_epoch=guard["finished_epoch"], windows_memory=guard.get("windows_memory"),
        completed_formal_repeats=35, resource_skipped_formal_repeats=10, pending_formal_repeats=5,
        interrupted_attempt=frozen["interrupted_attempt"], resume_plan="resume_wan_n8/plan.json"))
    rows = records + pending_rows(config)
    write_json(root / "summary.json", dict(status="resource_interrupted", rows=aggregate_missing(rows, 5), raw=rows,
        resource_interruption="RESOURCE_INTERRUPTION.json", resume_plan="resume_wan_n8/plan.json"))
    return frozen


def execute_resume(output):
    from run_performance import require_memory_guard
    guard = require_memory_guard()
    root, config, original_plan = original_state(output)
    resume = root / "resume_wan_n8"
    frozen = read_json(resume / "plan.json")
    if (resume / "STATUS.json").exists():
        raise FileExistsError("this recovery attempt already started; prior evidence will not be replaced")
    for evidence in frozen["source_files"]:
        if file_identity(Path(evidence["path"])) != evidence:
            raise ValueError("original evidence changed before resumption: " + evidence["path"])
    inputs = input_identity(Path(frozen["recovered_input_directory"]),
        original_plan["resource_guard"]["started_epoch"],
        (root / "runs/lan_model/sci_w28_n1/warmup/result.json").stat().st_mtime)
    if inputs != frozen["input_identity"]:
        raise ValueError("preserved input identity changed before resumption")
    records = original_records(root, config)
    network = next(value for value in config["networks"] if value["name"] == NETWORK)
    runtime = load_runtime()
    status = dict(status="running", started_epoch=time.time(), resource_guard=guard,
                  original_interruption="../RESOURCE_INTERRUPTION.json", completed_formal_repeats=0)
    write_json(resume / "STATUS.json", status)
    for index, tag in enumerate(TAGS):
        destination = resume / "runs" / tag
        if destination.exists():
            raise FileExistsError("recovery output already exists")
        row = runtime.run_point(Path(config["build"]), destination, "ours", 28, 8, network,
            [Path(value["path"]) for value in inputs], 128, "perf", config["timeout_seconds"])
        row.update(tag=tag, address_space_limit_bytes_per_party=config["address_space_limit_bytes_per_party"],
                   resource_guard=guard, recovery="original_preserved_inputs_new_protocol_material",
                   artifact=str((destination / "result.json").relative_to(root)))
        write_json(destination / "result.json", row)
        records.append(row)
        pending = pending_rows(config)[index + 1:]
        state = "complete_with_resource_failures" if not pending and row["status"] == "complete" else "partial_resume"
        write_json(root / "summary.json", dict(status=state, rows=aggregate_missing(records + pending, 5),
            raw=records + pending, resource_interruption="RESOURCE_INTERRUPTION.json", resume_plan="resume_wan_n8/plan.json"))
        if row["status"] != "complete":
            write_json(resume / "STATUS.json", dict(status, status="failed", failed_tag=tag,
                       reason=row.get("reason"), finished_epoch=time.time()))
            raise RuntimeError("recovery protocol failed; original and recovery evidence were retained")
        status["completed_formal_repeats"] += 1
        write_json(resume / "STATUS.json", status)
        print(json.dumps({key: row[key] for key in ("tag", "status", "total_ms")}), flush=True)
    write_json(resume / "STATUS.json", dict(status, status="complete", finished_epoch=time.time()))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    modes = parser.add_mutually_exclusive_group(required=True)
    modes.add_argument("--prepare", action="store_true")
    modes.add_argument("--execute", action="store_true")
    parser.add_argument("--private-shares", type=Path)
    args = parser.parse_args()
    if args.prepare:
        if args.private_shares is None:
            parser.error("--prepare requires the preserved --private-shares directory")
        prepare_resume(args.output, args.private_shares)
    else:
        execute_resume(args.output)


if __name__ == "__main__":
    main()
