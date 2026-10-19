#!/usr/bin/env python3
"""Run fresh online-optimization diagnostics and matched formal measurements."""
import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[3]
RUNTIME = REPO / "project/our/tools/segmented_dcf_runtime"
FROZEN_PLAN = HERE.parent / "paper_folded_activation_20261013/ticket11/frozen-plan-final.json"
DEFAULT_BINARY = HERE / "build/runtime/sdcf-runtime"
DEFAULT_BRIDGE = HERE / "build/core/libsegfss_sci_bridge.so"
DEFAULT_FIXTURE = HERE.parent / "paper_folded_activation_20261013/ticket11/final-build/fixture"
DEFAULT_HIDDEN = HERE.parent / "sci_sirnn_resident_20261012/reference/nonzero_hidden.i16"
MEMORY = 1610612736
ALLOCATOR = {"MALLOC_ARENA_MAX": "2", "MALLOC_TRIM_THRESHOLD_": "1", "MALLOC_MMAP_THRESHOLD_": "131072"}
NETWORKS = {"LAN": {"mbps": 1000, "rtt_ms": 0.5}, "WAN": {"mbps": 400, "rtt_ms": 4}}
OFFLINE_FLAGS = ["--fuse-sign-wrap", "--bit-block-mode", "direct_cot",
                 "--native-ring-cot", "--cache-lut-leaves"]
ONLINE_FLAGS = ["--folded-reuse-input-sign", "--folded-clip-bits", "22",
                "--lut-skip-zero", "--lut-simd"]
MAIN_ROWS = ("ours_online_before", "ours_online_optimized",
             "SCI_SiRNN_preprocessed", "SCI_SiRNN_unprocessed")


def write(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n")


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def specifications(include_ablations=False):
    rows = [("ours_online_before", "parallel-a2-full-bitdec", [])]
    if include_ablations:
        rows += [("ours_input_sign_reuse", "parallel-a2-full-bitdec", ONLINE_FLAGS[:1]),
                 ("ours_input_sign_reuse_clip22", "parallel-a2-full-bitdec", ONLINE_FLAGS[:3]),
                 ("ours_input_sign_reuse_clip22_lutskip", "parallel-a2-full-bitdec", ONLINE_FLAGS[:4])]
    return rows + [("ours_online_optimized", "parallel-a2-full-bitdec", ONLINE_FLAGS),
                   ("SCI_SiRNN_preprocessed", "parallel-a2-SCI_preprocessed", []),
                   ("SCI_SiRNN_unprocessed", "parallel-a2-SCI_preprocessed", [])]


def set_value(command, flag, value):
    if flag in command:
        command[command.index(flag) + 1] = str(value)
    else:
        command.extend([flag, str(value)])


def commands(mode, output, binary=DEFAULT_BINARY, fixture=DEFAULT_FIXTURE,
             hidden=DEFAULT_HIDDEN, include_ablations=False):
    frozen = json.loads(FROZEN_PLAN.read_text())
    rows = []
    for identity, source, flags in specifications(include_ablations):
        source_row = next(row for row in frozen["rows"] if row["id"] == source)
        for nonzero in ((False, True) if mode == "qualify" else (False,)):
            name = identity + ("_nonzero" if nonzero else "_zero") if mode == "qualify" else identity
            destination = output / name
            command = [str(value).replace("{output}", str(destination)) for value in source_row["command"]]
            command[:2] = [sys.executable, str(RUNTIME / "run.py")]
            for flag, value in (("--binary", binary), ("--fixture", fixture),
                                ("--queries", 2 if mode == "qualify" else 10),
                                ("--warmup", 0), ("--network-projection", "both")):
                set_value(command, flag, value)
            backend = source_row["backend"]
            if identity.startswith("ours_"):
                set_value(command, "--bit-block-mode", "direct_cot")
                command += [value for value in OFFLINE_FLAGS if value not in ("--bit-block-mode", "direct_cot")]
                command += flags
            if identity == "SCI_SiRNN_unprocessed":
                backend = "SCI"
                set_value(command, "--backend", backend)
                command.remove("--native-preprocessing")
            if mode == "qualify":
                command.append("--diagnostic")
            if nonzero:
                command += ["--hidden", str(hidden)]
            rows.append({"id": name, "configuration": identity, "backend": backend,
                         "activation_mode": "folded" if identity.startswith("ours_") else "sci",
                         "native_preprocessing_expected": backend != "SCI", "command": command,
                         "output": str(destination), "initial_hidden": "nonzero" if nonzero else "zero"})
    return rows


def live_build(binary, bridge):
    return {"binary": digest(binary), "bridge": digest(bridge)}


def input_identity(fixture, hidden):
    return {"fixture": {name: digest(fixture / name) for name in
                        ("client_input.i16", "server_model.i16", "contract.json")},
            "hidden": digest(hidden)}


def protocol_command(command):
    # Remove only diagnostic/sample/output selectors. Every implementation,
    # workload, fixture and resource option must match qualification exactly.
    result, index = [], 0
    while index < len(command):
        if command[index] == "--diagnostic":
            index += 1
        elif command[index] in ("--output", "--queries", "--hidden"):
            index += 2
        else:
            result.append(command[index])
            index += 1
    return result


def loaded_bridge(binary, bridge, environment):
    output = subprocess.check_output(["ldd", str(binary)], env=environment, text=True)
    matches = [line.split("=>", 1)[1].strip().split()[0] for line in output.splitlines()
               if "libsegfss_sci_bridge.so =>" in line]
    if len(matches) != 1 or not Path(matches[0]).is_file() or digest(Path(matches[0])) != digest(bridge):
        raise ValueError("runtime does not load the declared bridge: " + output)
    return str(Path(matches[0]).resolve())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("qualify", "formal"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--binary", type=Path, default=DEFAULT_BINARY)
    parser.add_argument("--bridge", type=Path, default=DEFAULT_BRIDGE)
    parser.add_argument("--fixture", type=Path, default=DEFAULT_FIXTURE)
    parser.add_argument("--hidden", type=Path, default=DEFAULT_HIDDEN)
    parser.add_argument("--qualification", type=Path, default=HERE / "qualification")
    parser.add_argument("--include-ablations", action="store_true")
    parser.add_argument("--dry-run", action="store_true", help="write commands without running parties")
    args = parser.parse_args()
    binary, bridge, fixture, hidden = (path.resolve() for path in
                                      (args.binary, args.bridge, args.fixture, args.hidden))
    output = args.output.resolve()
    identity = live_build(binary, bridge)
    inputs = input_identity(fixture, hidden)
    rows = commands(args.mode, output, binary, fixture, hidden, args.include_ablations)
    environment = {key: value for key, value in os.environ.items()
                   if not key.startswith(("SEGFSS_", "SDCF_", "MALLOC_")) and key not in
                   ("OMP_THREAD_LIMIT", "OMP_DYNAMIC", "GLIBC_TUNABLES", "GOOGLE30_PROXY_OFFSET", "SWEEP_CONNECT_OFFSET")}
    environment.update(OMP_NUM_THREADS="1", LD_LIBRARY_PATH=str(bridge.parent), **ALLOCATOR)
    actual_bridge = loaded_bridge(binary, bridge, environment)
    if args.mode == "formal" and not args.dry_run:
        qualification = json.loads((args.qualification / "STATUS.json").read_text())
        qplan = json.loads((args.qualification / "PLAN.json").read_text())
        required = {row["configuration"] for row in rows}
        passed = {state["id"] for state in qualification["rows"]
                  if state["status"] == "complete" and state.get("integer_audit") == "passed"}
        if qualification["status"] != "complete" or qualification["build"] != identity:
            raise ValueError("same-build qualification has not completed")
        if qplan["fixture"] != str(fixture) or qplan["hidden"] != str(hidden):
            raise ValueError("qualification fixture/hidden differs from formal inputs")
        if qplan.get("input_identity") != inputs:
            raise ValueError("fixture or hidden contents changed after qualification")
        if not all(config + suffix in passed for config in required for suffix in ("_zero", "_nonzero")):
            raise ValueError("every formal configuration requires zero/nonzero integer qualification")
        for row in rows:
            qualified = next(value for value in qplan["rows"] if value["id"] == row["configuration"] + "_zero")
            if protocol_command(qualified["command"]) != protocol_command(row["command"]):
                raise ValueError("formal protocol differs from the qualified command: " + row["id"])
    output.mkdir(parents=True, exist_ok=False)
    plan = {"mode": args.mode, "rows": rows, "build": identity, "binary": str(binary),
            "bridge": str(bridge), "loaded_bridge": actual_bridge, "fixture": str(fixture),
            "hidden": str(hidden), "input_identity": inputs, "allocator": ALLOCATOR, "networks": NETWORKS,
            "qualification_directory": str(args.qualification.resolve()) if args.mode == "formal" else None,
            "measurement": "unshaped localhost; measured process CPU and wall; causal communication projections",
            "aggregation": "10 fresh formal queries without warmup; max-party wall/CPU and total-party CPU; two-party sent bytes",
            "unprocessed_policy": "backend SCI without native preprocessing; protocol generation remains inside online; measured offline retained",
            "source_revision": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO, text=True).strip()}
    write(output / "PLAN.json", plan)
    status = {"status": "planned" if args.dry_run else "running", "build": identity,
              "rows": [{"id": row["id"], "status": "not_started"} for row in rows]}
    write(output / "STATUS.json", status)
    if args.dry_run:
        print(json.dumps({"status": "planned", "rows": len(rows), "plan": str(output / "PLAN.json")}))
        return 0
    with Path("/tmp/sdcf-measurement.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        for row, state in zip(rows, status["rows"]):
            if live_build(binary, bridge) != identity:
                raise ValueError("build changed during execution")
            if input_identity(fixture, hidden) != inputs:
                raise ValueError("inputs changed during execution")
            state.update(status="running", started_epoch=time.time())
            write(output / "STATUS.json", status)
            print("starting", row["id"], flush=True)
            launcher = ["/usr/bin/time", "-v", "-o", str(output / (row["id"] + ".controller.txt")),
                        "/usr/bin/prlimit", "--as=" + str(MEMORY), "--"] + row["command"]
            with (output / (row["id"] + ".runner.log")).open("w") as log:
                completed = subprocess.run(launcher, cwd=REPO, env=environment, stdout=log, stderr=subprocess.STDOUT)
            result_path = Path(row["output"]) / "result.json"
            result = json.loads(result_path.read_text()) if result_path.exists() else {}
            state.update(status=result.get("status", "failed"), exit_code=completed.returncode,
                         reason=result.get("reason"), finished_epoch=time.time())
            if completed.returncode or state["status"] != "complete":
                state["status"] = status["status"] = "failed"
                write(output / "STATUS.json", status)
                print("failed", row["id"], state, flush=True)
                return 1
            if args.mode == "qualify":
                audit_command = [sys.executable, str(HERE / "audit_integer.py"), "--directory", row["output"],
                                 "--fixture", str(fixture), "--activation-mode", row["activation_mode"],
                                 "--output", str(Path(row["output"]) / "INTEGER_AUDIT.json")]
                audit_result = subprocess.run(audit_command, cwd=REPO, env=environment, capture_output=True, text=True)
                (output / (row["id"] + ".audit.log")).write_text(audit_result.stdout + audit_result.stderr)
                state["integer_audit"] = "passed" if audit_result.returncode == 0 else "failed"
                if audit_result.returncode:
                    state["status"] = status["status"] = "failed"
                    write(output / "STATUS.json", status)
                    print(audit_result.stdout + audit_result.stderr, flush=True)
                    return 1
            write(output / "STATUS.json", status)
            print("completed", row["id"], flush=True)
    status["status"] = "complete"
    write(output / "STATUS.json", status)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
