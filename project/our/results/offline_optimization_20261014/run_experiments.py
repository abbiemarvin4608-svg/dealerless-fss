#!/usr/bin/env python3
"""Run matched fresh-material diagnostics and offline optimization ablations."""
import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[3]
RUNTIME = REPO / "project/our/tools/segmented_dcf_runtime"
HISTORICAL_PLAN = HERE.parent / "paper_folded_activation_20261013/ticket11/frozen-plan-final.json"
BINARY = HERE / "build/runtime/sdcf-runtime"
BRIDGE = HERE / "build/core/libsegfss_sci_bridge.so"
MEMORY = 1610612736
ALLOCATOR = dict(MALLOC_ARENA_MAX="2", MALLOC_TRIM_THRESHOLD_="1", MALLOC_MMAP_THRESHOLD_="131072")
HIDDEN = HERE.parent / "sci_sirnn_resident_20261012/reference/nonzero_hidden.i16"


def write(path, value):
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def specifications():
    return [
        ("ours_segmented_before", "parallel-a2-segmented-bitdec", []),
        ("ours_full10", "parallel-a2-full-bitdec", []),
        ("ours_full10_fused", "parallel-a2-full-bitdec", ["--fuse-sign-wrap"]),
        ("ours_full10_fused_blockcot", "parallel-a2-full-bitdec", ["--fuse-sign-wrap", "--bit-block-mode", "direct_cot"]),
        ("ours_full10_fused_bothcot", "parallel-a2-full-bitdec", ["--fuse-sign-wrap", "--bit-block-mode", "direct_cot", "--native-ring-cot"]),
        ("ours_optimized", "parallel-a2-full-bitdec", ["--fuse-sign-wrap", "--bit-block-mode", "direct_cot", "--native-ring-cot", "--cache-lut-leaves"]),
        ("SCI_SiRNN", "parallel-a2-SCI_preprocessed", []),
    ]


def commands(mode, output):
    historical = json.loads(HISTORICAL_PLAN.read_text())
    rows = []
    for identity, source, additions in specifications():
        if mode == "qualify" and identity not in ("ours_full10", "ours_optimized", "SCI_SiRNN"):
            continue
        frozen = next(row for row in historical["rows"] if row["id"] == source)
        hidden_modes = (False, True) if mode == "qualify" else (False,)
        for nonzero in hidden_modes:
            name = identity + ("_nonzero" if nonzero else "_zero") if mode == "qualify" else identity
            destination = output / name
            command = [str(part).replace("{output}", str(destination)) for part in frozen["command"]]
            command[1] = str(RUNTIME / "run.py")
            command[command.index("--binary") + 1] = str(BINARY)
            command[command.index("--queries") + 1] = "2" if mode == "qualify" else "10"
            for flag in ("--bit-block-mode",):
                if flag in additions:
                    command[command.index(flag) + 1] = additions[additions.index(flag) + 1]
            filtered = [value for index, value in enumerate(additions)
                        if value != "--bit-block-mode" and (index == 0 or additions[index - 1] != "--bit-block-mode")]
            command += filtered + ["--network-projection", "both"]
            if mode == "qualify":
                command += ["--diagnostic"]
            if nonzero:
                command += ["--hidden", str(HIDDEN)]
            rows.append(dict(id=name, configuration=identity, backend=frozen["backend"],
                             command=command, output=str(destination),
                             activation_mode="sci" if identity == "SCI_SiRNN" else "folded"))
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("qualify", "formal"))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    if args.mode == "formal":
        qualification = json.loads((HERE / "qualification/STATUS.json").read_text())
        if qualification["status"] != "complete":
            raise ValueError("current diagnostics have not passed")
        if qualification["build"] != {"binary": digest(BINARY), "bridge": digest(BRIDGE)}:
            raise ValueError("qualified build changed")
    rows = commands(args.mode, output)
    environment = {key: value for key, value in os.environ.items()
                   if not key.startswith(("SEGFSS_", "SDCF_", "MALLOC_")) and key not in
                   ("OMP_THREAD_LIMIT", "OMP_DYNAMIC", "GLIBC_TUNABLES", "GOOGLE30_PROXY_OFFSET", "SWEEP_CONNECT_OFFSET")}
    environment.update(OMP_NUM_THREADS="1", **ALLOCATOR)
    identity = dict(binary=digest(BINARY), bridge=digest(BRIDGE))
    plan = dict(mode=args.mode, rows=rows, build=identity, allocator=ALLOCATOR,
                networks={"LAN": dict(mbps=1000, rtt_ms=0.5), "WAN": dict(mbps=400, rtt_ms=4)},
                method="unshaped localhost with measured bytes and causal round projections",
                aggregation="10 fresh queries per formal row, no warmup; two-party max wall; total sends counted once",
                source_revision=subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO, text=True).strip())
    write(output / "PLAN.json", plan)
    status = dict(status="running", build=identity, rows=[dict(id=row["id"], status="not_started") for row in rows])
    write(output / "STATUS.json", status)
    with Path("/tmp/sdcf-measurement.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        for row, state in zip(rows, status["rows"]):
            if identity != dict(binary=digest(BINARY), bridge=digest(BRIDGE)):
                raise ValueError("build changed during execution")
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
                fixture = Path(row["command"][row["command"].index("--fixture") + 1])
                audit = ["python3", str(RUNTIME / "audit_folded_step.py"), "--directory", row["output"],
                         "--artifact", str(fixture), "--fixture", str(fixture),
                         "--activation-mode", row["activation_mode"], "--output", str(Path(row["output"]) / "INTEGER_AUDIT.json")]
                audit_result = subprocess.run(audit, cwd=REPO, env=environment, capture_output=True, text=True)
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
