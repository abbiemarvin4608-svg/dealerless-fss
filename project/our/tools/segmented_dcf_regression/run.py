#!/usr/bin/env python3
"""Run the SDCF numeric, comparison and truncation suites with one evidence report."""

import argparse
from contextlib import ExitStack
import json
import math
import os
from pathlib import Path
import re
import resource
import signal
import socket
import subprocess
import sys
import time


HERE = Path(__file__).resolve().parent
PROJECT = HERE.parents[1]
ROOT = PROJECT.parents[1]
SUITES = ("oracle_literals", "native_oracle", "comparison_local", "comparison_socket", "adapter_socket")


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def stop(process):
    try:
        os.killpg(process.pid, signal.SIGTERM)
    except ProcessLookupError:
        pass
    try:
        process.wait(timeout=1)
    except subprocess.TimeoutExpired:
        os.killpg(process.pid, signal.SIGKILL)
        process.wait()


def execute(commands, output, timeout, memory_gib=None):
    output.mkdir(parents=True, exist_ok=True)
    write_json(output / "commands.json", commands)
    start = time.monotonic()
    children, logs = [], []
    status, reason = "passed", None

    def limits():
        resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
        if memory_gib is not None:
            cap = int(memory_gib * (1 << 30))
            resource.setrlimit(resource.RLIMIT_AS, (cap, cap))

    try:
        with ExitStack() as stack:
            for index, command in enumerate(commands):
                path = output / ("process%d.log" % (index + 1))
                logs.append(path)
                log = stack.enter_context(path.open("w"))
                child = subprocess.Popen(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT,
                                         start_new_session=True, preexec_fn=limits)
                children.append(child)
                stack.callback(stop, child)
            while any(child.poll() is None for child in children):
                if any(child.poll() not in (None, 0) for child in children):
                    status, reason = "failed", "a process exited unsuccessfully"
                    break
                if time.monotonic() - start > timeout:
                    status, reason = "timeout", "suite wall-time budget exceeded"
                    break
                time.sleep(0.02)
    except OSError as error:
        status, reason = "unavailable", str(error)
    codes = [child.returncode for child in children]
    text = "\n".join(path.read_text(errors="replace") for path in logs)
    text += "\n" + "\n".join(path.read_text(errors="replace") for path in output.rglob("party*.log"))
    if status == "passed" and (len(codes) != len(commands) or any(code != 0 for code in codes)):
        status, reason = "failed", "every process must exit zero"
    if status == "failed" and re.search(r"bad_alloc|Cannot allocate memory|MemoryError", text):
        status, reason = "oom", "allocation failure under declared address-space cap"
    mismatch = next((line for line in text.splitlines()
                     if re.search(r"MISMATCH|\bFAIL\b|mismatch|Assertion .* failed", line)), None)
    if status == "passed" and mismatch:
        status, reason = "failed", "failure marker found despite a successful process exit"
    result = {"status": status, "reason": reason, "exit_codes": codes,
              "elapsed_seconds": time.monotonic() - start,
              "timeout_seconds": timeout, "memory_gib_per_process": memory_gib,
              "first_mismatch": mismatch, "logs": [path.name for path in logs]}
    write_json(output / "execution.json", result)
    return result


def available_port():
    # SCI IOPack also binds base+50 and base+100 for OT channels.
    for base in range(25000, 55000, 137):
        sockets = []
        try:
            for offset in (0, 1, 2, 3, 50, 100):
                sock = socket.socket()
                sockets.append(sock)
                sock.bind(("127.0.0.1", base + offset))
            return base
        except OSError:
            pass
        finally:
            for sock in sockets:
                sock.close()
    raise RuntimeError("no free local protocol port range")


def coverage_for(name, output):
    text = (output / "process1.log").read_text()
    if name == "oracle_literals":
        match = re.search(r"PASS literal vectors=(\d+) invalid contracts=(\d+)", text)
        if not match:
            raise ValueError("missing oracle literal completion record")
        return [{"backend": "plaintext_oracle", "execution": "local_literals",
                 "numeric_mode": "separate_sci_and_floor_literals", "values": int(match[1]),
                 "invalid_contracts": int(match[2]), "status": "passed"}]
    if name == "native_oracle":
        report = json.loads((output / "native" / "summary.json").read_text())
        if report.get("status") != "passed":
            raise ValueError("native SCI cross-check did not pass")
        rows = [{"backend": "SCI_native", "execution": "socket_2pc", "numeric_mode": "sci_compatible",
                 "values": report["checked_values"], "configurations": report["groups"],
                 "widened_groups": report["widened_groups"], "sci_commit": report["sci_commit"], "status": "passed"}]
        if report["unsupported_groups"]:
            rows.append({"backend": "SCI_native_direct", "execution": "not_executed",
                         "numeric_mode": "sci_compatible", "values": None,
                         "configurations": report["unsupported_groups"], "status": "unsupported",
                         "reason": report["native_direct_unsupported"]})
        return rows
    if name == "comparison_local":
        match = re.search(r"local_2pc_simulation comparisons=(\d+) assertions=(\d+) PASS", text)
        if not match or "segmented DCF: PASS" not in text:
            raise ValueError("missing local comparison completion record")
        return [{"backend": "segmented_and_dealer_full_tree", "execution": "local_2pc_simulation",
                 "numeric_mode": "unsigned_comparison", "values": int(match[1]), "assertions": int(match[2]),
                 "domain": "all alpha/query/nseg for bits 1..8", "status": "passed"}]
    if name == "comparison_socket":
        match = re.search(r"exhaustive_comparisons=(\d+) legacy_fulltree_comparisons=(\d+) PASS", text)
        if not match or "segmented DCF: PASS" not in text:
            raise ValueError("missing socket comparison completion record")
        rows = []
        for bits, nseg, count in re.findall(r"socket_2pc bits=(\d+) nseg=(\d+) count=(\d+)", text):
            rows.append({"backend": "full_tree" if int(nseg) == 1 else "segmented",
                         "execution": "socket_2pc", "numeric_mode": "unsigned_comparison",
                         "bits": int(bits), "nseg": int(nseg), "values": int(count), "status": "passed"})
        rows.append({"backend": "legacy_dfss_full_tree", "execution": "socket_2pc",
                     "numeric_mode": "unsigned_comparison", "values": int(match[2]),
                     "domain": "all alpha/query for bits 1..5", "status": "passed"})
        return rows
    matches = re.findall(r"COVERAGE backend=(\w+) numeric_mode=(\w+) configurations=(\d+) values=(\d+)", text)
    summary = re.search(r"SCI truncation: PASS configurations=(\d+) values=(\d+)", text)
    if not matches or not summary:
        raise ValueError("missing adapter coverage or completion record")
    rows = [{"backend": backend, "execution": "socket_2pc", "numeric_mode": mode,
             "configurations": int(configs), "values": int(values), "status": "passed"}
            for backend, mode, configs, values in matches]
    if sum(row["configurations"] for row in rows) != int(summary[1]) or sum(row["values"] for row in rows) != int(summary[2]):
        raise ValueError("adapter coverage totals differ from completed assertions")
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--build-dir", type=Path, default=PROJECT / "build")
    parser.add_argument("--sci", type=Path, default=Path("/home/jiang/EzPC/SCI"))
    parser.add_argument("--suites", nargs="+", choices=SUITES, default=list(SUITES))
    parser.add_argument("--no-build", action="store_true", help="use binaries from an already configured build of this checkout")
    parser.add_argument("--timeout", type=float, default=240)
    parser.add_argument("--memory-gib", type=float, default=1.5)
    args = parser.parse_args()
    if not math.isfinite(args.timeout) or args.timeout <= 0 or not math.isfinite(args.memory_gib) or args.memory_gib <= 0:
        parser.error("resource budgets must be finite and positive")
    output, build = args.output.resolve(), args.build_dir.resolve()
    output.mkdir(parents=True, exist_ok=False)
    identity = {"git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
                "tracked_status": subprocess.check_output(["git", "status", "--short", "--untracked-files=no"], cwd=ROOT, text=True),
                "command": [sys.executable, *sys.argv], "checkout": str(ROOT),
                "build_dir": str(build), "sci": str(args.sci.resolve()),
                "sci_git_commit": subprocess.check_output(["git", "-C", str(args.sci), "rev-parse", "HEAD"], text=True).strip(),
                "sci_reference_status": subprocess.check_output(["git", "-C", str(args.sci), "status", "--short", "--",
                    "src/BuildingBlocks/truncation.cpp", "src/BuildingBlocks/value-extension.cpp", "src/defines.h"], text=True)}
    patch = subprocess.check_output(["git", "diff", "HEAD", "--", "project/our"], cwd=ROOT, text=True)
    if patch:
        (output / "source.patch").write_text(patch)
        identity["source_patch"] = "source.patch"
    write_json(output / "identity.json", identity)
    report = {"status": "passed", "suites": {}, "coverage": [], "first_mismatch": None,
              "selected_suites": args.suites, "unselected_suites": [name for name in SUITES if name not in args.suites],
              "evidence_scope": "numeric/primitive regression; local simulation is separate from socket execution"}
    if not args.no_build:
        configure = execute([["cmake", "-S", str(PROJECT), "-B", str(build), "-DCMAKE_BUILD_TYPE=Release",
                              "-DEZPC_ROOT=" + str(args.sci.resolve().parent)]], output / "configure", args.timeout)
        compilation = execute([["cmake", "--build", str(build), "--target", "SCI_NUMERIC_ORACLE_TEST",
                                "SEGMENTED_DCF_TEST", "SCI_TRUNCATION_TEST", "-j2"]],
                              output / "build", args.timeout) if configure["status"] == "passed" else configure
    else:
        cache = build / "CMakeCache.txt"
        valid = cache.exists() and "CMAKE_HOME_DIRECTORY:INTERNAL=" + str(PROJECT) + "\n" in cache.read_text()
        compilation = {"status": "passed" if valid else "unavailable",
                       "reason": None if valid else "build is missing or belongs to another checkout"}
    report["build"] = compilation
    binary = build / "test" / "correctness"
    for name in args.suites:
        if compilation["status"] != "passed":
            result = {"status": "unavailable", "reason": "build did not complete", "values": None}
        else:
            destination = output / name
            if name == "native_oracle":
                commands = [[sys.executable, str(PROJECT / "tools/sci_oracle/run_crosscheck.py"),
                             "--sci", str(args.sci.resolve()), "--output", str(destination / "native"),
                             "--timeout", str(max(1, int(args.timeout)))]]
            elif name.endswith("socket"):
                executable = "SEGMENTED_DCF_TEST" if name == "comparison_socket" else "SCI_TRUNCATION_TEST"
                port = available_port()
                commands = [[str(binary / executable), str(role), str(port)] for role in (2, 3)]
            else:
                executable = "SCI_NUMERIC_ORACLE_TEST" if name == "oracle_literals" else "SEGMENTED_DCF_TEST"
                commands = [[str(binary / executable)]]
            result = execute(commands, destination, args.timeout, args.memory_gib)
            if result["status"] == "passed":
                try:
                    rows = coverage_for(name, destination)
                    for row in rows:
                        row["suite"] = name
                    report["coverage"].extend(rows)
                except (OSError, ValueError, KeyError) as error:
                    result.update(status="failed", reason="result validation: " + str(error))
        report["suites"][name] = result
        if result["status"] != "passed":
            report["coverage"].append({"suite": name, "status": result["status"], "values": None,
                                       "reason": result.get("reason")})
            report["status"] = "failed"
        if report["first_mismatch"] is None and result.get("first_mismatch"):
            report["first_mismatch"] = {"suite": name, "detail": result["first_mismatch"]}
        write_json(output / "report.json", report)
    if report["status"] == "passed":
        report["status"] = "partial" if report["unselected_suites"] else (
            "passed_with_unsupported" if any(row["status"] == "unsupported" for row in report["coverage"]) else "passed")
    write_json(output / "report.json", report)
    print(json.dumps({"status": report["status"], "report": str(output / "report.json")}))
    return 1 if report["status"] == "failed" else 0


if __name__ == "__main__":
    sys.exit(main())
