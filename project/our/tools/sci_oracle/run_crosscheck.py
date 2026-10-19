#!/usr/bin/env python3
"""Cross-check the frozen plaintext numeric oracle with real SCI two-party APIs."""

import argparse
import json
from pathlib import Path
import re
import socket
import subprocess


def run(args):
    here = Path(__file__).resolve().parent
    source = here.parents[1] / "src"
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=True)
    sci = args.sci.resolve()
    libs = sci / "build/lib"
    command = ["g++", "-O2", "-std=c++17", "-pthread", "-maes", "-msse4.1",
               "-mavx2", "-mrdseed", "-fopenmp", "-DSCI_OT=1",
               "-Wno-deprecated-declarations", "-I" + str(source),
               "-I" + str(sci / "src"), "-I" + str(sci / "extern/eigen"),
               str(here / "crosscheck.cpp"),
               str(sci / "src/BuildingBlocks/truncation.cpp"),
               str(sci / "src/BuildingBlocks/value-extension.cpp"),
               "-Wl,--start-group"]
    command += [str(libs / ("libSCI-" + name + ".a")) for name in
                ["OT", "LinearOT", "GC", "Math", "BuildingBlocks", "FloatingPoint"]]
    command += ["-Wl,--end-group", "-lssl", "-lcrypto", "-lgmp", "-o", str(out / "crosscheck")]
    (out / "build_command.json").write_text(json.dumps(command, indent=2) + "\n")
    with (out / "build.log").open("w") as log:
        subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, check=True)
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
    processes, logs = [], []
    try:
        for party in (1, 2):
            log = (out / f"party{party}.log").open("w")
            logs.append(log)
            processes.append(subprocess.Popen(
                [str(out / "crosscheck"), str(party), str(port), str(out / "vectors.csv")],
                stdout=log, stderr=subprocess.STDOUT))
        codes = [p.wait(timeout=args.timeout) for p in processes]
    finally:
        for p in processes:
            if p.poll() is None:
                p.kill()
            p.wait()
        for log in logs:
            log.close()
    summary = re.search(r"SUMMARY checked=(\d+) groups=(\d+) widened_groups=(\d+) native_direct_unsupported=(\d+) status=passed",
                        (out / "party2.log").read_text())
    report = {"status": "passed" if codes == [0, 0] and summary else "failed",
              "party_exit_codes": codes, "DIV_RESCALING": True,
              "sci_commit": subprocess.check_output(["git", "-C", str(sci), "rev-parse", "HEAD"], text=True).strip(),
              "oracle_header": "src/sci_numeric_oracle.h",
              "small_domain": "all residues and all shifts for input_bits=1..8; three additive decompositions per residue",
              "wide_domain": "boundary values and 64 fixed-seed random residues per width; shifts 0,1,bits/2,bits-1",
              "vector_file": "vectors.csv contains first 12 residues per group with all three share decompositions; exhaustive remainder is reproducible from crosscheck.cpp",
              "native_direct_unsupported": "signed div_pow2(bits=64, shift=63); oracle literal checked, not native validated",
              "edge_path": "signed div_pow2 shift=bits-1 with bits<64 is checked via SCI s_extend to bits+1, div_pow2, output reduction",
              "timeout_seconds_per_party": args.timeout}
    if summary:
        report.update(zip(["checked_values", "groups", "widened_groups", "unsupported_groups"], map(int, summary.groups())))
    (out / "summary.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    if report["status"] != "passed":
        raise SystemExit(1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sci", type=Path, default=Path("/home/jiang/EzPC/SCI"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--timeout", type=int, default=180)
    run(parser.parse_args())
