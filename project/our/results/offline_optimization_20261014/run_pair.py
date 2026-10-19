#!/usr/bin/env python3
"""Run the declared two-party correctness executables and retain both logs."""
import argparse
import json
import os
from pathlib import Path
import subprocess

HERE = Path(__file__).resolve().parent
CORE = HERE / "build/core"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("case", choices=("ot", "fusion", "bridge", "cache"))
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--port", type=int, default=34000)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    if args.case == "ot":
        binary = CORE / "test/correctness/SEGFSS_CORRECTNESS_TEST"
        commands = [[str(binary), "r=" + str(role), "p=" + str(args.port), "t=15"] for role in (2, 3)]
    elif args.case == "fusion":
        binary = CORE / "test/correctness/SCI_TRUNCATION_TEST"
        commands = [[str(binary), str(role), str(args.port), "fusion"] for role in (2, 3)]
    elif args.case == "cache":
        binary = CORE / "test/correctness/DPF_LUT_BATCH_TEST"
        commands = [[str(binary), "r=" + str(role), "p=" + str(args.port),
                     "mode=4", "q=10", "count=33", "tile=32"] for role in (2, 3)]
    else:
        binary = CORE / "SCI_SEGFSS_CONVERSION_TEST"
        commands = [[str(binary), str(role), str(args.port), "0"] for role in (1, 2)]
    environment = {key: value for key, value in os.environ.items()
                   if not key.startswith(("SEGFSS_", "SDCF_"))}
    environment.update(OMP_NUM_THREADS="1", SEGFSS_CGGM_THREADS="1")
    if args.case == "bridge":
        environment.update(SEGFSS_FUSE_SIGN_WRAP="1", SEGFSS_NATIVE_RING_COT="1",
                           SEGFSS_BIT_BLOCK_MODE="direct_cot", SEGFSS_MATERIAL_LAYOUT="contiguous",
                           SEGFSS_PREFETCH_BYTES="8388608", SEGFSS_TEST_MULTI_CONTEXT="1")
    logs = [(args.output / ("party%d.log" % role)).open("w") for role in (1, 2)]
    children = []
    try:
        children = [subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT, env=environment)
                    for command, log in zip(commands, logs)]
        codes = [child.wait(timeout=240) for child in children]
    except subprocess.TimeoutExpired:
        codes = [child.poll() for child in children]
    finally:
        for child in children:
            if child.poll() is None:
                child.kill()
                child.wait()
        for log in logs:
            log.close()
    passed = codes == [0, 0]
    (args.output / "result.json").write_text(json.dumps(dict(
        status="passed" if passed else "failed", exit_codes=codes,
        commands=commands, explicit_environment={key: value for key, value in environment.items()
                                               if key.startswith(("SEGFSS_", "OMP_"))}), indent=2) + "\n")
    print(args.case, "passed" if passed else "failed", codes, flush=True)
    if not passed:
        for role in (1, 2):
            print((args.output / ("party%d.log" % role)).read_text()[-3000:])
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
