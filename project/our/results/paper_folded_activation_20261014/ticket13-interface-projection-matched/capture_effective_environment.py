#!/usr/bin/env python3
"""Retain a whitelist of the two live protocol processes' fixed environment."""
import json
from pathlib import Path
import time

HERE = Path(__file__).resolve().parent
OUTPUT = HERE / "SCI_preprocessed"
NAMES = ("MALLOC_ARENA_MAX", "MALLOC_TRIM_THRESHOLD_", "MALLOC_MMAP_THRESHOLD_",
         "GLIBC_TUNABLES", "OMP_NUM_THREADS", "OMP_DYNAMIC", "OMP_MAX_ACTIVE_LEVELS",
         "SDCF_BACKEND", "SEGFSS_GENERATION_BUDGET_BYTES")


def capture():
    rows = []
    for root in Path("/proc").iterdir():
        if not root.name.isdigit():
            continue
        try:
            if (root / "cwd").resolve() != OUTPUT:
                continue
            argv = (root / "cmdline").read_bytes().decode().rstrip("\0").split("\0")
            if not any("sdcf-runtime" in part for part in argv):
                continue
            environment = dict(part.split("=", 1) for part in
                (root / "environ").read_bytes().decode().rstrip("\0").split("\0") if "=" in part)
            rows.append(dict(pid=int(root.name), argv=argv,
                whitelisted_effective_environment={key: environment.get(key) for key in NAMES}))
        except (OSError, ValueError):
            continue
    return rows


def main():
    deadline = time.monotonic() + 1800
    while time.monotonic() < deadline:
        if OUTPUT.exists():
            rows = capture()
            if len(rows) == 2:
                report = dict(schema_version=1,
                    source="live /proc/<pid>/environ and cmdline; whitelist only",
                    observed_epoch=time.time(), rows=rows)
                (OUTPUT / "EFFECTIVE_ENVIRONMENT.json").write_text(
                    json.dumps(report, indent=2, sort_keys=True) + "\n")
                print("captured SCI_preprocessed effective environment", flush=True)
                return 0
        time.sleep(0.5)
    raise RuntimeError("the two SCI protocol processes were not observed")


if __name__ == "__main__":
    raise SystemExit(main())
