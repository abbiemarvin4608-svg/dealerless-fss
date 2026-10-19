#!/usr/bin/env python3
"""Execute the two frozen LAN/WAN projection rows in one clean environment."""
import fcntl
import json
import os
from pathlib import Path
import subprocess
import time

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[4]
RUNTIME = REPO / "project/our/tools/segmented_dcf_runtime"
FROZEN = REPO / "project/our/results/paper_folded_activation_20261013/ticket11/frozen-plan-final.json"
MEMORY = 1610612736
ALLOCATOR = {
    "MALLOC_ARENA_MAX": "2",
    "MALLOC_TRIM_THRESHOLD_": "1",
    "MALLOC_MMAP_THRESHOLD_": "131072",
}
SELECTED = ("parallel-a2-segmented-bitdec", "parallel-a2-SCI_preprocessed")


def write(path, value):
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def main():
    plan = json.loads(FROZEN.read_text())
    rows = []
    for identity in SELECTED:
        frozen = next(row for row in plan["rows"] if row["id"] == identity)
        backend = frozen["backend"]
        output = HERE / backend
        if output.exists():
            raise ValueError("refusing to overwrite existing run: " + str(output))
        command = [str(part).replace("{output}", str(output)) for part in frozen["command"]]
        command[1] = str(RUNTIME / "run.py")
        command += ["--network-projection", "both"]
        resource_file = HERE / (backend + ".controller.resources.txt")
        launcher = ["/usr/bin/time", "-v", "-o", str(resource_file),
                    "/usr/bin/prlimit", "--as=" + str(MEMORY), "--"] + command
        rows.append(dict(frozen_row_id=identity, backend=backend, output=str(output),
                         controller_argv=command, launch_argv=launcher,
                         controller_address_space_limit_bytes=MEMORY,
                         protocol_timeout_seconds=600,
                         runner_log=str(HERE / (backend + ".runner.log"))))
    excluded_names = ("GOOGLE30_PROXY_OFFSET", "SWEEP_CONNECT_OFFSET", "OMP_THREAD_LIMIT",
                      "OMP_DYNAMIC", "GLIBC_TUNABLES")
    excluded_prefixes = ("SEGFSS_", "SDCF_", "MALLOC_")
    environment = {key: value for key, value in os.environ.items()
                   if not key.startswith(excluded_prefixes) and key not in excluded_names}
    environment["OMP_NUM_THREADS"] = "1"
    environment.update(ALLOCATOR)
    write(HERE / "OUTER_COMMANDS.json", dict(
        schema_version=1, frozen_plan=str(FROZEN), working_directory=str(REPO),
        order=[row["backend"] for row in rows], rows=rows,
        environment_sanitization=dict(remove_prefixes=list(excluded_prefixes),
                                      remove_names=list(excluded_names)),
        allocator_environment=ALLOCATOR,
        explicit_environment={"OMP_NUM_THREADS": "1", **ALLOCATOR},
        observed_removed_environment_names=sorted(key for key in os.environ
            if key.startswith(excluded_prefixes) or key in excluded_names),
        controller_environment_assertions=dict(glibc_tunables_absent="GLIBC_TUNABLES" not in environment,
                                               ld_preload_present="LD_PRELOAD" in environment),
        changes_from_frozen_rows=["entry point is current runtime run.py",
                                  "--network-projection both added"],
        measurement_lock="/tmp/sdcf-measurement.lock"))
    status = dict(schema_version=1, status="running", rows=[dict(backend=row["backend"],
        status="not_started", result=str(Path(row["output"]) / "result.json")) for row in rows])
    write(HERE / "RUN_STATUS.json", status)
    with Path("/tmp/sdcf-measurement.lock").open("a") as lock:
        for index, row in enumerate(rows):
            state = status["rows"][index]
            fcntl.flock(lock, fcntl.LOCK_EX)
            started = time.monotonic()
            try:
                state.update(status="running", started_epoch=time.time())
                write(HERE / "RUN_STATUS.json", status)
                with Path(row["runner_log"]).open("w") as log:
                    completed = subprocess.run(row["launch_argv"], cwd=REPO,
                                               stdout=log, stderr=subprocess.STDOUT,
                                               env=environment)
                path = Path(state["result"])
                result = json.loads(path.read_text()) if path.exists() else dict(
                    status="failed", reason="runner produced no result.json")
                state.update(status=result["status"], reason=result.get("reason"),
                             controller_exit_code=completed.returncode,
                             controller_elapsed_seconds=time.monotonic() - started,
                             result_status=result["status"],
                             completed_queries=len(result.get("measurements", {}).get("queries", [])))
                if completed.returncode and result["status"] == "complete":
                    state.update(status="failed", reason="controller exited nonzero")
                write(HERE / "RUN_STATUS.json", status)
                print(row["backend"], state["status"], flush=True)
                if state["status"] != "complete":
                    status["status"] = "failed"
                    write(HERE / "RUN_STATUS.json", status)
                    return 1
            finally:
                fcntl.flock(lock, fcntl.LOCK_UN)
    status["status"] = "complete"
    write(HERE / "RUN_STATUS.json", status)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
