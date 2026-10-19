#!/usr/bin/env python3
"""Paired real SCI/private-segmented-MSB sweep with immutable raw metrics."""

import argparse
import json
import os
import resource
import secrets
import struct
import subprocess
import sys
import tempfile
import time
from contextlib import ExitStack
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "google30_benchmark"))
from run import stop  # noqa: E402
from metrics import distribution  # noqa: E402


def private_files(tmp: Path, count: int = 100):
    """Create paired ephemeral additive shares over the full 32-bit ring.

    The same pair is reused for every backend/width point so paired results
    have identical inputs. ``secrets`` keeps the fixture non-deterministic;
    TemporaryDirectory removes both files after the sweep.
    """
    mask = (1 << 32) - 1
    shares = [[], []]
    for _ in range(count):
        value = secrets.randbits(32)
        left = secrets.randbits(32)
        shares[0].append(left)
        shares[1].append((value - left) & mask)
    paths = []
    for role, values in enumerate(shares, 1):
        path = tmp / f"party{role}.bin"
        path.write_bytes(struct.pack("<" + ("Q" * len(values)), *values))
        path.chmod(0o600)
        paths.append(path)
    return paths


def _write_result(path: Path, result: dict):
    path.write_text(json.dumps(result, indent=2) + "\n")


def run_point(build, out, backend, width, nseg, net, private_share_files,
              count=100, mode="perf", timeout=60):
    """Run one real two-party point and retain failure/censor evidence."""
    out.mkdir(parents=True, exist_ok=True)
    base = 27000
    mapping = ",".join(f"{base + 1000 + x}:{base + x}"
                       for x in [0, 3, 50, 100, 200])
    children = []
    status = "complete"
    reason = None
    start = time.monotonic()
    with ExitStack() as stack:
        relay_log = stack.enter_context((out / "relay.log").open("w"))
        relay_cmd = [
            "taskset", "-c", "4", sys.executable,
            str(HERE.parent / "google30_benchmark/link_proxy.py"),
            "--ports", mapping, "--rtt-ms", str(net["rtt_ms"]),
            "--mbps", str(net["mbps"]),
        ]
        relay = subprocess.Popen(relay_cmd, stdout=subprocess.PIPE,
                                 stderr=relay_log, text=True)
        stack.callback(stop, relay)
        if relay.stdout.readline().strip() != "READY":
            raise RuntimeError("relay startup failure")
        relay.stdout.close()
        commands = []
        for role, cpu in [(1, 0), (2, 2)]:
            env = dict(
                os.environ,
                OMP_NUM_THREADS="1",
                SWEEP_METRICS=str(out / f"party{role}.json"),
                SWEEP_PRIVATE_SHARE_FILE=str(private_share_files[role - 1]),
            )
            if role == 2:
                env.update(LD_PRELOAD=str(build / "connect_offset.so"),
                           SWEEP_CONNECT_OFFSET="1000")
            cmd = ["taskset", "-c", str(cpu), str(build / "SCI_MSB_SWEEP"),
                   str(role), str(base), backend, str(width), str(nseg),
                   str(count), mode]
            commands.append(cmd)
            party_log = stack.enter_context((out / f"party{role}.log").open("w"))

            def limits():
                resource.setrlimit(resource.RLIMIT_AS,
                                   (768 * 1024**2, 768 * 1024**2))

            child = subprocess.Popen(cmd, env=env, stdout=party_log,
                                     stderr=subprocess.STDOUT,
                                     preexec_fn=limits)
            children.append(child)
            stack.callback(stop, child)

        deadline = time.monotonic() + timeout
        while any(child.poll() is None for child in children):
            codes = [child.poll() for child in children]
            if any(code is not None and code != 0 for code in codes):
                status, reason = "failed", "party_nonzero_exit"
                break
            if time.monotonic() > deadline:
                status, reason = "censored", f"wall_timeout_{timeout}s"
                break
            time.sleep(0.02)
        for child in children:
            stop(child)

    # A child may fail immediately after the polling loop observes it. Always
    # derive final status from both exit codes and metrics.
    codes = [child.returncode for child in children]
    if status == "complete" and any(code != 0 for code in codes):
        status, reason = "failed", "party_nonzero_exit"

    result = dict(
        status=status,
        reason=reason,
        backend=backend,
        input_bits=width,
        comparison_bits=width - 1,
        nseg=nseg if backend == "ours" else None,
        count=count,
        mode=mode,
        network=net,
        commands=commands,
        relay_command=relay_cmd,
        exit_codes=codes,
        wall_seconds=time.monotonic() - start,
        completed_calls=0,
        total_ms=None,
        link_bytes=None,
        dependent_rounds=None,
    )
    if status != "complete":
        # Keep a short diagnostic in the machine-readable row while retaining
        # the complete party logs beside it.  This distinguishes expected
        # single-tree allocation failures from transport/time-limit failures.
        tails = {}
        for role in [1, 2]:
            log_path = out / f"party{role}.log"
            if log_path.exists():
                tails[f"party{role}"] = log_path.read_text(errors="replace")[-2048:]
        result["party_log_tails"] = tails
        if any("bad_alloc" in tail for tail in tails.values()):
            result["reason"] = "party_nonzero_exit:std::bad_alloc"
    if status == "complete":
        try:
            parties = [json.loads((out / f"party{role}.json").read_text())
                       for role in [1, 2]]
        except (OSError, json.JSONDecodeError) as exc:
            result.update(status="failed",
                          reason=f"missing_or_invalid_metrics:{type(exc).__name__}")
            _write_result(out / "result.json", result)
            return result
        for i in [0, 1]:
            if parties[i]["sent_bytes"] != parties[1 - i]["received_bytes"]:
                raise ValueError("unbalanced per-party counts")
        result.update(
            completed_calls=1,
            scalar_instances=count,
            total_ms=max(p["total_ms"] for p in parties),
            setup_ms=max(p["setup_ms"] for p in parties),
            offline_ms=(max(p["offline_ms"] for p in parties)
                        if backend == "ours" else None),
            online_ms=max(p["online_ms"] for p in parties),
            link_bytes=sum(p["sent_bytes"] for p in parties),
            setup_bytes=sum(p["setup_sent_bytes"] for p in parties),
            offline_bytes=(sum(p["offline_sent_bytes"] for p in parties)
                           if backend == "ours" else None),
            online_bytes=sum(p["online_sent_bytes"] for p in parties),
            peak_rss_kib=[p["peak_rss_kib"] for p in parties],
            opened_outputs=sum(p["opened_outputs"] for p in parties),
            mismatches=sum(p["mismatches"] for p in parties),
            rounds_status="unavailable: library I/O counters are not dependency rounds",
            keygen_and_masks_ms=(max(p["offline_ms"] for p in parties)
                                 if backend == "ours" else None),
            input_conversion_ms=None,
            scan_ms=None,
            phase_reason=("Input is already additive shares; offline includes "
                          "mask B2A/keygen/scan triples; masked input opening "
                          "and scan are inseparable within measured online; "
                          "performance output remains shared so no output "
                          "reconstruction."),
        )
        if mode == "check" and result["mismatches"]:
            result.update(status="failed", reason="mismatches")
    _write_result(out / "result.json", result)
    return result


def aggregate(out: Path):
    """Aggregate only repeat1..repeat3; retain every raw row."""
    rows = []
    for path in out.glob("runs/*/*/*/result.json"):
        row = json.loads(path.read_text())
        row["artifact"] = str(path.relative_to(out))
        row["tag"] = path.parent.name
        rows.append(row)
    groups = {}
    for row in rows:
        if row["tag"] not in ("repeat1", "repeat2", "repeat3"):
            continue
        key = (row["network"]["name"], row["backend"],
               row["input_bits"], row["nseg"])
        groups.setdefault(key, []).append(row)
    summaries = []
    for key, group in sorted(groups.items(), key=str):
        completed = [row for row in group if row["status"] == "complete"]
        summary = dict(network=key[0], backend=key[1], input_bits=key[2],
                       comparison_bits=key[2] - 1, nseg=key[3],
                       expected_repeats=3, completed=len(completed),
                       failed=len(group) - len(completed),
                       statuses=sorted({row["status"] for row in group}))
        for field in ["total_ms", "setup_ms", "offline_ms", "online_ms",
                      "link_bytes"]:
            values = [row[field] for row in completed
                      if row.get(field) is not None]
            summary[field] = distribution(values) if values else None
        summaries.append(summary)
    (out / "summary.json").write_text(
        json.dumps(dict(rows=summaries, raw=rows), indent=2) + "\n")
    return summaries


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    args.output.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="sci-sweep-private-") as tmp:
        files = private_files(Path(tmp), config["batch"])
        for net in config["networks"]:
            for width in config["widths"]:
                configs = [("sci", 1)] + [
                    ("ours", n) for n in config["nseg"] if n < width
                ]
                for backend, nseg in configs:
                    if args.smoke and (width != 16 or
                                       (backend == "ours" and nseg != 4)):
                        continue
                    label = f"{backend}_w{width}_n{nseg}"
                    base = args.output / "runs" / net["name"] / label
                    tags = ["check"] if args.smoke else [
                        "warmup", "repeat1", "repeat2", "repeat3"
                    ]
                    failed = False
                    for tag in tags:
                        path = base / tag
                        if failed:
                            path.mkdir(parents=True, exist_ok=True)
                            _write_result(path / "result.json", dict(
                                status="not_run",
                                reason="predeclared_skip_after_resource_failure",
                                backend=backend, input_bits=width,
                                comparison_bits=width - 1,
                                nseg=nseg if backend == "ours" else None,
                                count=config["batch"], mode="perf",
                                network=net, completed_calls=0,
                                total_ms=None, link_bytes=None))
                            continue
                        row = run_point(Path(config["build"]), path, backend,
                                        width, nseg, net, files,
                                        config["batch"],
                                        "check" if args.smoke else "perf",
                                        config["timeout_seconds"])
                        print(json.dumps({k: row.get(k) for k in
                                          ["backend", "input_bits", "nseg",
                                           "status", "total_ms", "link_bytes"]}),
                              flush=True)
                        failed = row["status"] != "complete"
    aggregate(args.output)
