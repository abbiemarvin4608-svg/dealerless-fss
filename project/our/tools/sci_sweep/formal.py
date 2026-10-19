#!/usr/bin/env python3
"""Frozen point-list runner for the complete width x segmentation sweep."""

import argparse
import importlib.util
import json
import tempfile
from pathlib import Path

SPEC = importlib.util.spec_from_file_location(
    "sweep_runtime", Path(__file__).parent / "run.py")
runtime = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(runtime)


def points_for(profile, cfg):
    widths = cfg["widths"]
    nsegs = cfg["nseg"]
    if profile == "complete":
        # SCI is the native monolithic/single-tree reference.  Segmented FSS
        # runs every legal nseg (< input width), including width 2's nseg=1.
        return [(width, backend, nseg, "perf", 3)
                for width in widths
                for backend, nseg in ([('sci', 1)] +
                                      [('ours', n) for n in nsegs if n < width])]
    if profile == "main16":
        return ([(16, "sci", 1, "perf", 3)] +
                [(16, "ours", n, "perf", 3) for n in [1, 2, 4, 8]])
    if profile == "boundary":
        return [(width, backend, nseg, "check", 0)
                for width in widths
                for backend, nseg in ([('sci', 1)] +
                                      [('ours', n) for n in nsegs if n < width])]
    if profile == "wide":
        return [(width, backend, 4, "perf", 0)
                for width in [20, 24, 25, 32]
                for backend in ["sci", "ours"]]
    raise ValueError(f"unknown profile {profile}")


def boundary_acceptance(rows):
    """Accept only completed, reconstructed check calls; keep failures unknown."""
    groups = {}
    for row in rows:
        if row["mode"] != "check":
            continue
        verified = (row["status"] == "complete" and
                    row["count"] >= 8 + 3 * (row["input_bits"] - 1) and
                    row.get("completed_calls") == 1 and
                    row.get("opened_outputs") == 2 * row["count"] and
                    row.get("mismatches") == 0)
        point = dict(
            backend=row["backend"], input_bits=row["input_bits"],
            comparison_bits=row["comparison_bits"], nseg=row["nseg"],
            count=row["count"], status=row["status"], verified=verified,
            mismatches=row.get("mismatches"), reason=row.get("reason"),
            artifact=row.get("artifact"))
        key = (row.get("network", {}).get("name"), row["backend"],
               row["input_bits"], row["nseg"])
        groups.setdefault(key, []).append(point)
    points = []
    for attempts in groups.values():
        verified_attempts = [attempt for attempt in attempts if attempt["verified"]]
        selected = dict(verified_attempts[0] if verified_attempts else attempts[0])
        selected["attempts"] = attempts
        points.append(selected)
    accepted = [point for point in points if point["verified"]]
    return dict(
        requested_points=len(points), verified_points=len(accepted),
        attempted_checks=sum(len(point["attempts"]) for point in points),
        unverified_points=len(points) - len(accepted),
        verified_scalar_instances=sum(point["count"] for point in accepted),
        all_requested_points_verified=bool(points) and len(accepted) == len(points),
        points=points,
        scope="Public correctness-only boundary inputs; unavailable points are not verified.")


def run(args):
    cfg = json.loads(args.config.read_text())
    check_widths = [width for width, _, _, mode, _ in args.points
                    if mode == "check"]
    if check_widths:
        # sweep_point.cpp emits 8 extremes, then (2**bit - 1, 2**bit,
        # 2**bit + 1) for every bit through the input sign boundary.
        required_count = 8 + 3 * (max(check_widths) - 1)
        if cfg["batch"] < required_count:
            raise ValueError(
                f"boundary checks require count >= {required_count}; "
                f"configured {cfg['batch']} omits the final sign boundary")
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=True)
    build = Path(cfg["build"])
    with tempfile.TemporaryDirectory(prefix="sci-sweep-private-") as tmpdir:
        files = runtime.private_files(Path(tmpdir), cfg["batch"])
        for net in cfg["networks"]:
            for width, backend, nseg, mode, repeats in args.points:
                point_name = f"{backend}_w{width}_n{nseg}"
                base = out / "runs" / net["name"] / point_name
                failed = False
                tags = (["check"] if mode == "check" else
                        ["warmup"] + [f"repeat{i}" for i in range(1, repeats + 1)])
                for tag in tags:
                    path = base / tag
                    if failed:
                        path.mkdir(parents=True, exist_ok=True)
                        runtime._write_result(path / "result.json", dict(
                            status="not_run",
                            reason="predeclared_skip_after_resource_failure",
                            backend=backend,
                            input_bits=width,
                            comparison_bits=width - 1,
                            nseg=nseg if backend == "ours" else None,
                            count=cfg["batch"], mode=mode, network=net,
                            completed_calls=0, total_ms=None, link_bytes=None))
                        continue
                    result = runtime.run_point(
                        build, path, backend, width, nseg, net, files,
                        cfg["batch"], mode, cfg["timeout_seconds"])
                    print(json.dumps({
                        "network": net["name"],
                        "point": point_name,
                        "run": tag,
                        "status": result["status"],
                        "ms": result["total_ms"],
                    }), flush=True)
                    failed = result["status"] != "complete"
        runtime.aggregate(out)
        if check_widths:
            summary_path = out / "summary.json"
            summary = json.loads(summary_path.read_text())
            summary["correctness"] = boundary_acceptance(summary["raw"])
            summary_path.write_text(json.dumps(summary, indent=2) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--profile", choices=["complete", "main16",
                                               "boundary", "wide"],
                        required=True)
    args = parser.parse_args()
    cfg = json.loads(args.config.read_text())
    args.points = points_for(args.profile, cfg)
    run(args)
