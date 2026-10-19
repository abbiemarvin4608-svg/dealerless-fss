#!/usr/bin/env python3
"""Reuse ticket10 evidence and freeze the missing MSB-input-28 experiment."""

import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import statistics
import tempfile

from analyze_rnn_widths import REPO, read_json, write_csv, write_json


HERE = Path(__file__).resolve().parent
DEFAULT_SOURCE = REPO / "project/our/results/ticket10_width_sweep"
DEFAULT_OUTPUT = REPO / "project/our/results/precision_motivation_final/primitive_ablation"


def reused_rows(source):
    config = read_json(source / "frozen_config.json")
    summary = read_json(source / "summary.json")
    if config["batch"] != 100 or config["repetitions"] != 3:
        raise ValueError("ticket10 original batch/repeat identity changed")
    rows = []
    for row in summary["rows"]:
        if row["input_bits"] < 16:
            continue
        point_records = [item for item in summary["raw"] if
               (item["network"]["name"], item["backend"], item["input_bits"], item["nseg"]) ==
               (row["network"], row["backend"], row["input_bits"], row["nseg"])]
        raw = [item for item in point_records if item["tag"] in ("repeat1", "repeat2", "repeat3")]
        warmup = [item for item in point_records if item["tag"] == "warmup"]
        if len(raw) != 3 or any(item["count"] != 100 for item in raw):
            raise ValueError("original raw repeat coverage differs from its summary")
        if len(warmup) != 1 or warmup[0]["count"] != 100:
            raise ValueError("original warmup coverage differs from the frozen plan")
        output = {"origin": "reused_ticket10", "batch": 100,
                  "input_bits": row["input_bits"], "comparison_bits": row["comparison_bits"],
                  "network": row["network"], "backend": row["backend"], "nseg": row["nseg"],
                  "formal_repeats": 3, "completed": row["completed"],
                  "statuses": ";".join(row["statuses"]),
                  "warmup_status": warmup[0]["status"], "warmup_reason": warmup[0].get("reason"),
                  "address_space_limit_bytes_per_party": config["address_space_limit_bytes"]}
        for metric in ("setup_ms", "offline_ms", "online_ms", "total_ms", "link_bytes"):
            output[metric + "_median"] = row[metric]["median"] if row.get(metric) else None
            values = [item[metric] for item in raw if item["status"] == "complete" and item.get(metric) is not None]
            if output[metric + "_median"] != (statistics.median(values) if values else None):
                raise ValueError("historical summary median differs from retained raw records")
        for metric in ("setup_bytes", "offline_bytes", "online_bytes"):
            values = [item[metric] for item in raw if item["status"] == "complete" and item.get(metric) is not None]
            output[metric + "_median"] = statistics.median(values) if values else None
        output["source_summary"] = str((source / "summary.json").relative_to(REPO))
        rows.append(output)
    return rows, config


def prepare(source, output, build):
    if (output / "missing_width_runs").exists():
        raise FileExistsError("protocol evidence already exists; its frozen configuration will not be replaced")
    rows, old = reused_rows(source)
    if 28 in {row["input_bits"] for row in rows}:
        raise ValueError("input28 already exists; no missing-width run is needed")
    output.mkdir(parents=True, exist_ok=True)
    write_csv(output / "reused_batch100.csv", rows)
    plan = {"status": "planned_not_started", "primitive": "private_MSB",
            "width_axis": "input_bits; internal comparison_bits = input_bits - 1",
            "widths": [28], "comparison_bits": [27], "nseg": [1, 2, 4, 8],
            "batch": 128, "warmups": 1, "repetitions": 5,
            "timeout_seconds": 60, "threads_per_party": 1,
            "address_space_limit_bytes_per_party": 768 * 1024**2,
            "whole_process_tree_memory_bytes": 2 * 1024**3,
            "whole_process_tree_memory_and_swap_bytes": 2560 * 1024**2,
            "networks": old["networks"], "build": str(build.resolve()),
            "boundary_network": "lan_model",
            "failure_policy": "Keep actual failure/timeout/OOM record; skip remaining repeats only after a resource failure. Never substitute timing estimates.",
            "reused_input_widths": sorted({row["input_bits"] for row in rows}),
            "reused_batch": 100, "cross_batch_speedup_permitted": False,
            "existing_comparison_domain_coverage": {"23": "MSB input24, batch100", "31": "MSB input32, batch100"}}
    write_json(output / "missing_width_config.json", plan)
    write_json(output / "manifest.json", {"status": "prepared_not_executed", "reused_rows": len(rows),
        "reused_complete_groups": sum(row["completed"] == 3 for row in rows),
        "reused_unavailable_groups": sum(row["completed"] == 0 for row in rows),
        "reused_actual_failed_warmups": sum(row["warmup_status"] == "failed" for row in rows),
        "reused_completed_formal_repeats": sum(row["completed"] for row in rows),
        "source": str(source.relative_to(REPO)), "new_protocol_runs": 0,
        "new_point_count": len(plan["networks"]) * 5, "missing_width_config": "missing_width_config.json"})
    (output / "README.zh.md").write_text("""# 实验三原语：复用结果与唯一缺项

`reused_batch100.csv` 保留原 ticket10 的 batch=100、一次预热和三次正式重复，原始结果未改写。
50 个复用配置中，44 个各完成三次正式重复；6 个单树配置（24/25/32 位、两种网络）在预热时真实 bad_alloc，随后 18 次正式重复未运行。它们是资源不可用记录，不是 50 个全部成功的测量点。全部汇总中位数已与保留的原始重复记录交叉核对。
复用输入位宽为 16/20/24/25/32；对应内部比较域分别为 15/19/23/24/31。
因此 23/31 位 FSS 比较域已有实测，不追加名为“输入23/31”的冗余测试。

只补缺失的 MSB 输入28位（内部比较域27位）：batch=128，SCI 与 ours nseg=1/2/4/8，LAN 0ms/1Gbps、受控 WAN 2ms/100Mbps，一次预热、五次新材料正式重复。
新 batch128 与旧 batch100 分栏报告，不跨 batch 计算加速比或拼成相同实验配置的曲线。该 batch 是原语标量数，不是完整 CNN 图像批量。

每方继承旧 runner 的 768MiB 地址空间上限；外层同时使用既有 memory_guard 限制整个实验树 2GiB RAM+512MiB swap。128个公开边界输入单独核对，不进入性能统计。单树失败保留原始 OOM/超时/退出码与预算，剩余重复记 not_run，不强行重试。

网络口径为本地真实 TCP 协议经应用负载 relay 的受控链路模型。WAN 每向加入 1ms 传播延迟，两方向分别共享 100Mbps 序列化时钟，所有 SCI/dFSS 通道共用相应方向限速。LAN 模型为零附加传播延迟、每向 1Gbps。测量包含真实协议执行与 relay 等待，但不是物理跨地域 WAN、tc 整形或完整 TCP 拥塞仿真；不将名义 RTT 当作实际测得的全链路 RTT。

旧 runner 的聚合只识别 repeat1..3，因此补测使用本工具独立聚合 repeat1..5，避免漏算。setup/offline/online/总时延、各阶段通信和内存均分开报告；没有因果轮数测量时保持缺失。

## 执行顺序

本目录只完成计划和旧数据分析，尚未执行新协议。先完成用户要求的精度实验，再检查/重建 SCI_MSB_SWEEP、libsegfss_sci_bridge.so 与 connect_offset.so，并在既有内存保护下执行下列命令。`--build` 指向这三个产物所在目录；配置文件会核对该路径。

```bash
python3 project/our/tools/imagenet_resnet/prepare_primitive_ablation.py --build /path/to/qualified-sweep-build
python3 project/our/tools/imagenet_resnet/memory_guard.py run --output /tmp/primitive28-memory-guard.json -- python3 project/our/tools/imagenet_resnet/prepare_primitive_ablation.py --execute-missing
```

实际补测结果将写入 `missing_width_runs/`。旧构建路径位于已丢失的临时目录，不能假定原二进制仍可执行；新构建须记录构建身份，不把它默认为原构建。
""")
    return plan


def load_runtime():
    spec = importlib.util.spec_from_file_location("missing_width_sweep", HERE.parent / "sci_sweep/run.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def build_identity(build):
    result = {}
    for name in ("SCI_MSB_SWEEP", "libsegfss_sci_bridge.so", "connect_offset.so"):
        path = build / name
        value = hashlib.sha256()
        with path.open("rb") as stream:
            for block in iter(lambda: stream.read(1 << 20), b""):
                value.update(block)
        result[name] = {"bytes": path.stat().st_size, "sha256": value.hexdigest()}
    return result


def aggregate_missing(records, repetitions):
    groups = {}
    for row in records:
        if row["tag"] not in {"repeat%d" % i for i in range(1, repetitions + 1)}:
            continue
        key = (row["network"]["name"], row["backend"], row["input_bits"], row["nseg"])
        groups.setdefault(key, []).append(row)
    summaries = []
    for key, rows in sorted(groups.items(), key=str):
        completed = [row for row in rows if row["status"] == "complete"]
        summary = dict(network=key[0], backend=key[1], input_bits=key[2], comparison_bits=key[2]-1,
                       nseg=key[3], batch=128, expected_repeats=repetitions, completed=len(completed),
                       unavailable=len(rows)-len(completed))
        for metric in ("setup_ms", "offline_ms", "online_ms", "total_ms", "setup_bytes", "offline_bytes", "online_bytes", "link_bytes"):
            values = [row[metric] for row in completed if row.get(metric) is not None]
            summary[metric] = {"n": len(values), "median": statistics.median(values), "min": min(values), "max": max(values)} if values else None
        summaries.append(summary)
    return summaries


def execute_missing(output):
    from run_performance import require_memory_guard
    guard = require_memory_guard()
    cfg = read_json(output / "missing_width_config.json")
    if cfg["widths"] != [28] or cfg["batch"] != 128 or cfg["repetitions"] != 5:
        raise ValueError("only the frozen missing-input28, batch128, five-repeat plan is supported")
    build = Path(cfg["build"])
    for name in ("SCI_MSB_SWEEP", "libsegfss_sci_bridge.so", "connect_offset.so"):
        if not (build / name).is_file():
            raise FileNotFoundError("qualified sweep build missing: " + str(build / name))
    run_root = output / "missing_width_runs"
    if run_root.exists():
        raise FileExistsError("use an unoccupied output directory; prior evidence will not be overwritten")
    run_root.mkdir()
    runtime = load_runtime()
    records = []
    write_json(run_root / "plan.json", dict(cfg, resource_guard=guard, build_identity=build_identity(build)))
    with tempfile.TemporaryDirectory(prefix="primitive28-private-") as temporary:
        shares = runtime.private_files(Path(temporary), cfg["batch"])
        for network in cfg["networks"]:
            for backend, nseg in [("sci", 1)] + [("ours", n) for n in cfg["nseg"]]:
                point = "%s_w28_n%d" % (backend, nseg)
                failure = None
                tags = ["warmup"] + ["repeat%d" % i for i in range(1, cfg["repetitions"]+1)]
                for tag in tags:
                    destination = run_root / "runs" / network["name"] / point / tag
                    if failure:
                        destination.mkdir(parents=True)
                        row = dict(status="not_run", reason="resource_failure_in_previous_attempt", previous_reason=failure,
                            backend=backend, input_bits=28, comparison_bits=27,
                            nseg=nseg if backend == "ours" else None, count=128, mode="perf", network=network)
                    else:
                        row = runtime.run_point(build, destination, backend, 28, nseg, network,
                                                shares, 128, "perf", cfg["timeout_seconds"])
                    row.update(tag=tag, address_space_limit_bytes_per_party=cfg["address_space_limit_bytes_per_party"], resource_guard=guard)
                    write_json(destination / "result.json", row)
                    records.append(row)
                    if row["status"] not in ("complete", "not_run"):
                        if row["status"] == "censored" or "bad_alloc" in (row.get("reason") or "") or -9 in row.get("exit_codes", []):
                            failure = row.get("reason") or "resource_termination"
                        else:
                            write_json(run_root / "summary.json", dict(status="failed", rows=aggregate_missing(records, 5), raw=records))
                            raise RuntimeError("unexpected protocol failure; inspect " + str(destination))
                    print(json.dumps({key: row.get(key) for key in ("backend", "input_bits", "nseg", "tag", "status", "total_ms")}), flush=True)
                if network["name"] == cfg["boundary_network"]:
                    destination = run_root / "boundary" / point
                    if failure:
                        destination.mkdir(parents=True)
                        check = dict(status="not_run", reason="resource_failed_performance_point_unverified",
                            previous_reason=failure, backend=backend, input_bits=28, comparison_bits=27,
                            nseg=nseg if backend == "ours" else None, count=128, mode="check", network=network,
                            mismatches=None, opened_outputs=0, completed_calls=0)
                    else:
                        check = runtime.run_point(build, destination, backend, 28, nseg, network,
                                                  shares, 128, "check", cfg["timeout_seconds"])
                    check.update(tag="check", resource_guard=guard,
                        address_space_limit_bytes_per_party=cfg["address_space_limit_bytes_per_party"],
                        verified=check["status"] == "complete" and check.get("mismatches") == 0 and
                                 check.get("opened_outputs") == 256 and check.get("completed_calls") == 1)
                    write_json(destination / "result.json", check)
                    records.append(check)
                    if not failure and not check["verified"]:
                        write_json(run_root / "summary.json", dict(status="failed_boundary", rows=aggregate_missing(records, 5), raw=records))
                        raise RuntimeError("boundary check failed; inspect " + str(destination))
                write_json(run_root / "summary.json", dict(status="running", rows=aggregate_missing(records, 5), raw=records))
    write_json(run_root / "summary.json", dict(status="complete_with_resource_failures" if any(row["status"] != "complete" for row in records) else "complete", rows=aggregate_missing(records, 5), raw=records))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--build", type=Path)
    parser.add_argument("--execute-missing", action="store_true", help="start the frozen input28 run; requires the memory guard")
    args = parser.parse_args()
    if args.execute_missing:
        execute_missing(args.output)
    else:
        plan = prepare(args.source, args.output, args.build or args.output / "build")
        print(json.dumps({"status": plan["status"], "input_bits": plan["widths"], "comparison_bits": plan["comparison_bits"], "batch": plan["batch"]}))


if __name__ == "__main__":
    main()
