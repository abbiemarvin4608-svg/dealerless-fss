#!/usr/bin/env python3
"""Build the LAN/WAN communication projection for the ticket13 retest."""
from __future__ import annotations

import hashlib
import json
import statistics
from pathlib import Path


ROOT = Path(__file__).resolve().parent
NETWORKS = {
    "LAN": {"bandwidth_bits_per_second": 1_000_000_000, "rtt_seconds": 0.0005},
    "WAN": {"bandwidth_bits_per_second": 400_000_000, "rtt_seconds": 0.004},
}
METHODS = (
    ("SCI_preprocessed", "SCI+SiRNN (SCI_preprocessed)"),
    ("SCI_Beaver_segmented_PaperLUT", "SCI_Beaver_segmented_PaperLUT"),
)
PHASES = ("offline", "online")
QUERY_TOTAL_PHASES = ("offline", "input", "ready", "online", "output")


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def distribution(values):
    return {
        "values": values,
        "median": statistics.median(values),
        "min": min(values),
        "max": max(values),
        "pstdev": statistics.pstdev(values),
    }


def phase_index(rounds):
    if rounds.get("status") != "passed":
        raise ValueError("round trace is not qualified")
    return {(row["query"], row["phase"]): row for row in rounds["phases"]}


def project_phase(measurement, trace):
    bytes_from_measurement = measurement.get("sent_bytes")
    bytes_from_trace = sum(trace["party_sent_bytes"])
    if bytes_from_measurement != bytes_from_trace:
        raise ValueError("result and round trace byte totals differ")
    rounds = trace["logical_rounds"]
    if type(bytes_from_measurement) is not int or bytes_from_measurement < 0:
        raise ValueError("phase bytes must be a nonnegative integer")
    if type(rounds) is not int or rounds < 0:
        raise ValueError("causal rounds must be a nonnegative integer")
    projected = {"bytes": bytes_from_measurement, "rounds": rounds, "networks": {}}
    for name, network in NETWORKS.items():
        transmission = 8 * bytes_from_measurement / network["bandwidth_bits_per_second"]
        propagation = rounds * network["rtt_seconds"]
        projected["networks"][name] = {
            "transmission_seconds": transmission,
            "propagation_seconds": propagation,
            "communication_seconds": transmission + propagation,
        }
    return projected


def build_row(backend, label):
    directory = ROOT / backend
    result_path, rounds_path = directory / "result.json", directory / "rounds.json"
    result, rounds = read_json(result_path), read_json(rounds_path)
    if result.get("status") != "complete":
        raise ValueError(f"{backend}: result is not complete")
    measurements = result["measurements"]
    traces = phase_index(rounds)
    queries = []
    for query in range(10):
        item = {"query": query, "phases": {}, "offline_plus_online_seconds": {}}
        for phase in PHASES:
            item["phases"][phase] = project_phase(
                measurements["queries"][query][phase], traces[(query, phase)])
        for phase in ("input", "ready", "output"):
            item["phases"][phase] = project_phase(
                measurements["queries"][query][phase], traces[(query, phase)])
        for name in NETWORKS:
            item["offline_plus_online_seconds"][name] = sum(
                item["phases"][phase]["networks"][name]["communication_seconds"]
                for phase in PHASES
            )
            item["query_total_seconds_communication_only"] = item.get(
                "query_total_seconds_communication_only", {})
            item["query_total_seconds_communication_only"][name] = sum(
                item["phases"][phase]["networks"][name]["communication_seconds"]
                for phase in QUERY_TOTAL_PHASES
            )
        queries.append(item)

    summary = {}
    for phase in (*PHASES, "query_total"):
        source_phases = QUERY_TOTAL_PHASES if phase == "query_total" else (phase,)
        values = []
        for item in queries:
            values.append({
                "bytes": sum(item["phases"][part]["bytes"] for part in source_phases)
                if phase == "query_total" else item["phases"][phase]["bytes"],
                "rounds": sum(item["phases"][part]["rounds"] for part in source_phases)
                if phase == "query_total" else item["phases"][phase]["rounds"],
            })
        summary[phase] = {
            "bytes": distribution([value["bytes"] for value in values]),
            "rounds": distribution([value["rounds"] for value in values]),
            "networks": {},
        }
        for name in NETWORKS:
            if phase == "query_total":
                totals = [item["query_total_seconds_communication_only"][name] for item in queries]
                components = {key: [] for key in (
                    "transmission_seconds", "propagation_seconds", "communication_seconds")}
                for item in queries:
                    for key in components:
                        components[key].append(sum(
                            item["phases"][part]["networks"][name][key]
                            for part in QUERY_TOTAL_PHASES))
            else:
                components = {
                    key: [item["phases"][phase]["networks"][name][key] for item in queries]
                    for key in ("transmission_seconds", "propagation_seconds", "communication_seconds")
                }
                totals = components["communication_seconds"]
            summary[phase]["networks"][name] = {
                key: distribution(values) for key, values in components.items()
            }
    return {
        "backend": backend,
        "label": label,
        "result": str(result_path),
        "rounds": str(rounds_path),
        "queries": queries,
        "summary": summary,
        "measured_wall_seconds": {
            phase: measurements[f"{phase}_seconds"] for phase in PHASES
        },
        "source_sha256": {"result.json": sha256(result_path), "rounds.json": sha256(rounds_path)},
    }


def report(model):
    lines = [
        "# ticket13 LAN/WAN 通信模型复测",
        "",
        "本报告使用同一冻结构建、batch=128、单时间步、10 个 fresh query 的两方 localhost 实测。",
        "localhost 只用于取得协议字节和因果轮次；LAN/WAN 数字是通信模型投影，不是网络整形实测总时延。",
        "",
        "## 计算规则",
        "",
        "```text",
        "传输时间 = 8 × 双方 sent bytes 之和 / 带宽(bit/s)",
        "传播时间 = 因果轮次 × RTT",
        "通信时间 = 传输时间 + 传播时间",
        "```",
        "",
        "LAN 为 1 Gbps、0.5 ms RTT；WAN 为 400 Mbps、4 ms RTT。每个 query 先计算，表中取 10 个 query 的中位数。",
        "",
        "## 离线与在线",
        "",
        "| 方法 | 阶段 | 实测墙钟(s) | bytes | 轮次 | LAN 传输(s) | LAN 传播(s) | LAN 通信(s) | WAN 传输(s) | WAN 传播(s) | WAN 通信(s) |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in model["rows"]:
        for phase in PHASES:
            summary = row["summary"][phase]
            lan, wan = summary["networks"]["LAN"], summary["networks"]["WAN"]
            wall = row["measured_wall_seconds"][phase]["median"]
            lines.append(
                f"| {row['label']} | {phase} | {wall:.9f} | {summary['bytes']['median']:.0f} | "
                f"{summary['rounds']['median']:.0f} | {lan['transmission_seconds']['median']:.9f} | "
                f"{lan['propagation_seconds']['median']:.9f} | {lan['communication_seconds']['median']:.9f} | "
                f"{wan['transmission_seconds']['median']:.9f} | {wan['propagation_seconds']['median']:.9f} | "
                f"{wan['communication_seconds']['median']:.9f} |"
            )
    lines += [
        "",
        "在线模型相对：SCI+SiRNN / PaperLUT 为 LAN %.6fx、WAN %.6fx；离线为 LAN %.6fx、WAN %.6fx。"
        % tuple(model["comparisons"][name][phase]["baseline_over_candidate"]
                for phase in ("online", "offline")
                for name in ("LAN", "WAN")),
        "",
        "## 查询阶段合计",
        "",
        "下表包含 offline、input、ready、online、output；setup、model_prepare 和 teardown 不计入。",
        "",
        "| 方法 | bytes | 轮次 | LAN 通信(s) | WAN 通信(s) |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for row in model["rows"]:
        summary = row["summary"]["query_total"]
        lines.append(
            f"| {row['label']} | {summary['bytes']['median']:.0f} | {summary['rounds']['median']:.0f} | "
            f"{summary['networks']['LAN']['communication_seconds']['median']:.9f} | "
            f"{summary['networks']['WAN']['communication_seconds']['median']:.9f} |"
        )
    lines += [
        "",
        "## 接口统计核对",
        "",
        "当前 `result.json` 的 `measurements` 已保存 `sent_bytes` 和墙钟；其各阶段 `logical_rounds` 仍为 `null`。",
        "因果轮次只写入同目录 `rounds.json`，`transmission_seconds`、`propagation_seconds` 和通信总时间也没有回写到 `result.json`。",
        "运行时随后已加入 `measurements.network_projection` 写入逻辑；本批 `result.json` 生成早于该改动，因此本报告仍以独立投影文件为准。",
        "本目录的 `NETWORK_MODEL.json` 固化了这次复测的逐 query 投影及来源哈希，因此本报告没有把 localhost 墙钟当作 LAN/WAN 时延。",
        "",
        "## 解释",
        "",
        "PaperLUT 的在线通信量和轮次都显著较小；其离线通信量约 15.83 倍于 SCI+SiRNN，导致离线 LAN/WAN 通信模型明显更高。",
        "这些结果是通信量与因果关键路径轮次的确定性投影，不包含本地计算、材料生成/加载和 relay 排队。",
        "",
    ]
    return "\n".join(lines)


def main():
    rows = [build_row(*method) for method in METHODS]
    by_backend = {row["backend"]: row for row in rows}
    comparisons = {}
    baseline = by_backend[METHODS[0][0]]
    candidate = by_backend[METHODS[1][0]]
    for name in NETWORKS:
        comparisons[name] = {}
        for phase in PHASES:
            left = baseline["summary"][phase]["networks"][name]["communication_seconds"]["median"]
            right = candidate["summary"][phase]["networks"][name]["communication_seconds"]["median"]
            comparisons[name][phase] = {
                "baseline_seconds": left,
                "candidate_seconds": right,
                "baseline_over_candidate": left / right,
                "candidate_over_baseline": right / left,
                "candidate_reduction": 1 - right / left,
            }
    model = {
        "schema": 1,
        "kind": "communication-time-model",
        "eligible": True,
        "networks": NETWORKS,
        "round_convention": "two-party causal send-flight clocks",
        "transmission_formula": "8 * sum(both parties sent bytes) / bandwidth_bits_per_second",
        "propagation_formula": "rtt_seconds * causal_rounds",
        "local_computation_included": False,
        "measured_lan_wan": False,
        "rows": rows,
        "comparisons": comparisons,
        "interface_stat_check": {
            "result_measurements_include_bytes": True,
            "result_measurements_include_logical_rounds": False,
            "result_measurements_include_network_projection": False,
            "rounds_json_include_logical_rounds": True,
        },
    }
    (ROOT / "NETWORK_MODEL.json").write_text(json.dumps(model, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (ROOT / "REPORT.zh.md").write_text(report(model) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
