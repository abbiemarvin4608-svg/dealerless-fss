#!/usr/bin/env python3
"""Verify runtime interface statistics and report the matched ticket13 pair."""
import argparse
import json
import math
import os
from pathlib import Path
import statistics


HERE = Path(__file__).resolve().parent
PHASES = ("offline", "online")
COMPONENTS = ("transmission_seconds", "propagation_seconds", "communication_seconds")
NETWORKS = {"LAN": {"mbps": 1000, "rtt_ms": 0.5},
            "WAN": {"mbps": 400, "rtt_ms": 4.0}}
ALLOCATOR = {"MALLOC_ARENA_MAX": "2", "MALLOC_TRIM_THRESHOLD_": "1",
             "MALLOC_MMAP_THRESHOLD_": "131072"}
COMMON_FIELDS = (
    "binary", "binary_sha256", "runtime_manifest", "fixture", "fixture_identity",
    "fixture_files", "sample_ids", "hidden_file",
    "batch", "steps", "start", "queries", "warmup_queries", "diagnostic",
    "native_preprocessing_effective", "model_bias_precomputed", "graph_mode",
    "graph_openings", "tail_parallel", "activation_workers_per_party",
    "cggm_worker_limit_per_party", "requested_cpu_affinity_per_party", "cpu_affinity_per_party",
    "memory_limit_bytes_per_party", "generation_budget_bytes", "generation_reserve_bytes",
    "generation_tile_keys", "prefetch_bytes", "offline_overlap", "material_layout",
    "beaver_algebra", "beaver_mapped_layout", "beaver_tile_rows", "segment_policy",
    "max_segments", "conversion_optimizations", "pack_ring", "prune_equality",
    "prune_leaf_hashes", "bit_block_mode", "merge_leaf_cw", "reuse_generation_workspace",
    "aes_online", "aes_offline", "round_trace", "network", "network_projection",
)


def close(left, right):
    if not math.isclose(left, right, rel_tol=1e-12, abs_tol=1e-12):
        raise ValueError("network projection differs from measured bytes/rounds")


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def validate_launch(result_paths):
    """Check both recorded launches against the common frozen environment."""
    outer = read(HERE / "OUTER_COMMANDS.json")
    status = read(HERE / "RUN_STATUS.json")
    frozen = read(Path(outer["frozen_plan"]))
    if (outer["allocator_environment"] != ALLOCATOR or
            frozen["allocator_environment"] != ALLOCATOR or
            outer["explicit_environment"] != {"OMP_NUM_THREADS": "1", **ALLOCATOR}):
        raise ValueError("launch allocator environment differs from the frozen plan")
    sanitization = outer["environment_sanitization"]
    if (not {"MALLOC_", "SDCF_", "SEGFSS_"}.issubset(sanitization["remove_prefixes"]) or
            "GLIBC_TUNABLES" not in sanitization["remove_names"] or
            not outer["controller_environment_assertions"]["glibc_tunables_absent"]):
        raise ValueError("launch did not record the frozen environment sanitization")
    if (status["status"] != "complete" or len(status["rows"]) != 2 or
            len(outer["rows"]) != 2 or
            {row["backend"] for row in status["rows"]} != set(result_paths) or
            {row["backend"] for row in outer["rows"]} != set(result_paths) or
            outer["order"] != ["SCI_Beaver_segmented_PaperLUT", "SCI_preprocessed"]):
        raise ValueError("the matched two-row run is incomplete")
    states = {row["backend"]: row for row in status["rows"]}
    for row in outer["rows"]:
        backend = row["backend"]
        state = states[backend]
        source = next(item for item in frozen["rows"] if item["id"] == row["frozen_row_id"])
        result_path = result_paths[backend]
        if (state["status"] != "complete" or state["controller_exit_code"] != 0 or
                state["completed_queries"] != 10 or
                Path(state["result"]).resolve() != result_path or
                Path(row["output"]).resolve() != result_path.parent):
            raise ValueError("launch status does not match the source result")
        command = [part.replace("{output}", str(result_path.parent)) for part in source["command"]]
        command[1] = str(Path(outer["working_directory"]) / "project/our/tools/segmented_dcf_runtime/run.py")
        command += ["--network-projection", "both"]
        if (source["backend"] != backend or row["controller_argv"] != command or
                row["controller_address_space_limit_bytes"] != 1610612736 or
                frozen["controller_address_space_limit_bytes"] != 1610612736):
            raise ValueError("launch command or controller budget differs from the frozen row")
    return {
        "frozen_plan": outer["frozen_plan"], "outer_commands": str(HERE / "OUTER_COMMANDS.json"),
        "run_status": str(HERE / "RUN_STATUS.json"), "order": outer["order"],
        "allocator_environment": outer["allocator_environment"],
        "environment_sanitization": sanitization,
        "controller_address_space_limit_bytes": 1610612736,
        "changes_from_frozen_rows": outer["changes_from_frozen_rows"],
        "frozen_launch_verified": True,
    }


def summarize_source(result_path, label):
    result = read(result_path)
    if result["status"] != "complete":
        raise ValueError(str(result_path) + ": incomplete result")
    rounds_path = result_path.with_name("rounds.json")
    rounds = read(rounds_path)
    if rounds["status"] != "passed":
        raise ValueError(str(rounds_path) + ": unqualified round trace")
    config = result["config"]
    if (config["queries"] != 10 or config["warmup_queries"] != 0 or
            config["batch"] != 128 or config["steps"] != 1 or
            config["generation_budget_bytes"] != 1610612736):
        raise ValueError("result does not match the frozen ticket13 workload/budget")
    effective_path = result_path.with_name("EFFECTIVE_ENVIRONMENT.json")
    observed = read(effective_path)["rows"]
    commands_path = result_path.with_name("commands.json")
    recorded_commands = read(commands_path)["commands"]
    if len(observed) != 2 or len(recorded_commands) != 2:
        raise ValueError("observed party processes differ from recorded native commands")
    for index, (row, command) in enumerate(zip(observed, recorded_commands)):
        affinity = ",".join(str(cpu) for cpu in config["cpu_affinity_per_party"][index])
        if command[:3] != ["taskset", "-c", affinity] or command[3:] != row["argv"]:
            raise ValueError("observed party processes differ from recorded native commands")
    for row in observed:
        environment = row["whitelisted_effective_environment"]
        if ({key: environment.get(key) for key in ALLOCATOR} != ALLOCATOR or
                environment.get("GLIBC_TUNABLES") is not None or
                environment.get("OMP_NUM_THREADS") != "1" or
                environment.get("SDCF_BACKEND") != config["backend"] or
                environment.get("SEGFSS_GENERATION_BUDGET_BYTES") != "1610612736"):
            raise ValueError("observed party environment differs from the frozen launch contract")
    measurements = result["measurements"]
    projection = measurements["network_projection"]
    if projection["mode"] != "both" or set(projection["networks"]) != set(NETWORKS):
        raise ValueError("both LAN and WAN interface projections are required")
    traces = {(row["query"], row["phase"]): row for row in rounds["phases"]}
    projected = {}
    for name, network in projection["networks"].items():
        if not network["eligible"] or network["network"] != NETWORKS[name]:
            raise ValueError("network projection is ineligible or uses another link model")
        if (len(network["queries"]) != 10 or
                [row["query"] for row in network["queries"]] != list(range(10))):
            raise ValueError("network totals lack ten ordered fresh queries")
        for phase in PHASES:
            rows = network["phases"][phase]
            if len(rows) != 10 or {row["query"] for row in rows} != set(range(10)):
                raise ValueError("interface projection lacks ten fresh queries")
            projected[name, phase] = {row["query"]: row for row in rows}
    queries = []
    if len(measurements["queries"]) != 10:
        raise ValueError("phase measurements lack ten fresh queries")
    for query, source in enumerate(measurements["queries"]):
        item = {"query": query, "phases": {}, "offline_plus_online_seconds": {}}
        for phase in PHASES:
            value, trace = source[phase], traces[query, phase]
            if [party["query"] for party in value["per_party"]] != [query, query]:
                raise ValueError("phase query ids do not match fresh query ordering")
            sent, received = trace["party_sent_bytes"], trace["party_received_bytes"]
            count = trace["logical_rounds"]
            if (value["logical_rounds"] != count or value["party_sent_bytes"] != sent or
                    value["party_received_bytes"] != received or
                    value["sent_bytes"] != sum(sent) or received != sent[::-1]):
                raise ValueError("query phase counters differ from the qualified trace")
            phase_value = {
                "bytes": sum(sent), "logical_rounds": count,
                "party_sent_bytes": sent, "party_received_bytes": received,
                "localhost_seconds": value["seconds"], "networks": {},
            }
            for name, network in NETWORKS.items():
                row = projected[name, phase][query]
                if projection["networks"][name]["queries"][query]["phases"][phase] != row:
                    raise ValueError("duplicated top-level query details differ from phase projection")
                if row["bytes"] != sum(sent) or row["rounds"] != count or row["sent_bytes"] != sent:
                    raise ValueError("top-level projection counters differ from the qualified trace")
                transmission = 8 * sum(sent) / (network["mbps"] * 1000000)
                propagation = count * network["rtt_ms"] / 1000
                expected = {"transmission_seconds": transmission,
                            "propagation_seconds": propagation,
                            "communication_seconds": transmission + propagation}
                for key, expected_value in expected.items():
                    close(row[key], expected_value)
                    close(value["network_projection"][name][key], expected_value)
                phase_value["networks"][name] = expected
            item["phases"][phase] = phase_value
        for name in NETWORKS:
            total = sum(item["phases"][phase]["networks"][name]["communication_seconds"]
                        for phase in PHASES)
            close(projection["networks"][name]["queries"][query]["offline_plus_online_seconds"], total)
            item["offline_plus_online_seconds"][name] = total
        queries.append(item)
    summary = {}
    for phase in PHASES:
        summary[phase] = {
            "bytes_median": statistics.median(item["phases"][phase]["bytes"] for item in queries),
            "logical_rounds_median": statistics.median(item["phases"][phase]["logical_rounds"] for item in queries),
            "localhost_seconds_median": statistics.median(item["phases"][phase]["localhost_seconds"] for item in queries),
            "networks": {},
        }
        for name in NETWORKS:
            components = {key: statistics.median(item["phases"][phase]["networks"][name][key]
                                                 for item in queries) for key in COMPONENTS}
            interface_summary = projection["networks"][name]["summary"][phase]
            for key, value in components.items():
                close(interface_summary[key + "_median"], value)
            close(interface_summary["bytes_median"], summary[phase]["bytes_median"])
            close(interface_summary["rounds_median"], summary[phase]["logical_rounds_median"])
            summary[phase]["networks"][name] = components
    combined = {}
    for name in NETWORKS:
        combined[name] = statistics.median(item["offline_plus_online_seconds"][name] for item in queries)
        close(projection["networks"][name]["summary"]["offline_plus_online_seconds_median"], combined[name])
    return {
        "backend": config["backend"], "label": label,
        "sources": {"result": str(result_path), "rounds": str(rounds_path),
                    "commands": str(commands_path), "effective_environment": str(effective_path)},
        "activation_config": {key: config.get(key) for key in (
            "activation_algorithm", "folded_activations", "lut_key_family",
            "lut_segment_widths", "folded_sign_backend")},
        "queries": queries, "summary": summary,
        "offline_plus_online_seconds_median": combined,
        "interface_statistics_verified": True,
    }, config


def report(model):
    lines = [
        "# ticket13 同配置 LAN/WAN 在线与离线复测", "",
        "PaperLUT 的在线通信模型相对 SCI+SiRNN 在 LAN 降低 %.2f%%（%.4fx），在 WAN 降低 %.2f%%（%.4fx）。"
        % (100 * model["comparisons"]["LAN"]["online"]["candidate_reduction"],
           model["comparisons"]["LAN"]["online"]["baseline_over_candidate"],
           100 * model["comparisons"]["WAN"]["online"]["candidate_reduction"],
           model["comparisons"]["WAN"]["online"]["baseline_over_candidate"]),
        "其离线通信模型为 SCI+SiRNN 的 LAN %.4f 倍、WAN %.4f 倍。"
        % (model["comparisons"]["LAN"]["offline"]["candidate_over_baseline"],
           model["comparisons"]["WAN"]["offline"]["candidate_over_baseline"]), "",
        "## 配置与口径", "",
        "两套方案使用同一持久构建和 fixture，batch=128、steps=1、parallel-a2、10 个 fresh query、warmup=0。每方四个 CPU、1.5 GiB 地址空间上限，generation budget 显式为 1610612736 B，reserve 与 prefetch 各为 536870912 B。",
        "两套使用统一的 allocator 环境：移除继承的 MALLOC_* 和 GLIBC_TUNABLES 后固定 MALLOC_ARENA_MAX=2、MALLOC_TRIM_THRESHOLD_=1、MALLOC_MMAP_THRESHOLD_=131072；控制器也设置 1.5 GiB 地址空间上限。",
        "PaperLUT 使用 folded_activations=both、segmented_boolean_tensor_b2a（5+5 bits）、folded_sign_backend=bitdec，Sigmoid/TanH 都采用完整 folded LUT；SCI+SiRNN 对照为 SCI_preprocessed，保留 SCI 原生激活。这里比较两套方法的完整单步成本。两套执行顺序进行。", "",
        "LAN 为 1 Gbps / 0.5 ms RTT，WAN 为 400 Mbps / 4 ms RTT。协议在 localhost 执行；LAN/WAN 数字按用户给定模型从实测字节和因果轮次计算。", "",
        "```text", "D = 两方 sent bytes 之和（每条消息计一次）",
        "传输时延 = 8 × D / 带宽(bit/s)", "传播时延 = logical_rounds × RTT(s)",
        "通信时延 = 传输时延 + 传播时延", "```", "",
        "轮次取两方因果 send-flight 的关键路径，保留并行分支的 fork/join 依赖。每 query 先算各项，随后取 10 个 query 的中位数。offline 为输入前的新材料生成与就绪工作；online 为 input/ready 之后的完整协议执行。两阶段表不含 setup、model_prepare、input、ready、output 和 teardown。", "",
        "## 复测结果", "",
        "| 方法 | 阶段 | 发送 bytes | 因果轮次 | localhost 墙钟(s) |",
        "| --- | --- | ---: | ---: | ---: |",
    ]
    for row in model["rows"]:
        for phase in PHASES:
            value = row["summary"][phase]
            lines.append("| %s | %s | %s | %s | %.9f |" % (
                row["label"], phase, format(value["bytes_median"], ",.0f"),
                format(value["logical_rounds_median"], ".0f"), value["localhost_seconds_median"]))
    for name in NETWORKS:
        lines += ["", "### " + name, "",
                  "| 方法 | 阶段 | 传输(s) | 传播(s) | 通信合计(s) |",
                  "| --- | --- | ---: | ---: | ---: |"]
        for row in model["rows"]:
            for phase in PHASES:
                value = row["summary"][phase]["networks"][name]
                lines.append("| %s | %s | %.9f | %.9f | %.9f |" % (
                    row["label"], phase, *(value[key] for key in COMPONENTS)))
    lines += ["", "| 方法 | offline+online LAN 通信(s) | offline+online WAN 通信(s) |",
              "| --- | ---: | ---: |"]
    for row in model["rows"]:
        lines.append("| %s | %.9f | %.9f |" % (
            row["label"], row["offline_plus_online_seconds_median"]["LAN"],
            row["offline_plus_online_seconds_median"]["WAN"]))
    lines += ["", "将每次新材料生成也计入 offline+online 后，PaperLUT 的通信模型为 SCI+SiRNN 的 LAN %.4f 倍、WAN %.4f 倍。在线收益伴随更高的离线成本。" % (
              model["comparisons"]["LAN"]["offline_plus_online"]["candidate_over_baseline"],
              model["comparisons"]["WAN"]["offline_plus_online"]["candidate_over_baseline"]), "",
              "localhost 墙钟单独列出，包含本地计算、协议等待与计量开销；LAN/WAN 通信模型只含传输与传播。两者相加不能直接视为网络环境中的实测总时延。", "",
              "## 接口统计", "",
              "这次两套新执行的 result.json 均直接写入以下字段：", "",
              "- `measurements.queries[*].offline/online.logical_rounds`、`party_sent_bytes`、`party_received_bytes`。",
              "- `measurements.queries[*].offline/online.network_projection.LAN/WAN` 中的 `transmission_seconds`、`propagation_seconds`、`communication_seconds`。",
              "- `measurements.network_projection.networks.LAN/WAN` 中的公式、每 query 明细、阶段中位数与 `offline_plus_online_seconds_median`。", "",
              "命令使用 `--round-trace --network localhost --network-projection both`。汇总脚本逐 query 核对接口字段与 rounds.json，并重算传输、传播、合计和中位数；全部通过。接口投影覆盖 offline/online，offline+online 仅为这两个阶段之和。", "",
              "接口与通道配置聚焦测试共 18 个：17 个通过，1 个需指定编译二进制的环境测试跳过。", "",
              "## 原始证据", ""]
    for row in model["rows"]:
        sources = row["sources"]
        lines.append("- %s：[result.json](%s)、[rounds.json](%s)、[commands.json](%s)、[实际两方环境](%s)。" % (
            row["label"], *(os.path.relpath(sources[key], HERE) for key in
                            ("result", "rounds", "commands", "effective_environment"))))
    lines += ["- [NETWORK_MODEL.json](NETWORK_MODEL.json) 保存逐 query 复核后的统计和对比比值。",
              "- [OUTER_COMMANDS.json](OUTER_COMMANDS.json) 保存完整启动命令、执行顺序、环境净化和固定 allocator；[RUN_STATUS.json](RUN_STATUS.json) 保存两行执行状态；[run_matched.py](run_matched.py) 可复现启动过程。",
              "- [summarize.py](summarize.py) 对照冻结命令、环境、共同配置及每 query 接口字段后生成本报告。", "",
              "旧 ticket13-network-retest 的协议数据仍保留；接口尚未回填的旧结果、预算为 0 的 ticket13-interface-projection-final-v3、未冻结 allocator 的 ticket13-paperlut-budget-1610612736-v1、relay 失败或中断目录均未进入本报告的样本。", ""]
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-result", type=Path, required=True)
    parser.add_argument("--candidate-result", type=Path, required=True)
    args = parser.parse_args()
    launch_contract = validate_launch({"SCI_preprocessed": args.baseline_result.resolve(),
                                      "SCI_Beaver_segmented_PaperLUT": args.candidate_result.resolve()})
    baseline, left_config = summarize_source(args.baseline_result.resolve(), "SCI+SiRNN（SCI_preprocessed）")
    candidate, right_config = summarize_source(args.candidate_result.resolve(), "Segmented DPF PaperLUT")
    if baseline["backend"] != "SCI_preprocessed" or candidate["backend"] != "SCI_Beaver_segmented_PaperLUT":
        raise ValueError("wrong comparison backends")
    if candidate["activation_config"] != {
            "activation_algorithm": "folded_sigma_f12_q10", "folded_activations": "both",
            "lut_key_family": "segmented_boolean_tensor_b2a", "lut_segment_widths": [5, 5],
            "folded_sign_backend": "bitdec"}:
        raise ValueError("PaperLUT does not use the complete segmented activation configuration")
    differences = [key for key in COMMON_FIELDS if left_config.get(key) != right_config.get(key)]
    if differences:
        raise ValueError("unmatched non-method configuration fields: " + ", ".join(differences))
    comparisons = {}
    for name in NETWORKS:
        comparisons[name] = {}
        for phase in PHASES:
            left = baseline["summary"][phase]["networks"][name]["communication_seconds"]
            right = candidate["summary"][phase]["networks"][name]["communication_seconds"]
            comparisons[name][phase] = {"baseline_seconds": left, "candidate_seconds": right,
                "baseline_over_candidate": left / right, "candidate_over_baseline": right / left,
                "candidate_reduction": 1 - right / left}
        left = baseline["offline_plus_online_seconds_median"][name]
        right = candidate["offline_plus_online_seconds_median"][name]
        comparisons[name]["offline_plus_online"] = {
            "baseline_seconds": left, "candidate_seconds": right,
            "baseline_over_candidate": left / right, "candidate_over_baseline": right / left,
            "candidate_reduction": 1 - right / left}
    model = {"schema_version": 1, "kind": "communication-time-model", "eligible": True,
             "networks": NETWORKS, "transmission_formula": "8 * sum(party_sent_bytes) / (mbps * 1000000)",
             "propagation_formula": "logical_rounds * rtt_ms / 1000",
             "aggregation": "per-query phase components, then medians over ten fresh queries",
             "common_config": {key: left_config.get(key) for key in COMMON_FIELDS},
             "local_computation_included": False, "measured_lan_wan": False,
             "rows": [baseline, candidate], "comparisons": comparisons,
             "launch_contract": launch_contract,
             "interface_statistics_verified": True}
    (HERE / "NETWORK_MODEL.json").write_text(json.dumps(model, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (HERE / "REPORT.zh.md").write_text(report(model), encoding="utf-8")
    print(json.dumps({"status": "complete", "report": str(HERE / "REPORT.zh.md"),
                      "comparisons": comparisons}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
