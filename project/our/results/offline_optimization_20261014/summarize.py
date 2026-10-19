#!/usr/bin/env python3
"""Summarize the matched seven-row offline experiment from recorded evidence."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import statistics

HERE = Path(__file__).resolve().parent
PHASES = ("offline", "online", "offline_plus_online")
NETWORKS = {"LAN": {"mbps": 1000, "rtt_ms": 0.5},
            "WAN": {"mbps": 400, "rtt_ms": 4}}
ORDER = ("ours_segmented_before", "ours_full10", "ours_full10_fused",
         "ours_full10_fused_blockcot", "ours_full10_fused_bothcot",
         "ours_optimized", "SCI_SiRNN")
LABELS = dict(zip(ORDER, ("5+5 LUT 原配置", "full10 算术 LUT", "+ sign/wrap 融合",
                        "+ block COT", "+ arithmetic COT", "+ 叶缓存（最终）", "SCI/SiRNN")))
COMMON_FIELDS = ("batch", "steps", "start", "queries", "warmup_queries", "hidden_file",
                 "fixture_identity", "fixture_files", "binary_sha256", "memory_limit_bytes_per_party",
                 "cpu_affinity_per_party", "graph_mode", "activation_workers_per_party")


def read(path):
    return json.loads(path.read_text())


def exactly(rows, **identity):
    selected = [row for row in rows if all(row.get(key) == value for key, value in identity.items())]
    if len(selected) != 1:
        raise ValueError("missing/duplicate evidence: " + str(identity))
    return selected[0]


def distribution(values):
    if not values:
        return None
    values = sorted(values)
    position = (len(values) - 1) * 0.95
    lower, upper = math.floor(position), math.ceil(position)
    p95 = values[lower] + (values[upper] - values[lower]) * (position - lower)
    return {"n": len(values), "median": statistics.median(values), "p95": p95,
            "min": values[0], "max": values[-1]}


def projection(sent_bytes, rounds, specification):
    transfer = 8 * sent_bytes / (specification["mbps"] * 1e6)
    latency = rounds * specification["rtt_ms"] / 1000
    return {"transfer_seconds": transfer, "round_seconds": latency,
            "total_seconds": transfer + latency}


def inventory(parties):
    result = {"zero_online_generation": True, "generation_delta_fields": [],
              "one_use": True, "balance_records": {}, "verification_flags": {},
              "applicable_verification_flags": []}
    seen_deltas = set()
    for party_index, party in enumerate(parties):
        online = [row for row in party.get("phases", []) if row.get("phase") == "online"]
        for row in online:
            for key, value in row.items():
                if key.endswith("_generation_epoch_delta"):
                    seen_deltas.add(key)
                    result["zero_online_generation"] &= value == 0
        for key in ("folded_activation_consumed", "dpf_lut_consumed"):
            for row in party.get(key, []):
                for field in ("online_generated", "online_generation_epoch_delta", "online_loaded_bytes"):
                    if field in row:
                        result["zero_online_generation"] &= row[field] == 0
        for key in ("folded_activation_balances", "dpf_lut_balances", "native_material_balances",
                    "native_triple_balances"):
            balances = party.get(key, [])
            result["balance_records"].setdefault(key, []).append(len(balances))
            for row in balances:
                result["one_use"] &= row.get("remaining") == 0
                if "requested" in row:
                    result["one_use"] &= row.get("consumed") == row["requested"]
                    result["one_use"] &= row.get("generated") == row.get("consumed", 0) + row.get("padding_discarded", 0)
                else:
                    result["one_use"] &= row.get("generated") == row.get("consumed")
                if "generated_elements" in row:
                    result["one_use"] &= row["generated_elements"] == row.get("consumed_elements")
                for field in ("online_generated", "online_generation_epoch_delta", "online_loaded_bytes"):
                    if field in row:
                        result["zero_online_generation"] &= row[field] == 0
        for key, value in party.items():
            if key.endswith("_verified") and isinstance(value, bool):
                result["verification_flags"].setdefault(key, []).append(value)
                # Inactive backends deliberately report verified=false.
                applicable = (key != "dpf_lut_verified" or bool(party.get("dpf_lut_balances"))) and \
                             (key != "folded_activation_verified" or bool(party.get("folded_activation_balances")))
                if applicable:
                    result["one_use"] &= value
                    if key not in result["applicable_verification_flags"]:
                        result["applicable_verification_flags"].append(key)
    result["generation_delta_fields"] = sorted(seen_deltas)
    result["zero_online_generation_evidence_present"] = bool(seen_deltas)
    result["one_use_evidence_present"] = any(any(counts) for counts in result["balance_records"].values())
    return result


def extract(declaration, state, plan, expected_query_count=10, require_nondiagnostic=True):
    path = Path(declaration["output"])
    result = read(path / "result.json")
    parties = read(path / "parties.json")
    config = result["config"]
    measurements = result["measurements"]
    rounds = result["round_measurements"]
    checks = {"complete": state["status"] == result.get("status") == "complete",
              "exit_codes": state.get("exit_code") == 0 and result.get("party_exit_codes") == [0, 0],
              "fresh_queries": config.get("queries") == expected_query_count and config.get("warmup_queries") == 0 and
                               len(measurements["queries"]) == expected_query_count,
              "workload": config.get("batch") == 128 and config.get("steps") == 1 and config.get("start") == 0 and
                          config.get("hidden_file") is None and (not require_nondiagnostic or not config.get("diagnostic")),
              "causal_rounds": rounds.get("status") == "passed",
              "frozen_binary": config.get("binary_sha256") == plan["build"]["binary"],
              "memory_limit": config.get("memory_limit_bytes_per_party") == 1610612736 and
                              result.get("address_space_hard_limit_bytes_per_party") == 1610612736,
              "unshaped_network": config.get("network", {}).get("name") == "localhost"}
    bridge_sources = config.get("runtime_manifest", {}).get("sources", {})
    embedded_bridge = {name: value for name, value in bridge_sources.items()
                       if name.endswith("libsegfss_sci_bridge.so")}
    # The runtime source manifest was written at runtime compilation. A later
    # core rebuild can leave this embedded source hash stale. PLAN/STATUS and
    # the runner's per-row live-build gate bind the actual shared library.
    bridge_matches = {name: hashlib.sha256(Path(name).read_bytes()).hexdigest() == plan["build"]["bridge"]
                      for name in embedded_bridge if Path(name).is_file()}
    checks["frozen_bridge_current_file"] = bool(bridge_matches) and all(bridge_matches.values())
    if not all(checks.values()):
        raise ValueError("%s contract failed: %s" % (declaration["id"],
            ", ".join(name for name, passed in checks.items() if not passed)))
    queries = []
    for query_id, measured in enumerate(measurements["queries"]):
        query = {"query": query_id, "phases": {}}
        for phase in ("offline", "online"):
            raw = measured[phase]
            participants = raw["per_party"]
            if len(participants) != 2:
                raise ValueError("missing two-party phase measurements")
            trace = exactly(rounds["phases"], query=query_id, phase=phase)
            party_sent = [row["sent_bytes"] for row in participants]
            party_received = [row["received_bytes"] for row in participants]
            sent_bytes = sum(party_sent)
            wall = max(row["seconds"] for row in participants)
            if sent_bytes != raw["sent_bytes"] or wall != raw["seconds"] or \
                    party_sent != trace["party_sent_bytes"] or party_received != trace["party_received_bytes"] or \
                    party_sent != list(reversed(party_received)):
                raise ValueError("phase traffic/wall differs between measurement and causal trace")
            logical_rounds = trace["logical_rounds"]
            query["phases"][phase] = {"sent_bytes": sent_bytes, "party_sent_bytes": party_sent,
                "party_received_bytes": party_received, "wall_seconds": wall,
                "party_wall_seconds": [row["seconds"] for row in participants],
                "causal_rounds": logical_rounds,
                "network_projection": {name: projection(sent_bytes, logical_rounds, specification)
                    for name, specification in NETWORKS.items()}}
        offline, online = (query["phases"][phase] for phase in ("offline", "online"))
        query["phases"]["offline_plus_online"] = {
            "sent_bytes": offline["sent_bytes"] + online["sent_bytes"],
            "party_sent_bytes": [offline["party_sent_bytes"][i] + online["party_sent_bytes"][i] for i in (0, 1)],
            "wall_seconds": offline["wall_seconds"] + online["wall_seconds"],
            "wall_rule": "sum of the two phase maxima; excludes input/ready/output/setup",
            "causal_rounds": offline["causal_rounds"] + online["causal_rounds"],
            "round_rule": "sum of separate offline/online phase causal depths",
            "network_projection": {name: {field: offline["network_projection"][name][field] +
                online["network_projection"][name][field] for field in ("transfer_seconds", "round_seconds", "total_seconds")}
                for name in NETWORKS}}
        query["query_total_seconds"] = measured["query_total"]["seconds"]
        queries.append(query)
    statistics_by_phase = {}
    for phase in PHASES:
        statistics_by_phase[phase] = {field: distribution([q["phases"][phase][field] for q in queries])
            for field in ("sent_bytes", "causal_rounds", "wall_seconds")}
        statistics_by_phase[phase]["network_projection"] = {name: {
            field: distribution([q["phases"][phase]["network_projection"][name][field] for q in queries])
            for field in ("transfer_seconds", "round_seconds", "total_seconds")} for name in NETWORKS}
    resources = []
    observed_peaks = result.get("process_resource_peaks", [{}, {}])
    for index, party in enumerate(parties):
        observed = observed_peaks[index]
        resources.append({"party": index + 1,
            "rss_peak_bytes": max(observed.get("rss_peak_bytes", 0), party.get("peak_rss_kib", 0) * 1024),
            "vm_peak_bytes": max(observed.get("vm_peak_bytes", 0), party.get("vm_peak_bytes", 0)),
            "threads_peak": observed.get("threads_peak")})
    accounting = inventory(parties)
    return {"id": declaration["id"], "status": "complete", "directory": str(path), "checks": checks,
            "common_config": {field: config.get(field) for field in COMMON_FIELDS},
            "options": {field: config.get(field) for field in ("backend", "lut_key_family", "bit_block_mode",
                "native_ring_cot", "cache_lut_leaves", "fuse_sign_wrap", "conversion_optimizations")},
            "queries": queries, "statistics": statistics_by_phase,
            "query_total_seconds": distribution([q["query_total_seconds"] for q in queries]),
            "resources_per_party": resources, "inventory": accounting,
            "setup": measurements.get("setup"), "model_prepare": measurements.get("model_prepare"),
            "first_query": measurements.get("first_query"),
            "session_total_seconds": measurements.get("total_seconds"),
            "embedded_bridge_source_hashes": embedded_bridge,
            "embedded_bridge_hash_matches_runner": all(value == plan["build"]["bridge"] for value in embedded_bridge.values())}


def comparisons(rows):
    pairs = [(ORDER[index - 1], ORDER[index], "cumulative_ablation") for index in range(1, 6)]
    pairs += [("ours_segmented_before", "ours_optimized", "final_vs_segmented"),
              ("SCI_SiRNN", "ours_optimized", "final_vs_SCI_SiRNN")]
    output = []
    for before, after, kind in pairs:
        comparison = {"baseline": before, "candidate": after, "kind": kind, "phases": {}}
        if rows[before]["status"] != "complete" or rows[after]["status"] != "complete":
            comparison["status"] = "unavailable"
            output.append(comparison)
            continue
        comparison["status"] = "complete"
        for phase in PHASES:
            baseline = rows[before]["statistics"][phase]
            candidate = rows[after]["statistics"][phase]
            metrics = {field: (baseline[field]["median"], candidate[field]["median"])
                       for field in ("wall_seconds", "sent_bytes", "causal_rounds")}
            metrics.update({name + "_projection_seconds": (baseline["network_projection"][name]["total_seconds"]["median"],
                candidate["network_projection"][name]["total_seconds"]["median"]) for name in NETWORKS})
            comparison["phases"][phase] = {field: {"baseline_median": left, "candidate_median": right,
                "baseline_over_candidate": left / right if right else None,
                "reduction_fraction": 1 - right / left if left else None} for field, (left, right) in metrics.items()}
        output.append(comparison)
    return output


def number(value, digits=4):
    return "缺失" if value is None else format(value, ".%df" % digits)


def report(data):
    lines = ["# 离线优化累计消融：batch=128、单个 recurrent timestep", ""]
    if data["status"] == "complete":
        final = data["rows"]["ours_optimized"]["statistics"]
        before = data["rows"]["ours_segmented_before"]["statistics"]
        sci = data["rows"]["SCI_SiRNN"]["statistics"]
        def network_ratio(phase, network):
            return final[phase]["network_projection"][network]["total_seconds"]["median"] / \
                   sci[phase]["network_projection"][network]["total_seconds"]["median"]
        lines += ["四条路径均已实现并完成复测。相对 5+5 原配置，离线通信减少 **%s%%**，"
                  "本机离线墙钟中位数由 **%s s 降至 %s s（减少 %s%%）**。" % (
            number(100 * (1 - final["offline"]["sent_bytes"]["median"] / before["offline"]["sent_bytes"]["median"]), 2),
            number(before["offline"]["wall_seconds"]["median"]), number(final["offline"]["wall_seconds"]["median"]),
            number(100 * (1 - final["offline"]["wall_seconds"]["median"] / before["offline"]["wall_seconds"]["median"]), 2)), "",
            "相对 SCI/SiRNN，最终版的 LAN/WAN 在线通信投影分别为其 **%s%% / %s%%**，"
            "离线投影分别为其 **%s× / %s×**；离线＋在线合计分别为其 **%s× / %s×**。"
            "在线优势仍伴随较高的每查询预处理成本。" % (
                number(100 * network_ratio("online", "LAN"), 2), number(100 * network_ratio("online", "WAN"), 2),
                number(network_ratio("offline", "LAN"), 3), number(network_ratio("offline", "WAN"), 3),
                number(network_ratio("offline_plus_online", "LAN"), 3), number(network_ratio("offline_plus_online", "WAN"), 3)), ""]
    lines += ["七行复用同一 fixture、runtime/core 构建、每方 4 个逻辑 CPU、1.5 GiB 地址空间上限及分配器配置。"
        "每行 10 次 fresh query，无 warmup；隐藏状态采用同一默认值。结果为本机未限速实测。", "",
        "墙钟取每个 query 两方阶段墙钟的最大值；通信取两方发送字节之和，仅计一次，"
        "不再加入接收字节。轮次来自完整因果轨迹，不能求和各线程轮次。"
        "p95 采用排序后线性插值。比值是 baseline/candidate 的中位数比，未计算置信区间。", "",
        "LAN（1000 Mbps、RTT 0.5 ms）和 WAN（400 Mbps、RTT 4 ms）使用 "
        "8 × 两方发送字节 / 带宽 + RTT × 因果轮次，仅为通信模型投影；"
        "不是 shaped 网络墙钟，也未加入或替代本机计算时间。", "",
        "状态：`%s`。" % data["status"], "",
        "| 配置 | 阶段 | 实测墙钟 median / p95 (s) | 两方发送 median / p95 (MiB) | 因果轮次 median / p95 | LAN / WAN 投影 median (s) |",
        "|---|---|---:|---:|---:|---:|"]
    for identity in ORDER:
        row = data["rows"][identity]
        if row["status"] != "complete":
            lines.append("| %s | %s | — | — | — | — |" % (LABELS[identity], row["status"]))
            continue
        for phase in PHASES:
            values = row["statistics"][phase]
            wall, sent, rounds = (values[field] for field in ("wall_seconds", "sent_bytes", "causal_rounds"))
            lines.append("| %s | %s | %s / %s | %s / %s | %s / %s | %s / %s |" % (
                LABELS[identity], {"offline": "离线", "online": "在线", "offline_plus_online": "离线＋在线"}[phase], number(wall["median"]), number(wall["p95"]),
                number(sent["median"] / (1 << 20), 2), number(sent["p95"] / (1 << 20), 2),
                number(rounds["median"], 0), number(rounds["p95"], 1),
                number(values["network_projection"]["LAN"]["total_seconds"]["median"]),
                number(values["network_projection"]["WAN"]["total_seconds"]["median"])))
    lines += ["", "| 比较（baseline → candidate） | 阶段 | 墙钟倍数 / 降幅 | 通信倍数 / 降幅 | 轮次倍数 | LAN / WAN 投影倍数 |",
              "|---|---|---:|---:|---:|---:|"]
    for comparison in data["comparisons"]:
        if comparison["status"] != "complete":
            continue
        for phase, values in comparison["phases"].items():
            wall, sent, rounds = (values[field] for field in ("wall_seconds", "sent_bytes", "causal_rounds"))
            lines.append("| %s → %s | %s | %s× / %s%% | %s× / %s%% | %s× | %s× / %s× |" % (
                LABELS[comparison["baseline"]], LABELS[comparison["candidate"]], {"offline": "离线", "online": "在线", "offline_plus_online": "离线＋在线"}[phase],
                number(wall["baseline_over_candidate"], 3), number(100 * wall["reduction_fraction"], 2),
                number(sent["baseline_over_candidate"], 3), number(100 * sent["reduction_fraction"], 2),
                number(rounds["baseline_over_candidate"], 3),
                number(values["LAN_projection_seconds"]["baseline_over_candidate"], 3),
                number(values["WAN_projection_seconds"]["baseline_over_candidate"], 3)))
    lines += ["", "离线＋在线本机墙钟是同一个 query 两个阶段最大值的和，"
              "不是 session 或完整 query 墙钟；网络投影也先逐 query 求和，再统计 median/p95。", "",
              "负降幅表示退化。累计消融表依次增加 full10、融合、block COT、arithmetic COT、叶缓存；"
              "每行都是独立的 fresh 材料运行。", "",
              "| 配置 | 两方 RSS 峰值 (MiB) | 两方 VmPeak (MiB) | 在线零生成证据 | 一次消费证据 |",
              "|---|---:|---:|---|---|"]
    for identity in ORDER:
        row = data["rows"][identity]
        if row["status"] != "complete":
            continue
        evidence = row["inventory"]
        zero = "通过" if evidence["zero_online_generation_evidence_present"] and evidence["zero_online_generation"] else "缺失或失败"
        once = "通过" if evidence["one_use_evidence_present"] and evidence["one_use"] else "缺失或失败"
        lines.append("| %s | %s | %s | %s | %s |" % (LABELS[identity],
            " / ".join(number(p["rss_peak_bytes"] / (1 << 20), 2) for p in row["resources_per_party"]),
            " / ".join(number(p["vm_peak_bytes"] / (1 << 20), 2) for p in row["resources_per_party"]), zero, once))
    lines += ["", "| 配置 | setup (s) | model_prepare (s) | 首次请求 (s) | session (s) |",
              "|---|---:|---:|---:|---:|"]
    for identity in ORDER:
        row = data["rows"][identity]
        if row["status"] != "complete":
            continue
        lines.append("| %s | %s | %s | %s | %s |" % (LABELS[identity],
            number((row.get("setup") or {}).get("seconds")), number((row.get("model_prepare") or {}).get("seconds")),
            number((row.get("first_query") or {}).get("seconds")), number(row.get("session_total_seconds"))))
    lines += ["", "setup、model_prepare、首次请求和 session 为单次 session 原始计时，"
              "不是 10-query 中位数。首次请求已包含 setup，不能重复相加。", ""]
    if any(row.get("embedded_bridge_hash_matches_runner") is False for row in data["rows"].values()):
        lines += ["runtime 编译时内嵌的 bridge source hash 早于最终 core 重建，"
                  "与最终共享库标识不同。实际共享库由 PLAN/STATUS、runner 每行启动前的文件标识核验"
                  "及汇总时文件标识核验绑定；两份 source hash 均保留于 METRICS.json。", ""]
    lines += ["材料一次消费由运行器公开计划与 offline/online 对应检查、bridge 在 query end 的库存检查，"
              "以及适用子系统的 balances/verified 记录共同验证；各子系统的显式 balance 记录数量不同。", "",
              "构建标识与分配器配置记录于 PLAN.json；逐 query 值、字段来源、"
              "累计比较及资源/库存记录保存在 METRICS.json。"
              "SCI/SiRNN 使用原生激活，ours 使用 folded f12/q10 激活；"
              "两者整数算法的区别保留，不能将跨算法比较解释为单个开关的收益。", ""]
    return "\n".join(lines)


def summarize(directory, allow_incomplete=False):
    plan, execution = read(directory / "PLAN.json"), read(directory / "STATUS.json")
    if plan.get("mode") != "formal" or [row["id"] for row in plan["rows"]] != list(ORDER):
        raise ValueError("expected the declared seven-row formal experiment")
    if execution.get("build") != plan["build"]:
        raise ValueError("execution build differs from frozen plan")
    if not allow_incomplete and execution.get("status") != "complete":
        raise ValueError("formal experiment is incomplete; use --allow-incomplete for a pending report")
    rows, errors = {}, []
    common = None
    for declaration in plan["rows"]:
        state = exactly(execution["rows"], id=declaration["id"])
        if state["status"] != "complete":
            rows[declaration["id"]] = {"id": declaration["id"], "status": state["status"], "reason": state.get("reason")}
            continue
        try:
            row = extract(declaration, state, plan)
            if common is None:
                common = row["common_config"]
            if row["common_config"] != common:
                raise ValueError("common fixture/build/workload/resources differ between rows")
            rows[declaration["id"]] = row
        except (ValueError, KeyError, OSError) as error:
            errors.append({"id": declaration["id"], "reason": str(error)})
            rows[declaration["id"]] = {"id": declaration["id"], "status": "invalid", "reason": str(error)}
    if errors and not allow_incomplete:
        raise ValueError(str(errors))
    return {"schema": 1, "status": "complete" if all(r["status"] == "complete" for r in rows.values()) else "incomplete",
        "directory": str(directory), "build": plan["build"], "allocator": plan["allocator"], "common_config": common,
        "network_specs": NETWORKS, "rows": rows, "comparisons": comparisons(rows), "errors": errors,
        "sources": {"phase_wall": "result.json measurements.queries[q][phase].per_party[*].seconds: max",
                    "sent_bytes": "same per_party[*].sent_bytes: sum, cross-checked with round_measurements.phases.party_sent_bytes",
                    "causal_rounds": "result.json round_measurements.phases[query=q,phase=phase].logical_rounds",
                    "resources": "max result.json process_resource_peaks and parties.json VmPeak/peak_rss_kib",
                    "zero_online_generation": "parties.json online phase *_generation_epoch_delta and activation stock records",
                    "one_use": "parties.json *_balances and *_verified flags",
                    "configuration": "PLAN.json/STATUS.json and each row result.json config"},
        "statistics_method": "median; p95 linear interpolation at sorted index 0.95*(n-1); no CI",
        "combined_phase_method": "per-query sum of offline and online max-party phase walls and projections, then median/p95; excludes other phases",
        "projection_method": "8*both-party-sent-bytes/(Mbps*1e6) + causal-rounds*RTT-ms/1000; communication model only"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--directory", type=Path, default=HERE / "formal")
    parser.add_argument("--allow-incomplete", action="store_true")
    args = parser.parse_args()
    directory = args.directory.resolve()
    data = summarize(directory, args.allow_incomplete)
    (directory / "METRICS.json").write_text(json.dumps(data, ensure_ascii=False, indent=2, sort_keys=True) + "\n")
    (directory / "REPORT.zh.md").write_text(report(data))
    print(json.dumps({"status": data["status"], "rows": len(data["rows"]),
                      "metrics": str(directory / "METRICS.json"), "report": str(directory / "REPORT.zh.md")}, ensure_ascii=False))


if __name__ == "__main__":
    main()
