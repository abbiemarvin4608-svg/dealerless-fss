#!/usr/bin/env python3
"""Audit executed RNN conversion domains without running a protocol."""

import argparse
import csv
import json
import math
from pathlib import Path
import statistics


HERE = Path(__file__).resolve().parent
REPO = HERE.parents[3]
DEFAULT_SOURCE = REPO / "project/our/results/online_optimization_20261014/formal/ours_online_optimized"
DEFAULT_OUTPUT = REPO / "project/our/results/precision_motivation_final/experiment2"
CALL_NAMES = {0: "div_pow2_toward_zero", 1: "truncate_floor",
              2: "truncate_and_reduce_floor", 3: "convert"}
CONTRACT_FIELDS = ("operator", "index", "slot", "count", "input_bits", "output_bits",
                   "shift", "call", "numeric_mode", "comparison_backend",
                   "low_comparison_bits", "sign_comparison_bits", "wrap_comparison_bits",
                   "low_nseg", "sign_nseg", "wrap_nseg", "low_widths", "sign_widths",
                   "wrap_widths", "fused_sign_wrap")


def read_json(path):
    return json.loads(Path(path).read_text())


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")


def write_csv(path, rows):
    if not rows:
        raise ValueError("cannot export an empty evidence table")
    with Path(path).open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def index_calls(rows):
    indexed = {}
    for row in rows:
        key = (row["query"], row["index"], row["slot"])
        if key in indexed:
            raise ValueError("duplicate conversion identity: %r" % (key,))
        indexed[key] = row
    return indexed


def contract(row):
    return {key: row[key] for key in CONTRACT_FIELDS}


def checked_time(row, field):
    value = row[field]
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
        raise ValueError("invalid measured duration: " + field)
    return value


def analyze(config, parties):
    if (config.get("batch"), config.get("queries"), config.get("steps")) != (128, 10, 1):
        raise ValueError("expected the completed batch128, ten-query, one-timestep RNN record")
    if len(parties) != 2 or {party.get("party") for party in parties} != {1, 2}:
        raise ValueError("exactly two distinct parties are required")
    parties = sorted(parties, key=lambda party: party["party"])
    calls_by_party, details_by_party = [], []
    for party in parties:
        if party.get("status") != "complete":
            raise ValueError("incomplete protocol record")
        calls = index_calls(party["conversion_calls"])
        offline = index_calls(party["conversion_offline_calls"])
        details = index_calls(party["conversion_details"])
        if set(calls) != set(offline) or set(calls) != set(details):
            raise ValueError("offline/online/detail identities differ")
        if {key[0] for key in calls} != set(range(config["queries"])):
            raise ValueError("query coverage differs from the frozen configuration")
        for key, call in calls.items():
            if contract(call) != contract(offline[key]):
                raise ValueError("offline and online contracts differ")
            if call["comparison_backend"] != "segmented_dcf" or call["numeric_mode"] != "sci_compatible":
                raise ValueError("unexpected conversion backend or numeric contract")
            detail = details[key]
            if detail["timing_kind"] != "serial_phase_wall" or detail["raw_secret_openings"] != 0:
                raise ValueError("unexpected timing or opening contract")
            expected_keys = call["count"] * sum(call[field] for field in ("low_nseg", "sign_nseg", "wrap_nseg"))
            if detail["comparison_keys"] != expected_keys:
                raise ValueError("executed comparison keys disagree with the domain plan")
            checked_time(detail, "total_online_ms")
            checked_time(detail, "comparison_online_ms")
        calls_by_party.append(calls)
        details_by_party.append(details)
    if set(calls_by_party[0]) != set(calls_by_party[1]) or any(
            contract(call) != contract(calls_by_party[1][key]) for key, call in calls_by_party[0].items()):
        raise ValueError("the parties disagree on executed public conversion contracts")

    reference = calls_by_party[0]
    slots = sorted({key[1:] for key in reference})
    operator_rows = []
    for index, slot in slots:
        calls = [reference[(query, index, slot)] for query in range(config["queries"])]
        first = calls[0]
        if any(contract(call) != contract(first) for call in calls):
            raise ValueError("conversion contract changed between queries")
        row = {"operator": first["operator"], "index": index, "slot": slot,
               "input_bits": first["input_bits"], "output_bits": first["output_bits"],
               "shift": first["shift"], "semantics": CALL_NAMES[first["call"]],
               "scalar_instances_per_query": first["count"], "executed_queries": len(calls),
               "low_comparison_bits": first["low_comparison_bits"],
               "sign_comparison_bits": first["sign_comparison_bits"],
               "wrap_comparison_bits": first["wrap_comparison_bits"],
               "contains_domain_gt20": max(first[field] for field in
                   ("low_comparison_bits", "sign_comparison_bits", "wrap_comparison_bits")) > 20,
               "high_domain_segment_keys_per_party_per_query": first["count"] * sum(
                   first[segment] for width, segment in (("low_comparison_bits", "low_nseg"),
                       ("sign_comparison_bits", "sign_nseg"), ("wrap_comparison_bits", "wrap_nseg")) if first[width] > 20),
               "all_segment_keys_per_party_per_query": details_by_party[0][(0, index, slot)]["comparison_keys"]}
        for party, details in enumerate(details_by_party, 1):
            row[f"party{party}_conversion_work_ms_median"] = statistics.median(
                details[(query, index, slot)]["total_online_ms"] for query in range(config["queries"]))
        operator_rows.append(row)

    domain_rows = []
    for width in sorted({row["sign_comparison_bits"] for row in operator_rows if row["sign_comparison_bits"] > 20}):
        matches = [row for row in operator_rows if row["sign_comparison_bits"] == width]
        count = sum(row["scalar_instances_per_query"] for row in matches)
        domain_rows.append({"comparison_bits": width, "call_sites_per_query": len(matches),
                            "logical_comparisons_per_query": count,
                            "logical_comparisons_all_queries": count * config["queries"],
                            "queries": config["queries"], "batch": config["batch"],
                            "segment_key_shares_per_party_per_query": sum(
                                row["high_domain_segment_keys_per_party_per_query"] for row in matches)})
    if {row["comparison_bits"]: row["logical_comparisons_per_query"] for row in domain_rows} != {23: 6528, 31: 51200}:
        raise ValueError("the frozen RNN high-domain inventory changed")

    query_rows, paired_stage_rows = [], []
    high_slots = {(row["index"], row["slot"]) for row in operator_rows if row["contains_domain_gt20"]}
    for query in range(config["queries"]):
        phases = []
        for party_index, party in enumerate(parties):
            details = details_by_party[party_index]
            selected = [details[(query, index, slot)] for index, slot in slots]
            high = [details[(query, index, slot)] for index, slot in high_slots]
            total = sum(row["total_online_ms"] for row in selected)
            high_total = sum(row["total_online_ms"] for row in high)
            comparisons = sum(row["comparison_online_ms"] for row in selected)
            high_comparisons = sum(row["comparison_online_ms"] for row in high)
            if total <= 0 or comparisons <= 0:
                raise ValueError("cannot compute work shares from zero measured work")
            phase = [row for row in party["phases"] if row["query"] == query and row["phase"] == "online"]
            if len(phase) != 1:
                raise ValueError("missing or duplicated whole-stage online measurement")
            phases.append(phase[0])
            query_rows.append({"party": party["party"], "query": query,
                "all_conversion_calls": len(selected), "high_domain_conversion_calls": len(high),
                "all_conversion_serial_work_ms": total,
                "high_domain_conversion_serial_work_ms": high_total,
                "high_domain_conversion_work_share_percent": 100 * high_total / total,
                "all_comparison_serial_work_ms": comparisons,
                "high_domain_containing_comparison_serial_work_ms": high_comparisons,
                "high_domain_containing_comparison_work_share_percent": 100 * high_comparisons / comparisons,
                "whole_stage_online_wall_ms_context_only": checked_time(phase[0], "seconds") * 1000,
                "pure_high_domain_dcf_ms": None})
        if any(phases[i]["sent_bytes"] != phases[1-i]["received_bytes"] for i in range(2)):
            raise ValueError("whole-stage peer communication counts differ")
        paired_stage_rows.append({"query": query,
            "whole_stage_online_wall_ms": max(phase["seconds"] for phase in phases) * 1000,
            "whole_stage_online_sent_bytes": sum(phase["sent_bytes"] for phase in phases),
            "whole_stage_online_cpu_seconds": sum(phase["process_cpu_seconds"] for phase in phases)})
    return {"status": "complete_existing_record_analysis", "batch": config["batch"],
            "queries": config["queries"], "recurrent_timesteps": config["steps"],
            "protocol_runs_started": 0, "operators": operator_rows, "domains": domain_rows,
            "conversion_work": query_rows, "whole_stage_context": paired_stage_rows,
            "accounting": {
                "logical_instances": "One two-party instance counted once, never summed across parties.",
                "work_share_denominator": "Sum of serial_phase_wall durations of all conversion calls within the same party and query.",
                "work_share_numerator": "Same durations restricted to conversions containing a >20-bit domain; includes their low-domain and conversion work.",
                "not_whole_stage_fraction": "Calls can overlap across contexts; summed durations are not wall time or CPU time and are not divided by whole-stage wall time.",
                "pure_high_domain_dcf_timing": "Unavailable: comparison_online_ms combines the low/sign/wrap comparisons of a conversion.",
                "offline_attribution": "Not joined: preprocess records lack query/context/call identity for an unambiguous attribution."}}


def report(result):
    lines = ["# 实验二：已有混合位宽 RNN 的实际 FSS 域", "",
        "来源为已完成的 Google-30 / FastGRNN 单步执行：batch=128，10 次全新 query，双方均 complete。本次只解析既有记录，未启动协议。", "",
        "## 实际调用", "",
        "| 算子 | 输入→输出 | 右移 | 低位比较域 | 符号比较域 | 每 query 标量数 |", "|---|---:|---:|---:|---:|---:|"]
    for row in result["operators"]:
        if row["contains_domain_gt20"]:
            lines.append(f"| {row['operator']} | {row['input_bits']}→{row['output_bits']} | {row['shift']} | {row['low_comparison_bits']} | {row['sign_comparison_bits']} | {row['scalar_instances_per_query']:,} |")
    lines += ["", "每 query 的 23 位逻辑比较为 6,528 次，31 位为 51,200 次，合计 57,728 次；两方共同完成的实例只计一次。10 次 query 对应 65,280 和 512,000 次。23 位域拆为 [6,6,6,5]，31 位域拆为 [8,8,8,7]。", "",
        "这些行采用 SCI 兼容的有符号 div_pow2，向零舍入并保持输出环。输入张量与激活接口的 8/16 位不代表所有转换比较均为低位宽。这个结论限定于该实际整数执行路径，不是对所有等价实现的位宽下界。", "",
        "对照行：36→32、右移 4 的 truncate_and_reduce 仅调用低 4 位比较；29→24、右移 5 仅调用低 5 位比较。16→36 扩环使用输入的 15 位符号比较。现有记录没有 32→16 转换，不能将 23/31 位域泛化到任意缩环 floor 操作。", "",
        "## 转换开销占比", "",
        "下表先在每个 query 内求和、求比例，再汇报 10 次中位数。分子是含高位域的 6 个完整转换调用，分母是全部 19 个转换调用。", "",
        "| 方 | 全部转换串行阶段工作量中位数 (ms) | 含高位域转换工作量中位数 (ms) | 工作量占比中位数 |", "|---|---:|---:|---:|"]
    for party in (1, 2):
        rows = [row for row in result["conversion_work"] if row["party"] == party]
        median = lambda field: statistics.median(row[field] for row in rows)
        lines.append(f"| {party} | {median('all_conversion_serial_work_ms'):.3f} | {median('high_domain_conversion_serial_work_ms'):.3f} | {median('high_domain_conversion_work_share_percent'):.2f}% |")
    wall = statistics.median(row["whole_stage_online_wall_ms"] for row in result["whole_stage_context"])
    lines += ["", f"整个在线阶段的双方较大墙钟中位数为 {wall:.3f} ms，仅作背景。转换调用可跨 context 并行，以上相加值不是整阶段墙钟或 CPU 时间，不作为在线总时延百分比。", "",
        "comparison_online_ms 仍同时包含每个转换的低位域与高位域比较；现有统计不能单独拆出纯 23/31 位 DCF 耗时，也不能据此精确分摊其通信或离线成本。CSV 中的 comparison 工作量比例沿用相同边界，不冒充纯高域比例。", "",
        "## 产物", "", "- `operators.csv`：全部 19 个转换位置、实际域、逻辑次数、每方分段 key 数和调用工作量。",
        "- `domains.csv`：23/31 位域的实例总量，双方不重复计数。",
        "- `conversion_work.csv`：每方每 query 的转换工作量与占比。",
        "- `whole_stage_context.csv`：真实整阶段墙钟、CPU、通信，单独存放。",
        "- `analysis.json`：上述结构化结果、来源和计量边界。", ""]
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    result = analyze(read_json(args.source / "config.json"), read_json(args.source / "parties.json"))
    try:
        result["source"] = str(args.source.resolve().relative_to(REPO))
    except ValueError:
        result["source"] = str(args.source.resolve())
    args.output.mkdir(parents=True, exist_ok=True)
    for filename, field in (("operators.csv", "operators"), ("domains.csv", "domains"),
                            ("conversion_work.csv", "conversion_work"), ("whole_stage_context.csv", "whole_stage_context")):
        write_csv(args.output / filename, result[field])
    write_json(args.output / "analysis.json", result)
    (args.output / "REPORT.zh.md").write_text(report(result))
    print(json.dumps({"status": result["status"], "output": str(args.output), "domains": result["domains"]}))


if __name__ == "__main__":
    main()
