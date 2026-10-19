#!/usr/bin/env python3
"""Summarize only existing precision-motivation experiment artifacts."""
import argparse
import csv
import json
from pathlib import Path
import statistics

from summarize import extract_run


REPO = Path(__file__).resolve().parents[4]
MODELS = ("resnet50", "resnet152")
BACKENDS = ("SCI", "SCI_preprocessed", "SCI_ours_beaver")
WRAP_SCOPES = ("parameter_encoding", "input_encoding", "dot_before_truncation",
               "bias_add", "residual_add", "avgpool_sum_before_division")


def read(path):
    return json.loads(Path(path).read_text()) if Path(path).is_file() else None


def write_json(path, document):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(document, ensure_ascii=False, indent=2) + "\n")
    temporary.replace(path)


def write_csv(path, rows):
    fields = list(dict.fromkeys(key for row in rows for key in row)) or ["status"]
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows({key: json.dumps(value, ensure_ascii=False) if isinstance(value, (dict, list)) else value
                         for key, value in row.items()} for row in rows)


def accuracy_rows(scan, anchors):
    rows = []
    for model in MODELS:
        selection = read(scan / model / "selected.json") or {}
        chosen = {row["bits"]: row for row in selection.get("rows", [])}
        for bits in (16, 20, 24, 28, 32):
            selected = chosen.get(bits, {})
            frac = 12 if bits == 32 else selected.get("frac")
            source = (anchors / (model + "_w32_f12") / "accuracy.json" if bits == 32 else
                      scan / model / "validation" / ("w%d_f%s" % (bits, frac)) / "accuracy.json")
            evidence = read(source) if frac is not None else None
            row = dict(model=model, bits=bits, selected_frac=frac,
                       role="reused_fixed_anchor" if bits == 32 else "calibration_selected",
                       status="pending", calibration_top1_pct=selected.get("calibration_top1_pct"),
                       calibration_samples=selected.get("calibration_samples"), completed_calibration_candidates=None,
                       pruned_calibration_candidates=None, resolved_calibration_candidates=None,
                       samples=evidence.get("samples") if evidence else None,
                       expected_samples=50000, fp32_top1_pct=None, fp32_top5_pct=None,
                       fixed_top1_pct=None, fixed_top5_pct=None, delta_top1_pp=None,
                       agreement_pct=None, overflow_status="missing", source=str(source) if frac is not None else None)
            if bits != 32:
                candidates = list((scan / model / "calibration").glob("w%d_f*/metrics.json" % bits))
                candidates = [read(path) or {} for path in candidates]
                row["completed_calibration_candidates"] = sum(bool(candidate.get("complete")) for candidate in candidates)
                row["pruned_calibration_candidates"] = sum(candidate.get("status") == "pruned" for candidate in candidates)
                row["resolved_calibration_candidates"] = row["completed_calibration_candidates"] + row["pruned_calibration_candidates"]
            complete = bool(evidence and evidence.get("status") == "complete"
                            and evidence.get("evidence_scope") == "formal" and evidence.get("samples") == 50000)
            if evidence and (evidence.get("model"), evidence.get("bits"), evidence.get("frac")) != (model, bits, frac):
                row.update(status="invalid", reason="accuracy identity differs from selected configuration")
                complete = False
            elif evidence:
                row["status"] = "reused_complete" if complete and bits == 32 else "complete" if complete else evidence.get("status", "pending")
                if row["status"] == "complete" and not complete:
                    row.update(status="incomplete", reason="not a complete formal 50000-image result")
            for scope in WRAP_SCOPES:
                for field in ("wrapped", "total", "rate_pct"):
                    row[scope + "_" + field] = None
            if complete:
                row.update(fp32_top1_pct=evidence["fp32"]["top1_pct"], fp32_top5_pct=evidence["fp32"]["top5_pct"],
                           fixed_top1_pct=evidence["fixed"]["top1_pct"], fixed_top5_pct=evidence["fixed"]["top5_pct"],
                           delta_top1_pp=evidence["fixed"]["top1_pct"] - evidence["fp32"]["top1_pct"],
                           agreement_pct=evidence["top1_agreement_pct"])
                counts = evidence.get("overflow_counts", {})
                row["overflow_status"] = "measured" if counts else "missing_historical_run" if bits == 32 else "missing"
                for scope in WRAP_SCOPES:
                    for field in ("wrapped", "total", "rate_pct"):
                        row[scope + "_" + field] = counts.get(scope, {}).get(field)
            rows.append(row)
    return rows


def primitive_rows(directory):
    reused_path = directory / "reused_batch100.csv"
    with reused_path.open(newline="") if reused_path.exists() else open("/dev/null") as stream:
        reused = list(csv.DictReader(stream))
    plan = read(directory / "missing_width_config.json") or {}
    source = directory / "missing_width_runs" / "summary.json"
    evidence = read(source) or {}
    observed = {(row["network"], row["backend"], row["nseg"]): row for row in evidence.get("rows", [])}
    new = []
    for network in plan.get("networks", [{"name": "lan_model"}, {"name": "low_latency_wan_model"}]):
        for backend, nseg in [("sci", None)] + [("ours", n) for n in plan.get("nseg", [1, 2, 4, 8])]:
            measured = observed.get((network["name"], backend, nseg))
            raw = [item for item in evidence.get("raw", []) if item.get("network", {}).get("name") == network["name"]
                   and item.get("backend") == backend and item.get("nseg") == nseg]
            row = dict(origin="new_missing_width", input_bits=28, comparison_bits=27, batch=128,
                       network=network["name"], backend=backend, nseg=nseg, status="pending",
                       expected_repeats=plan.get("repetitions", 5), completed=None,
                       source=str(source), raw_statuses=sorted({item["status"] for item in raw}),
                       raw_reasons=sorted({item["reason"] for item in raw if item.get("reason")}))
            if measured:
                row["completed"] = measured["completed"]
                row["status"] = ("complete" if measured["completed"] == row["expected_repeats"] else
                                 "pending" if "resource_interruption_pending_resume" in row["raw_reasons"] else
                                 "unavailable" if measured["completed"] == 0 and raw else "partial")
            for metric in ("setup_ms", "offline_ms", "online_ms", "total_ms", "link_bytes",
                           "setup_bytes", "offline_bytes", "online_bytes"):
                row[metric + "_median"] = (measured.get(metric) or {}).get("median") if measured else None
            new.append(row)
    return reused, new, evidence.get("status", "pending")


def performance_rows(directory):
    rows, phases, details = [], [], []
    for path in sorted(directory.glob("*/STATUS.json")):
        status = read(path)
        if status.get("model") not in MODELS or status.get("backend") not in BACKENDS or status.get("diagnostic"):
            continue
        row = dict(model=status["model"], backend=status["backend"], bits=status.get("bits"), frac=status.get("frac"),
                   batch=status.get("batch"), queries=status.get("queries"), status=status.get("status"),
                   reason=status.get("reason"), source=str(path.parent), total_wall_seconds=None,
                   launcher_wall_seconds=None, total_sent_bytes=None, total_process_cpu_seconds=None,
                   online_wall_seconds_median=None, output_values_checked=None,
                   evidence_scope=status.get("evidence_scope"), protocol_started=status.get("protocol_started"),
                   resource_preflight=status.get("resource_preflight"))
        try:
            if status.get("status") == "resource_unavailable":
                preflight = status.get("resource_preflight") or {}
                if (status.get("backend") != "SCI_preprocessed"
                        or (status.get("bits"), status.get("frac")) != (32, 12)
                        or status.get("protocol_started") is not False
                        or status.get("evidence_scope") != "public_shape_resource_preflight_not_measurement"
                        or preflight.get("status") != "resource_unavailable" or not preflight.get("violations")):
                    raise ValueError("resource-unavailable row requires an explicit unstarted 32/12 resource preflight")
                rows.append(row)
                continue
            result = extract_run(path.parent)
            if (result.get("bits"), result.get("frac")) != (32, 12):
                raise ValueError("expected the frozen 32/12 CNN configuration")
            details.append(result)
            row["status"] = result["status"]
            if result["status"] == "complete":
                measured = result["samples"]
                row.update(total_wall_seconds=result.get("protocol_process_wall_seconds"),
                           launcher_wall_seconds=result.get("launcher_wall_seconds"),
                           total_sent_bytes=sum(value["sent_bytes"] for value in measured),
                           total_process_cpu_seconds=sum(value["cpu_seconds_total"] for value in measured),
                           online_wall_seconds_median=(result["statistics"].get("online", {}).get("wall_seconds") or {}).get("median"),
                           output_values_checked=(result.get("output_audit") or {}).get("values_checked"))
                for value in measured:
                    phases.append(dict(model=row["model"], backend=row["backend"], source=str(path.parent),
                        query=value["query"], phase=value["phase"], wall_seconds=value["wall_seconds"],
                        cpu_seconds_total=value["cpu_seconds_total"], sent_bytes=value["sent_bytes"],
                        causal_rounds=value.get("causal_rounds")))
        except (ValueError, KeyError, OSError, TypeError) as error:
            row.update(status="invalid_or_incomplete", reason=str(error))
        rows.append(row)
    present = {(row["model"], row["backend"]) for row in rows}
    for model in MODELS:
        for backend in BACKENDS:
            if (model, backend) not in present:
                rows.append(dict(model=model, backend=backend, bits=32, frac=12, batch=1, queries=None,
                                 status="pending", reason="no qualified run artifact", source=None,
                                 total_wall_seconds=None, launcher_wall_seconds=None, total_sent_bytes=None,
                                 total_process_cpu_seconds=None, online_wall_seconds_median=None, output_values_checked=None,
                                 evidence_scope=None, protocol_started=None, resource_preflight=None))
    return rows, phases, details


def display(value):
    if value is None or value == "":
        return "pending"
    return ("%.4f" % value).rstrip("0").rstrip(".") if isinstance(value, float) else str(value)


def table(headers, rows):
    return ["| " + " | ".join(headers) + " |", "|" + "---|" * len(headers)] + [
        "| " + " | ".join(display(value).replace("|", "/").replace("\n", " ") for value in row) + " |" for row in rows]


def run(args):
    output = args.output_dir or args.root / "results"
    output.mkdir(parents=True, exist_ok=True)
    preparation = read(args.root / "data/DOWNLOAD_STATUS.json") or {}
    accuracy = accuracy_rows(args.width_scan_dir or args.root / "experiment1/width_scan", args.anchor_dir)
    rnn_path = args.root / "experiment2/analysis.json"
    rnn = read(rnn_path) or {"status": "pending", "domains": []}
    reused, new, primitive_status = primitive_rows(args.root / "primitive_ablation")
    cnn, phases, details = performance_rows(args.performance_dir or args.root / "performance")
    complete_accuracy = sum(row["status"] in ("complete", "reused_complete") for row in accuracy)
    complete_cnn = sum(any(row["model"] == model and row["backend"] == backend and row["status"] == "complete"
                           for row in cnn) for model in MODELS for backend in BACKENDS)
    unavailable_cnn = sum(
        not any(row["model"] == model and row["backend"] == backend and row["status"] == "complete" for row in cnn)
        and any(row["model"] == model and row["backend"] == backend and row["status"] == "resource_unavailable" for row in cnn)
        for model in MODELS for backend in BACKENDS)
    counts_complete = all(row["bits"] == 32 or all(row[scope + "_total"] is not None for scope in WRAP_SCOPES) for row in accuracy)
    done = (complete_accuracy == 10 and counts_complete and complete_cnn + unavailable_cnn == 6 and reused
            and rnn["status"] == "complete_existing_record_analysis"
            and primitive_status in ("complete", "complete_with_resource_failures"))
    completed_status = ("complete_with_resource_failures" if unavailable_cnn
                        or primitive_status == "complete_with_resource_failures" else "complete")
    state = dict(status=completed_status if done else "pending", experiment1=dict(completed=complete_accuracy, expected=10,
                 calibration_data=preparation, rows=accuracy),
                 experiment2=dict(status=rnn["status"], source=str(rnn_path), analysis=rnn),
                 experiment3=dict(primitive_status=primitive_status, reused_rows=reused, missing_width_rows=new,
                                  completed_cnn=complete_cnn, resource_unavailable_cnn=unavailable_cnn,
                                  resolved_cnn=complete_cnn + unavailable_cnn, expected_cnn=6,
                                  cnn_rows=cnn, qualified_run_details=details))
    write_json(output / "status.json", state)
    for name, rows in (("experiment1_accuracy.csv", accuracy), ("experiment2_domains.csv", rnn.get("domains", [])),
                       ("primitive_reused_batch100.csv", reused), ("primitive_new_batch128.csv", new),
                       ("cnn_performance.csv", cnn), ("cnn_phases.csv", phases)):
        write_csv(output / name, rows)
    lines = ["# 位宽实验结果与当前状态", "", "总状态：`%s`。本报告只汇总已保存的真实产物，不启动实验。" % state["status"], "",
             "## 实验一：统一环位宽", "", "完整精度行 %d/10；其中32/12为原结果复用，缺失回绕统计不补零。" % complete_accuracy, ""]
    if preparation:
        lines += ["训练校准数据：状态 `%s`，已保存 %s 张、%s 类；原因：%s。" % (
            preparation.get("status"), preparation.get("images", preparation.get("samples", 0)),
            preparation.get("sampled_classes", preparation.get("classes", 0)), preparation.get("reason", "N/A")), ""]
    lines += table(["模型", "bits/s", "状态", "校准Top-1", "验证样本", "FP32 Top-1", "定点Top-1/Top-5", "一致率"], [
        [r["model"], "%s/%s" % (r["bits"], display(r["selected_frac"])), r["status"],
         "N/A" if r["bits"] == 32 else r["calibration_top1_pct"], r["samples"],
         r["fp32_top1_pct"], "%s / %s" % (display(r["fixed_top1_pct"]), display(r["fixed_top5_pct"])), r["agreement_pct"]] for r in accuracy])
    lines += ["", "回绕的完整计数、分母和比例见 `experiment1_accuracy.csv`。点积分母是截断前完整点积的输出标量数，不是逐项MAC次数。", "",
              "## 实验二：混合位宽RNN记录", "", "来源：`%s`；状态：`%s`。" % (rnn_path, rnn["status"]), ""]
    lines += table(["比较域位宽", "每步逻辑比较", "实测query数", "batch"], [[r["comparison_bits"], r["logical_comparisons_per_query"], r["queries"], r["batch"]] for r in rnn.get("domains", [])])
    work = rnn.get("conversion_work", [])
    if work:
        lines += ["", "各party/query中，包含>20位域的转换调用占全部转换串行工作时间的比例，中位数为 %.4f%%。" % statistics.median(r["high_domain_conversion_work_share_percent"] for r in work),
                  "该值包括这些调用的低位域及转换工作，不是高位DCF独占时间，也不是整步墙钟占比。高位DCF独占时延仍为缺失。"]
    lines += ["", "## 实验三：原语与CNN成本", "", "复用原语配置 %d 行，保留原batch100及失败记录；详见 `primitive_reused_batch100.csv`。" % len(reused),
              "缺失输入28位对应比较域27位，新batch128补测状态：`%s`。两种batch不计算交叉加速比。" % primitive_status, ""]
    lines += table(["网络", "后端", "分段", "状态", "完成重复", "offline中位ms", "online中位ms"], [
        [r["network"], r["backend"], "N/A" if r["backend"] == "sci" else r["nseg"],
         r["status"], r["completed"], r["offline_ms_median"], r["online_ms_median"]] for r in new])
    lines += ["", "CNN已实测并完整核验 %d/6；资源预检不可用 %d/6；已处理 %d/6。" %
              (complete_cnn, unavailable_cnn, complete_cnn + unavailable_cnn),
              "只读取本次performance目录，不纳入迁移小图和旧暂停批次。资源不可用行未启动协议，N/A表示无实测值。", ""]
    lines += table(["模型", "后端", "状态", "协议总墙钟s", "online中位s", "双方发送bytes", "输出核对数", "原因"], [
        [r["model"], r["backend"], r["status"]] + ["N/A" if r["status"] == "resource_unavailable" else r[key]
            for key in ("total_wall_seconds", "online_wall_seconds_median", "total_sent_bytes", "output_values_checked")]
        + [r.get("reason") or "N/A"] for r in cnn])
    constraints = [[r["model"], r["backend"], violation.get("constraint"), violation.get("required_bytes"),
                    violation.get("available_bytes")] for r in cnn if r["status"] == "resource_unavailable"
                   for violation in r["resource_preflight"]["violations"]]
    if constraints:
        lines += ["", "以下是公开图形状的资源预检，所需字节数属于静态估算：", ""]
        lines += table(["模型", "后端", "资源约束", "所需bytes（估算）", "可用bytes"], constraints)
    lines += ["", "各阶段墙钟、CPU和通信见 `cnn_phases.csv`；总墙钟缺失时不以阶段最大值之和替代。完整来源、资源记录和未完成原因保留于 `status.json`。", ""]
    (output / "REPORT.zh.md").write_text("\n".join(lines))
    print(json.dumps({"status": state["status"], "accuracy_rows_complete": complete_accuracy,
                      "cnn_combinations_complete": complete_cnn, "cnn_resource_unavailable": unavailable_cnn,
                      "report": str(output / "REPORT.zh.md")}, ensure_ascii=False))
    return state


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=REPO / "project/our/results/precision_motivation_final")
    parser.add_argument("--width-scan-dir", type=Path)
    parser.add_argument("--performance-dir", type=Path)
    parser.add_argument("--anchor-dir", type=Path, default=REPO / "project/our/results/imagenet_resnet_20261015/accuracy")
    parser.add_argument("--output-dir", type=Path)
    run(parser.parse_args())


if __name__ == "__main__":
    main()
