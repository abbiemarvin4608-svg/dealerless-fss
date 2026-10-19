#!/usr/bin/env python3
"""Publish the ReLU-only ablation report after both full model runs finish."""
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parent
MODELS = ("resnet50", "resnet152")


def read_csv(path):
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def integer(row, key):
    return int(float(row[key]))


def load_predictions(path):
    saved = torch.load(path, map_location="cpu", weights_only=False)
    return saved["predictions"], saved["targets"]


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def verify_predictions(directory, row, reference, baseline):
    predictions, targets = load_predictions(directory / row["prediction_file"])
    require(torch.equal(targets, reference[1]), "prediction target/order mismatch")
    require(len(targets) == 3925, "full 3925-image validation is required")
    correct = int(predictions.eq(targets).sum())
    require(integer(row, "top1_correct") == correct, "saved predictions disagree with correct count")
    checks = {
        "top1_pct": 100.0 * correct / len(targets),
        "prediction_agreement_pct": 100.0 * int(predictions.eq(reference[0]).sum()) / len(targets),
        "baseline_prediction_agreement_pct": 100.0 * int(predictions.eq(baseline[0]).sum()) / len(targets),
        "delta_from_wide_baseline_pp": 100.0 * (
            correct - int(baseline[0].eq(targets).sum())) / len(targets),
    }
    for key, value in checks.items():
        require(math.isclose(float(row[key]), value, rel_tol=0, abs_tol=1e-8),
                f"saved predictions disagree with {key}")
    return predictions, targets


def verify_reused_baseline(directory, model, row, metadata, baseline):
    if row["evaluation"] != "historical_control_reuse":
        require(row["evaluation"] == "measured", "unknown wide baseline provenance")
        return "本次完整实测"
    source_path = Path(row["source_prediction_file"])
    source_predictions, source_targets = load_predictions(source_path)
    require(torch.equal(source_predictions, baseline[0]) and torch.equal(source_targets, baseline[1]),
            "copied historical control differs from source prediction vectors")
    source_directory = source_path.parents[3]
    previous = json.loads((source_directory / f"uniform_storage_{model}_metadata.json").read_text())
    keys = ("model", "weights", "weights_url", "transform", "dataset", "dataset_root",
            "class_to_imagenet", "classifier_classes", "calibration_samples",
            "validation_samples", "register_bits", "fp32_arithmetic", "integer_backend")
    require(all(previous.get(key) == metadata.get(key) for key in keys),
            "historical control metadata differs from current experiment")
    require(previous.get("arithmetic") ==
            "uniform_ell_bit_storage_wide_integer_dot_then_floor_rescale",
            "historical control must use wide accumulation and saturation")
    # The runner additionally checks its legacy arithmetic identity and a
    # freshly computed first batch before it accepts this source vector.
    for split in ("calibration", "validation"):
        name = f"{model}_{split}_manifest.csv"
        require(read_csv(source_directory / name) == read_csv(directory / name),
                "historical control sample manifest differs")
    return f"历史控制组复用；来源 `{source_path}`"


def collect(directory):
    states = {}
    for model in MODELS:
        path = directory / f"relu_{model}_status.json"
        require(path.exists(), f"{model} 尚无完成状态；暂不生成 RESULTS_CN.md")
        state = json.loads(path.read_text())
        require(state.get("status") == "completed",
                f"{model} 状态为 {state.get('status')}；暂不生成 RESULTS_CN.md")
        require(state.get("evidence_scope") == "formal"
                and state.get("calibration_samples") == 512
                and state.get("validation_samples") == 3925,
                f"{model} 不是完整校准/验证实验")
        states[model] = state
    rows = read_csv(directory / "final_results.csv")
    calibration = read_csv(directory / "calibration_candidates.csv")
    result = {}
    for model, state in states.items():
        metadata = json.loads((directory / f"relu_{model}_metadata.json").read_text())
        require(metadata.get("base_ell") == 24 and metadata.get("base_s") == 15
                and metadata.get("activation_bits") == 16
                and metadata.get("register_bits") == 128
                and metadata.get("arithmetic_mode") == "wide"
                and metadata.get("overflow_mode") == "saturate"
                and metadata.get("activation_scope") == "every_signed_relu_input_and_output",
                f"{model} experiment semantics differ from the requested ablation")
        candidates = [row for row in calibration if row["model"] == model
                      and row["variant"] == "relu_quantized"]
        require(len(candidates) == 16 and {integer(row, "activation_s") for row in candidates} == set(range(16)),
                f"{model} needs all 16 training-calibration candidates")
        require(all(integer(row, "samples") == 512 and row["status"] == "completed"
                    and row["evaluation"] == "measured" for row in candidates),
                f"{model} calibration candidates must be independently measured")
        selected = integer(max(candidates, key=lambda row: (
            integer(row, "top1_correct"), -integer(row, "activation_s"))), "activation_s")
        require(selected == state["selected_s"] and state["preset_s"] == 12,
                f"{model} calibration selection mismatch")
        model_rows = [row for row in rows if row["model"] == model]
        expected_s = {selected, 12}
        require(len(model_rows) == 2 + len(expected_s), f"{model} unexpected validation rows")
        require(all(row["split"] == "validation" and row["status"] == "completed"
                    and integer(row, "samples") == 3925
                    and integer(row, "base_ell") == 24 and integer(row, "base_s") == 15
                    for row in model_rows), f"{model} validation scope mismatch")
        fp32_rows = [row for row in model_rows if row["variant"] == "fp32"]
        wide_rows = [row for row in model_rows if row["variant"] == "wide_baseline"]
        require(len(fp32_rows) == len(wide_rows) == 1, f"{model} controls are missing/duplicated")
        ablations = [row for row in model_rows if row["variant"] == "relu_quantized"]
        require({integer(row, "activation_s") for row in ablations} == expected_s,
                f"{model} selected/preset validation points mismatch")
        fp32_row, wide_row = fp32_rows[0], wide_rows[0]
        reference = load_predictions(directory / fp32_row["prediction_file"])
        baseline = load_predictions(directory / wide_row["prediction_file"])
        provenance = verify_reused_baseline(directory, model, wide_row, metadata, baseline)
        for row in model_rows:
            verify_predictions(directory, row, reference, baseline)
        for row in ablations:
            require(row["evaluation"] == "measured" and integer(row, "activation_ell") == 16
                    and integer(row, "register_overflows") == 0,
                    f"{model} ablation must be newly measured without accumulator overflow")
            expected_calls = {"resnet50": 49, "resnet152": 151}[model]
            batches = (3925 + metadata["batch_size"] - 1) // metadata["batch_size"]
            require(integer(row, "relu_calls") == expected_calls * batches,
                    f"{model} ReLU invocation coverage mismatch")
        result[model] = {"metadata": metadata, "selected": selected, "fp32": fp32_row,
                         "wide": wide_row, "ablations": ablations, "provenance": provenance,
                         "candidates": candidates}
    return result


def report(results):
    text = [
        "# 仅限制 ReLU 为 16 位的 ResNet 精度实验结果", "",
        "两个模型均完成：训练集 512 张校准图像独立扫描 ReLU 的小数位数 "
        "s=0…15，再对训练集选中的点和预设 Q(16,12) 测试全部 3925 张验证图像；"
        "两点相同时只测一次。验证集不参与选点。", "",
        "| 模型 | 设置 | 验证正确数 / 3925 | Top-1 (%) | 相对宽累加基线 (百分点) | 与 FP32 预测一致率 (%) | 与宽累加基线预测一致率 (%) |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for model, values in results.items():
        rows = [("FP32", values["fp32"]), ("全网 Q(24,15) 宽累加基线", values["wide"])]
        for row in sorted(values["ablations"], key=lambda row: integer(row, "activation_s")):
            s = integer(row, "activation_s")
            roles = []
            if s == values["selected"]:
                roles.append("训练集选点")
            if s == 12:
                roles.append("预设点")
            rows.append((f"仅 ReLU Q(16,{s})（{'、'.join(roles)}）", row))
        for label, row in rows:
            text.append(f"| {model} | {label} | {integer(row, 'top1_correct')} / 3925 | "
                        f"{float(row['top1_pct']):.4f} | {float(row['delta_from_wide_baseline_pp']):+.4f} | "
                        f"{float(row['prediction_agreement_pct']):.4f} | "
                        f"{float(row['baseline_prediction_agreement_pct']):.4f} |")
    text.extend(["", "改变的仅是每次 ReLU 的有符号输入和输出表示。网络输入、参数、BN 系数、"
                 "卷积/全连接及其他层输出继续使用 Q(24,15)，点积采用经溢出检查的 128 位逻辑临时累加器，"
                 "点积结束后 floor 缩放一次并饱和写回。MaxPool、残差加法和池化规则保持基线设置。"
                 "ReLU 之外没有改为 16 位。", "",
                 "ReLU 的 16 位包含符号位，表示范围是 "
                 "$[-2^{15-s},32767/2^s]$，步长是 $2^{-s}$。输入整数编码先 floor 转换到 "
                 "Q(16,s)，超界时饱和，然后执行 max(0,x)，最后用精确整数移位恢复到基线的 15 位小数编码。"
                 "这不是截取低 16 位，转换过程没有浮点计算。ResNet50/152 使用 ReLU，"
                 "本实验不包含 sigmoid/tanh 或查找表。", "",
                 "| 模型 | ReLU Q(16,s) | ReLU 输入元素总数 | 正数上界裁剪数 (%) | 负数下界裁剪数 (%) | 正值降精度数 (%) | 正值归零数 (%) |",
                 "|---|---|---:|---:|---:|---:|---:|",])
    for model, values in results.items():
        for row in sorted(values["ablations"], key=lambda row: integer(row, "activation_s")):
            text.append(f"| {model} | Q(16,{integer(row, 'activation_s')}) | {integer(row, 'relu_input_total')} | "
                        f"{integer(row, 'relu_upper_clipped')} ({float(row['relu_upper_clipping_pct']):.6f}) | "
                        f"{integer(row, 'relu_lower_clipped')} ({float(row['relu_lower_clipping_pct']):.6f}) | "
                        f"{integer(row, 'relu_positive_rounded')} ({float(row['relu_positive_rounding_pct']):.6f}) | "
                        f"{integer(row, 'relu_positive_zeroed')} ({float(row['relu_positive_zeroing_pct']):.6f}) |")
    text.extend(["", "上表统计所有验证图像、所有 ReLU 调用中的元素事件；同一元素可能同时计入正值降精度和正数上界裁剪，"
                 "各列不是互斥分类。负数下界饱和保持负号，随后都输出零，因此它本身不改变 ReLU 输出。"
                 "正数上界裁剪会直接改变输出，正值 floor 降精度及归零也可能影响后续预测；"
                 "应结合这些统计和 Top-1/预测一致率判断 16 位的影响，不能把所有负值裁剪都当作精度损失。", ""])
    for model, values in results.items():
        chosen = next(row for row in values["candidates"] if integer(row, "activation_s") == values["selected"])
        text.append(f"{model}：训练集选择 s={values['selected']}，校准正确 "
                    f"{integer(chosen, 'top1_correct')}/512（{float(chosen['top1_pct']):.4f}%）；"
                    "同分取较小 s。宽累加基线来源：" + values["provenance"] + "。")
        text.append("")
    text.extend([
        "历史基线没有在此次实验中重复完整推理。runner 在复用前核对权重、预处理、"
        "数据集映射、算术设置、样本顺序和目标向量，并用当前实现独立重算首批预测作检查。"
        "本汇总再次检查源预测与复制预测一致、元数据和样本清单一致，并从保存的预测向量重新核算表内准确率和一致率。"
        "所有 16 位校准点及最终验证消融点均为本次独立实测。若复现时选择重新测量基线，该来源会标为本次完整实测。", "",
        "本实验使用 torchvision ImageNet-1K V1 预训练权重、原始 1000 类分类头，不重新训练。"
        "准确率是在 ImageNette2-160 的完整验证集上测得，标签映射回 ImageNet 类别；"
        "它不是完整 ImageNet 验证集准确率。", "",
        "这是明文整数的数值精度实验。宽累加使用安全 MPC 时需要相应的安全扩环、乘法和截断协议；"
        "此处的 128 位逻辑临时累加器不代表已运行 128 位秘密共享网络，耗时也不能作为 MPC 性能。", "",
        "复现命令、代码、校准候选、样本清单、逐图预测和测试日志见 "
        "[README_REPRODUCE_CN.md](README_REPRODUCE_CN.md)。所有原始数字保存在 "
        "`final_results.csv`、`validation_results.csv` 和 `calibration_candidates.csv`。", "",
    ])
    return "\n".join(text)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", type=Path,
                        default=ROOT / "results" / "relu16_wide_imagenette")
    args = parser.parse_args()
    directory = args.results_dir.resolve()
    try:
        results = collect(directory)
    except (RuntimeError, OSError, KeyError, ValueError) as error:
        parser.exit(1, f"尚不能发布最终结果：{error}\n")
    output = directory / "RESULTS_CN.md"
    temporary = output.with_suffix(".md.tmp")
    temporary.write_text(report(results), encoding="utf-8")
    temporary.replace(output)
    print(output)


if __name__ == "__main__":
    main()
