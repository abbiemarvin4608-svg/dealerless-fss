#!/usr/bin/env python3
"""Recompute the fixed ReLU+MaxPool experiment report from saved predictions."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import torch

from run_e0_uniform_storage import prediction_metrics


def read_rows(path):
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def require_equal(actual, expected, message):
    if abs(float(actual) - float(expected)) > 1e-8:
        raise RuntimeError(message)


def summarize(results_dir, models):
    grouped, all_rows = {}, []
    for model in models:
        prefix = results_dir / f"relu_maxpool_{model}"
        state = json.loads(Path(str(prefix) + "_status.json").read_text())
        metadata = json.loads(Path(str(prefix) + "_metadata.json").read_text())
        if state["status"] != "completed":
            raise RuntimeError(f"{model}: experiment is not complete")
        if (metadata["base_ell"], metadata["base_s"], metadata["activation_bits"],
                metadata["activation_s"], metadata["arithmetic_mode"],
                metadata["overflow_mode"]) != (24, 15, 16, 6, "wide", "saturate"):
            raise RuntimeError(f"{model}: report requires the requested Q(16,6) on Q(24,15)")
        rows = read_rows(Path(str(prefix) + "_runs.csv"))
        variants = {row["variant"]: row for row in rows}
        if len(rows) != 3 or set(variants) != {"fp32", "wide_baseline", "relu_maxpool_quantized"}:
            raise RuntimeError(f"{model}: expected all three validation configurations")
        manifest = read_rows(results_dir / f"{model}_validation_manifest.csv")
        if len(manifest) != 3925:
            raise RuntimeError("This report requires all 3925 validation images")
        targets = torch.tensor([int(item["imagenet_target"]) for item in manifest])
        saved = {}
        for variant, row in variants.items():
            if row["status"] != "completed" or row["split"] != "validation":
                raise RuntimeError("unfinished validation row")
            bundle = torch.load(results_dir / row["prediction_file"], map_location="cpu", weights_only=False)
            if not torch.equal(bundle["targets"], targets):
                raise RuntimeError("prediction targets differ from the validation manifest")
            saved[variant] = bundle["predictions"]
        fp32, baseline = saved["fp32"], saved["wide_baseline"]
        baseline_metrics = prediction_metrics(baseline, targets)
        for variant, row in variants.items():
            measured = prediction_metrics(saved[variant], targets, fp32)
            paired = prediction_metrics(saved[variant], targets, baseline)
            for key, value in measured.items():
                require_equal(row[key], value, f"{model} {variant}: inconsistent {key}")
            require_equal(row["baseline_prediction_agreement_pct"], paired["prediction_agreement_pct"],
                          "inconsistent baseline agreement")
            require_equal(row["delta_from_wide_baseline_pp"],
                          measured["top1_pct"] - baseline_metrics["top1_pct"],
                          "inconsistent baseline accuracy change")
            if variant != "fp32" and int(float(row["register_overflows"])):
                raise RuntimeError("wide temporary register overflow")
        joint = variants["relu_maxpool_quantized"]
        if (int(joint["activation_ell"]), int(joint["activation_s"]),
                int(float(joint["maxpool_bits"])), int(float(joint["maxpool_s"])),
                int(float(joint["relu_bits"])), int(float(joint["relu_s"])),
                joint["arithmetic_mode"], joint["overflow_mode"]) != (16, 6, 16, 6, 16, 6, "wide", "saturate"):
            raise RuntimeError(f"{model}: saved validation row has a different format")
        expected_relu_modules, expected_relu_calls = {"resnet50": (17, 49), "resnet152": (51, 151)}[model]
        batches = (3925 + metadata["batch_size"] - 1) // metadata["batch_size"]
        for key, value in {"relu_module_count": expected_relu_modules,
                           "relu_calls": expected_relu_calls * batches,
                           "maxpool_module_count": 1, "maxpool_calls": batches,
                           "maxpool_same_format_checked_calls": batches}.items():
            require_equal(joint[key], value, f"{model}: incomplete operator coverage")
        for key in ("maxpool_same_format_input_violations", "maxpool_same_format_output_violations"):
            if int(float(joint[key])):
                raise RuntimeError(f"MaxPool format check failed: {key}")
        for key in ("maxpool_input_rounded", "maxpool_input_clipped",
                    "maxpool_output_rounded", "maxpool_output_clipped"):
            if int(float(joint[key])):
                raise RuntimeError(f"MaxPool added quantization changes: {key}")
        for suffix in ("input", "output"):
            require_equal(joint[f"maxpool_same_format_checked_{suffix}_total"],
                          joint[f"maxpool_{suffix}_total"], "incomplete MaxPool format coverage")
        grouped[model] = variants
        all_rows.extend(rows)

    with (results_dir / "final_results.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(dict.fromkeys(key for row in all_rows for key in row)))
        writer.writeheader()
        writer.writerows(all_rows)
    text = [
        "# 仅 ReLU 与 MaxPool 使用 Q(16,6) 的完整验证结果", "",
        "固定使用有符号 16 位、6 位小数；s=6 由用户预先指定，没有校准扫描或验证集选点。"
        "其余网络继续使用 Q(24,15)，卷积和全连接在经检查的宽临时寄存器中累加，"
        "点积结束后一次 floor 缩放并饱和写回。", "",
        "| 模型 | 配置 | 正确数 / 3925 | Top-1 (%) | 相对宽基线 (百分点) | FP32 预测一致率 (%) | 宽基线预测一致率 (%) |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    labels = {"fp32": "FP32", "wide_baseline": "全网 Q(24,15) 宽累加基线",
              "relu_maxpool_quantized": "仅 ReLU + MaxPool Q(16,6)"}
    for model, variants in grouped.items():
        for variant in labels:
            row = variants[variant]
            text.append(f"| {model} | {labels[variant]} | {int(float(row['top1_correct']))} / 3925 | "
                        f"{float(row['top1_pct']):.4f} | {float(row['delta_from_wide_baseline_pp']):+.4f} | "
                        f"{float(row['prediction_agreement_pct']):.4f} | "
                        f"{float(row['baseline_prediction_agreement_pct']):.4f} |")
    text.extend(["", "本次独立实测联合限制的完整验证预测。FP32 和宽累加控制组经元数据、"
                 "样本清单、目标向量和正确数核对后复用；宽累加控制组另用当前实现重算首批预测。"
                 "来源记录在模型结果 CSV 和 metadata 中；"
                 "汇总程序根据逐图预测和样本清单重新计算了准确率与预测一致率。", "",
                 "## 量化与覆盖", "",
                 "Q(16,6) 包含符号位，范围为 [-512,511.984375]，步长为 1/64。"
                 "每个受限算子的输入整数编码 floor 除以 512、饱和到 [-32768,32767]，"
                 "在该范围执行 ReLU 或 MaxPool，然后输出乘以 512 恢复为 Q(24,15)。"
                 "转换和池化全程使用整数。平均池化、BN、残差加法、卷积和全连接不改为 16 位。", "",
                 "| 模型 | ReLU 输入元素数 | 正值降精度数 | 正值归零数 | ReLU 上/下界裁剪 | ReLU 调用次数 | MaxPool 调用次数 |",
                 "|---|---:|---:|---:|---:|---:|---:|",])
    for model, variants in grouped.items():
        row = variants["relu_maxpool_quantized"]
        count = lambda key: int(float(row[key]))
        text.append(f"| {model} | {count('relu_input_total')} | {count('relu_positive_rounded')} | "
                    f"{count('relu_positive_zeroed')} | {count('relu_upper_clipped')} / {count('relu_lower_clipped')} | "
                    f"{count('relu_calls')} | {count('maxpool_calls')} |")
    text.extend(["", "| 模型 | MaxPool 输入元素数 | MaxPool 输出元素数 | 输入舍入 / 裁剪 | 输入 / 输出格点违规 |",
                 "|---|---:|---:|---:|---:|"])
    for model, variants in grouped.items():
        row = variants["relu_maxpool_quantized"]
        count = lambda key: int(float(row[key]))
        text.append(f"| {model} | {count('maxpool_input_total')} | {count('maxpool_output_total')} | "
                    f"{count('maxpool_input_rounded')} / {count('maxpool_input_clipped')} | "
                    f"{count('maxpool_same_format_input_violations')} / {count('maxpool_same_format_output_violations')} |")
    text.extend(["", "ResNet50 的 17 个 ReLU 模块每图调用 49 次，ResNet152 的 51 个模块每图调用 151 次。"
                 "两个模型都仅有 stem 的一个 MaxPool，每图调用一次；表中的调用次数累计所有验证批次。", "",
                 "MaxPool 紧跟同格式 ReLU；全部输入和输出均通过 Q(16,6) 范围及格点检查。"
                 "同一格式的 floor/饱和转换是幂等的，MaxPool 仅从输入中选择最大值，"
                 "因此它没有增加新的舍入或裁剪误差。", "",
                 "## 适用范围", "",
                 "使用 torchvision ImageNet-1K V1 预训练权重和原始 1000 类分类头，不重新训练。"
                 "数据为 ImageNette2-160 的完整 3925 张验证图像，标签映射到原 ImageNet 类别。"
                 "这些数字不是完整 ImageNet 验证集准确率。", "",
                 "这是明文整数数值精度实验；安全 MPC 的宽环、扩环、乘法和截断协议不在本次测试范围内。"
                 "复现命令和证据路径见 [README_REPRODUCE_CN.md](README_REPRODUCE_CN.md)。", "",])
    (results_dir / "RESULTS_CN.md").write_text("\n".join(text))
    return grouped


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", type=Path, default=Path(__file__).resolve().parent / "results" / "relu_maxpool16_s6_wide_imagenette")
    parser.add_argument("--models", default="resnet50,resnet152")
    args = parser.parse_args()
    summarize(args.results_dir.resolve(), args.models.split(","))
    print(f"Wrote {args.results_dir / 'RESULTS_CN.md'}")


if __name__ == "__main__":
    main()
