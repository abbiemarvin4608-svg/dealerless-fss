#!/usr/bin/env python3
"""Verify completed E0-U prediction evidence and compare historical strict E0-Q."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
import torch


def summarize(root: Path, strict_path: Path) -> None:
    final = pd.read_csv(root / "e0_uniform_storage_final.csv")
    candidates = pd.read_csv(root / "e0_uniform_storage_candidates.csv")
    strict = pd.read_csv(strict_path)
    comparison = []
    arithmetic_values = set(final.arithmetic)
    if len(arithmetic_values) != 1 or set(candidates.arithmetic) != arithmetic_values:
        raise RuntimeError("result files mix arithmetic modes")
    arithmetic = next(iter(arithmetic_values))
    ring = arithmetic == "uniform_ell_bit_ring_mac_then_signed_floor_rescale"
    audit_counts = {}
    for model in ("resnet50", "resnet101", "resnet152"):
        state = json.loads((root / f"uniform_storage_{model}_status.json").read_text())
        metadata = json.loads((root / f"uniform_storage_{model}_metadata.json").read_text())
        if state["status"] != "completed" or metadata["evidence_scope"] != "formal":
            raise RuntimeError(f"{model}: formal experiment is not complete")
        if state["arithmetic"] != arithmetic or metadata["arithmetic"] != arithmetic:
            raise RuntimeError(f"{model}: arithmetic metadata mismatch")
        model_candidates = candidates[candidates.model == model]
        expected = {(ell, s) for ell in (12, 16, 20, 24, 28, 32) for s in range(ell)}
        actual = set(zip(model_candidates.ell, model_candidates.s))
        if actual != expected or len(model_candidates) != len(expected):
            raise RuntimeError(f"{model}: calibration grid incomplete or duplicated")
        if not model_candidates.samples.eq(512).all():
            raise RuntimeError(f"{model}: calibration sample count differs")
        # Audit every calibration prediction too, rather than trusting the
        # selected aggregate row alone. Both splits have frozen FP32 evidence.
        all_rows = pd.read_csv(root / f"uniform_storage_{model}_runs.csv")
        if len(all_rows) != 138:
            raise RuntimeError(f"{model}: expected 132 calibration and 6 validation runs")
        for split, samples in (("calibration", 512), ("validation", 3925)):
            ref = torch.load(root / "predictions" / model / f"fp32_{split}.pt",
                             map_location="cpu", weights_only=False)
            labels, fp = ref["targets"], ref["predictions"]
            if labels.numel() != samples:
                raise RuntimeError(f"{model} {split}: wrong reference sample count")
            for record in all_rows[all_rows.split == split].to_dict("records"):
                item = torch.load(root / record["prediction_file"], map_location="cpu",
                                  weights_only=False)
                if not torch.equal(item["targets"], labels) or item["arithmetic"] != arithmetic:
                    raise RuntimeError(f"{model} {split}: inconsistent prediction evidence")
                pred = item["predictions"]
                if pred.shape != labels.shape:
                    raise RuntimeError("prediction vector has wrong shape")
                derived = {"top1_correct": int(pred.eq(labels).sum()),
                           "prediction_agreement_pct": 100 * int(pred.eq(fp).sum()) / samples,
                           "fp32_right_fixed_wrong": int((fp.eq(labels) & ~pred.eq(labels)).sum()),
                           "fixed_right_fp32_wrong": int((~fp.eq(labels) & pred.eq(labels)).sum())}
                for key, value in derived.items():
                    if abs(float(record[key]) - value) > 1e-8:
                        raise RuntimeError(f"{model} {split}: {key} differs from raw evidence")
                if int(record["register_overflows"]):
                    raise RuntimeError("invalid register arithmetic")
                if ring and (record["evaluation"] != "measured" or
                             int(record["register_bits"]) != int(record["ell"]) or
                             int(record["max_register_bits"]) > int(record["ell"]) or
                             record["max_register_bits_kind"] != "logical_ring_width"):
                    raise RuntimeError("strict ring run used widened logical arithmetic or width reuse")
        audit_counts[model] = {"calibration_rows": 132, "validation_rows": 6,
                               "raw_prediction_records_checked": len(all_rows)}
        reference = torch.load(root / "predictions" / model / "fp32_validation.pt",
                               map_location="cpu", weights_only=False)
        targets, fp32 = reference["targets"], reference["predictions"]
        if targets.numel() != 3925:
            raise RuntimeError(f"{model}: validation sample count differs")
        model_final = final[final.model == model]
        if set(model_final.ell) != {12, 16, 20, 24, 28, 32} or len(model_final) != 6:
            raise RuntimeError(f"{model}: selected validation grid incomplete")
        for row in model_final.to_dict("records"):
            ell, s = int(row["ell"]), int(row["s"])
            calibrated = model_candidates[model_candidates.ell == ell].sort_values(
                ["top1_correct", "s"], ascending=[False, True]).iloc[0]
            if s != int(calibrated.s):
                raise RuntimeError(f"{model} ell={ell}: selected s differs from training choice")
            evidence = torch.load(root / row["prediction_file"], map_location="cpu",
                                  weights_only=False)
            if not torch.equal(evidence["targets"], targets):
                raise RuntimeError(f"{model} ell={ell}: prediction target ordering differs")
            prediction = evidence["predictions"]
            ok, ref_ok = prediction.eq(targets), fp32.eq(targets)
            measured = {
                "top1_pct": 100.0 * int(ok.sum()) / 3925,
                "fp32_top1_pct": 100.0 * int(ref_ok.sum()) / 3925,
                "prediction_agreement_pct": 100.0 * int(prediction.eq(fp32).sum()) / 3925,
                "fp32_right_fixed_wrong": int((ref_ok & ~ok).sum()),
                "fixed_right_fp32_wrong": int((ok & ~ref_ok).sum()),
            }
            for key, value in measured.items():
                if abs(float(row[key]) - value) > 1e-8:
                    raise RuntimeError(f"{model} ell={ell}: {key} disagrees with predictions")
            if int(row["register_overflows"]) or row["evidence_scope"] != "formal":
                raise RuntimeError(f"{model} ell={ell}: invalid arithmetic or evidence scope")
            old = strict[(strict.model == model) & (strict.ell == ell)].iloc[0]
            comparison.append({
                "model": model, "ell": ell, "uniform_s": s,
                "uniform_top1_pct": measured["top1_pct"],
                "uniform_fp32_top1_pct": measured["fp32_top1_pct"],
                "uniform_agreement_pct": measured["prediction_agreement_pct"],
                "strict_s": int(old.s), "strict_top1_pct": float(old.accuracy_pct),
                "strict_fp32_top1_pct": float(old.float_accuracy_pct),
                "uniform_minus_strict_pp": measured["top1_pct"] - float(old.accuracy_pct),
                "max_register_bits": int(row["max_register_bits"]),
                "max_register_bits_kind": row.get("max_register_bits_kind", "observed_prefix"),
                "register_overflows": int(row["register_overflows"]),
                "arithmetic": arithmetic,
                **{f"{kind}_{'wrapped' if ring else 'clipped'}":
                   int(row[f"{kind}_{'wrapped' if ring else 'clipped'}"])
                   for kind in ("offline", "input", "activation")},
                "evaluation": row["evaluation"],
            })
    table = pd.DataFrame(comparison).sort_values(["model", "ell"])
    table.to_csv(root / "uniform_vs_strict.csv", index=False)
    lines = ["# 全网统一定点存储：结果与严格累加对照", "",
             "三个模型各有 132 个完整训练校准点；每个位宽独立选择 s 后，"
             "在全部 3925 张 ImageNette 验证图上评估。主表的准确率、一致率和"
             "配对错误数已从保存的逐样本预测独立复算。", "",
             ("主实验采用原始乘积、累加和池化求和均 mod 2^ell 的严格环规则，"
              "按 ell 位有符号解释后在点积结束时缩放一次。" if ring else
              "本目录的历史实验采用宽临时点积，结束后缩放并饱和写回。") +
             "历史严格饱和对照按其训练选点"
             "报告逐乘积缩放、逐加法饱和的结果；两组使用相同预训练模型及数据，"
             "每行 s 可能不同。历史对照没有逐样本预测，且未记录 TF32 设置，"
             "因此保留其原 FP32 基线，并比较已有的聚合准确率。", "",
             "| 模型 | ell | 主实验 s | 主实验 Top-1 % | FP32 一致率 % | 严格对照 s | 严格对照 Top-1 % |",
             "|---|---:|---:|---:|---:|---:|---:|"]
    for row in table.to_dict("records"):
        lines.append(f"| {row['model']} | {row['ell']} | {row['uniform_s']} | "
                     f"{row['uniform_top1_pct']:.3f} | {row['uniform_agreement_pct']:.3f} | "
                     f"{row['strict_s']} | {row['strict_top1_pct']:.3f} |")
    lines += ["", ("全部严格环配置的逻辑寄存器为 ell 位，模回绕不会被算作执行错误。"
              "所有配置均实测，没有跨位宽复用。wrapped 表示后端计算受模回绕影响的"
              "输出数，不是饱和裁剪，也不是各后端可比较的原始 MAC 事件总数。"
              if ring else "所有历史主实验选点的临时寄存器溢出计数为 0。临时位宽字段的"
              "`observed_prefix` 为实测峰值，`proven_upper_bound` 为保守安全上界，"
              "上界不用于判断最小必需累加器位宽。逐层输出仍按 ell 位"
              "饱和，参数和层输出裁剪统计见 `e0_uniform_storage_final.csv`。") +
              "数据集为 ImageNette；结果不能代替完整 ImageNet-1K 的准确率。", ""]
    (root / "UNIFORM_VS_STRICT_CN.md").write_text("\n".join(lines))
    (root / "evidence_audit.json").write_text(json.dumps(
        {"status": "passed", "arithmetic": arithmetic, "models": audit_counts,
         "calibration_rows": len(candidates), "validation_rows": len(final)}, indent=2) + "\n")
    print(f"Verified 396 calibration rows and 18 full-validation rows; wrote {root / 'uniform_vs_strict.csv'}")


def main() -> None:
    base = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", type=Path,
                        default=base / "results" / "e0_uniform_ring_imagenette")
    parser.add_argument("--strict-csv", type=Path,
                        default=base / "results" / "e0_q_fixed_strict_imagenette_all" / "e0_q_fixed_final.csv")
    args = parser.parse_args()
    summarize(args.results_dir.resolve(), args.strict_csv.resolve())


if __name__ == "__main__":
    main()
