# E0-U 与历史严格定点实验的证据对应

本文原记录对应已完成的 E0-U-S（宽临时点积＋饱和写回），并非严格环计算。
2026-10-07 曾按用户要求切换至 E0-U-R：所有乘积与累加器 mod 2^ell
后再缩放，逻辑累加器位宽是 ell。2026-10-08 用户恢复宽临时累加方案，
当前默认再次为 E0-U-S 的 `wide + saturate`，对应原目录
`results/e0_uniform_storage_imagenette`。严格环仍可显式选用，结果目录
`results/e0_uniform_ring_imagenette` 与宽累加目录分开；目前只有执行语义
验证，没有完整严格环准确率。原始 CSV 与原结论保留。详见 `STRICT_RING_CN.md`。

本记录核对历史 E0-U-S 与 E0-Q-fixed 的模型、样本选择和原始证据。

## 模型、数据和选择协议

两个 runner 都直接复用 `run_e0_q_all.py` 的 `MODEL_SPECS`、`build_model` 和 `load_imagenette`。当前代码中的模型是 torchvision ResNet-50/101/152 `IMAGENET1K_V1`，分类头保留原始 1000 类；输入预处理统一是 `weights.transforms()`：短边 resize 256、中心 crop 224、bilinear，mean `[0.485,0.456,0.406]`，std `[0.229,0.224,0.225]`。

| 模型 | 权重文件 |
|---|---|
| ResNet-50 | `https://download.pytorch.org/models/resnet50-0676ba61.pth` |
| ResNet-101 | `https://download.pytorch.org/models/resnet101-63fe2227.pth` |
| ResNet-152 | `https://download.pytorch.org/models/resnet152-394f9c45.pth` |

当前 ImageNette 训练 split 有 9469 张，完整 validation 有 3925 张。训练校准512张按 ImageFolder 类别顺序逐类轮流取样，`n01440764`、`n02102040` 各52张，另外8类各51张。前三个模型生成的样本索引、路径和标签映射一致。开头12个训练索引为 `0,963,1918,2911,3769,4710,5666,6627,7558,8509,1,964`；第一个训练路径为 `n01440764/ILSVRC2012_val_00000293.JPEG`。文件名中含 `val` 来自原始 ImageNet 文件名，该文件在 ImageNette 的 `train` split 中。

两种实验都按 `ell ∈ {12,16,20,24,28,32}` 枚举全部合法 `s=0..ell-1`，训练校准 Top-1 最大者入选，同分选较小 `s`；validation 不参与选点。历史strict合并候选包含396个完成行，每模型132行，样本数均512。其18个最终行每行均为完整3925张validation。

历史strict没有保存样本 manifest 或模型 metadata；代码、日志与样本数支持选择规则一致，无法从旧文件独立追溯当时每张图像内容。新runner保存逐模型 manifest 和 metadata，并在续跑时核对它们。

## 历史原始文件

下面路径均相对当前 `tools/precision` 目录。

| 用途 | 原始路径 |
|---|---|
| 三模型全部strict校准候选 | `results/e0_q_fixed_strict_imagenette_all/e0_q_fixed_candidates.csv` |
| 三模型strict完整验证结果 | `results/e0_q_fixed_strict_imagenette_all/e0_q_fixed_final.csv` |
| ResNet-50 原始逐模型目录 | `results/e0_q_fixed_strict_imagenette/` |
| ResNet-101 原始逐模型目录 | `results/e0_q_fixed_strict_imagenette_resnet101/` |
| ResNet-152 原始逐模型目录 | `results/e0_q_fixed_strict_imagenette_resnet152/` |
| ResNet-50 执行日志 | `results/e0_q_fixed_strict_imagenette_resnet50.log`、`results/e0_q_fixed_strict_imagenette_resnet50_final32.log` |
| ResNet-101 执行日志 | `results/e0_q_fixed_strict_imagenette_resnet101.log` |
| ResNet-152 执行日志 | `results/e0_q_fixed_strict_imagenette_resnet152.log` |
| strict中文报告 | `../../results/e0_q_fixed_strict_imagenette_report_20260910.md` |
| 旧浮点算子Q-all完整结果 | `results/e0_q_imagenette_formal/e0_q_all_final.csv` |

合并strict与逐模型源中的 `s,accuracy_pct,float_accuracy_pct,calibration_accuracy_pct,samples` 逐项一致。不要将 `e0_q_resnet50_formal/` 作为完整对照：该目录实际只有256张校准、1000张验证，与此次512/3925协议不同。

历史strict的FP32标量基线为：

| 模型 | 训练校准 Top-1 (%) | 完整validation Top-1 (%) |
|---|---:|---:|
| ResNet-50 | 71.6796875 | 75.4140127388535 |
| ResNet-101 | 74.609375 | 78.1146496815287 |
| ResNet-152 | 73.6328125 | 77.4777070063694 |

旧strict与Q-all驱动未记录或显式禁用TF32。旧Q-all的ResNet-101 FP32为78.0891719745223%，比strict少1张正确样本，其余两个完整validation基线相同。新runner显式关闭matmul/cuDNN TF32并设置 `float32_matmul_precision=highest`，保存实际FP32结果及策略；各历史表的原基线应保留，不能把1张样本的差异自动归因于定点语义。

## 新旧结果字段映射

| 比较内容 | 历史strict字段 | 新E0-U字段 |
|---|---|---|
| 网络与格式 | `model,ell,s` | `model,ell,s` |
| 定点Top-1 | `accuracy_pct` | `top1_pct` |
| 同次FP32基线 | `float_accuracy_pct` | `fp32_top1_pct` |
| 相对基线差值 | `delta_pp` | `delta_pp` |
| 验证图像数 | `samples` | `samples` |
| 训练选择的准确率 | `calibration_accuracy_pct`（final） | `calibration_top1_pct`（selected表） |
| 参数/输入裁剪 | `offline_clipped,offline_total,input_clipped,input_total` | 同名字段 |
| 算术语义 | `strict_q_mul_then_ell_bit_saturating_add` | `uniform_ell_bit_storage_wide_integer_dot_then_floor_rescale` |
| 累加器 | `accumulator_bits=ell` | `register_bits=128` |
| 逐项饱和 | `mul_saturations,add_saturations` | 不适用；完整点积后记录写回 `activation_clipped,activation_total` |
| 与FP32预测一致率 | 未保存 | `prediction_agreement_pct` |
| 配对正确/错误样本 | 未保存 | `fp32_right_fixed_wrong,fixed_right_fp32_wrong` |
| 所需存储/寄存器位数 | 未保存 | `max_storage_bits,max_register_bits,max_register_bits_kind` |
| 整数执行实现 | CUDA ordered scalar MAC | `integer_backend`，后端统计另保留 requested/effective 字段 |
| 实际测量/等价重用 | candidates `measurement,reused_from_ell` | `evaluation,reused_from_ell,prediction_file` |
| 完整/小样本证据 | 依靠样本数与目录区别 | `evidence_scope=formal/diagnostic` |

历史strict没有保存任何逐样本FP32或定点预测 `.pt/.npy`，也没有agreement与配对字段，不能从聚合准确率反推出这些统计。若需要新旧语义的逐样本对照，只须在同一manifest上重评历史选定 `(ell,s)` 并保存预测；不需要重跑396行历史选点网格。本记录不启动这些补充评估。

## E0-U 完成后报告来源

主准确率和位宽比较从 `results/e0_uniform_storage_imagenette/e0_uniform_storage_final.csv` 读取，训练选择从 `uniform_storage_<model>_selected.csv` 读取。报告应展示 `model,ell,s,top1_pct,fp32_top1_pct,delta_pp,prediction_agreement_pct,fp32_right_fixed_wrong,fixed_right_fp32_wrong`；同时提供 `offline/input/activation` 裁剪计数及比例、`register_overflows,max_storage_bits,max_register_bits,max_register_bits_kind,register_bits,integer_backend`。寄存器位宽须区分标量路径的 `observed_prefix` 实测值与整数GEMM路径的 `proven_upper_bound` 保守上界。

每个正式结果应关联完整validation的 `samples=3925`、`evidence_scope=formal`、完成状态、权重与预处理metadata、两个split的manifest和 `prediction_file`。`evaluation=exact_reuse` 保留原测量来源；`elapsed_seconds` 只是明文模拟运行时间，不转换为安全两方时延。新增或恢复任务的设备、batch、Torch版本、GPU与FP32策略由 `_status.json` 中的 `runtime_context` 给出。
