# 10: 冻结验证集的明文模型质量

**What to build:** 使用新的 folded 整数网络对冻结验证集运行完整递归序列和分类头，给出替换激活后的模型质量及既有门槛是否通过的可复现结论。

**Blocked by:** 02: 冻结激活整数参考与整图回放。

**Status:** closed

**Labels:** completed

- [x] 使用同一冻结 checkpoint、量化图、归一化与 official valid 的全部 6798 条，记录样本顺序和来源；每条完整运行 99 步与分类头。
- [x] 两种激活都使用 02 冻结的 Q7/Q12 folded 映射，其他图规则不变；不从旧 SCI 整数图质量结果推导新结果。
- [x] 保存逐样本预测及可用于差异定位的 logits、Top-1、相对同 checkpoint FP32 的百分点损失，以及与旧 SCI 整数网络的预测一致率。
- [x] 按既有 1.0 个百分点损失门槛给出通过或失败；valid 结果出来后不修改 q/f、表项、量化或模型，不选择更有利 checkpoint。
- [x] 结果明确标记为明文完整序列质量证据，安全执行样本数为零，不用于推导安全 99 步网络性能。
- [x] 两种 LUT 构造及两种符号协议在已证明同一整数映射后共用本项质量结果，不为同一明文算法重复评估。
- [x] 交付新结果、逐样本证据和复现命令。门槛失败也完成本调查票，但新数值程序的模型推荐资格记为失败，交由 12 如实汇总。

**Scope boundary:** 不依赖安全激活接入、驻留、并行或性能票；质量失败不授权重新调参。

**Resolution:** 2026-10-13 完成全 6798 条/99 步及分类头明文回放。folded Top-1 为 88.9967637540%，相对冻结 FP32 的损失为 0.3383348044 个百分点，通过 1.0 个百分点门槛；与同源重跑 SCI 预测一致率 98.7496322448%。无 valid 后调参，安全样本数为零。

**Evidence:** [模型质量报告](../../../project/our/results/paper_folded_activation_20261013/ticket10/REPORT.zh.md)、[机器可读结果](../../../project/our/results/paper_folded_activation_20261013/ticket10/summary.json)、[逐样本来源与预测](../../../project/our/results/paper_folded_activation_20261013/ticket10/predictions.csv)。
