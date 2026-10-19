# 09: 完成 MP-SPDZ Google-30 安全推理

**What to build:** 将 #08 的真实 MP-SPDZ OT 后端扩展为完整 Google-30 FastGRNN 安全推理，使用独立评估产物给出可与其他后端对照的质量、正确性和端到端成本结果。

**Blocked by:** #02 生成独立评估的 Google-30 质量产物；#08 接通 MP-SPDZ 单个 FastGRNN 时间步

**Status:** partial — bounded verification stopped at 16/6798 samples by user-selected budget

**Report:** [ticket 09 report](../../../project/our/results/ticket09_mpspdz_full/REPORT_CN.md)

- [ ] 完整网络使用 #02 的独立评估产物及冻结整数图，沿用 #01 的契约定义；核验新产物实际位宽与尺度，完成 99 个时间步与分类头，不将历史模型的权重或输出强加给新产物。
- [ ] 全流程运行 MP-SPDZ 两方半诚实 OT 协议，模型与输入保持私有且不引入可信经销者、明文旁路或教程替身。
- [ ] 在独立评估集上输出逐样本定点预测、FP32 Top-1、相对 FP32 的 loss_pp 和 prediction agreement，并注明是否达到 1.0 pp 主门槛。
- [x] 对代表性样本检查重构后的整数输出；sample 0 的 133,779 个中间/最终整数与 SCI oracle 逐值一致。完整评估集仍未完成，ledger 会在 mismatch 时阻止质量排名。
- [ ] 端到端结果分别报告 offline、online、total 延迟、offline/online 通信、依赖轮数和峰值内存，且包含输入转换、OT、扩环、截断和输出重构成本。
- [ ] 首查询与跨查询摊销结果分开记录，明确初始化的摊销次数和规则；一次性密钥、掩码和三元组按每次执行的新消耗计费，不重复使用。
- [x] 固定 localhost、CPU 6/8、单线程、B=1 三次和 B=4 基准一次，另完成限时真实 B=4 批次四次（样本 0--15）；保存原始双方日志、配置、版本和资源上限。
- [x] 限时 ledger 已持久化 16/6798 个真实安全预测，逐样本与 SCI oracle 一致；`complete` 与完整质量字段保持为空，停止原因和后续可恢复 supervisor 已记录。
- [ ] 对不可运行、超时或 OOM 的配置标记 unavailable 并说明原因，不用 structure-only、明文时间或单样本 smoke 结果填充完整安全推理结果。
