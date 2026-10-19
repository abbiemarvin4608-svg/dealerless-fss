# 09: 分段 FSS 符号测试候选

**What to build:** 在完整 folded 激活中用既有私有分段 DReLU/MSB 实现两个符号判断，保留同一数值函数并交付相对 BitDec 的真实阶段成本，明确是否值得采用。

**Blocked by:** 07: 双激活 Batch=128 联合驻留。

**Status:** closed

**Labels:** completed

- [x] 增加显式 `sign_backend=segmented_fss`，用私有比较得到包含零点的 d 和裁剪条件 i；幅值、裁剪、LUT、BitXA 恢复与接口缩放均保持冻结整数语义。
- [x] 复用已验证的分段比较协议、显式上下文和外层公开分段配置，不依赖未掩码的秘密比较。LUT 分段参数与符号比较配置分别记录。
- [x] planner 只生成所选符号后端所需材料，替换 BitDec 库存而非叠加；相关 mask/key/triples 输入前就绪，一次消费，成本计入本激活。
- [x] 两种 LUT 下两种激活的全部 65536 原生编码与 BitDec 及独立 oracle 一致；包括最小负数、零、裁剪边界和负 TanH 最后舍入。
- [x] 在现有串行完整 batch=128 单步中通过默认与非零隐状态、连续 fresh query 和联合资源核验，不等待 08 才开始实现或局部验收。
- [x] 固定同一 LUT、输入、资源和串行调度，与 BitDec 进行独立匹配成本测量；记录两次符号测试及完整激活的离线/在线字节、因果轮次和墙钟。
- [x] 按既有阶段准入方法判断该候选的收益、退化或未分辨，保留实际失败状态。变化属于符号协议实现优化，不能套用论文 BitDec 的成本式或宣称完整激活一轮。
- [x] 交付可复用的配置与局部证据供 12 评估；性能退化仍可作为正确实现的实验候选交付，不强行成为默认。

**Scope boundary:** 不阻塞 11 中固定 BitDec 的分段/不分段 LUT 四行比较。与 08 并行调度的组合正确性及拟推荐配置的最终成本在 12 核验。

## 2026-10-13 续办状态

ticket09 原始构建的原生全域、显式上下文及八行完整 batch=128 单步资格均通过。八行覆盖两种 LUT、串行/parallel-a2、默认/非零隐状态，每行两次 fresh query；每查询 172928 个整数零差异，最大 party VmPeak 982069248 字节，低于 1.5 GiB 限额。该资格的具名来源复用与当前持久构建的新诊断见 `ticket11/qualification-reuse-final.json`。原始证据见 [验收说明](../../../project/our/results/paper_folded_activation_20261013/ticket09/README.md) 与 `ticket09/step-final/qualification.json`。

**Resolution:** 最终 `formal-final/` 的串行、并行 BitDec 父配置与分段符号候选均通过材料、资源及计量资格；父配置各有三组 30 次 fresh query，候选各有一组 10 次。父/候选整步在线比在串行为 0.8913，95% BCa [0.8239, 0.9300]；并行为 0.9020 [0.8397, 0.9833]，两者均退化。整步离线墙钟和离线 LAN/WAN 通信模型在两种调度下也均退化，按冻结规则判为 `experimental_regression`。这两对不触发追加采样，符号候选作为正确但未获默认准入的实验配置交付。旧 `formal/` 初始批次曾得到串行 0.970959 [0.902356, 1.034366]、并行 1.055432 [1.024154, 1.088322]；这些数值只描述历史构建，不与最终批次混合。

**Evidence:** [最终逐阶段成本与比较](../../../project/our/results/paper_folded_activation_20261013/ticket11/formal-final/REPORT.md)、`ticket11/formal-final/analysis.json` 的 `sequential-a1-sign`、`parallel-a2-sign`，以及对应四行全部原始 `result.json`、双方日志及无损轨迹。旧 [初始批次报告](../../../project/our/results/paper_folded_activation_20261013/ticket11/formal/REPORT.md)保留为历史诊断。完整激活与两个符号测试分别保留成本，不把并行 worker 的轮次相加。
