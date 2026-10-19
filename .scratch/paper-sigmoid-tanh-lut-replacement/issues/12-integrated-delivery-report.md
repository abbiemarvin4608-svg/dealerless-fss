# 12: 集成交付与推荐配置报告

**What to build:** 交付用户可以按具名配置实际运行的完整论文激活后端，汇总两种 LUT、符号候选、并行、质量和成本证据，明确每种配置的适用范围与推荐资格。

**Blocked by:** 09: 分段 FSS 符号测试候选；10: 冻结验证集的明文模型质量；11: 分段与不分段的匹配性能实测。

**Status:** closed

**Labels:** completed

- [x] 汇合两种 LUT、原生 Sigmoid/TanH、batch=128 驻留、四参与者并行和显式符号后端，命令及日志完整标明数值算法、LUT family、段数、符号后端和调度。
- [x] 在最终集成版本验证此前未覆盖的“分段符号＋并行”组合，覆盖两种 LUT、默认及非零隐状态、连续请求、材料和资源；已验证且未变更路径复用对应证据，不无条件重复全部大规模测试。
- [x] 若拟推荐分段符号并行组合，补齐其相对匹配 BitDec 父配置的正式阶段成本和统计；局部符号收益不能直接推导并行组合默认资格。没有组合实测时保留具名实验配置。
- [x] 从原始证据分别给出协议/整数正确、输入前驻留、资源、模型质量和阶段性能判定。模型门槛失败或阶段退化如实保留，不用其他维度改善抵消。
- [x] 默认优化准入依据具名父配置：离线与在线通信模型及墙钟无可复现退化，至少一项改善；区间跨 1 记为未分辨。LUT 推荐不预设分段胜出，无合格新推荐时保留明确可运行的实验配置与结论。
- [x] 最终中文报告明确回答不分段 LUT 的在线效率、离线交换代价、并行收益和完整激活退出 SCI 旧链的覆盖；公开表内积无需 Beaver、完整激活仍有比较/BitXA/截断的事实保持一致。
- [x] 交付配置清单、构建与运行说明、阶段数据及证据索引，保留旧 SCI 和内部小表 DPF 的名称、结果与显式对照入口；不把一个组合的资格传播到未独立测试开关。
- [x] 完成范围明确为完整单时间步安全执行及完整明文序列质量。99 步安全整网、分类头安全执行和论文正文同步仍在本批之外，不修改或关闭父规格及历史工单。

**Scope boundary:** 正确但退化、在线未分辨或模型质量未达标都可以形成完整交付结论；不得为了关闭工单伪造推荐资格或擅自调参。

## 2026-10-13 续办状态

最终分段符号八行资格通过；16:45（北京时间）刷新交付摘要后，全部规定的整数、驻留、资源、协议、显式上下文、原生全域与模型质量门槛通过。复现配置与构建说明已交付，正式性能与推荐仍待 11 的队列结果。当前报告见 [交付报告](../../../project/our/results/paper_folded_activation_20261013/ticket12/REPORT.zh.md)；未验收的分段符号 parallel-a1 仍保留 `unqualified_experimental`。

17:22 交付生成器与运行说明已接入 `frozen-plan-budget-fixed.json`、`formal-budget-fixed/analysis.json`、`qualification-reuse.json` 和当前新诊断。生成器核对新计划/性能 manifest、共同构建及通过资格身份，保留历史非零状态与 parallel-a1 覆盖及来源局限；十五项轻量测试通过。新正式批次正在运行，当前报告仍是旧批次的待完成版本，最终成本与推荐报告在新采样终止后生成。

17:55 新正式批次已结束但未满足交付门槛：9 行合格，1 行 parallel-a2/full/BitDec 因 query9 地址空间 OOM；性能分析为 `incomplete/pending_extra_groups`，因此未生成新的 summary/REPORT，也未传播该组合的推荐资格。需在明确 allocator/虚拟地址高水位限制后重跑缺失行或新冻结批次。

后续旧 `formal-allocator-clean/` 的缺失 `/tmp` binary 补测、改写清单身份及失败组重置均已审计归档，详见 `ticket11/historical-cohort-audit.json`。初始 DPF 材料预算失败、修复后 query9 OOM、allocator 未净化中断和隔离诊断的历史记录继续保留；隔离诊断的 `/tmp` 原结果现已丢失，当前报告将其标为 pending/unavailable，不把它当作可核验的正式证据。旧批次样本不进入当前性能。新持久构建后的 `formal-final-invalid-analyzer-mismatch-20261013/` 与 `formal-final-invalid-qualification-selection-20261013/` 分别保留分析器冻结身份变化、误选旧 step-final 资格导致的两次中断，原证据完整排除，资格遗漏已纠正。

唯一当前交付来源为 `ticket11/final-build/{core,runtime,fixture}`、`ticket11/frozen-plan-final.json`、`ticket11/formal-final/analysis.json`。计划冻结 `qualification-reuse-final.json` 以及 `final-build/qualification-budget/qualification.json`、`final-build/qualification-folded/qualification.json` 三份通过资格；复用记录重验新/旧来源与原执行身份，保留历史 runtime 编译命令未记录及旧 `/tmp` 实体丢失的局限。新诊断结果不能传播给未独立验收的开关。

## 2026-10-14 终局与 Resolution

最终 `ticket11/formal-final/analysis.json` 为 `complete`，十行全部合格、18 组共 180 次 fresh query、11 项具名比较完整且测量缺口为零；四次反序补测后两对 LUT 在线比较仍未分辨，`sampling_status=exhausted_budget`。由唯一最终计划、三份资格与本批原始结果生成的 [中文集成交付报告](../../../project/our/results/paper_folded_activation_20261013/ticket12/REPORT.zh.md)和 `ticket12/summary.json` 均为 `complete`；配置、构建、运行与复现入口见 [交付说明](../../../project/our/results/paper_folded_activation_20261013/ticket12/README.zh.md)。历史失败、中断与失效批次仍保留且不进入性能统计。

Segmented DPF (5,5) Sigmoid/TanH 在 parallel-a2、BitDec 下的整步在线中位数为 0.207982 秒，完整 Sigmoid/TanH 为 0.089768/0.089408 秒，离线为 11.731217 秒。相同在线核的完整 10-bit LUT 在线为 0.210444 秒；匹配整步在线分段/完整比 0.9883 [0.9124, 1.0299]，未分辨。完整 LUT 离线为 7.539369 秒，离线分段/完整比 1.5560 [1.4792, 1.6033]；两者在线通信相同，分段离线通信更高。两种符号候选的整步在线及离线墙钟均退化；并行调度的在线改善伴随离线通信退化。SCI 具名对照显示在线收益，但数值激活算法不同且离线阶段有退化。

最终推荐状态为 `no_qualified_new_default`：`sequential-a1-segmented-bitdec` 保留 `qualified_reference`，新组合按具名父配置标为未分辨、退化或未测实验配置。parallel-a1 分段符号组合的整数/驻留/资源与阶段成本仍未独立验收，不传播其他组合资格。模型质量为 6798 条完整 99 步明文 folded Top-1 88.996764%，相对 FP32 损失 0.338335 个百分点；安全执行及性能范围仅 batch=128 单时间步。父规格、历史工单、99 步安全整网、分类头安全执行和论文正文均维持既定范围。
