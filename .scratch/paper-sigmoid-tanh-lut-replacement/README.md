# 论文 Sigmoid/TanH 完整 LUT 替换 SCI 激活

**日期:** 2026-10-13 启动；2026-10-14 收尾

**工单方式:** 项目本地 Markdown。

**Status:** complete

**Labels:** completed

| 规格 | 状态 | 交付目标 |
| --- | --- | --- |
| [完整激活替换与并行优化方案](spec.md) | ready-for-agent | 将论文 f12/q10 folded Sigmoid/TanH 接入当前 Beaver＋分段 FSS 运行时，比较分段/不分段 DPF one-hot 查表，完成整算子替换与四核并行。 |

方案已冻结 Q7/Q12 数值适配、显式上下文、两种 LUT 构造的材料生命周期、已有并行能力及本次优化、正确性/质量/性能验收。新后端命名为 `SCI_Beaver_segmented_PaperLUT`，名称中的 segmented 指外层分段 FSS，LUT 构造单独标记；历史 `SCI_Beaver_segmented_DPF_LUT` 仍表示 SCI 内部指数与倒数小表替换。

按用户补充要求，两个 5-bit 段与完整 10-bit 不分段算术 DPF 都是必测候选，各自运行串行和并行配置，比较 LUT、完整激活与整步在线耗时，并报告离线代价和内存。两者共用在线表示与计算核，推荐构造由实测决定。

主验收沿用此前已确认的两方 FastGRNN/Google-30 单时间步，batch=128，默认与非零隐状态。新增明文全序列质量评估用于检查激活近似变化。

12 张工单已按用户的 implement-spec 指令发布，见[工单依赖与发布索引](TICKET_PLAN.md)。12/12 张已按各自证据关闭；LAN/WAN 是新实测字节和因果轮次的通信模型。

09 交付正确的分段符号实验候选；最终匹配计量的串行与并行整步在线、离线墙钟均退化，未获默认准入。11 的旧 DPF 预算缺陷已修复；随后 full-LUT parallel-a2 在旧批次 query9 地址空间 OOM，失败记录保留。旧 `formal-allocator-clean/` 的 `/tmp` 构建丢失、缺失二进制补测以及改写清单/失败组状态均已审计归档，不用于最终推断。后续两次因冻结后分析器变化、资格选择遗漏而中断的尝试也完整归档。

最终执行入口为 `ticket11/final-build/{core,runtime,fixture}`、`ticket11/frozen-plan-final.json` 和 `ticket11/formal-final/`。计划内三份资格为 `qualification-reuse-final.json`、`final-build/qualification-budget/qualification.json`、`final-build/qualification-folded/qualification.json`，均通过。正式十行全部合格；两对 LUT 补测至每行三组后，在线差异仍未分辨，补测状态为 `exhausted_budget`。分段 DPF (5,5) 的并行整步在线中位数为 0.207982 秒，完整 Sigmoid/TanH 分别为 0.089768/0.089408 秒；与完整 10-bit LUT 的匹配比较未证实在线加速，离线墙钟和通信成本更高。最终结论为 `no_qualified_new_default`，保留具名实验配置。详见[实施记录](EXECUTION.md)、[正式性能报告](../../project/our/results/paper_folded_activation_20261013/ticket11/formal-final/REPORT.md)和[中文集成交付报告](../../project/our/results/paper_folded_activation_20261013/ticket12/REPORT.zh.md)。

追加 ticket13 已完成 SCI+SiRNN 与 `SCI_Beaver_segmented_PaperLUT` 的新接口复测：统一 1.5 GiB 生成预算、allocator 与 parallel-a2，每套 10 次 fresh query。按 LAN 1 Gbps/0.5 ms RTT、WAN 400 Mbps/4 ms RTT，PaperLUT 在线通信模型为对照的 1/7.2350 和 1/3.9323，离线则高 15.7830 和 15.6798 倍。逐 query 因果轮次、双端字节和传输/传播/通信三时间已直接写入 `result.json`，详见[匹配复测报告](../../project/our/results/paper_folded_activation_20261014/ticket13-interface-projection-matched/REPORT.zh.md)。
