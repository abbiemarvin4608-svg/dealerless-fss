# SCI+SiRNN 预处理基线规格与工单

**日期:** 2026-10-12（Asia/Shanghai）

**发布状态:** completed

**工单方式:** 项目本地 Markdown；沿用既有 ready-for-agent 标签含义。

| 规格 | 状态 | 交付目标 |
| --- | --- | --- |
| [SCI+SiRNN 预处理基线与 Batch=128 公平对照](spec.md) | closed | 已实现真实输入前预处理与共享策略，并完成 batch=128 单时间步的匹配复测。 |

用户已确认以下八张工单的粒度与阻塞依赖，现已逐票发布。完整交付说明和规格覆盖见 [工单拆分索引](TICKET_PLAN.md)。

| 工单 | 阻塞工单 | 状态 |
| --- | --- | --- |
| [01：输入前材料生命周期与二选一 GOT 闭环](issues/01-prepared-got-lifecycle.md) | 无 | closed |
| [02：多选一 LUT 与批量 GOT 位乘法预处理](issues/02-prepared-lut-batched-got.md) | 01 已完成 | closed |
| [03：B2A 与乘法的算术 COT 预处理](issues/03-prepared-arithmetic-cot.md) | 01 已完成 | closed |
| [04：比较材料与完整整数转换预处理](issues/04-prepared-comparison-conversion.md) | 02、03 已完成 | closed |
| [05：原生 Sigmoid/TanH 内部预处理闭环](issues/05-prepared-native-activations.md) | 04 已完成 | closed |
| [06：共享预处理策略贯通四后端完整单步](issues/06-shared-preprocessing-step.md) | 05 已完成 | closed |
| [07：激活 worker 与两分支预处理库存隔离](issues/07-context-parallel-preprocessing.md) | 06 已完成 | closed |
| [08：Batch=128 新样本公平对照与中文报告](issues/08-batch128-fair-report.md) | 07 已完成 | closed |

01 至 08 已合入 `integration/sci-sirnn-preprocessed`，全部工单关闭。四后端、a4-sequential/a2-parallel 两种调度及默认/非零隐状态的 16 行 batch=128 完整诊断通过逐整数、上下文库存和四核/内存资格验收；8 行正式主表各完成十个 fresh query，各行串行运行。三个预处理后端在输入前实际生成 OT 与安全 bit-triple，输入后所有上下文生成增量为零。结果见[中文报告](../../project/our/results/sci_sirnn_preprocessed_20261012/ticket08_fair/REPORT.zh.md)、[24 行资格证据](../../project/our/results/sci_sirnn_preprocessed_20261012/ticket08_fair/ELIGIBILITY.json)和[执行记录](EXECUTION.md)。

2026-10-12 已复核并收紧[公平基准与阶段边界](TICKET_PLAN.md)：SCI_preprocessed 必须在查询输入/隐状态装载前实际生成随机 OT、标准/相关 bit-triple 等材料，两个 ours 的剩余原生调用获得相同预处理机会；输入相关修正及输入后的材料读取等工作计在线。各后端按实际协议生成材料，不要求材料数量相同。

主测试接口沿用现有两方 FastGRNN 单时间步，默认与非零隐状态均核验整数结果。主表保留默认 SCI，并比较 SCI_preprocessed、SCI_ours 与 SCI_Beaver_segmented；后两者同样启用适用的共享原生 SCI 预处理。顺序与两分支配置各行十个 fresh query、无预热，完整报告离线、在线、直接查询总成本、setup、通信和内存。

规格由 to-spec 发布，经 to-tickets 拆分后按 implement-spec 完成代码、真实验收与新样本报告。相对 SCI_preprocessed，Beaver+分段 FSS 在 a4/a2 的在线加速为 1.870x/1.451x，在线通信减少 58.56%，直接查询总时间增加 220.83%/262.38%；SCI_ours 本轮未取得在线时延优势。SDCF-09 的 99 步整图、分类头和质量验收继续延期。

相关已完成工作见 [SCI/FSS 运行时优化工单](../sci-fss-runtime-optimization/README.md) 与 [历史 batch=128 报告](../../project/our/results/runtime_optimization_20261010/REPORT_CN.md)。历史样本不进入新主表。
