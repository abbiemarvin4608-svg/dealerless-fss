# 08: Batch=128 新样本公平对照与中文报告

**What to build:** 重新测量默认 SCI、完整预处理 SCI 和两个明确命名的 ours 后端，报告双方获得相同预处理机会后的在线收益及离线、总成本交换。

**Blocked by:** 07：激活 worker 与两分支预处理库存隔离。

**Status:** closed

- [x] 正式开跑前冻结同一模型、输入、整数契约、Release 构建配置、各后端实际材料覆盖和全部共享开关；所有新主表行使用同一最终构建，历史样本仅作参考。匹配计算任务、调度、资源和预处理机会；各协议的材料数量、内部工作量和离线时长按实际记录，不强行配成相同。
- [x] 在 localhost、batch=128 单时间步下，a4-sequential 和 a2-parallel 各重新测量默认 SCI、SCI_preprocessed、SCI_ours 和 SCI_Beaver_segmented；后两者主表启用适用的共享原生预处理。
- [x] 沿用服务方 CPU 0,2,4,10、客户端 CPU 6,8,12,16，每方四核及 1.5 GiB RLIMIT_AS；每行十个 fresh query、warmup=0、600 秒时间预算。正式行串行运行，与编译及其他重负载隔离。
- [x] 主表资格要求对应整数诊断、内部材料覆盖和资源核验通过；双方日志证明本查询随机 OT、标准/相关 bit-triple 等库存已经实际生成，并完成 material-ready 后才装载查询输入/隐状态。严格预处理行从 prepare 完成至输出结束的 OT 扩展、一次性三元组库存生成与补货为零；默认 SCI 明确作为未启用每查询预处理的控制组。
- [x] 所有后端使用相同阶段边界，在线快照先于 enter_online；选择修正、消息/相关量适配、比较 AND 的打开及修正、输入后的材料读取与等待、已完成材料的 PRG/密钥评价全部计在线，input/ready 不隐藏协议成本。一次性材料生成按查询计离线，不借 setup 摊掉。
- [x] 保存逐查询双方日志、公开计划、生成/消费覆盖、命令、构建与配置身份；按同查询双方最大时延统计中位数、min/max 和离散程度，通信为双方发送字节之和。
- [x] 报告 setup、offline/material-ready、输入/ready、online、output、直接 query total、首请求和 session total、分阶段通信及峰值 RSS。online 在秘密份额计算及完成同步后结束，各行同样将最终公开结果重构计入 output，直接 query total 包含该输出成本；直接总时间独立测量，不把阶段中位数相加或将 socket 调用数当作协议轮数。
- [x] 中文报告在每个匹配 cohort 内分别计算默认 SCI→SCI_preprocessed 和 SCI_preprocessed→各 ours 的在线时延、在线通信及离线/总成本变化；区分 SCI_ours 的转换替换与 SCI_Beaver_segmented 的乘法加转换组合。在线加速倍数为基线在线时延除以 ours 在线时延，时延下降比例为一减去 ours 在线时延除以基线在线时延；主要比较使用新测 SCI_preprocessed，不与历史或另一调度行混算。
- [x] 不从历史在线时间扣减理论 λ 项估计性能；差异落在波动范围内时明确不确定，无收益或总成本增加也如实报告。未完成行保留真实失败状态与 null 完成成本。
- [x] 所有前置工单自带局部正确性和材料证据，本票只做最终集成资格核对与新样本复测，不代替补做前置验收；不要求无关开关全排列或跨查询流水。
- [x] 结果范围限定两方 batch=128 单时间步；SDCF-09 的 99 步整图、分类头及质量验收继续延期，不更新整网性能或 Top-1 主张。


## Implementation and validation

Implementation branch: `ticket08/fair-audit-report` (based on integration `4c605ff`).

Integrated implementation: `4a6aa63` on `integration/sci-sirnn-preprocessed`.

[Chinese report](../../../project/our/results/sci_sirnn_preprocessed_20261012/ticket08_fair/REPORT.zh.md),
[eligibility audit](../../../project/our/results/sci_sirnn_preprocessed_20261012/ticket08_fair/ELIGIBILITY.json),
[reproduction entry](../../../project/our/results/sci_sirnn_preprocessed_20261012/ticket08_fair/README.md)
and [red/green validation](../../../project/our/results/sci_sirnn_preprocessed_20261012/ticket08_fair/VALIDATION.md)
record all eight completed formal rows and sixteen qualified integer diagnostics.

Each formal row runs ten fresh queries with the same final native runtime/bridge,
matching cohort, four CPUs/party and 1.5 GiB RLIMIT_AS. Raw logs prove actual
query material generation and consumption; the archived driver reads query/hidden
inputs after material-ready. Strict native and bridge generation are zero in
input, ready, online and output. Original commands/configs and all 24 two-party
logs are preserved; archived runner/reference imports survive worktree cleanup.

Relative to SCI_preprocessed, Beaver+segmented FSS gives 1.870x / 1.451x online
speedup (a4 / a2), 46.51% / 31.06% latency reduction and 58.56% lower online
communication. Its directly measured query total increases 220.83% / 262.38%.
Conversion-only SCI_ours has 7.88% / 20.09% higher online medians and 2.49% lower
online communication; the a4 sample ranges overlap. No online latency advantage
is claimed for that backend. Full setup/first-query/session time and actual
non-overlapping session byte totals, every phase distribution and RSS are saved.

The frozen artifact audit exits zero with 8 formal + 16 diagnostic eligible rows.
Five focused qualification/report tests pass; the new complete-session test was
observed red before the aggregation change. An independent arithmetic review
agrees with all medians, ratios and eight session byte totals. SDCF-09 remains
deferred and no whole-network or Top-1 claim changes.
