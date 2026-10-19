# 10: 集成消融与 Batch=128 最终对照

**What to build:** 汇总已单独验证的优化，在匹配资源和数值算法下完成必要组合及四路后端的新样本对照，报告 Batch=128 单时间步的在线、离线、总成本与配置建议。

**Blocked by:** 02：转换通信合批与位打包；04：Beaver 本地矩阵乘法与布局优化；06：cGGM 批量 AES；08：隔离上下文并执行图分支并行；09：离线行分块与材料重叠准备。

**Status:** resolved

- [x] 汇集每张前置工单及其各子优化的局部正确性、性能、资源和失败证据；通过传递依赖同时覆盖计量、连续布局/预取、按域分段/离线树分块和激活并行，避免重复设置无意义阻塞边。合并工单不合并掉各优化的归因记录。
- [x] 验证多域扫描+双输入扩环、连续布局+批量 AES、分段策略+离线 tile、激活并行+图分支调度及离线 tile+重叠准备；按实际计划检查轮数、输出映射、调用覆盖和材料一次消费，不做所有开关的笛卡尔积。
- [x] 经现有原语、完整转换与两方单时间步接口验证整数语义；默认及非零隐状态覆盖 15 张量、172928 个整数诊断，正式性能关闭中间张量揭示。
- [x] 主表比较 SCI、SCI_ours、SCI_Beaver、SCI_Beaver_segmented，保留优化前组合行；对四路使用相同共享激活/图调度配置。转换比较 fixed-2、fixed-4、depth-cap-8，并保持公开 ScalarMul 和原生 SCI 激活数值算法。
- [x] 遵循冻结环境：服务方 CPU 0,2,4,10、客户端 CPU 6,8,12,16；每方四核及 1.5 GiB RLIMIT_AS；图并行两分支各最多 2 参与者，离线 Beaver/FSS 两任务各最多 2 参与者，内部库线程纳入配额。正式测量串行运行。
- [x] 单步所有匹配行重测 10 个无预热 fresh query，不混入历史五查询样本；16 位 DCF count128/12800 分别 30/10 次，signed32 shift14 count128 转换 5 次。基本原语/转换 120 秒，宽域及单步 600 秒每行。网络按 localhost、sirnn_lan（RTT 0.8 ms，3016 Mbps）、wan（RTT 2 ms，100 Mbps）分表，后两者只补测通信相关对照且覆盖全部通道。
- [x] 保存原始样本、构建/模型/输入身份、命令、实际域/段宽/参与者、发送字节和可分离轮数、峰值 RSS；按双方同查询最大时延统计中位数、min/max和离散程度。报告 setup、offline/material-ready、online、直接 query/session total、首请求及适用的稳态吞吐，重叠工作不重复计作墙钟成本。
- [x] 开跑前声明在线优先与离线/总成本优先的推荐准则；仅正确性和资源验收通过的配置参与推荐。差异未超出波动时报告持平或不确定，不预设加速比例；没有满足准则的收益则保留原推荐配置。
- [x] 两段宽域或单步未完成时保留 oom/timeout/unsupported/unavailable、失败位置及 null 完成成本，不用小 count 替代；发布中文结果报告和逐工单状态，区分分段、Beaver、通信合批及共享并行收益。SDCF-09 整图/质量验证继续延期，不更新论文整图结论。

## Answer

已完成 final2 的两种共享调度配置：顺序图与两分支图各包含 SCI、SCI_ours、SCI_Beaver、SCI_Beaver_segmented 四路后端，正式行均为 10 次 fresh query，无预热；默认及非零隐状态诊断逐值通过。报告、原始汇总、资格文件、保留日志重审和两条独立审查见 [Batch=128 报告](../../../project/our/results/runtime_optimization_20261010/REPORT_CN.md)、[final2](../../../project/our/results/runtime_optimization_20261010/final2/summary.json)、[资格清单](../../../project/our/results/runtime_optimization_20261010/final2/ELIGIBILITY.json)、[原始审计](../../../project/our/results/runtime_optimization_20261010/FINAL_RAW_AUDIT.json) 和 [审查记录](../../../project/our/results/runtime_optimization_20261010/review/)。

在线优先的观测最快行是 `SCI_Beaver_segmented-a2-parallel`：online 中位数 0.469814 s，相对匹配 SCI 的 1.149810 s 为 2.447×；顺序图为 0.491166 s，相对 1.425539 s 为 2.902×。组合后端 offline 分别为 6.095547/6.230367 s，直接 query total 为 6.570007/6.720596 s；SCI 的直接总成本仍最低。fixed-2 tile8 的完整 Batch=128 单步在 600 s 预算内 timeout，完成成本保持 null。历史 SDCF-09 继续延期。
