# 12: 补齐 Google-30 三后端主表

**What to build:** 将 `SCI + ours`、SCI/SiRNN OT 和 MP-SPDZ 三个同威胁模型后端放入同一 Google-30 主比较，统一质量门槛和端到端计量，并清楚标出不可用配置与上下文结果。

**Blocked by:** #09 完成 MP-SPDZ Google-30 安全推理；#11 完成 Google-30 双后端性能对照

**Status:** partial — MP-SPDZ unavailable under the user-selected bounded budget

**Report:** [ticket 12 report](../../../project/our/results/ticket12_main_table/REPORT_CN.md)

- [x] 单独报告相对约定参考的数值误差，注明比较张量、尺度、单位和统计方式；数值误差与 Top-1、agreement 分开，不以分类指标替代。
- [x] 主表恰好纳入 `SCI + ours`、SCI/SiRNN OT 和 MP-SPDZ 三个两方、无可信经销者、半诚实后端，并为每行记录威胁模型和输出揭示规则。
- [x] 三个后端使用同一 Google-30 FP32 checkpoint、数据划分和独立评估流程；系统主表允许各自选择达标配置，同图控制表固定同一原生量化产物。每行注明所属轨道，不强制系统主表统一高位宽。
- [x] 每个后端同时报告 Top-1、loss_pp、prediction agreement，并按 ≤1.0 pp 门槛区分达标、未达标和 unavailable，不因性能结果事后改变准入规则。
- [x] 每个可运行后端报告 offline、online、total 延迟、双方通信、依赖轮数、峰值内存及首查询/摊销查询成本，计量边界保持一致。
- [ ] 同图行满足 #11 的逐算子和整图一致性，系统轨各产物与其冻结参考核验；MP-SPDZ 满足 #09 的真实 OT 和全网安全执行条件。只有数值契约匹配的对照用于隔离协议差异。
- [x] 主表之外单列 ABY-64 和 dealer-assisted FSS，明确其控制位宽或信任假设差异，不将其混入同威胁模型排名。
- [x] 对超时、OOM、缺失完整模型或无法满足数值契约的 baseline 保留原始原因和资源上限，禁止用 smoke、structure-only 或明文结果补齐表格。
- [x] 产出可追溯的主表、原始日志索引、配置清单和复现命令，并确认结论只覆盖实际完成的 Google-30 任务和测量条件。
