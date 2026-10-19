# 07: 激活 worker 与两分支预处理库存隔离

**What to build:** 让四后端在现有激活并行和两分支单步调度中，按真实协议上下文正确消费预处理库存，并保持匹配 CPU、内存与通信计量。

**Blocked by:** 06：共享预处理策略贯通四后端完整单步。

**Status:** closed

**Resolved:** 2026-10-12; integrated branch tip `5409fb0` on `integration/sci-sirnn-preprocessed`.

- [x] 按公开 worker 分块、方向和图分支给每个真实原生 Math/OT 上下文分配独立库存和游标；两方路由一致，不依赖任务完成顺序。
- [x] 支持既有 a4-sequential 和 a2-parallel 共享配置：顺序图激活上限四，两分支各最多两个计算参与者；默认 SCI 与各预处理后端获得同样调度能力。
- [x] 每方四核、1.5 GiB RLIMIT_AS 下，协议线程、Eigen、OpenMP、激活、材料及其他计算 worker 均纳入总配额；不能按任务分别超配。
- [x] 所有原生、反向及新增通道的阶段通信完整计量；重叠任务时间不相加充当墙钟时间，输入后的材料装载与等待保留在线。
- [x] 在两 cohort 对四后端分别完成 batch=128 默认和非零隐状态的逐整数诊断，核验 15 张量、172928 个整数一致及查询材料消费平衡。
- [x] 通过真实两方调用验证激活余数分块、两分支执行顺序变化和连续查询；错误上下文或方向的材料不能被其他任务消费。对启用共享预处理的三个后端，全部上下文从 prepare 完成至输出结束的 OT 扩展、一次性三元组库存生成与补货保持为零，包含 input/ready 间隙；默认 SCI 控制组保留实际在线行为并记录。
- [x] 输出实际参与者、上下文、库存规模、padding 弃置、峰值 RSS 与覆盖证据；OOM、timeout 或不匹配资源不能取得匹配主表资格。

**Evidence:** [完整上下文验收](../../../project/our/results/sci_sirnn_preprocessed_20261012/ticket07_contexts/README.md)；16 行完整诊断均通过 fresh SCI / frozen native 逐整数与库存、配额、RSS 核验，真实两方余数及错误路由 harness 通过。冻结 runtime 与 bridge 未在诊断和正式计划之间重构建。SDCF-09 继续延期。
