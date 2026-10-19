# 08: 隔离上下文并执行图分支并行

**What to build:** 在现有两方 Batch=128 单时间步入口中，先隔离图任务状态并验证顺序执行，再让真实独立分支按依赖并行完成；四路后端在相同四核预算下运行，并分别计量上下文隔离与分支并行的影响。

**Blocked by:** 07：原生激活常驻并行。

**Status:** resolved

- [x] 首先提供隔离任务上下文的顺序执行模式，每个任务具有唯一调用身份及独立的当前层、转换计划和材料游标；保留原有顺序模式作为对照。隔离后的顺序模式通过整数诊断和消费检查后，再开启图分支并行。
- [x] 复用 07 的通道所有权与计算配额机制，明确任务到协议连接的两方映射。独立连接由对应上下文管理，共用连接通过有序协议调度发送和接收，不能多个任务直接并发访问同一有状态协议对象；日志和计数按任务归属记录后汇总。
- [x] keys、mask、triples 按查询及任务新鲜分配、一次消费，仅工作缓冲可在所有权明确时复用。SCI、SCI_ours、SCI_Beaver、SCI_Beaver_segmented 使用相同图依赖与调用身份体系，根据身份及依赖验证覆盖和消费数量，不假定日志仍是全局串行顺序。
- [x] 按真实依赖调度输入投影与隐状态投影两条链，以及共同线性输出之后的两条激活分支；先检验同层独立 opening 合批，再评估独立通道执行，并保留实际协议计划和逻辑轮数。后继任务等待输入就绪，普通 Beaver 跨层依赖和循环时间步间递归依赖继续保留。
- [x] 并行主候选固定两条分支、每条最多两个计算参与者，分支内 activation 和 cGGM 上限均为 2，共用配额同时约束 Eigen、显式 OpenMP 和调用线程。记录请求与实际并发；仅设置四核 affinity 不足以通过资源验收，无法满足配额的模式标为 unsupported。
- [x] 用默认和非零隐状态下的现有单步诊断核对 15 张量、172928 个整数，原有顺序、隔离顺序和并行模式与 SCI 数值参考逐值一致，并验证重复查询的输出顺序、调用覆盖和材料消费。保持位宽、尺度、舍入及原生激活算法，正式计量关闭中间张量揭示。
- [x] 隔离顺序模式与图并行可独立配置，保留原有顺序、仅隔离顺序、隔离后两分支三组局部对照，其他参数匹配；四路后端同步重测并补充激活与图调度组合证据。分别报告隔离开销和新增并行收益，不能只给累计最终配置。
- [x] 使用冻结 Batch=128 模型、量化输入与 Release 环境、不整形 localhost；服务方 CPU 为 0,2,4,10，客户端为 6,8,12,16，每方 RLIMIT_AS 为 1.5 GiB、总预算四个计算参与者。每个单步行 10 个新鲜查询、无预热、600 秒预算，测量串行进行且无并发重负载。
- [x] 记录 setup、offline、material-ready、input/ready、online、output、直接 query/session total、首查询、分支工作时间、全部通道通信、逻辑轮数、峰值 RSS 和实际参与者；不将重叠任务耗时相加冒充墙钟时延。保存原始样本，逐查询取双方最大时延后报告中位数、范围和离散程度，通信取双方发送字节之和。
- [x] 交付可复现局部对照和配置建议，收益、持平及回退均报告，不以必然加速作为验收条件；失败行保留真实状态和 null 完成时延。最高验收范围为现有两方单时间步入口及原语接口，SDCF-09 整图验证继续延期。

## Answer

已交付原顺序、隔离顺序、两分支并行模式和同层 opening 合批控制。Peer/context 拥有独立调用身份、协议状态、转换计划和材料游标，公共依赖约束后继任务；两分支各至多两个计算参与者，activation/cGGM/native OT 请求与实际团队均受同一配额约束，Eigen 内部并行关闭。上下文与并发修正已合入集成分支。

[graph_schedule](../../../project/our/results/runtime_optimization_20261010/graph_schedule/summary.json)保存四路后端、顺序/隔离/并行/opening 合批共 16 个正式十查询行；[graph_diagnostics](../../../project/our/results/runtime_optimization_20261010/graph_diagnostics/summary.json)与[整数核验](../../../project/our/results/runtime_optimization_20261010/graph_diagnostics/integer_verification.json)保留 36 个默认/非零正确性行，每查询核对 15 张量/172928 个整数。[上下文 bridge](../../../project/our/results/runtime_optimization_20261010/context_bridge/runtime-opt-multi-context-green-results/summary.json)、[generation participant-cap 回归](../../../project/our/results/runtime_optimization_20261010/context_bridge/runtime-opt-generation-cap-green-pair/summary.json)和[byte 协议检查](../../../project/our/results/runtime_optimization_20261010/review_verification/byte_core_protocol/summary.json)补充真实两方独立上下文、预算和消费证据。

[匹配图调度对照](../../../project/our/results/runtime_optimization_20261010/graph_schedule/graph_comparison.json)保留隔离开销、分支时间、直接 query、全部通道通信、资源和回退；opening 合批对 SCI/SCI_ours 接近持平或回退，对 Beaver 行约有 1–2% 收益。不同冻结队列分别存档，最终 byte transport 队列由 10 汇总，不以并行任务工作时间相加代替墙钟时延。历史 SDCF-09 整图验证继续延期。
