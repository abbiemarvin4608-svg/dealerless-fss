# 09: 离线行分块与材料重叠准备

**What to build:** 先将矩阵 Beaver OT 的公开行 tile 配置化并独立验证，再以独立协议上下文重叠准备同查询 Beaver 与 FSS 材料；通过现有两方单时间步入口完整消费材料，在固定预算下比较分块和重叠各自的离线成本、首请求时延与查询总成本。

**Blocked by:** 05：按域分段与离线树分块。

**Status:** resolved

- [x] 先公开配置 Beaver OT 行 tile，双方依据一致公开计划确定分块和顺序；运行前冻结包含现有默认值的有限候选集合。覆盖小于一个 tile、非整 tile 尾部、非方阵及单步实际形状，记录块数、尾块大小、临时工作集、常驻材料、离线通信批次及逻辑轮数。
- [x] 通过现有 Beaver 原语两方整数重构与 SCI 交叉检查验证不同 tile，覆盖环回绕及不同累加宽度，并在现有运行入口消费生成材料。保持环宽、符号扩环、矩阵累加、截断和无经销协议契约，核对产出顺序、消费数量及实际后端；正确性和工作集记录完成后再接入重叠调度。
- [x] 为同查询 Beaver/FSS 准备任务建立独立有状态协议上下文和固定两方通道映射，不允许并发共用 LinearOT/IOPack。该工单自行完成离线任务隔离，不依赖 08 的在线图分支并行；全部通道纳入资源和通信记录。
- [x] 复用 05 的有界 FSS tile 调度与内存预留，联合 Beaver tile 工作集控制展开缓冲、最终材料和通道上下文的总驻留量，并实测组合峰值。主候选为两个准备任务、每任务最多两个计算参与者，cGGM、Eigen、显式 OpenMP、协议计算及调用线程均计入任务配额，每方总计四核且 RLIMIT_AS 为 1.5 GiB。
- [x] 每个查询新鲜分配 keys、mask、triples 并一次消费，材料归属按公开调用计划确定，不受任务完成先后影响；使用现有生命周期检查核对消费。跨查询预取仅为独立可选模式，不阻塞本工单，仅在具备新鲜材料分配和有界队列时另行启用。
- [x] 默认和非零隐状态下，用现有单步诊断核对 15 张量、172928 个整数，分块前后及顺序/重叠准备结果与 SCI 数值参考逐值一致。四路后端的图调度与 activation 设置保持匹配，保留公开 ScalarMul、原生激活和全部定点契约，正式计量关闭中间张量揭示。
- [x] Beaver tile 与材料重叠具有独立配置，分别做逐 tile 对照、固定 tile 的顺序/重叠对照，并验证必要的分段与 tile 组合；保留原生 SCI、仅 Beaver、仅 FSS 和组合后端的匹配控制。各行仅改变对应优化，失败行保留，不用小 count 原语数据代替单步结果。
- [x] 使用冻结 Batch=128 环境和不整形 localhost，服务方 CPU 为 0,2,4,10，客户端为 6,8,12,16；每个单步行 10 个新鲜查询、无预热、600 秒预算，基本原语每行 120 秒。新增原语形状的样本数和候选集合在运行前声明，正式测量串行进行。
- [x] 输出 setup、逐查询材料生成工作、offline/material-ready 墙钟时间、input/ready、等待、online、output、直接 query/session total、首请求、全部通道通信、峰值 RSS 和实际参与者；重叠工作时间不相加作为墙钟成本。若另测跨查询模式，另列稳态吞吐并保留全部离线成本；保存原始样本，按双方最大时延报告中位数/范围/离散程度，通信取双方发送字节之和。
- [x] 根据预声明目标分别给出 tile 与重叠配置建议，报告收益、持平或回退，不要求必然加速；oom、timeout、unsupported 等失败记录真实状态及 null 完成时延。资源目标未完成时明确限制，SDCF-09 整图验证继续延期。

## Answer

已交付 Beaver OT 行 tile 1/8/32/128（兼容默认 32）、确定的公开分块顺序、独立 Beaver/FSS 准备上下文和有界同查询重叠。两任务各至多两个计算参与者，全部材料与 tile 工作集进入联合预算，通道由独立 owner 管理，最终材料按公开调用身份一次消费；不包含跨查询流水。

[正式报告](../../../project/our/results/runtime_optimization_20261010/offline_tiling_overlap/REPORT.md)和[summary](../../../project/our/results/runtime_optimization_20261010/offline_tiling_overlap/summary.json)保留全部 11 个预声明完成行，每行 10 个新鲜查询、无预热、600 秒预算、每方 1.5 GiB；最大所测 RSS 为 384480 KiB。[Beaver 原语](../../../project/our/results/runtime_optimization_20261010/beaver_all_tiles_correctness/summary.json)核对 278824 个重构整数；[tile/overlap 默认与非零诊断](../../../project/our/results/runtime_optimization_20261010/offline_tiling_overlap_diagnostics/integer_comparison.json)以及[最终 overlap 诊断](../../../project/our/results/runtime_optimization_20261010/offline_overlap_final_diagnostics/integer_comparison.json)均通过，每查询 15 张量/172928 整数与 SCI 一致。

tile32 的顺序→重叠使 offline 中位数下降 3.64%、直接 query 中位数下降 3.02%，online 上升 1.96%；首请求未改善。tile8 是较低内存候选，tile32 保持兼容默认，重叠为可选配置。历史 per-matrix Beaver receive 计数不可用，不能把旧日志的零解释为零通信；完整 phase traffic 已计入正式结果，`a39164f` 为后续构建补齐 owned receive 计数。资源、失败和总成本限制均在报告中保留；历史 SDCF-09 继续延期。
