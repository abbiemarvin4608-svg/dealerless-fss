# 07: 原生激活常驻并行

**What to build:** 在现有两方 Batch=128 单时间步入口中，用常驻、独立协议 worker 对 Sigmoid/TanH 的连续元素区间并行求值；四路后端获得相同的激活并行能力，并能单独比较激活 worker 为 1 与 4 的正确性、在线成本和新增初始化成本。

**Blocked by:** 01：补齐计量并冻结匹配基线。

**Status:** resolved

- [x] 保持 Sigmoid/TanH 原生 SCI 数值算法、位宽、尺度、舍入和输出顺序。单参与者直接调用，多参与者按连续元素区间使用常驻 worker；支持非整分片尾部及重复查询。
- [x] 每个 worker 拥有独立 SCI Math/IO/OT 上下文和固定两方通道映射。激活内部始终走 native SCI，不通过修改全局线程数让未分片的 Beaver 或转换 hook 并发执行；调用身份、日志和计数按 worker 收集。
- [x] SCI、SCI_ours、SCI_Beaver、SCI_Beaver_segmented 均支持相同的 activation=1/4 配置。并发配额包含调用线程、cGGM、Eigen 和显式 OpenMP 等实际计算参与者；每方最多四个参与者，记录请求值、实际值及内部并行配置。
- [x] 使用现有单时间步诊断，在默认和非零隐状态下核对 15 张量、172928 个整数；逐值比较 activation=1/4 与原生 SCI 数值参考，验证跨查询调用覆盖完整及一次性材料新鲜消费。正式计时关闭中间张量揭示。
- [x] 新增协议上下文初始化计入 setup，所有实际通道的通信与内存纳入总计；输出独立激活耗时和 setup、offline、material-ready、online、output、直接 query/session total、首查询及峰值 RSS。并行子任务工作时间与墙钟时间分别记录。
- [x] 局部消融固定其他配置，仅改变 activation=1/4，并为四路后端同步重测。使用冻结模型、量化输入和 Release 环境；服务方 CPU 为 0,2,4,10，客户端为 6,8,12,16，每方 RLIMIT_AS 为 1.5 GiB，主网络为不整形 localhost。
- [x] 每个单步对照行运行 10 个新鲜查询、无预热，墙钟预算 600 秒；正式测量串行进行。保存原始样本，逐查询取双方时延最大值后报告中位数、范围与离散程度，通信取双方发送字节之和；不混入历史五查询样本。
- [x] 提交激活成本占比、在线/离线/直接总成本和额外 setup/内存的匹配对照，允许持平或回退，未验证配置不成为默认。失败行保留真实状态和 null 完成时延；验收止于现有单时间步及原语接口，SDCF-09 继续延期。

## Implementation Evidence

- Runtime capability test first failed against the frozen baseline (missing activation worker controls), then passed against the worker build.
- Existing public runtime tests passed. A two-query Batch=5 test compares activation=1/4/3, including a nonuniform three-way tail; all 30 tensors match.
- `/tmp/runtime-opt-07-verification/summary.json`: all 16 diagnostic rows passed (four backends, default/nonzero hidden state, activation=1/4, two fresh queries). Each query matches the frozen native SCI reference at 15 tensors and 172928 integers.
- Frozen implementation binary: `/tmp/runtime-opt-07-runtime/sdcf-runtime`; linked preprocessing bridge: `/tmp/runtime-opt-01-build/libsegfss_sci_bridge.so`. New contexts, resident worker startup and public configuration handshake belong to setup. All SCI channels enter the existing atomic phase counters.
- Default activation remains 1. Native one-worker activation retains two OT participants; parallel activation executes both OT directions serially within each worker, and Eigen internal parallelism is disabled. Native activation detail records replace legacy finalize activation totals.
- Matched ten-query secure activation rows are retained in activation/; diagnostic time is not used as performance evidence. SDCF-09 remains deferred.

## Answer

已交付四路后端共用的 activation=1/4 控制、常驻 native SCI Sigmoid/TanH worker、独立 Math/IO/OT 上下文和固定两方通道。Batch=5 的 activation=1/4/3 公共测试覆盖非整分片尾部与重复查询；数值算法、位宽、尺度及输出顺序保持一致。上下文和常驻池启动计入 setup，全部 SCI 通道进入 phase meter；Eigen 内部并行关闭，实际计算参与者受四核配额约束。

[正式激活消融](../../../project/our/results/runtime_optimization_20261010/activation/summary.json)包含四路后端 × worker 1/4 的 8 个匹配十查询行；[默认/非零隐状态诊断](../../../project/our/results/runtime_optimization_20261010/activation/diagnostics/summary.json)全部 16 行通过，每查询 15 张量/172928 整数与 native SCI 一致。正式行无预热、不揭示中间张量，保留双方样本、直接 query/setup/first-query、全部通道通信和峰值 RSS。

worker 1→4 的 online 中位数在 SCI/SCI_ours/SCI_Beaver/SCI_Beaver_segmented 分别为 1.8313→1.3940、1.6477→1.2760、1.5062→1.0304、1.2529→0.8240 s；新增 setup 约 0.50 s，并增加 RSS。单查询总成本与初始化分别记录；最终匹配队列由 10 汇总。最高验收范围为现有单时间步及原语接口，SDCF-09 继续延期。
