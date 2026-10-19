# 06: cGGM 批量 AES

**What to build:** 利用连续密钥与查询批量视图，在现有常驻 worker 的连续任务块内共同推进独立 cGGM 路径，并对离线同层节点使用批量 AES，提高本地哈希吞吐；两方 DCF、完整转换和单时间步继续得到相同整数输出。

**Blocked by:** 03：连续材料与有界预取。

**Status:** resolved

- [x] 使用 03 提供的连续批量视图与受所有权管理的工作缓冲，复用现有批量 AES 能力；在线按深度推进多条独立路径，离线整层节点及叶子哈希使用同一批量内核，保留标量候选供逐值与性能对照。
- [x] 固定密钥哈希定义、密码学参数、纠正字应用和输出顺序保持不变；相同输入的标量与批量输出逐位一致，覆盖混合深度、不等宽段、尾块和大小任务。若启用由公开路径方向决定的叶子哈希裁减，单独核对其结果与哈希计数。
- [x] 通过现有 cGGM/DCF 公共接口及真实两方完整转换检查独立 oracle；同时验证 signed32、shift14 的 low/sign/wrap 修正和舍入，避免仅以标量实现作为唯一正确性依据。
- [x] 不改变协议逻辑轮数、双方材料对应关系或新鲜度要求；跨查询只复用缓冲容量，keys、mask、triples 每查询新生成且一次消费。混合深度任务完成后恢复原查询及段顺序。
- [x] 继续使用常驻池和短任务直接执行策略，批量 AES 不自行创建嵌套线程池；记录请求 worker 上限、实际参与者、各阶段哈希量与工作缓冲峰值，每方总 CPU 与内存预算保持一致。
- [x] 在 03 基线上分别测量仅在线批量、仅离线批量及二者集成，固定预取开关并保留布局加批量 AES 的组合结果，区分数据布局与吞吐贡献。16 位 DCF 按 count128/count12800、fixed-2/fixed-4、cGGM 上限 1/4 的匹配行分别运行 30/10 个新鲜查询；完整转换运行 5 个新鲜查询，无预热。
- [x] 用现有 Batch=128 单时间步验证集成和默认/非零隐状态逐算子整数一致，正式计量关闭中间张量揭示；受影响单步性能行与匹配基线各使用 10 个新鲜查询，不把历史五查询数据混入新结果。
- [x] 沿用规格冻结机器、构建、四核配额、每方 1.5 GiB 地址空间和不整形 localhost；基本原语/转换每行 120 秒、单步每行 600 秒。保存双方对齐的原始样本、哈希量、通信、RSS、setup/offline/online/直接总时延及中位数/范围/离散程度，失败完成时延为 null；变慢也如实报告，保持已验证默认配置直到新模式验收完成。

## Implementation Evidence

- Independent online/offline flags are `SEGFSS_AES_ONLINE=0|1` and `SEGFSS_AES_OFFLINE=0|1`, both default 0. The runtime runner forwards `--aes-online` and `--aes-offline` and records them; it rejects requested AES against binaries lacking the capability.
- Online compact and legacy public batch interfaces advance bounded groups of 256 mixed-depth paths per resident participant. Both leaf directions use the unchanged fixed-key hash; output remains in original key order. Offline node vectors and bounded leaf blocks reuse cryptoTools vector AES. No nested threads or new protocol rounds are introduced.
- TDD red: existing public cGGM/DCF test failed to compile because promised AES mode/hash/scratch accounting was absent. Green: local public test passes exact individual-share equality for depths 1/4/7/16/64 and batch lengths 1/7/8/9/257/640 with worker limits 1/4, plus the independent unsigned-comparison oracle (669924 comparisons / 2679696 assertions).
- Primitive samples and conversion preprocessing/online detail log internal and leaf hashes, requested AES modes, actual participants and additional AES scratch capacity. Final two-party, bridge, Batch=128 diagnostics and formal ablations are retained below. Historical implementation notes: `/tmp/runtime-opt-ticket06-notes.md`.

## Answer

已交付默认关闭的独立 online/offline AES 开关。在线通过常驻参与者推进有界混合深度路径块；离线节点和叶子块复用相同 fixed-key AES 定义。输出顺序、correction words、协议轮数、新鲜材料契约保持一致，不建立嵌套线程池；模式、hash 量、scratch 和实际参与者均进入日志。

本地公共接口覆盖 depth 1/4/7/16/64，batch 1/7/8/9/257/640 和 worker 1/4，包含逐位共享结果与独立 unsigned-comparison oracle。[64 行原语/转换消融](../../../project/our/results/runtime_optimization_20261010/primitive_ablations-recheck/summary.json)分别保留 online-only、offline-only 和 both 模式，DCF count128/count12800 分别 30/10 个新鲜查询、完整转换 5 个。[kernel/AES 单步消融](../../../project/our/results/runtime_optimization_20261010/kernel_aes/summary.json)为 7 个正式十查询行，[16 行默认/非零诊断](../../../project/our/results/runtime_optimization_20261010/kernel_aes_diagnostics/summary.json)全部通过，每查询 15 张量/172928 整数与 SCI 一致。

[归因报告](../../../project/our/results/runtime_optimization_20261010/kernel_aes/report.md)显示 online AES 的本地 cGGM eval 下降约 42–43%，但 online-only/offline-only/both 的直接 query 中位数均高于控制；both 的直接总成本上升 3.02%。这些收益和回退完整保留，默认开关继续关闭；若目标为本地 online eval，可单独选择 online 模式。SDCF-09 继续延期。
