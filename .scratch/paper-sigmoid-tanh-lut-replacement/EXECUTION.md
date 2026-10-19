# 完整 folded 激活实施记录

日期：2026-10-13 启动，2026-10-14 收尾。用户授权：实施 12 张工单，报告完整 FSS Sigmoid/TanH 与 SCI/SiRNN 的离线、在线及 LAN/WAN 成本。

集成分支：`integration/paper-folded-activation-lut`。代码起点：`c4c67cb`；工单发布提交：`ca99afe`。用户已有的运行时优化 README 修改不纳入本批提交。

## 当前进度

| 工单 | 执行状态 | 证据 |
| --- | --- | --- |
| 01 | closed | 实现提交 `7861876` 已合入；双非默认上下文完整激活、两次 fresh query、默认通道隔离及原语回归证据见 [01 工单](issues/01-context-isolated-activation.md)。 |
| 02 | closed | 实现提交 `886478a`、CLI 换行修正 `b3946eb` 已合入；验收与冻结来源见 [02 工单](issues/02-folded-integer-oracle.md)。 |
| 03 | closed | 实现提交 `59028d9` 已合入；两次新查询穷举 q10 全域及尾部、12 类库存错配通信前拒绝、完整 Q12 双上下文并发及历史 9/14/16 回归日志均通过；证据见 [03 工单](issues/03-full-q-lut-lifecycle.md)。 |
| 04 | closed | 实现提交 `c3b2767` 已合入；两构造两次新查询穷举 q10 全域及尾部、14 类库存错配与 3 类配置通信前拒绝、完整 Q12 双上下文并发同值回归通过，分段每元素 1024 AND 与 1024 输出 B2A 实际计量，四 CPU affinity 与 1.5 GiB AS 下通过；证据见 [04 工单](issues/04-segmented-lut-lifecycle.md)。 |
| 05 | closed | 实现与验收提交 `22319c4` 已合入；完整原生 Sigmoid 在随机分享下穷举 65536 编码零差异，分段构造与历史 SCI/DPF 边界回归通过；batch=1 默认及非零隐状态各两次新查询的全部 15 张量零差异，离线/在线 18 阶段流量守恒与库存/epoch 检查通过；证据见 [05 工单](issues/05-native-folded-sigmoid.md)。 |
| 06 | closed | 实现与验收提交 `a75443d` 已合入；完整原生 TanH 的 65536 编码随机分享测试及 batch=1 默认/非零隐状态各两次新查询通过，最终审计六项协议检查全真；适配器、符号/裁剪、LUT 与实际共享准备阶段流量计量完整；证据见 [06 工单](issues/06-native-folded-tanh.md)。 |
| 07 | closed | 实现与验收提交 `d16f31b` 已合入；两种 LUT 构造的完整原生 Sigmoid/TanH 全 65536 编码通过，四组 batch=128 默认/非零隐状态各两次 fresh query 的全部 15 张量、172928 个整数均零差异，联合驻留及零在线生成/加载/等待通过，最大 VmPeak 为 761470976 字节，35 个聚焦测试通过；证据见 [07 工单](issues/07-batch128-joint-residency.md)。 |
| 10 | closed | 实现提交 `e3b9740` 已合入；6798 条完整 99 步明文 folded Top-1 为 88.996764%，相对 FP32 损失 0.338335 个百分点，通过 1.0 门槛。14 个数值与质量报告测试复核通过；证据见 [10 工单](issues/10-frozen-model-quality.md)。 |
| 08 | closed | 实现与验收提交 `3631f42` 已合入；两种 LUT、三种调度、两种隐状态在 batch=1 与 batch=128 各 12 行、每行两次 fresh query 全部通过；259 元素 130/129 分片与 cap-one transport/hash 验证通过，43 个聚焦测试通过，最大 party VmPeak 936857600 字节。原始轨迹压缩后两种 LUT 代表行重分析一致；完整因果 scope 和局部匹配计量见 [08 工单](issues/08-four-participant-parallelism.md)。 |
| 09 | closed | 原生全域、显式上下文及八行两次 fresh query 单步资格通过，最大 party VmPeak 982069248 字节。`formal-final/` 中符号候选相对匹配 BitDec 父配置的串行、并行整步在线比分别为 0.8913 [0.8239, 0.9300]、0.9020 [0.8397, 0.9833]，均退化；离线墙钟及离线 LAN/WAN 也退化，保持 `experimental_regression`。旧批次 1.055432 等数字仅属历史诊断。 |
| 11 | closed | 持久共同构建、三份冻结资格及 `formal-final/` 的十行全部合格；四条 LUT 构造行各三组，其余六行各一组，共 18 组、180 次 fresh query。两对 LUT 在线比较补测至上限仍未分辨，`sampling_status=exhausted_budget`；旧失败与中断批次不进入最终统计。[正式报告](../../project/our/results/paper_folded_activation_20261013/ticket11/formal-final/REPORT.md)。 |
| 12 | closed | [最终中文报告](../../project/our/results/paper_folded_activation_20261013/ticket12/REPORT.zh.md)与 `summary.json` 已由最终计划、资格和 `formal-final/analysis.json` 生成，状态 `complete`；结论为 `no_qualified_new_default`。未测或未独立验收的组合维持实验/待资格状态；安全性能仅单时间步。 |
| 13（追加） | closed | SCI+SiRNN 与 Segmented DPF PaperLUT 在统一预算及 allocator 下重新执行，共 20 次 fresh query；轮次、双端字节及 LAN/WAN 三项时间直接写入接口并逐 query 复核通过。[复测报告](../../project/our/results/paper_folded_activation_20261014/ticket13-interface-projection-matched/REPORT.zh.md)。 |

## 验证与计量

- 沿用用户认可的真实两方激活接口、batch=128 完整单时间步及默认/非零隐状态。
- 最终性能在代码冻结、正确性和材料/资源资格通过后执行；正式测量不与编译、质量计算或其他重负载并发。
- 当前冻结网络参数为 LAN 1 Gbit/s、RTT 0.5 ms；WAN 400 Mbit/s、RTT 4 ms。
- LAN/WAN 通信模型来自新实测双端发送字节及因果关键路径轮次：8×bytes/bandwidth＋RTT×rounds。模型与网络整形实测明确区分，不将 localhost 墙钟简单加进通信模型冒充总时延。
- 每阶段同时保留真实本地两方墙钟、字节、轮次、峰值内存及材料资格；性能报告分别提供激活和整步范围。
- 原生 SCI+SiRNN 预处理、Beaver＋分段 FSS＋原生 SCI 激活、旧 SCI 内部小表 DPF 以及新分段/不分段 folded LUT 均具名对照。
- 新完整整数网络的 6798 条、99 步质量只计明文评估；安全性能为单时间步。

全部工单按实际代码及证据关闭，阶段退化、质量失败或未分辨结论保持原始结果。

## 2026-10-13 续办记录（历史过程）

最终符号全输入验证用时 134.397 秒，3 个测试通过；原始日志位于 `ticket09/checks/native-final.log`。`contexts-final/qualification.json` 的两构造双上下文测试通过。

首次最终单步行的两方均正常完成，材料与张量原始日志及 914267 events/方的完整轨迹保留在 `ticket09/step-controller-oom/`。随后外部 Python 轨迹分析器的逐事件对象/依赖图分配耗尽主机内存，控制进程中断；该行没有被记为最终资格或正式性能。旧分析器在 1.5 GiB 限额下重现 `MemoryError`；紧凑 phase DAG 修复后在 768 MiB 限额下完成 1820622 events 重放，与历史 phase/scope 结果完全一致，18 个聚焦测试通过。证据见 `ticket09/checks/trace-memory-validation.json`。该修改只涉及外部证据处理，不改变两方协议或构建。

`ticket09/step-final/qualification.json` 于 16:07 写入最终八行通过结果；`ticket11/frozen-plan.json` 于 16:11 冻结共同构建、资格、十行命令、顺序及统计规则。正式测量进程继续执行，已完成行的原始轨迹逐组无损归档，失败、未完成及必要追加组均保留。

16:45 刷新 `ticket12/summary.json` 与 `REPORT.zh.md`：原生全域、显式上下文、最终符号调度、整数、输入前驻留、资源、协议与模型质量均通过，待正式性能证据。历史小表 DPF 对照在 query0 预处理时双方退出 2，原始日志报 `DPF resident materials exceed reserved budget`；其失败记录及不完整轨迹保留，不替换为小 batch 或历史性能。

16:48 核查时八行完成、一行失败、最后的并行分段符号行运行中。旧 DPF 对照失败已定位到 `native_dpf_lut.cpp` 的预留及预检硬编码每材料 256 字节，而当前 `DpfPublicLutMaterial::residentBytes()` 使用实际 336 字节对象开销：四槽共驻留 53899776 字节，超过宣告预留 48766976 字节。双方 VmPeak 为 825778176/772399104 字节，未达到地址空间上限。尚需修复预留估算并用独立版本证据重测；当前冻结二进制、失败行和其余样本保留。

交付生成器已修复内存汇总：优先正式分析中的 `conservative_process_resource_peaks`，兼容旧字段并取所有合格组的峰值；9 个轻量单元测试通过。该修复仅涉及 ticket12，未改变本轮冻结计量输入。

第二次续办核查：首次队列已经结束，正式报告和 analysis 已生成；09 的两对符号成本完整，阶段准入均为 `experimental_regression`，按规格完成调查票并关闭。总计关闭 10/12 张（01–10）。11 的两对 LUT 在线比较仍需追加组，旧 DPF 失败按事前停止规则阻断了整个批次的补测；本次修复后另建版本与批次，不覆盖旧证据。

DPF 元数据预留修复已完成：native 预留与 planned 计算共同调用 bridge 的实际材料布局 API，删除 256 字节硬编码。三项聚焦回归覆盖九种 sigmoid/tanh/combined 模式，先复现原始 prepare 拒绝再全部通过；邻近七项库存/overlay 测试通过。独立新桥接真实返回 336 字节，新运行时 binary 为 `364344c3efa8ba509e1e4f390a8a9a651e148b293a8af8f16dc04d0a1070cc7f`。

`ticket11/diagnostic-budget-fixed/qualification.json` 的默认及非零隐状态各两次 fresh query 全部通过，每查询 172928 个整数零差异；两方完整材料、零在线生成/加载/等待、联合 prefix、四 CPU 和 1.5 GiB 地址空间资格全真。DPF 每方驻留 53899776 字节，新预留 54910976 字节，最大 party VmPeak 828833792 字节。新构建正在完成四条 folded LUT 串行/并行选择性诊断，并准备具名历史资格复用记录和独立正式重测。

17:12 新构建的四条 folded LUT 诊断全部完成：两构造的 sequential-a1 与 parallel-a2 各两次 fresh query，每查询 15 张量、172928 个整数零差异，驻留/资源资格通过，最大 party VmPeak 932716544 字节。`ticket11/qualification-reuse.json` 精确记录三处预算修复、新诊断身份、七份历史资格及其二十条原执行身份；旧桥接源码采用 Git 基线重建、外部依赖仅有当前快照的局限明确保留，历史资格不称为新执行。

17:12 `ticket11/frozen-plan-budget-fixed.json` 冻结新共同构建、上述通过资格和全部十行测量命令；`ticket11/formal-budget-fixed/` 正式启动，每行十次 fresh query，所有样本全新执行。初始十行完成并全部合格后，按原冻结规则选择反序补测，每行最多三组；不混入旧批次样本。

17:52 新正式批次十行已结束：9 行 `qualified`，1 行 `oom`。失败行是完整 10-bit LUT、parallel-a2、BitDec；q0–q8 完成，q9 TanH slot3 预处理时 party2 `std::bad_alloc`，VmPeak 1606959104 字节，距离 1.5 GiB 上限仅剩 3653632 字节，新的分配请求失败。q0–q8 的 DPF 工作集、材料驻留、folded balance 与旧批次一致；新 party2 虚拟地址空间每查询约增加 100 MiB，旧批次约 50 MiB，现有证据只支持 allocator/虚拟地址高水位差异待归因，不能把它归因到三项 DPF 预算修复。`analysis.json` 保持 `incomplete/pending_extra_groups`，不生成最终推荐或关闭 11/12。

当时的 allocator 隔离诊断记录使用同一二进制、十次新查询及相同协议/资源参数，只固定 `MALLOC_ARENA_MAX=2`、`MALLOC_TRIM_THRESHOLD_=1`、`MALLOC_MMAP_THRESHOLD_=131072`；记录显示全部完成，两方 VmPeak 为 665567232/676253696 字节，后续 prefix 地址空间保持稳定。原结果曾位于 `/tmp/paper-folded-allocator-trim-full-a2-20261013/`，现已丢失，不能作为当前可核验的正式证据，亦不纳入统计。RAII/材料释放链完整；主线程准备、多个执行线程释放选择向量使 allocator arena 保留是当时的排障解释，但三变量同时改变，不能单独归因某变量。

新的共同环境同时剥离继承 `MALLOC_*` 与 `GLIBC_TUNABLES`，再固定三项 allocator 参数。最初未净化的 `formal-allocator-fixed/` 因环境契约不完整中止，保留初始行原始轨迹、不纳入统计；当时的 `frozen-plan-allocator-clean.json` 和 `formal-allocator-clean/` 随后启动十行及必要补测。该旧批次现已作废，详见以下审计和当前入口。allocator 参数只改变分配/资源/性能行为，不改变协议或整数映射。

## 持久构建与当前正式批次

续办审计发现旧 `/tmp/paper-folded-budget-fixed-{core,runtime}` 和 fixture 已消失。`formal-allocator-clean/` 的六次补测启动日志保留了缺失旧二进制错误；其 `manifest.json` 曾被改写为旧 rounds 持久运行时/fixture 路径，却保留不匹配的声明二进制 SHA，失败组状态也被重置为 `not_started`。这种清单身份伪造及状态改写不能恢复资格。原清单 `manifest.json.original`、失败组备份 `summary.json.with-failed-groups`、改写文件和全部启动日志均保留，审计索引为 `ticket11/historical-cohort-audit.json`，旧补测工具原文保存在 `ticket11/historical-attempt-tools/`。旧失败批次、allocator 隔离诊断和这一无效批次的样本全部排除当前正式统计。

共同构建已在项目内 `ticket11/final-build/core` 与 `runtime` 重新生成，使用 Release bridge 和 `project/our/tools/sci_conversion/build.py`，并保存实际构建命令、运行时 manifest 与来源 hashes。`final-build/fixture` 来自持久 `project/our/results/sci_sirnn_rounds_20261012/fixture`，模型、输入及 contract 与原冻结 fixture 相同。新 binary SHA 为 `f48eef822754de789ab73850e058255ee75ef3e611a4eba9d7e5afe286821918`，bridge SHA 为 `35c65b724bc948692c6eab5c88c7ecd4b9197eed398cfdbe8a7069947d139fb6`。

`final-build/build-input-reuse.json` 和桥输入对照记录证明：62 项运行时输入中仅重新构建的桥输出身份变化，146 项桥源码/依赖及规范化编译链接输入一致；generated、compiler、DIV_RESCALING 与 numeric_mode 相同。旧 `/tmp` 实体不称为仍存在；历史 runtime 实际编译命令未保留，不能宣称完整历史命令一致。原资格记录的 Git 基线重建及外部依赖快照局限继续保留。

新构建的 `final-build/qualification-budget/qualification.json` 验证历史小表 DPF 默认/非零状态，各两次 fresh query；`final-build/qualification-folded/qualification.json` 验证两种 LUT 的 BitDec 默认状态、sequential-a1/parallel-a2 共四行，各两次 fresh query。整数、驻留、资源全部通过。`requalify_rebuild.py` 重验旧 config proof、62/146 项来源、七份历史资格及二十条原执行身份，再生成通过的 `ticket11/qualification-reuse-final.json`，保留历史非零状态、parallel-a1 及 FSS 符号资格的具名复用范围。

持久构建后的第一份正式尝试已归档为 `ticket11/formal-final-invalid-analyzer-mismatch-20261013/`：首行完成后，冻结的 `analyze.py` 发生变化，下一行启动前身份验证停止。第二份尝试已归档为 `ticket11/formal-final-invalid-qualification-selection-20261013/`：冻结时误只选择旧 `ticket09/step-final` 资格，发现遗漏后主动停止。两份对应计划及原始命令、manifest、summary、日志和排除声明全部保留；两批样本均不进入当前统计。

唯一最终正式入口为 `ticket11/frozen-plan-final.json` 与 `ticket11/formal-final/`，计划准确冻结 `qualification-reuse-final.json`、`final-build/qualification-budget/qualification.json`、`final-build/qualification-folded/qualification.json` 三份资格及持久 runtime/bridge/fixture。全部十行在统一 allocator 环境下重新执行，再依事前规则选择反序新组；失败即停止、每行最多三组等规则沿用原计划。该批次已终局；结果见下文。

## 2026-10-14 终局

`formal-final/summary.json`、`ELIGIBILITY.json` 与 `analysis.json` 记录十行均合格、状态 `complete`；四条 LUT 构造行各三组，另六行各一组，共 18 组、180 次 fresh query，warmup=0，11 项具名比较完整、测量缺口为零。四次反序补测决策及全部八个追加组均保留。两项 LUT 构造比较达到每行三组上限后仍为 `unresolved`，故抽样终态为 `exhausted_budget`，不能将离线改善解释成在线加速。来源为 [工单 11 正式报告](../../project/our/results/paper_folded_activation_20261013/ticket11/formal-final/REPORT.md)及同目录的 `analysis.json`。

采用 Segmented DPF (5,5) LUT、BitDec 符号和 parallel-a2 时，整步离线/在线中位数为 11.731217/0.207982 秒；完整 Sigmoid/TanH 在线分别为 0.089768/0.089408 秒。相同在线编码与核的完整 10-bit DPF LUT 为 7.539369/0.210444 秒，完整 Sigmoid/TanH 为 0.090128/0.090120 秒。冻结匹配比较的分段/完整整步在线比为 0.9883，95% BCa [0.9124, 1.0299]，未分辨；完整 Sigmoid、TanH 与两个 LUT 在线 event window 的区间也均跨 1。整步离线分段/完整比为 1.5560 [1.4792, 1.6033]，说明完整 LUT 离线墙钟更低。两种构造整步在线均发送 5860322 字节、134 因果轮次，LAN/WAN 通信模型相同；分段离线发送 2938230430 字节，完整 LUT 为 1851664030 字节。

Segmented DPF 的并行整步在线相对三种 SCI 具名对照的描述性中位数比分别为 2.7834（原生预处理）、1.4369（Beaver＋分段 FSS＋SCI 原生激活）、1.1288（SCI 内部小表 DPF）；离线墙钟比分别为 0.1624、0.5384、0.6122。SCI 对照的数值激活算法不同，且冻结 SCI 比较的 BCa 候选为完整 10-bit LUT；这些 Segmented/SCI 数字仅是描述性比值，没有对应的统计准入。并行相对串行的整步在线改善伴随离线通信模型退化；两种符号候选整步在线也退化。完整结论与逐阶段成本见 [工单 12 中文报告](../../project/our/results/paper_folded_activation_20261013/ticket12/REPORT.zh.md)及 `summary.json`，推荐状态为 `no_qualified_new_default`。安全实测限定 batch=128 单时间步；6798 条、99 步是明文质量评估。

## 2026-10-14 ticket13 网络模型复测

已使用当前统计接口重新执行 `SCI_preprocessed`（SCI+SiRNN）与 `SCI_Beaver_segmented_PaperLUT`，唯一当前入口为 `ticket13-interface-projection-matched/`。两套均使用同一持久构建与 fixture、batch=128、steps=1、parallel-a2、10 个 fresh query、warmup=0，顺序执行；两方均退出 0，`result.status=complete`、`rounds.status=passed`。PaperLUT 的 Sigmoid/TanH 都采用完整 folded LUT，构造为 `segmented_boolean_tensor_b2a`（5+5 bits），符号后端为 BitDec。

generation budget 显式固定为 `1610612736 B`，reserve/prefetch 各为 `536870912 B`；每方四个 CPU、1.5 GiB 地址空间上限，控制器也限制为 1.5 GiB。继承的 `MALLOC_*` 和 `GLIBC_TUNABLES` 经净化后，固定 `MALLOC_ARENA_MAX=2`、`MALLOC_TRIM_THRESHOLD_=1`、`MALLOC_MMAP_THRESHOLD_=131072`。完整命令与环境见 [OUTER_COMMANDS.json](../../project/our/results/paper_folded_activation_20261014/ticket13-interface-projection-matched/OUTER_COMMANDS.json)，两套实际双端环境另有 live `/proc` 快照。

计算规则为 `传输时延 = 8 × 双方 sent bytes 之和 / 带宽(bit/s)`、`传播时延 = 因果轮次 × RTT`、`通信时延 = 传输时延 + 传播时延`。LAN 使用 1 Gbps、0.5 ms RTT，WAN 使用 400 Mbps、4 ms RTT。轮次取保留 fork/join 依赖的因果关键路径，10 个 query 逐个计算后取中位数。localhost 墙钟独立保留；LAN/WAN 是按用户公式计算的通信模型，不包含本地计算，也不是 relay 整形实测总时延。

| 方法 | 阶段 | bytes | 因果轮次 | 本地墙钟中位数(s) | LAN 通信模型(s) | WAN 通信模型(s) |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| SCI+SiRNN (`SCI_preprocessed`) | offline | 185,571,244 | 70 | 1.680032026 | 1.519569952 | 3.991424880 |
| SCI+SiRNN (`SCI_preprocessed`) | online | 91,429,766 | 185 | 0.483954080 | 0.823938128 | 2.568595320 |
| `SCI_Beaver_segmented_PaperLUT` | offline | 2,938,230,430 | 955 | 11.606024504 | 23.983343440 | 62.584608600 |
| `SCI_Beaver_segmented_PaperLUT` | online | 5,860,322 | 134 | 0.202816914 | 0.113882576 | 0.653206440 |

在线通信模型中 SCI+SiRNN / PaperLUT 为 LAN `7.234980x`、WAN `3.932287x`，对应下降 86.18% 和 74.57%；PaperLUT 离线通信模型为对照的 LAN `15.782981x`、WAN `15.679766x`。offline+online 每 query 求和后的中位数为 SCI+SiRNN `2.343508080/6.560020200 s`（LAN/WAN）、PaperLUT `24.097226016/63.237815040 s`，候选总成本分别为对照的 `10.282544x/9.639881x`。此合计仅含 offline 与 online 两阶段。完整传输/传播分量、原始结果与逐 query 明细见[复测报告](../../project/our/results/paper_folded_activation_20261014/ticket13-interface-projection-matched/REPORT.zh.md)及 [NETWORK_MODEL.json](../../project/our/results/paper_folded_activation_20261014/ticket13-interface-projection-matched/NETWORK_MODEL.json)。

### 接口统计核对

计算逻辑已写入 [run.py](../../project/our/tools/segmented_dcf_runtime/run.py)，本次两套均使用 `--round-trace --network localhost --network-projection both`，新 `result.json` 直接包含：

- `measurements.queries[*].offline/online.logical_rounds`、`party_sent_bytes`、`party_received_bytes`。
- `measurements.queries[*].offline/online.network_projection.LAN/WAN` 中的 `transmission_seconds`、`propagation_seconds`、`communication_seconds`。
- `measurements.network_projection.networks.LAN/WAN` 中的公式、逐 query 明细、阶段中位数及 `offline_plus_online_seconds_median`。

独立汇总对照冻结启动命令、预算、allocator 实际环境、共同配置及全部 20 个 query 的 trace/接口字段；三项时间、阶段中位数和逐 query 合计均通过。接口与通道配置测试 18 个：17 通过，1 个需指定编译二进制的环境测试跳过。使用说明已更新到[运行器 README](../../project/our/tools/segmented_dcf_runtime/README.md)。

旧 `ticket13-network-retest/` 保留接口回填前的历史报告；`ticket13-interface-projection-final-v3/` 两套 budget 均为 0，只用于接口验证；`ticket13-paperlut-budget-1610612736-v1/` 未冻结 allocator，仅为预算诊断。这些样本均未混入当前 matched 统计。
