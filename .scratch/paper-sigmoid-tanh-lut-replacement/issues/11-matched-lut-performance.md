# 11: 分段与不分段的匹配性能实测

**What to build:** 用新鲜两方执行证据回答“1024 项表采用不分段 10-bit DPF 是否在线更快”，同时呈现其离线代价、内存与相对 SCI 的完整方法成本。

**Blocked by:** 08: 四参与者图并行与激活分片。

**Status:** closed

**Labels:** completed

- [x] 测量前冻结可执行构建、配置、样本顺序、统计方法和既有 LAN/WAN 参数。正式诊断确认两种 LUT 均正确、资源合格且输入前材料就绪，正式计时关闭中间值揭示。
- [x] 必须完成 (5,5) 分段和完整 10-bit 不分段各自的图串行/单 worker、图并行/双 worker 四行；每行相同 batch=128、输入、表、BitDec 符号后端、32 位 packed 布局、在线核和索引编码。
- [x] 至少增加一条双分支/每激活单 worker 的匹配归因行，解释双分支与元素分片的各自收益。重测原生预处理 SCI、Beaver＋分段 FSS＋原生 SCI 激活、历史 SCI 内部小表 DPF 三种具名对照。
- [x] 对照使用相同四参与者、1.5 GiB 地址空间以及适用的公共调度/模型预计算权限；不通过更换符号后端、编码、编译选项或内积布局使 LUT 构造对照混杂。
- [x] 每行十个 fresh query、warmup=0、600 秒预算；不同构造的行按冻结顺序交错安排，执行仍串行并避开构建重负载。保留 OOM、timeout 和未完成时延，不以小 batch 替换。
- [x] 分别报告 setup、模型/公共表准备、每查询离线、在线、总成本与首请求；拆出 keygen、全域物化、AND/B2A 或 payload、接口转换、符号/BitXA/截断、LUT 开值及本地内积。
- [x] 记录实际通信字节、关键路径因果轮次、峰值 RSS/AS 及同时计算参与者。相同编码下两构造在线通信理论相同；十位紧密位计数与当前两字节编码分清，未实现的打包不算收益。
- [x] 以分段耗时/不分段耗时及既有置信区间分别判断 LUT、完整 Sigmoid/TanH 与整步在线快慢，结论可为任一方更快或未分辨；仅离线改善不能报告为在线加速。
- [x] 明确两方案同时改变树分段和算术 payload/B2A 构造，离线差异称为 LUT 构造比较；相同在线核若有稳定差异，保留缓存、缓冲或调度的定位证据。
- [x] 交付原始逐查询结果、模型估算、复现配置与中文成本报告，并给出各具名父子配置的阶段资格。比较 SCI 时说明数值算法已变化；正式结论不依赖历史小表或旧激活结果代替新样本。

**Scope boundary:** 固定 BitDec 完成本票，不等待 09 的符号候选或 10 的明文质量。最终部署推荐在 12 汇总所有门槛后决定。

## 2026-10-13 续办状态

16:11（北京时间）冻结共同构建、两份通过资格、十行命令与顺序、LAN/WAN 参数和 BCa/补测规则，正式队列已启动。16:45 核查时前七行已完成；历史 `SCI_Beaver_segmented_DPF_LUT` 行在 query0 预处理时因 `DPF resident materials exceed reserved budget` 双方退出 2，失败日志及不完整轨迹保留；串行分段符号候选正在运行，并行候选待启动。每行十次 fresh query。实时行/组状态见 `project/our/results/paper_folded_activation_20261013/ticket11/formal/summary.json`，完成状态不代替后续原始证据资格与匹配比较。完整复现见 [计量说明](../../../project/our/results/paper_folded_activation_20261013/ticket11/README.md)。

首次队列已结束并生成报告：九行 `qualified`，旧 DPF 行失败。符号两对成本结论已足以关闭 09；两对 LUT online 仍未分辨，初始失败按冻结停止规则阻断了必要补测。本次通过窄桥接 API 查询实际材料元数据，修正 native 预留/预检的 256 字节过时假定（当前为 336）。独立新构建 `/tmp/paper-folded-budget-fixed-{core,runtime}` 的 DPF 默认/非零 batch128 两次新查询均通过：每查询 172928 个整数零差异，真实 DPF 驻留 53899776 字节低于新预留 54910976 字节，最大 party VmPeak 828833792 字节。证据见 `ticket11/diagnostic-budget-fixed/qualification.json`。新同版本十行与必要补测另建输出目录，旧批次保留。

17:12 新构建四条 folded LUT 串行/并行诊断全部通过；具名历史资格复用记录 `ticket11/qualification-reuse.json` 通过，保留原执行身份及来源局限。新计划 `ticket11/frozen-plan-budget-fixed.json` 已冻结，新鲜十行正式队列在 `ticket11/formal-budget-fixed/summary.json` 更新进度；全部初始行合格后继续按冻结规则执行必要反序补测。

17:52 新同版本十行已完成 9 行 `qualified`，1 行 `oom`。失败的 `parallel-a2-full-bitdec` 在 query9 的 TanH slot3 预处理阶段抛出 `std::bad_alloc`；party2 VmPeak 为 1606959104 字节，距 1610612736 字节上限只剩 3653632 字节，party1 为 971350016 字节。q0–q8 的材料、工作集和生成/消费平衡与旧行一致；新 party2 每查询约增加 100 MiB 虚拟地址高水位，旧行约 50 MiB。该行不具备完整十查询资格，补测未启动，`analysis.json` 保持 `incomplete/pending_extra_groups`。

当时记录的同二进制隔离十查询在固定三项 glibc allocator 参数后通过，两方 VmPeak 665567232/676253696 字节；原 `/tmp` 结果现已丢失，不能作为当前可核验的正式证据，也不计入统计。此后曾启动 `formal-allocator-clean/`；该旧批次已因缺失二进制补测及清单/组状态改写而判为无效，最终验收入口见下文。

旧 `/tmp` 构建消失后，`formal-allocator-clean/` 的六次补测启动均保留缺失旧 binary 的错误。其 manifest 曾改写为其他持久 runtime/fixture 路径但声明 SHA 不符，失败组状态被改成 `not_started`；原 manifest、失败组备份、改写清单和日志均保留。审计见 `ticket11/historical-cohort-audit.json`，历史工具原文见 `historical-attempt-tools/`。这种清单身份伪造与状态改写不能补齐正式证据，不进入最终统计。

当前持久构建在 `ticket11/final-build/{core,runtime,fixture}`。新 DPF 默认/非零两行、新 folded BitDec 两种 LUT × 两种调度四行诊断通过，分别为 `final-build/qualification-budget/qualification.json` 与 `final-build/qualification-folded/qualification.json`。`qualification-reuse-final.json` 重验来源和历史原执行身份，绑定新构建，保留原资格来源局限及历史 runtime 编译命令缺口；旧 `/tmp` 实体未伪称仍存在。

新构建后的两次中断尝试完整归档：`formal-final-invalid-analyzer-mismatch-20261013/` 首行完成后因冻结的 `analyze.py` 改变而验证停止；`formal-final-invalid-qualification-selection-20261013/` 因误只引用旧 step-final 资格而主动停止。两个批次及对应计划的原证据完整保留且显式排除，资格遗漏已纠正。

唯一最终计划为 `ticket11/frozen-plan-final.json`，准确冻结上述三份资格；输出为 `ticket11/formal-final/`。完整十行及必要反序补测已在统一 allocator 环境下串行执行，组状态见 `formal-final/summary.json`。历史失败与中断批次保留但不进入当前统计。

## 2026-10-14 终局与 Resolution

最终 `formal-final/analysis.json` 为 `complete`，十行全部 `qualified`，11 项具名比较完整，测量缺口为零。四条 LUT 构造行各三组、另六行各一组，共 18 组、180 次 fresh query；四次反序补测决策保留在 `summary.json`。两对 LUT 构造比较均因每行三组上限结束，`sampling_status=exhausted_budget`、`additional_groups_status=blocked_by_group_budget`，在线判定保持 `unresolved`。最终样本均来自持久共同构建；原始逐查询数据、日志与轨迹见 `formal-final/`，未混入旧批次。

分段/完整 10-bit LUT 的整步在线比在串行为 0.9931 [0.9635, 1.0430]，并行为 0.9883 [0.9124, 1.0299]，均未分辨；完整 Sigmoid/TanH 及两个 LUT 在线联合窗口的匹配区间也均跨 1。整步离线比分别为 1.5151 [1.4513, 1.5558] 与 1.5560 [1.4792, 1.6033]，完整 LUT 离线墙钟更低。并行分段 DPF 的整步在线中位数为 0.207982 秒，完整 Sigmoid/TanH 分别为 0.089768/0.089408 秒；整步离线为 11.731217 秒。两构造在线通信同为 5860322 字节、134 因果轮次，LAN/WAN 模型相同；分段离线字节为 2938230430，完整 LUT 为 1851664030。并行配置相对串行的整步在线改善伴随离线通信模型退化；相对 SCI 的在线收益不能抵消离线退化，具名准入以 [正式逐阶段报告](../../../project/our/results/paper_folded_activation_20261013/ticket11/formal-final/REPORT.md) 与 `analysis.json` 为准。

本票交付原始组数据、资格、资源、阶段成本和复现入口；[工单 12 中文报告](../../../project/our/results/paper_folded_activation_20261013/ticket12/REPORT.zh.md)负责最终配置推荐，当前为 `no_qualified_new_default`。SCI 对照的数值激活算法不同；独立 Sigmoid/TanH 离线墙钟不可用，不从阶段 work 伪造。
