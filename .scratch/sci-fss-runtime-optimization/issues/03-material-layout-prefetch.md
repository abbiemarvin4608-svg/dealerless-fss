# 03: 连续材料与有界预取

**What to build:** 让 DCF 和完整转换通过稳定的连续材料视图与可复用工作区完成查询，再在公开内存预算内提前装载本查询材料，减少小读取、复制、临时分配和在线等待。布局与预取分别计量，材料前置的成本完整保留在查询总成本中。

**Blocked by:** 01：补齐计量并冻结匹配基线。

**Status:** resolved

- [x] 先完成离线材料存储、桥接读取和在线 eval 共用的稳定连续批量视图，包含 correction words、元数据与明确区间所有权；现有公共调用语义保持一致，为 05 的离线分块及 06 的批量 AES 提供同一材料契约。
- [x] 在线按公开 shift/width 直接从 query 取位，或使用连续 bit 缓冲；消除逐段查询向量、逐把密钥的小读取和重复复制。scan scratch、层缓冲与输出工作区可复用容量，生命周期和并发所有权明确。
- [x] 内部材料格式若改变，离线写入与在线读取同步迁移并检测格式及契约不匹配；不静默误读旧材料。缓冲容量可复用，keys、mask、triples 按查询新鲜分配且恰好消费一次，异常或读取失败后不可重用已消费材料。
- [x] 在连续视图验收后实现可关闭的同查询有界预取与释放策略；预算包含已驻留材料、scratch、协议上下文及其他对象，记录预算和实际驻留，避免将全时间步材料直接常驻内存。跨查询重叠仍属后续可选范围。
- [x] 预取保持材料归属和消费顺序；提前终止、预算不足、读取失败或重复调用时，已装载和已消费材料状态清楚，现有顺序装载模式继续可用。新模式不得以超额驻留绕过每方地址空间限制。
- [x] 输入确认前装载计入 offline/material-ready，确认后发生的读取与等待计入 online；setup、首查询、直接 query/session total 均覆盖真实成本。保留与旧边界可对齐的装载计量，不因移动计时起点宣称总开销减少。
- [x] 经现有真实两方 bridge 生命周期、DCF 和完整转换接口验证大小批次、非整齐段宽、输出顺序、重复新鲜查询与异常后的不可重用；结果匹配独立比较 oracle 和 SCI-compatible 整数语义。默认及非零隐状态 Batch=128 单步的调用覆盖、材料消费和整数结果通过。
- [x] 分别保留原布局、仅连续布局、连续布局加预取的匹配对照，记录材料装载/反序列化、查询准备、eval、scan、释放、预取等待、驻留字节及首查询成本；布局收益与材料前置收益分开解释，未验证模式不成为默认配置。
- [x] 使用规格冻结机器、构建、模型、输入、每方四核及 1.5 GiB RLIMIT_AS，不整形 localhost，正式测量串行且关闭中间张量揭示。受影响 16 位 DCF count128/count12800 行分别使用 30/10 个新鲜查询，signed32、shift14、count128 完整转换 5 个；单步匹配行各 10 个，均无预热，不混入历史五查询样本。
- [x] 基本原语/转换每行预算 120 秒、单步每行 600 秒；保存双方对齐后的原始样本、实际参与者、通信轮数/字节、峰值 RSS、setup/offline/online/直接总时延，并汇总双方时延最大值的中位数、范围和离散程度。OOM/timeout 等失败保留真实状态及 null 完成时延，online 下降但总成本未改善也如实报告。

## Implementation Evidence

实现提交 `9f2fffb`、`5f5ff97`，并在 `fcac230` 合入 02 的共享 scan workspace 契约。保留 legacy/关闭预取默认；连续布局使用按偏移拥有的 metadata/CW 批量材料，直接读取公开 query 位。桥接使用版本与完整转换契约标记的 spool，以及 Context 拥有的可复用输入/scan 工作区。

`--material-layout contiguous --prefetch-bytes 67108864` 启用受限的本查询材料前缀装载，不创建异步 loader，也不跨查询预取。预算检查同时考虑队列容量、当前进程虚拟地址空间、RLIMIT_AS 及在线工作区预留；不足时继续顺序读取。日志明确将预装载计入 offline，剩余读取计入 online，分别保留 preload/load/resident/budget 指标。

已通过本地 mixed-depth cGGM 回归；真实两方 DCF 小域 6372 次比较及 count12800；完整转换 1417 配置、64137 整数值；融合/位打包组合 16 配置、100 值；两方 bridge 的 legacy、连续加预取、1 字节预算回退、新鲜查询、契约失败后不可重用。默认隐状态组合后端与 SCI 的 15 张量、172928 值诊断一致。

正式匹配计量与默认/非零隐状态诊断已完成，闭环证据与范围见下方 Answer；诊断计时不用于性能结论。

## Answer

已交付按偏移拥有的连续 metadata/correction-word 材料视图、公开 query 位直接读取、Context 所有的可复用输入/scan 工作区，以及格式版本和完整转换契约检查。实现提交为 `9f2fffb`、`5f5ff97`，共享 scan workspace 契约在 `fcac230` 合入。keys、mask、triples 仍按查询新鲜生成并一次消费，失败后不能重用已消费材料。

可关闭预取为**同查询有界材料前缀装载**；预算包含队列、驻留材料、在线工作区、进程地址空间和 RLIMIT_AS，不足时回退顺序读取。装载计入 offline，剩余读取与等待计入 online；没有异步 loader 或跨查询预取。[原语消融](../../../project/our/results/runtime_optimization_20261010/primitive_ablations-recheck/summary.json)、[11 行单步匹配消融](../../../project/our/results/runtime_optimization_20261010/conversion_material/summary.json)、[18 行默认/非零诊断](../../../project/our/results/runtime_optimization_20261010/conversion_material_diagnostics/summary.json)和[真实两方 bridge/context 检查](../../../project/our/results/runtime_optimization_20261010/context_bridge/byte_transport/summary.json)保存布局、预取、异常生命周期及整数语义证据。

组合后端的 legacy/contiguous/contiguous+prefetch 直接 query 中位数为 7.5666/6.9548/7.2142 s；预取降低 online 中位数 0.9666→0.9545 s，却增加匹配布局控制的总成本。SCI_ours 的对应直接总成本为 6.5971/6.2402/6.0218 s。保留两种结果，预取仍为可选配置，不将前置材料成本解释为消失。SDCF-09 继续延期。
