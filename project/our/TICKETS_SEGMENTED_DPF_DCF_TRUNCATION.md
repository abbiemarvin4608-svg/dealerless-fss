# 分段 DPF-based DCF 截断实施工单

日期：2026-10-09（Asia/Shanghai）  
状态：已检查并写入，实施与验收待完成

执行调整（2026-10-09，用户明确要求）：本轮跳过 SDCF-09 的完整评估集整图正确性与质量验证，继续实施其他工单并报告 Batch=128 相对 SCI/SiRNN 的在线/离线实测。SDCF-11 在本轮交付待质量准入的性能对比，SDCF-12 更新实现与这类受限证据；两者不得将跳过 09 写成通过质量门槛。算子及单时间步正确性检查照常执行。集成分支为 `integration/sdcf-batch128`，本文件作为本轮本地工单记录。

补充复测（2026-10-10，用户要求）：已实现常驻 cGGM 线程池及按批次分配连续任务，完成 nseg=2/4、线程上限1/4与同环境 SCI/SiRNN、Beaver 控制组的 Batch=128 原语和单时间步测量。[常驻池复测报告](results/segmented_dpf_dcf_truncation/10/persistent_pool_recheck_20261010/REPORT_CN.md)保留首轮调度阈值依据及最终复测；两段单步在统一 1.5 GiB 预算下 OOM，未填入完成时延。四段完成配置独立诊断通过。本项不改变 SDCF-09/11/12 的整图质量准入状态。

后续运行时优化（2026-10-10）：[SCI/SiRNN + 分段 FSS + Beaver 运行时优化方案](../../.scratch/sci-fss-runtime-optimization/spec.md)已在 `integration/sci-fss-runtime-optimization` 分支完成本轮实现与验收，覆盖通信合批、材料布局、按域分段、计算与内存优化，以及激活/图调度并行；Batch=128 原语和单时间步的四路对比见[运行时优化报告](results/runtime_optimization_20261010/REPORT_CN.md)。本轮沿用现有原语和两方单时间步验收接口，不改变本批历史 SDCF-09/11/12 的整图质量准入状态，SDCF-09 继续延期。

本批落实[分段截断专项需求](REQUIREMENTS_SEGMENTED_DPF_DCF_TRUNCATION.md)及[多位宽推理需求](REQUIREMENTS_MULTIWIDTH_DEALERLESS_INFERENCE.md)，保留 12 条工单的目标和顺序。编号使用 `SDCF-01` 至 `SDCF-12`，与历史 `results/ticket01_*` 至 `results/ticket12_*` 独立；历史报告中的“完成”不自动构成本批验收。

## 检查结论

- 原依赖图无环。SDCF-08 增加 SDCF-04 前置，确保接入 segmented truncation 前通过正确性回归。
- SDCF-09 补入 full-tree 后端的整图验证，使 SDCF-11 的四路主表都有相同的正确性与质量准入证据。
- SDCF-01 按 SCI 实际调用点冻结舍入语义，不能把整个 SCI 图概括为一种右移规则。SDCF-02/10 明确比较域宽度与张量环宽的区别。
- SDCF-08 按 ScalarMul 操作数的保密属性分流。原生 Google-30 的公开 `zeta/nu` 保持公开常量本地乘法；私密标量使用 Beaver，单列验证，不能把公开常量改成私密权重来制造 OT 对照。
- SDCF-05/11 固定阶段计量、一次性材料和摊销规则；无法测得的字段不填零。SDCF-09/12 区分本批新结果与历史结果。

## 共同验收约定

- `ready` 表示没有本批前置阻塞；`blocked` 表示前置工单尚未按本批标准验收，不代表没有可复用代码。只有对应交付、测试和证据齐全才改为 `done`。
- `comparison_backend` 与 `numeric_mode` 分开配置：前者区分 full-tree/segmented，后者区分 `sci_compatible`/`exact_floor_diagnostic`。四路主表全部使用 `sci_compatible`；floor 诊断独立记录。
- 同图轨固定模型、量化产物、样本顺序、mixed-width 图、权重/输入持有方式、输出揭示规则和两方半诚实无经销安全设置。比较重构后的整数输出，不比较随机份额本身。
- 中间值重构仅用于独立正确性测试。正式安全计量关闭诊断揭示，只记录协议允许打开的值的类型和数量，不记录原始秘密值。
- 新证据计划写入 `results/segmented_dpf_dcf_truncation/01/` 至 `12/`，本次仅登记路径。每份验收报告关联源码版本或源码快照、配置、命令、原始两方日志、结果文件和退出状态；不得覆盖历史 `ticketXX` 证据。
- 工单状态与运行状态分别记录。运行失败、`timeout`、`oom`、`unsupported`、`unavailable`、`quality_failed` 保留原因和资源预算；缺失值使用 `null`。失败运行没有完成时延，也不参与达标速度排名。
- 本批验收不要求获得加速结论。资源不可用或质量失败的配置可以作为完整记录的一部分，但不能据此宣称该配置已完成正确性或安全性能验证。

## SDCF-01 冻结 SCI 兼容截断与扩环契约

**Blocked by:** 无  
**状态:** `done`；证据：[01 验收记录](results/segmented_dpf_dcf_truncation/01/summary.json)，227,518 个真实 SCI 交叉检查输出通过；原生 64/63 边界单列不支持。

**交付:** 固定 `sci_compatible` 的负数截断、环宽、符号扩环、wrap correction、输出环和 `shift` 语义，生成可调用的参考 oracle。逐调用点记录 SCI 版本、构建配置（包括 `DIV_RESCALING`）、对应函数、输入/输出符号与尺度、乘积/累加器宽度、截断顺序和舍入方式；保留数学 floor/缩环的 `exact_floor_diagnostic` 作为独立诊断。

**验收:**

- 覆盖正负数、非整除负数、零、最小/最大值、wrap 和图中各类环宽；小域穷举，大域使用边界及固定随机向量。文档与 oracle 对 SCI 参考输出逐值一致。
- 明确 `shift=0` 的恒等/转换分支、`0 < shift < input_bits` 的有效截断域及非法参数行为；图中负 shift 归入左移/尺度转换分支，不为零或负位比较域构造 DCF。
- 按调用链验证规则组合，例如累加阶段的 floor 缩环与后续 `div_pow2` 向零截断；不能用宿主语言 `/` 或 `>>` 的默认行为代替 SCI 语义。
- oracle 可由后续测试直接调用，并保存与冻结 SCI 参考交叉验证的向量和报告；不是只与自身公式比较。

## SDCF-02 实现分段 DPF-based DCF 比较原语

**Blocked by:** SDCF-01  
**状态:** `done`；证据：[02 验收记录](results/segmented_dpf_dcf_truncation/02/README.md)，小域穷举、真实两方完整树对照和 12,800 元素批次通过。

**交付:** 在截断的 `shift` 位比较域上按 `nseg` 生成浅层相关 DPF/cGGM 树，在线完成分段评估和跨段 DCF 合并；约束 `1 <= nseg <= shift`。比较定义为无符号域内的 `1{z < alpha}`，有符号数值解释由截断适配器负责。优先复用已有 segment layout、cGGM 和 Boolean scan 接口。

**验收:**

- `nseg=1` 与完整树 DCF 的重构输出等价；对 1 至 8 位域穷举 `alpha`、查询值和所有合法 `nseg`，均与严格 `<` 参考一致。
- 支持段宽不均，固定连续高位优先布局，各段宽度之和为 `shift`，并记录每段宽度；不增加 `shift % nseg == 0` 的限制。非法段数报错，不静默裁剪。覆盖相等、零、最大值和跨段边界。
- 段内局部比较/相等关系及跨段合并共同恢复完整比较；不能把单个 DPF 点函数输出直接当作 DCF 的区间比较。
- Boolean triples 等合并材料在 offline 生成；日志能确认每段树深、实际段数和跨段依赖层。

## SDCF-03 接通分段截断单元素与批量接口

**Blocked by:** SDCF-01、SDCF-02  
**状态:** `done`；证据：[03 验收记录](results/segmented_dpf_dcf_truncation/03/README.md)，1,416 配置、64,136 个重构值及 12,800 元素批次通过。

**交付:** 将分段 DCF 接入截断的掩码、借位、算术 correction、符号扩环和输出转换，提供 offline、single/batch online、释放和一次性材料消费检查。材料显式保存数值模式、输入/输出环宽、`shift`、段布局、掩码和消费状态。

**验收:**

- single 与 batch 的重构结果一致，均匹配 SDCF-01 oracle；输出环由契约决定，不默认固定为 `input_bits - shift`。
- 相同契约、同一依赖层的输入批量打开；不同位宽、shift 或模式显式分组，记录各组大小。
- 正式执行不额外打开原始秘密输入；重复消费、同一 batch 重复引用材料、释放后使用和输入/材料契约不匹配均失败。失败后已参与协议的材料不能重新进入可用池。

## SDCF-04 完成截断与扩环正确性回归套件

**Blocked by:** SDCF-03  
**状态:** `done`；证据：[04 统一报告](results/segmented_dpf_dcf_truncation/04/run/report.json)，五套件通过，原生 SCI 64/63 单列 `unsupported`，首错为空。

**交付:** 为 full-tree DPF-based DCF、segmented DPF-based DCF 和 SCI-compatible adapter 建立统一测试入口，分别运行比较原语与完整截断/扩环的契约测试。

**验收:**

- 覆盖段边界、符号位、wrap、shift 边界、负数余数、输入/输出环转换及 single/batch；SCI 模式逐值匹配 SCI oracle。
- floor 模式使用独立数学参考，并包含与 SCI 向零规则结果不同的非整除负数用例，验证两种模式不会混用。
- 报告测试域、向量数、各后端覆盖率和第一处不一致；资源限制或未支持宽度明确标记，不能算通过。

## SDCF-05 建立 Batch=128 依赖层及阶段计量框架

**Blocked by:** SDCF-01  
**状态:** `done`；证据：[05 验收记录](results/segmented_dpf_dcf_truncation/05/REPORT.md)，Batch=128 单时间步、非零隐状态诊断和网络转发检查通过；原生 SCI 逻辑轮数尚不可分离，保留 `null`。

**交付:** 将 MatMul、MulCir、ScalarMul、扩环、截断和转换按依赖层批量执行，分离 setup、offline、online、total、分阶段通信、逻辑轮数、I/O 调用数和峰值内存。固定四路后端共用的调度与计量接口。

**验收:**

- `Batch=128` 表示 128 条独立序列；99 个递归时间步保留真实依赖。四路使用相同层边界和批处理分组，不能仅批量激活、其余算子逐样本循环。
- 在线计时在输入确认且所需 offline 材料就绪后开始。Beaver triples、分段密钥、B2A、Boolean triples 和掩码的生成独立计量，不能放进 fused online。
- SCI 原生在线 OT 的实际耗时保留在在线阶段，不能事后估算扣减。无法独立分出的子项标 `null/not_separable`；只有确认该阶段未发生工作时才记零。
- 记录 base OT/初始化、输入准备/分享、输出重构和 teardown 的边界；total 直接测量，同时保留阶段明细，避免重叠分项或双方阶段最大值重复相加。
- 通信主口径为双方发送字节之和，另存各方发送/接收；延迟保留双方值及汇总规则，峰值内存记录每方 RSS 和报告口径。逻辑依赖轮数独立于 socket I/O 调用数。
- 在正式测量前冻结机器、编译、线程、LAN/WAN 参数、预热/重复次数、统计方法、超时及内存预算。新鲜一次性材料按每次查询消耗，只摊销可复用 setup，不复用密钥、掩码或 triples。

## SDCF-06 完成 SCI + ours 单个 FastGRNN 时间步

**Blocked by:** SDCF-04、SDCF-05  
**状态:** `done`；证据：[06 验收记录](results/segmented_dpf_dcf_truncation/06/REPORT.md)，统一 SCI-compatible 转换桥接、C ABI 材料消费回归及 Batch=128 非零隐状态单步通过。

**交付:** 在原生 mixed-width 图中接通分段截断/扩环，保留 SCI 线性层和激活数值算法，完成一个真实 Batch=128 时间步。这里的 `SCI + ours` 仍使用 SCI 乘法，用于隔离转换协议的接入正确性。

**验收:**

- 独立测试运行逐算子重构整数输出与 SCI 一致，覆盖实际时间步输入及非零隐状态；差异定位到样本、算子、位宽、尺度和截断模式。
- 输出完整阶段计量和调用覆盖日志，列出实际替换的截断/扩环、比较域宽度、段数及仍走 SCI 的调用；仅替换 MSB 或配置 `nseg` 不算截断接通。
- 正式计量关闭中间值揭示，使用 SDCF-05 的批处理边界和阶段接口。

## SDCF-07 接入 MatMul Beaver 后端

**Blocked by:** SDCF-01、SDCF-05  
**状态:** `done`；证据：[07 验收记录](results/segmented_dpf_dcf_truncation/07/REPORT.md)，35,992 个矩阵输出及 Batch=128 非零隐状态单步的 172,928 个算子输出匹配 SCI；本项只替换 MatMul。

**交付:** 在相同权重持有方式和 mixed-width contract 下，为 MatMul 生成离线 Beaver 三元组，在线按依赖层批量打开 `E=X-A`、`F=Y-B`。本工单先保留 SCI 原生后续转换，以形成独立乘法控制组。

**验收:**

- MatMul 的乘积、累加及最终输出均与 SCI OT 逐值一致；覆盖有符号操作数、真实矩阵形状、累加长度、环回绕和舍入阶段。
- 先按 SCI 契约处理输入扩环、乘积环和累加器环；不能用同一窄环 Beaver 乘法默认替代 mixed-width MatMul。
- 三元组在两方无可信经销设置下真实生成，保持权重与输入的持有/分享规则。setup/offline/online、通信和内存独立记录，并记录批量 E/F 打开及每次查询的新材料消费。

## SDCF-08 接入 MulCir 与 ScalarMul Beaver 后端

**Blocked by:** SDCF-04、SDCF-07  
**状态:** `done`；证据：[08 验收记录](results/segmented_dpf_dcf_truncation/08/REPORT.md)，MulCir/私密 ScalarMul Beaver、公开 ScalarMul 本地路径及 Batch=128 单步回归通过。

**交付:** 为逐元素乘法和标量乘法接入同一乘法契约，显式区分公开和私密操作数；秘密乘法使用 Beaver，处理乘积宽度、扩环和后续 segmented DCF truncation。保留原生 SCI 转换分支，供 `SCI + Beaver` 对照使用。

**验收:**

- MulCir 和 ScalarMul 的重构输出与 SCI 逐值一致，覆盖广播、负数、乘积宽度、wrap 和后续 shift；完整树与分段树使用相同 SCI-compatible 转换契约。
- 原生 Google-30 的 `zeta/nu` 为公开常量，ScalarMul 保持本地乘法，其扩环/截断仍接入共同接口并计量。私密标量 Beaver 路径另测；不能声称原图公开 ScalarMul 存在被替换的 OT 成本。
- 乘法 triples 与激活已有 triples 分开计数，记录真实秘密乘法的批量 E/F 打开、批大小和消费数量；不能把激活预处理计为乘法预处理。
- 在进入 SDCF-09 前，补做接通 MatMul、MulCir、ScalarMul 和分段截断的 Batch=128 单时间步逐算子回归。

## SDCF-09 完成 Google-30 整图正确性与质量验证

**Blocked by:** SDCF-06、SDCF-08  
**状态:** `deferred_by_user`

**交付:** 完成 99 步 FastGRNN、分类头和 ArgMax 的四路整图验证：SCI/SiRNN OT、SCI + Beaver、SCI + Beaver + full-tree DPF-based DCF、SCI + Beaver + segmented DPF-based DCF。使用独立于模型/量化选择的评估产物计算 Top-1 和 prediction agreement。

**验收:**

- 四路使用同一冻结模型、量化图、预处理、标签映射和完整评估样本清单；安全整图输出逐样本匹配 SCI 整数参考，同图后端 prediction agreement 必须为 100%。同时记录已完成/总样本数和首个差异。
- 完整评估按 Batch=128 分批；尾批采用公开固定策略，padding 不计入 Top-1 或样本数，也不混入完整 Batch=128 的性能统计。
- 主表配置相对同一 FP32 checkpoint 满足 `loss_pp = FP32_Top1_pct - fixed_Top1_pct <= 1.0`，同时分别报告相对 FP32 和 SCI 基线的 agreement。最终评估集不能用于事后选择位宽、尺度或模型。
- SeeDot/FLOATEXP 历史预测不能代替 SCI 整数 oracle；独立明文质量产物可以复用，但不能代替新后端的完整安全推理覆盖。
- 超过质量门槛标记 `quality_failed`，整数不一致另记正确性失败；未完成安全全量覆盖或资源失败的后端保留不可用状态，不进入 SDCF-11 的达标排名。

## SDCF-10 完成比较位宽与分段数原语成本扫描

**Blocked by:** SDCF-04、SDCF-05  
**状态:** `done`；证据：[10 扫描记录](results/segmented_dpf_dcf_truncation/10/REPORT.md)，71 条预声明配置完成真实两方扫描，完成/OOM/失败状态均保留。

**交付:** 对 full-tree 与 segmented DPF-based DCF 扫描真实 16/20/24/32 位比较域及可行 `nseg`，补充原生图实际调用域，记录树深、段布局、跨段 scan、转换、分阶段通信和内存成本。

**验收:**

- 至少包含各比较域的 `nseg=1` 尝试，并预先记录候选段数与资源预算；可完成配置验证输出一致，再统计成本。
- 显式记录 `comparison_bits`。独立 DCF 扫描的域宽为比较输入宽度；截断调用中则为 `shift`，不是张量 `input_bits`。若测 32 位截断比较，需使用支持 `shift=32` 且输入环更宽的真实契约。
- 不沿用 legacy 截断 `shift <= 24` 的实现限制来冒充 32 位已测；超过实现上限记 `unsupported`，不能伪报 `timeout` 或 `oom`。未支持、超时或内存不足保留状态、预算和已产生的日志，不填零、不外推成完成值、不用模拟值替代。
- 分开报告段内树评估、跨段合并、B2A/correction 和扩环/输出转换；完整树与分段树的计量范围一致。

## SDCF-11 完成四路 Google-30 性能归因主表

**Blocked by:** SDCF-09、SDCF-10  
**状态:** `blocked`；证据：[11 受限报告](results/segmented_dpf_dcf_truncation/11/REPORT.md)。SDCF-09 延期使整图质量准入仍缺失；本轮仅报告 Batch=128 单时间步成本，full-tree OOM 保留为不可用行。

**交付:** 在同一 Batch=128、机器、线程、输入顺序及 LAN/WAN 条件下比较四路：SCI/SiRNN OT、SCI + Beaver、SCI + Beaver + full-tree DPF-based DCF、SCI + Beaver + segmented DPF-based DCF。

**验收:**

- 每个参与速度排名的配置都有 SDCF-09 的对应同图正确性与质量证据；full-tree 不能只凭 primitive 通过就进入整图主表。四路表保留失败/不可用行并注明原因，不用其他配置结果填补。
- 分别报告 setup/offline/online/total、分阶段通信、逻辑依赖轮数、I/O 调用数、每方峰值内存、重复次数和离散程度；未完成必要计量的配置不进入完整指标排名。
- 报告首查询总成本、会话总查询数和每查询摊销口径；一次性材料每查询新鲜生成并消耗，离线批量生成的成本按实际消耗分摊，不视为可重复使用。
- 以 OT 对 Beaver 归因秘密乘法预处理，以 Beaver + full-tree 对 Beaver + segmented 归因域分段，以 Beaver 对其截断替换版本分析转换成本；不跨不同数值图或批处理边界计算收益。
- 所有结论来自真实两方执行。收益、无收益及回退均如实记录，不预设分段 FSS 或 Beaver 加速；网络模拟和实机链路明确区分。

## SDCF-12 根据完成证据更新论文与实验记录

**Blocked by:** SDCF-09、SDCF-10、SDCF-11  
**状态:** `blocked`；证据：[12 证据边界记录](results/segmented_dpf_dcf_truncation/12/REPORT.md)。因 SDCF-09 延期，论文只能更新到单步与 primitive 证据边界，不能写入整图质量或端到端速度主张。

**交付:** 按完成证据更新方法、截断协议命名、实验表、主张边界和失败/不可用配置说明，建立论文数字到本批原始日志的对应关系。

**验收:**

- 不再把完整树实现称为分段截断；区分 DPF/cGGM 密钥材料、DCF 比较语义、SCI-compatible 适配器和 floor 诊断。
- 不把局部 primitive、结构模拟、明文质量或单时间步结果写成端到端安全加速；区分公开 ScalarMul 与秘密乘法的 Beaver 收益。
- 论文每个新数字可追溯至配置、命令和原始日志；注明全量安全覆盖、质量门槛、阶段边界、网络条件及摊销规则。历史数字保留原来源，不冒充本批复测。
- 质量失败、超时、OOM 和不可用配置保留说明；证据仅支持部分后端或条件时，主张限定到实际完成范围。修改后的 LaTeX、表格和实验记录一致并完成构建检查。

## 现有证据入口

以下文件供实现和检查复用，不表示本批工单已验收：

- [现有完整树截断实现](src/exact_truncation.cpp)：当前输出缩环、`shift <= 24` 和 single/batch 接口的基线。
- [历史算子契约生成器](tools/google30_replay/freeze_contract.py)：原生权重持有方式和公开学习标量的记录入口。
- [SCI 真实计量说明](tools/google30_benchmark/README.md)：现有阶段、两方通信和未分离字段的说明。
- [历史工单 01 至 03](results/TICKETS_01_03_REPORT_CN.md)及[历史工单 04 至 12](results/TICKETS_04_12_REPORT_CN.md)：旧编号与已有产物，不继承为本批完成状态。
- [历史 Batch=128 阶段报告](results/batch128_phase_google30/REPORT_CN.md)：已有局部比较/MSB 替换，不能代替本批完整截断及 Beaver 四路验证。
