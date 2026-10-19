# 分段 DPF-based DCF 截断需求

状态：已确认需求，待实现与验证  
日期：2026-10-09（Asia/Shanghai）

本文是
[多位宽两方无经销隐私推理需求](REQUIREMENTS_MULTIWIDTH_DEALERLESS_INFERENCE.md)
中截断、扩环和乘法对照的专项需求。它将当前“相关 DPF 密钥 + DCF 评估”的截断实现升级为论文提出的分段 FSS 域实现，并固定 Q1、Q3、Q6 的验收口径。

实施拆分见[分段截断 12 条工单](TICKETS_SEGMENTED_DPF_DCF_TRUNCATION.md)，使用独立编号 `SDCF-01` 至 `SDCF-12`。工单已写入，尚未据此验收实现；历史同号工单的完成状态不继承。本批补齐 full-tree 整图正确性准入，并按原生公开/私密操作数属性区分 ScalarMul 的本地乘法与 Beaver 路径。

## Problem Statement

当前 `exactTruncate` 使用相关 DPF/GGM 树生成比较密钥，再调用 DCF 评估器得到借位比特。它在函数语义上是 DPF-based DCF，但比较域仍是一棵完整深度的 GGM 树，没有使用论文提出的域分段来降低单棵树的深度。因此论文中关于“分段 FSS 缩短 GGM 树”的贡献尚未覆盖截断路径。

此外，截断不能只替换一个 MSB 或比较接口。它同时决定：

- 有符号输入的扩环和回绕修正；
- 低位比较产生的借位；
- 输出是否保留原环或缩到较小环；
- 负数截断采用向零、floor 或其他舍入规则；
- 乘法、截断和下一层输入之间的位宽契约。

如果这些语义发生变化，即使最终分类结果接近，也不能把性能差异归因于 FSS 后端。当前实现还需要把“分段 FSS 的协议收益”和“把乘法从在线 OT 移到离线 Beaver”的收益分开。

## Solution

实现一个保持整数语义不变的分段 DPF-based DCF 截断后端，并建立可归因的安全后端对照。

### 分段 DPF-based DCF

对于需要右移 `s` 位的有符号秘密值 `x`，离线阶段生成随机低位掩码、算术掩码份额、布尔比较掩码、分段相关 DPF/cGGM 密钥和跨段合并所需的 Boolean Beaver 材料。将 `s` 位比较域按论文的 `nseg` 配置切分为连续段，每段使用较浅的 DPF/cGGM 树；在线阶段对公开的掩码低位按段评估，并用前缀比较关系合并各段结果，得到与完整 DCF 相同的借位分享。

这里的命名必须区分两个层次：

- **DPF/GGM key material**：用于生成和保存每个分段树的相关密钥；
- **DCF evaluation**：输出 `1{x < alpha}` 形式的比较结果，供截断借位修正使用。

因此实现和论文统一称为 **segmented DPF-based DCF truncation**，不能简称为“DPF 截断”而隐藏其比较函数语义，也不能把当前单棵完整树实现误称为分段截断。

### SCI 兼容整数语义（Q1）

主实验必须保持 SCI 原生混合位宽图的精确整数语义。分段 FSS 只替换协议后端，不得改变数值契约。兼容模式至少固定：

- 输入和输出环宽、符号解释及扩环目标；
- SCI 的有符号 `div_pow2`/截断规则，包括负数和非整除值的行为；
- wrap/carry 修正、截断位置、缩放因子和输出重构规则；
- 乘法结果的乘积宽度、累加器宽度及后续截断位置。

当前 `exactTruncate` 的 floor 且缩环行为若与 SCI 兼容规则不同，必须通过显式语义适配器区分：

1. `sci_compatible`：用于主表和 SCI + ours，逐算子结果应与 SCI 参考整数结果一致；
2. `exact_floor_diagnostic`：保留数学 floor/缩环语义，作为独立诊断，不能混入主比较。

分段 DCF 只负责产生比较/借位分享；有符号扩环和算术 correction 仍由完整截断协议完成。不得把“替换 MSB 接口”当作完整扩环替换。

### Beaver 归因控制（Q3）

乘法对照必须使用与 SCI 原生 Google-30 图完全相同的混合位宽图、权重持有方式、输入/输出揭示规则和数值契约，并覆盖：

- `MatMul`；
- `MulCir`/逐元素乘法；
- `ScalarMul`。

至少保留以下三路：

1. `SCI/SiRNN OT`：原生 SCI 在线 OT 乘法与原生转换；
2. `SCI + Beaver`：相同 SCI 转换，乘法三元组离线生成，在线批量打开 `E=X-A` 与 `F=Y-B`；
3. `SCI + Beaver + segmented DPF-based DCF`：在第 2 路基础上替换截断/扩环中可替换的比较和借位协议。

三路必须分别报告 setup、offline、online、total、通信量、依赖轮数和峰值内存。离线 Beaver、DPF/cGGM 密钥、B2A、Boolean triples 和掩码生成不得计入 fused online；每个一次性材料按查询消耗，跨查询摊销时明确摊销次数。该设计不预设分段 FSS 或 Beaver 一定更快，收益必须通过对照归因。

正式性能比较先固定 `Batch=128`，并按真实依赖层批量执行 MatMul、MulCir、ScalarMul、位宽转换、扩环和截断。SCI/SiRNN OT、SCI + Beaver、full-tree DPF-based DCF 和 segmented DPF-based DCF 必须使用相同的批处理边界；不能让只有激活跨样本批量、其他算子逐样本循环的实现进入主排名。

### 质量门槛（Q6）

所有进入性能主表的配置必须满足相对于同一 FP32 checkpoint 的 Top-1 损失不超过 `1.0` percentage point，并同时报告 prediction agreement。训练/校准集可用于选尺度和位宽，最终评估集不得用于事后选点。无法达到门槛的配置保留在诊断结果中并标记 `quality_failed`，不得参与达标配置的速度排名。

## User Stories

1. 作为论文作者，我希望截断路径也使用论文提出的 FSS 域分段，从而让“缩短 GGM 树深度”的贡献覆盖实际推理主路径。
2. 作为协议实现者，我希望使用分段相关 DPF/cGGM 密钥构造 DCF 比较，从而在保持比较语义的同时控制单棵树深度。
3. 作为协议实现者，我希望明确区分 DPF 密钥材料和 DCF 评估函数，从而避免论文把比较协议误写成点函数协议。
4. 作为数值验证者，我希望分段 DCF 的借位结果与完整深度 DCF 在整个小域上完全一致，从而证明分段只改变协议实现。
5. 作为数值验证者，我希望 `nseg=1` 时分段实现退化为已验证的完整树语义，从而有清晰的回归基线。
6. 作为安全推理实现者，我希望分段截断保留 SCI 的混合位宽、符号扩环、回绕修正和缩放位置，从而可以在同一整数图上替换后端。
7. 作为安全推理实现者，我希望批量截断在同一位宽和 shift 下合并 masked opening、DCF 评估和借位修正，从而避免把逐元素循环误算成协议轮数。
8. 作为实验设计者，我希望分别测量原生 SCI 截断、当前完整树 DPF-based DCF 截断和分段 DPF-based DCF 截断，从而识别域分段本身的贡献。
9. 作为实验设计者，我希望比较 `SCI/SiRNN OT`、`SCI + Beaver` 和 `SCI + Beaver + segmented DPF-based DCF`，从而区分乘法预处理和转换协议的收益。
10. 作为实验设计者，我希望 Beaver 对照覆盖 MatMul、MulCir 和 ScalarMul，从而不把只优化矩阵乘法的结果泛化到整个时间步。
11. 作为基准维护者，我希望三路后端使用相同的权重持有方式、混合位宽图和输出规则，从而保证性能比较公平。
12. 作为性能分析者，我希望单独看到 setup、offline、online 和 total，避免把离线密钥和三元组生成错误地算成在线延迟。
13. 作为性能分析者，我希望记录比较域位宽、段数、段宽、DCF 扫描轮数、通信字节和峰值内存，从而解释分段后的成本变化。
14. 作为性能分析者，我希望在 LAN 和 WAN、固定 batch 和重复次数下测量，从而区分计算开销与 RTT 敏感性。
15. 作为精度实验者，我希望所有主表配置相对同一 FP32 checkpoint 的 Top-1 损失不超过 1 pp，从而避免用低质量配置换取速度。
16. 作为精度实验者，我希望同时查看 prediction agreement，从而发现 Top-1 恰好相同但逐样本决策已经改变的情况。
17. 作为论文作者，我希望把 floor/缩环模式和 SCI 兼容模式分开记录，从而避免负数截断差异污染协议结论。
18. 作为回归维护者，我希望分段截断覆盖零、正负边界、段边界、最大/最小环值和非整除负数，从而及时发现符号或借位错误。
19. 作为安全审计者，我希望正式执行只打开协议允许的掩码值和规定的中间比特，不打开原始私密输入，从而保持既有泄露边界。
20. 作为复现者，我希望每个结果行包含数值契约、段数、离线/在线边界、命令和状态，从而可以追溯到具体实现和配置。
21. 作为论文审稿人，我希望看到分段树深度、跨段合并成本和转换成本的完整分解，从而判断加速是否来自真实的 FSS 域分段。

## Implementation Decisions

### 协议接口

- 新增独立的 segmented DCF truncation material 类型，不复用无法表达段布局的单树 key 类型。
- material 至少记录输入位宽、输出位宽、shift、比较域位宽、段数、各段宽度、输入掩码份额、符号/借位修正份额、分段密钥和跨段 Boolean Beaver 游标。
- 提供 offline、single online、batch online、释放和 consumed-state 检查接口。重复使用一次性材料必须报错。
- 保留兼容适配层，独立选择协议后端 `legacy_full_tree`/`segmented_dcf` 与数值模式 `sci_compatible`/`exact_floor_diagnostic`；主实验目标配置使用 `segmented_dcf + sci_compatible`。

### 分段比较构造

- 比较域按论文的连续高位优先布局切分；段宽和段数由显式配置给出并写入结果日志。
- 每段生成 Boolean-leaf correlated DPF/cGGM key，在线只对公开掩码后的段值进行本地树评估。
- 跨段使用前缀关系 `(L_a,e_a) o (L_b,e_b) = (L_a xor (e_a and L_b), e_a and e_b)` 或等价实现合成全域比较结果。
- 跨段 AND 所需的 Boolean triples 在 offline 生成；在线批量处理同层依赖，报告逻辑依赖轮而非底层 socket 调用数。
- `nseg=1` 必须与完整树 DCF 的重构输出一致；任意合法 `nseg` 必须与完整 DCF 在相同 alpha 和公开查询点上输出相同比较比特。

### 有符号截断与扩环

- segmented DCF 输出的是借位/比较分享，不直接决定有符号输出的最终环解释。
- 需要独立实现 SCI 兼容的 sign extension、wrap correction、目标环转换和负数截断规则。
- 任何 floor、向零、随机舍入或饱和规则都作为数值契约字段记录；不得以“DPF/DCF 等价”掩盖规则差异。
- 若无法在同一接口同时支持不同规则，优先完成 `sci_compatible`，将现有 floor/缩环实现保留为诊断后端。

### Beaver 乘法接口

- MatMul、MulCir、ScalarMul 均使用同一 mixed-width contract 和原生权重持有模式；秘密乘法生成 Beaver 材料。原生公开 `zeta/nu` 的 ScalarMul 保持本地乘法，其扩环/截断仍计量；私密标量 Beaver 路径单列验证，不改变主图参数的保密属性。
- 首期端到端比较固定 `Batch=128`，以依赖层为单位批量生成和消耗材料；单元素或逐样本结果只作为 primitive/诊断结果。
- 对矩阵乘法优先使用合并的 `E/F` 打开；逐元素和标量路径也必须记录是否合并打开，不能把两次底层发送直接称为两轮协议。
- Beaver 后端的离线材料不能被激活协议的 triples 充当，也不能把只覆盖激活的预处理描述为整网乘法预处理。
- 乘法结果进入 segmented DCF truncation 前，必须验证乘积宽度、扩环和 shift 与 SCI 参考一致。

### 实验矩阵

主表至少包含：

| 后端 | 乘法 | 截断/扩环 | 用途 |
|---|---|---|---|
| SCI/SiRNN OT | 原生 OT | 原生 SCI | 原生基线 |
| SCI + Beaver | 离线 Beaver | 原生 SCI | 乘法预处理控制 |
| SCI + Beaver + full-tree DPF-based DCF | 离线 Beaver | 当前完整树 DPF-based DCF | 截断协议控制 |
| SCI + Beaver + segmented DPF-based DCF | 离线 Beaver | 分段 DPF-based DCF | 目标后端 |

在原生 mixed-width Google-30 图上进行主比较；统一存储/I-O 16 位、32 位及 strict same-ring 仅作为宽度敏感性和诊断控制。MP-SPDZ、ABY-64、dealer-assisted FSS 按主需求的威胁模型和可用状态单列，不与上述四路混排。

### 计量口径

每个结果至少记录：`setup_ms`、`offline_ms`、`online_ms`、`total_ms`、分阶段通信字节、逻辑依赖轮数、底层 I/O 调用数、峰值内存、查询数和摊销次数。分阶段至少包含：base OT/初始化、DPF/cGGM 密钥、B2A、Boolean triples、Beaver triples、masked input opening、分段树评估、跨段 scan、扩环、截断 correction 和输出转换。

在线输入 `x` 确认后才开始计 online；任何一次性 FSS 密钥、掩码、B2A 或 Beaver 生成都计 offline。若实现把材料生成和执行放在同一进程，必须通过显式阶段计时器拆分，不能用总进程时间替代 online。

## Testing Decisions

### 单元与性质测试

- 对 1--8 位比较域穷举 alpha、查询值和所有合法段数，验证分段 DCF 与完整 DCF 的重构比较位相同。
- 对 `nseg=1` 做回归测试，验证输出、借位和错误状态与 legacy full-tree 实现一致。
- 覆盖段宽不均、最高段/最低段边界、alpha 为 0/最大值、查询为 0/最大值和重复 key 消耗。
- 对有符号截断覆盖正数、负数、零、恰好整除、负余数、最小值、最大值、环回绕和 shift 边界。
- 分别验证 `sci_compatible` 与 `exact_floor_diagnostic`，不能用一个模式的期望值测试另一个模式。

### 协议集成测试

- 在单个 FastGRNN 时间步中逐算子比较 SCI 与 `SCI + Beaver + segmented DPF-based DCF` 的重构整数输出。
- 在完整 Google-30 固定评估集上比较定点预测向量；任何差异必须定位到算子、样本、位宽、scale 或舍入规则。
- 验证 MatMul、MulCir、ScalarMul 三种 Beaver 路径在同一 mixed-width contract 下的正确性和材料消耗。
- 记录安全执行的打开值类型和数量，确认没有额外打开原始私密输入。

### 精度与性能测试

- 使用独立于模型/量化选点的评估产物，报告 FP32 Top-1、定点 Top-1、loss_pp 和 prediction agreement。
- 只有 loss_pp <= 1.0 pp 的配置进入达标性能排名；0.5 pp 作为敏感性分析，不改变主门槛。
- 在相同硬件、线程、batch、LAN/WAN 和重复次数下比较四路后端，并分别报告 offline/online/total。
- 扫描真实比较位宽和 `nseg`，至少覆盖 16、20、24、32 位调用域及可行段数；记录超时、内存不足和未实现状态。
- 用单独的消融表拆分：乘法从 OT 改为 Beaver 的收益、完整树到分段树的收益、扩环/截断转换的收益。

## Out of Scope

- 不把当前单树 DPF-based DCF 实现直接称为论文的分段 FSS 结果。
- 不改变 Google-30 的模型权重、混合位宽图、激活近似或权重持有方式来制造协议收益。
- 不在本需求中重写 Sigmoid/TanH LUT、完整 CNN executor 或量化训练流程；这些属于独立工单。
- 不把独立 floor/缩环诊断、SCI 原生逐调用点截断规则和 strict same-ring 诊断混成一个 baseline；SCI 兼容图保留其内部既定的 floor 与向零截断顺序。
- 不把 Beaver 三元组离线生成误认为通信和密码计算都消失；离线成本、存储和生成通信必须报告。
- 不预设 segmented DPF-based DCF 必然降低总延迟；如果跨段 scan 或转换成本抵消树深度收益，结果应如实记录。
- 不将 Top-1 接近、少量样本一致或 structure-only 结果当作完整安全端到端正确性。
- 不把 ABY-64 或 dealer-assisted FSS 的结果放入同威胁模型主排名。

## Further Notes

当前实现中，`exactTruncateOffline` 使用 `keyGenCorrelatedDPFBit`，而 `exactTruncate` 使用 `evalCorrelatedDCFBit`。这说明已有代码的底层密钥材料和比较函数已经分层，但还没有把比较域替换为多棵浅分段树。新增实现应优先复用现有分段 cGGM、segment layout 和 Boolean scan 接口，避免重新设计第二套 FSS 树协议。

论文中的结果解释应遵循以下顺序：

1. 先证明 segmented DPF-based DCF 与 full-tree DCF 的整数输出等价；
2. 再证明 SCI-compatible 截断与 SCI 原生 mixed-width 图一致；
3. 再用 Beaver 控制组隔离乘法预处理收益；
4. 最后报告分段树深度、跨段合并和转换成本是否带来端到端收益。

无论最终结果是否优于 SCI/SiRNN OT，需求都视为完成；论文主张必须服从实际 Top-1、agreement、offline/online 和通信证据。
