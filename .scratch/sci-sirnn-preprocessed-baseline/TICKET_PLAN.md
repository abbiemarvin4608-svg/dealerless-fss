# SCI+SiRNN 预处理基线工单拆分索引

**日期:** 2026-10-12（Asia/Shanghai）

**Status:** completed

**拆分确认:** 用户已于 2026-10-12 认可八张工单的粒度、阻塞依赖及本地发布。

**来源:** [已发布并确认测试接口的规格](spec.md)。

八张工单已按直接阻塞依赖完成并合入 `integration/sci-sirnn-preprocessed`，全部关闭；每票保留真实两方行为、材料消费及阶段计量验收。最终[中文报告](../../project/our/results/sci_sirnn_preprocessed_20261012/ticket08_fair/REPORT.zh.md)与[24 行资格证据](../../project/our/results/sci_sirnn_preprocessed_20261012/ticket08_fair/ELIGIBILITY.json)覆盖八行正式主表和十六行默认/非零隐状态诊断。

1. **[输入前材料生命周期与二选一 GOT 闭环](issues/01-prepared-got-lifecycle.md)**。Blocked by：无。交付：先整理兼容材料接口，再完成真实随机 OT 离线生成、在线选择修正、一次性消费和计量闭环，默认 SCI 保持可运行。
2. **[多选一 LUT 与批量 GOT 位乘法预处理](issues/02-prepared-lut-batched-got.md)**。Blocked by：01。交付：真实查表与 matmul_multiplexer 批量 GOT 位乘法使用已生成库存，覆盖多选一、长消息、混合位宽和尾部。
3. **[B2A 与乘法的算术 COT 预处理](issues/03-prepared-arithmetic-cot.md)**。Blocked by：01。交付：真实 B2A 及普通/批量 COT 乘法路径保留模环结果，扩展提前完成，真实相关量适配留在线。
4. **[比较材料与完整整数转换预处理](issues/04-prepared-comparison-conversion.md)**。Blocked by：02、03。交付：提前生成标准/相关 bit-triple，贯通比较、wrap、截断和扩环，保持完整 SCI 整数语义。
5. **[原生 Sigmoid/TanH 内部预处理闭环](issues/05-prepared-native-activations.md)**。Blocked by：04。交付：完整原生激活的内部 LUT、转换、乘法与比较使用预生成材料，整数输出不变。
6. **[共享预处理策略贯通四后端完整单步](issues/06-shared-preprocessing-step.md)**。Blocked by：05。交付：顺序两方 batch=128 单步支持 SCI_preprocessed 与两个 ours 后端的共享原生预处理，保留默认 SCI，核验完整材料计划和整数结果。
7. **[激活 worker 与两分支预处理库存隔离](issues/07-context-parallel-preprocessing.md)**。Blocked by：06。交付：四后端在既有顺序/两分支调度下正确分配并消费各上下文库存，完成 batch=128、默认/非零状态及资源核验。
8. **[Batch=128 新样本公平对照与中文报告](issues/08-batch128-fair-report.md)**。Blocked by：07。交付：两 cohort、四后端、每行十个无预热新鲜查询，完整报告离线、在线、总成本、通信和内存，并以预处理 SCI 为主要比较参照。

## 公平基准与阶段边界核查

2026-10-12 已按用户要求复核并收紧验收。相同基准指同一冻结模型、单时间步图、batch=128、输入/隐状态、整数语义、共享调度、资源、安全假设和查询输入前的预处理机会；各后端按实际协议生成材料，材料数量、内部调用量及离线时长分别记录。

| 后端 | 查询输入前实际生成的材料 | 输入后保留的工作 |
| --- | --- | --- |
| 默认 SCI | 无每查询预生成库存；可复用 Base OT/上下文计 setup | 原生 OT 扩展、材料生成与全部原生执行，作为默认实现控制组 |
| SCI_preprocessed | 完整原生调用所需随机 OT、标准/相关比较 bit-triple 等库存 | 原生乘法、转换和激活的输入相关选择/相关量修正、消息适配及打开 |
| SCI_ours | 图级 FSS 转换材料，加剩余原生调用的随机 OT/bit-triple 库存 | FSS 图级转换评价，加原生乘法、激活及其输入相关执行 |
| SCI_Beaver_segmented | 图级 Beaver 算术三元组、分段 FSS 转换材料，加剩余原生调用的随机 OT/bit-triple 库存 | Beaver 打开与乘法、分段 FSS 评价、原生激活及其输入相关执行 |

- 01、03–06 明确要求 prepare 实际完成库存生成，双方 material-ready 后才允许装载本次查询输入/隐状态。只配置缓存或保存尚未执行扩展的种子不能通过验收；比较 bit-triple 也必须已经生成，不能只预备生成它们所用的 OT。
- 01、06–08 要求相同计量边界：在线快照先于 enter_online 本身；输入后的选择修正、消息/相关量适配、比较打开、库存读取、等待与释放计在线。input/ready 不隐藏协议工作。online 结束于秘密份额计算及完成同步，最终公开结果重构统一计 output，直接 query total 包含输出成本。
- 严格库存检查覆盖 prepare 完成至查询输出结束，包括 input/ready 间隙；禁止继续 OT 扩展、生成一次性库存或补货。已完成材料的 PRG/哈希与密钥评价照常执行并计在线，不能借此延后完成随机 OT 或 bit-triple 的生成。
- 06 为两个 ours 提供相同的原生预处理策略，只准备其剩余原生调用；已被 Beaver/FSS 替换的调用不再生成原生库存，Beaver/FSS 自身离线用到的 OT 不重复计费。
- 08 重新实测全部四行，以 SCI_preprocessed 为主要参照，分别报告在线、离线和直接 query total。每查询一次性材料成本计离线，不藏入 setup；不从历史时间或理论通信量估算新结果。

上述实验契约已通过真实材料生成、输入后零生成、完整整数诊断和新样本性能复测验收；SCI_preprocessed 的新实测结果及双方预处理后的 ours 对比已发布。

原源码核查发现的三项差距均已由对应工单完成：

- [原生 SCI adapter](../../project/our/tools/segmented_dcf_runtime/backend_adapter.h) 与共享[后端 adapter](../../project/our/tools/sci_beaver/adapter.h) 已贯通真实材料生成与计数；01–05 完成 OT/bit-triple 预处理，06 注册完整基线和两种 ours 的共享策略。
- [两方 driver](../../project/our/tools/segmented_dcf_runtime/driver.cpp.in) 已在输入装载前执行 prepare、seal_prepare、同步和 material-ready；在线快照仍先于 enter_online。
- [计量 guard](../../project/our/tools/segmented_dcf_runtime/measurement.h) 已核验原生 OT/bit-triple 与 Beaver/FSS 全部上下文，从 prepare 完成至输出结束的生成增量；严格预处理行 input/ready/online/output 均为零。

## 阻塞依据与执行前沿

执行前沿已清空：01–08 全部关闭。以下保留原始阻塞关系及局部/完整验收边界，执行证据见 [EXECUTION.md](EXECUTION.md)。

- 01 是可立即开始的首票；兼容接口整理与第一条真实 GOT 闭环在同一票内先后完成，避免单独交付只有接口的横向重构。
- 01 完成后，02 和 03 可独立并行。03 的局部 COT 调用链不需要 02 的 KKOT/GOT 库存；其他协议可在诊断中保留原在线路径并明确标为局部覆盖。
- 04 的完整转换需要 02 的比较叶节点/bit-triple 生成 OT，以及 03 的 B2A/COT；两条边均是必要阻塞。
- 05 通过 04 的传递依赖已经获得 GOT 与 COT，不重复列 02、03。06、07、08 同样仅列直接前置。
- 01–05 的局部实现不得冒充完整 SCI_preprocessed；06 完成顺序完整单步，07 才取得两种正式共享调度的资格。
- 08 只汇集已验收的局部结果并复测最终集成，不承担所有前置工单的正确性补课。正式性能运行保持串行，开发与轻量审查可并行。

## 规格覆盖

| 规格目标 | 工单 | 用户故事 |
| --- | --- | --- |
| 真正输入前生成、setup/离线/在线计量、一次性材料与错误处理 | 01、08 | 4、5、6、13、14、15、17、22、25 |
| 多选一/批量 GOT、算术 COT 与安全 bit-triple | 02、03、04 | 7、8、9、24 |
| 原生激活内部覆盖 | 05 | 10 |
| 共享预处理、明确后端名称、完整单步数值契约 | 06 | 1、2、3、11、18、19 |
| worker/分支隔离、材料资源与输入后装载 | 06、07 | 12、16、17 |
| 新样本、公平对照、统计/首请求/会话、失败与证据边界 | 08 | 20、21、22、23、25、26、27、28 |

SDCF-09 继续延期；本批编号 01–08 仅属于预处理基线工单，不改变历史 SDCF 工单和上一批优化工单的状态。
