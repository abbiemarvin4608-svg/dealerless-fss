# 多位宽两方无经销隐私推理工单

需求来源：[多位宽两方无经销隐私推理需求](../../project/our/REQUIREMENTS_MULTIWIDTH_DEALERLESS_INFERENCE.md)。

2026-10-08（Asia/Shanghai）：用户已确认以下 21 张工单的粒度和阻塞关系。每张工单为独立 Markdown；编号按依赖顺序排列，编号本身不表示必须串行执行。

## 执行规则

- **01–08、10–11 已完成**；本批执行范围为04–12。09按用户选择限时验证，正式主表暂列unavailable；12仅完成SCI/SCI + ours双后端行，三后端排名不可用；13–21本次未执行。
- 前一批见 [01–03 执行报告](../../project/our/results/TICKETS_01_03_REPORT_CN.md)；本批状态与比较见 [04–12 执行报告](../../project/our/results/TICKETS_04_12_REPORT_CN.md)。
- `ready-for-agent` 表示描述已准备好交付执行，不表示阻塞项已经完成，也不表示实现已完成。
- 同图比较要求重构整数和最终预测一致；系统比较采用同 checkpoint 的 Top-1 损失 ≤1 pp，并另报 agreement 和数值误差。
- 第一阶段目标后端是 `SCI + ours`。历史 Google-30 复现只作协议图锚点；正式质量排名使用独立评估产物。
- 工单中的未达标、超时和不支持配置需要如实记录；不得以得出有利加速结论作为验收条件。
- 每张工单在已有行为接口上验证完整路径。64 位扩展、整段 SIGMA 激活替换和完整独立 ours 后端不是首期必需项。

## 工单索引

| 编号 | 工单 | 阻塞项 | 独立交付 |
|---|---|---|---|
| 01 | [回放 Google-30 原生整数图并冻结数值契约](issues/01-google30-native-replay.md) | 无 | 历史整数图回放、逐算子数值契约、比较调用统计。 |
| 02 | [生成独立评估的 Google-30 质量产物](issues/02-google30-independent-quality.md) | 无 | 独立于最终评估集选点的模型与量化产物及质量报告。 |
| 03 | [修正当前论文的已完成范围](issues/03-paper-current-scope.md) | 无 | 依据已有证据修正文稿的位宽必要性和完成状态。 |
| 04 | [打通 SCI 真实推理的完整计量](issues/04-sci-measurement.md) | 01 | 真实 SCI 推理的 LAN/WAN 重复计量与可追溯日志。 |
| 05 | [在 SCI 中接通一次分段比较或 MSB](issues/05-sci-segmented-compare.md) | 01 | 一个使用真实私密输入的分段比较/MSB 调用闭环。 |
| 06 | [验证 SCI + ours 单个 FastGRNN 时间步](issues/06-hybrid-fastgrnn-step.md) | 05 | 保留 SCI 数值算法的完整 FastGRNN 时间步。 |
| 07 | [完成 SCI + ours 整网一致性](issues/07-hybrid-google30-full.md) | 02、06 | 99 步序列、分类和完整固定测试集的一致性报告。 |
| 08 | [接通 MP-SPDZ 单个 FastGRNN 时间步](issues/08-mpspdz-fastgrnn-step.md) | 01 | 真实 OT 预处理的 MP-SPDZ FastGRNN 时间步。 |
| 09 | [完成 MP-SPDZ Google-30 安全推理](issues/09-mpspdz-google30-full.md) | 02、08 | MP-SPDZ 完整序列、分类与正确性证据。 |
| 10 | [测量比较位宽与分段数成本曲线](issues/10-comparison-width-sweep.md) | 04、05 | 实际位宽及 16/20/24/32 位的分段成本曲线。 |
| 11 | [完成 Google-30 双后端性能对照](issues/11-google30-two-backends.md) | 04、07、10 | SCI 与 SCI + ours 的同图对照与达标配置系统比较。 |
| 12 | [补齐 Google-30 三后端主表](issues/12-google30-three-backends.md) | 09、11 | 补入 MP-SPDZ 的同口径三后端结果。 |
| 13 | [完成统一存储 16 位精度闭环](issues/13-uniform16-quality.md) | 01、02 | 允许宽累加和逐张量尺度的统一 16 位存储质量结果。 |
| 14 | [完成统一存储 32 位精度闭环](issues/14-uniform32-quality.md) | 13 | 统一 32 位存储质量结果，64 位为条件扩展。 |
| 15 | [完成严格同环数值诊断](issues/15-strict-ring-diagnostic.md) | 14 | 全程同环的独立诊断与误差来源分析。 |
| 16 | [比较位宽配置的真实安全成本](issues/16-width-secure-cost.md) | 07、09、10、14 | 逐配置安全成本及宽域和转换后窄域方案对照。 |
| 17 | [接通 ResNet-50 安全残差块](issues/17-resnet50-secure-block.md) | 11 | Q24.15/ReLU16 安全残差块的完整可验证路径。 |
| 18 | [完成 ResNet-50 整网验证](issues/18-resnet50-full.md) | 17 | ResNet-50 完整网络的定点质量与安全执行验证。 |
| 19 | [完成 ResNet-50 双轨性能结果](issues/19-resnet50-performance.md) | 18 | ResNet-50 原生配置和同图两轨性能结果。 |
| 20 | [扩展 ResNet-152 质量与成本验证](issues/20-resnet152-extension.md) | 19 | 复用 CNN 路径的 ResNet-152 质量与成本结果。 |
| 21 | [根据完成证据更新论文主张与实验](issues/21-paper-evidence-update.md) | 03、12、15、16、20 | 以完成证据更新论文、表格与结论。 |

## 可并行的工作

- 01 原生图回放、02 正式质量产物、03 文稿范围修正可直接并行。
- 01 后，SCI 比较接入、MP-SPDZ 时间步和 SCI 计量可以并行。
- 13—15 数值实验无需等待 Google-30 安全性能主表；11 双后端测量无需等待 09 MP-SPDZ 整网。
- Google-30 的 SCI/ours 闭环完成后进入 CNN 分支；17—20 与补齐三后端表、位宽诊断可以并行。
- 21 汇总所有已确认范围的最终证据；03 负责不等待新实验即可修正的既有表述。

## 验收与记录

按各工单的复选项验收。保存数值契约、配置、命令、原始日志和结果来源；验证模式中的中间值重构不进入正式性能测量。只摊销可摊销的初始化，单次使用的密钥、掩码和三元组按查询消耗计费。

具体线程数、batch、LAN/WAN 参数、预热与重复次数、统计方法及资源上限在正式运行前配置并冻结。本工单集没有将讨论中的示例数值当作已执行实验。

