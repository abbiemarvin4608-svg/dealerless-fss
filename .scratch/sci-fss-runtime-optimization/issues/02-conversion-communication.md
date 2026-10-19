# 02: 转换通信合批与位打包

**What to build:** 通过多域同层 scan 合批、MulCir 双输入扩环合批和 Boolean 打开位打包，减少真实两方完整转换的通信等待与载荷。三项优化可独立启用，组合模式按真实依赖计量，保留 SCI 定点整数输出及一次性材料契约。

**Blocked by:** 01：补齐计量并冻结匹配基线。

**Status:** resolved

- [x] 经现有完整转换接口，让同时就绪的 low/sign/wrap 进入局部求值、当前 scan 层准备、统一交换和当前层完成的闭环；各域保留独立材料区间与输出映射，不要求外部调用者管理 scan 层。
- [x] 多域 scan 依赖深度由各域深度之和变为最大深度，普通 Beaver 的后续层继续等待上一层。三个四段独立域的 scan 从 6 轮降为 2 轮，含向零修正的完整 signed div_pow2 从 1+6+1+1=9 轮降为 1+2+1+1=5 轮，以真实两方统计验证。
- [x] MulCir 按输入/输出位宽、符号、数值模式与转换操作分组；两个已就绪的同契约输入一次批量扩环后按原顺序切回，异构契约分别处理，存在数据依赖的调用保持原依赖。转换计划、离线材料数量、桥接消费、调用身份与覆盖检查同步支持分组。
- [x] 单独启用 MulCir 合批时，原两次各 6 轮的同契约扩环变为一次 6 轮；与多域 scan 同时启用时，重新核对分组映射和完整执行计划的真实轮数，不将两项独立降轮数直接相加。
- [x] scan E/F 与转换 correction 的布尔打开采用一致的 packed-bit 顺序及尾部约定；每条 N 个布尔值的数据载荷为向上取整的 N/8 字节，消息头另计，并以实际发送计数核验。普通 byte 接口语义兼容，位打包本身不改变依赖轮数，传输工作缓冲可复用并减少不必要中转复制。
- [x] 经现有 DCF、完整转换、Beaver MulCir 与 bridge 接口完成真实两方重构和独立 oracle/SCI-compatible 交叉检查，覆盖小域穷举、零位域、奇数节点、不等宽及混合段数、异构契约、符号边界、不规则 batch，以及布尔长度 1、7、8、9 的尾部处理。
- [x] 每查询 keys、mask、triples 及布尔相关材料均新鲜且一次消费；无遗漏、错位或跨查询复用。通过默认及非零隐状态 Batch=128 单时间步检查真实调用覆盖、输出顺序和整数结果，不改变乘法环宽、舍入、截断或原生激活算法。
- [x] 提供三项独立开关，保留未优化对照；分别报告仅多域 scan、仅双输入扩环、二者组合、仅位打包及本工单累计配置。组合覆盖围绕明确交互展开，不要求所有开关全排列；打包/解包和等待成本保留在 online 与直接 query total 内。
- [x] 使用规格冻结构建、模型、输入、每方四核与 1.5 GiB RLIMIT_AS，正式测量串行、无预热且不揭示中间值。16 位 DCF count128/count12800、fixed-2/fixed-4、cGGM 上限 1/4 的受影响匹配行分别使用 30/10 个新鲜查询；signed32、shift14、count128 完整转换 5 个，单步 10 个，基本原语/转换每行 120 秒、单步每行 600 秒。
- [x] 主对照为不整形 localhost；通信专项补测 sirnn_lan 的 RTT 0.8 ms/3016 Mbps 与 wan 的 RTT 2 ms/100 Mbps，只比较相同配置融合前后，所有通道经过对应链路并纳入计数，网络结果分表。
- [x] 保存双方对齐的原始样本、材料量、实际参与者、逻辑轮数、发送载荷与消息头、双方发送字节总和、RSS、setup/offline/online/直接 query/session total；按双方时延最大值报告中位数、范围和离散程度。收益及回退均保留，失败使用真实状态与 null 完成时延，未验证配置不成为默认设置。

## Answer

已交付多域 scan、同契约 MulCir 双输入扩环、Boolean 位打包的独立开关及必要交互组合。三个四段独立域的 scan 由 6 轮变为 2 轮；完整 signed div_pow2 按真实依赖由 9 轮变为 5 轮。双输入扩环以合并计划和独立消费区间执行，组合轮数以实际协议统计为准。`da1a421` 补齐直接 byte transport，packed tail 和实际 wire 计数经真实两方测试核对。

[原语与完整转换消融](../../../project/our/results/runtime_optimization_20261010/primitive_ablations-recheck/summary.json)保留 64 个完成行：DCF count128/count12800 分别 30/10 个新鲜查询，转换每行 5 个；[单步局部消融](../../../project/our/results/runtime_optimization_20261010/conversion_material/summary.json)保留 11 个匹配十查询行，[默认/非零隐状态诊断](../../../project/our/results/runtime_optimization_20261010/conversion_material_diagnostics/summary.json)全部 18 行完成。[byte 协议回归](../../../project/our/results/runtime_optimization_20261010/review_verification/byte_core_protocol/summary.json)覆盖 35 配置/212 值、segmented/full-tree bridge、多上下文及 1/7/8/9 位尾部。[网络补测](../../../project/our/results/runtime_optimization_20261010/network_supplements/summary.json)在 sirnn_lan 与 wan 上分别比较融合前后，每行 5 个新鲜查询。

通信专项减少依赖轮数和布尔载荷，并不保证 localhost 直接查询总成本下降；逐项与组合的回退保留在原始样本。旧冻结构建的局部消融与 byte transport 后的最终匹配队列分别保存，不混合样本。验收止于现有接口和 Batch=128 单时间步；历史 SDCF-09 继续延期。
