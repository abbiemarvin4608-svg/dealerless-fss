# 02: 多选一 LUT 与批量 GOT 位乘法预处理

**What to build:** 让真实 SCI 查表和使用批量 GOT 的位乘法调用，在输入前建立所需随机 OT 库存，输入后取得与默认协议相同的查表或乘法结果。

**Blocked by:** 01：输入前材料生命周期与二选一 GOT 闭环。

**Status:** closed

**Resolved:** 2026-10-12; implementation `608b1c1`, branch tip `b41783a`, integrated on `integration/sci-sirnn-preprocessed`.

**Evidence:** [真实 LUT 与批量 GOT 验收](../../../project/our/results/sci_sirnn_preprocessed_20261012/ticket02_got/REPORT_CN.md)。真实 AuxProtocols LUT 覆盖 radix 2..256；LinearOT::matmul_multiplexer 覆盖两个矩阵布局与两个 OT 方向，另验收混合位宽和超过 128 bit 的聚合消息。成功模式各执行两个新鲜查询，输入后各阶段生成 epoch 为零；错误分组和重复消费明确失败。本票保留默认路径，不代表完整单步预处理。

- [x] 将新鲜随机 OT 生成和在线输入相关适配贯通实际 IKNP GOT、KKOT 多选一，以及长消息和混合消息位宽的批量 GOT；不能只打开旧普通 GOT 缓存开关。
- [x] 通过真实两方 LUT 调用覆盖实际使用的公开表规模、选择索引和消息位宽，查表输出与默认 SCI 一致。
- [x] 单独通过真实两方 matmul_multiplexer 位乘法调用验证 send/recv_batched_got 路径；LUT 测试不能替代此验收。
- [x] 在线保留真正的消息、选择与掩码适配；多选一和批量模式按各自协议及消息长度计费，不套用普通二选一的一位修正公式。
- [x] 各 radix、方向、有效数量、取整生成量和消息分组都有材料消费覆盖；实际发送字节包括尾部填充，查询之间不复用随机材料。
- [x] 使用既有两方行为入口验证不规则 count、长消息、混合位宽与查询连续消费；局部开启预处理时保持整数结果一致，相关 GOT 路径在线不扩展、不补货。
- [x] 交付局部正确性、阶段通信和调用覆盖日志；未涉及的 COT、比较或其他协议仍明确记在在线，局部模式不冒充完整预处理基线。
