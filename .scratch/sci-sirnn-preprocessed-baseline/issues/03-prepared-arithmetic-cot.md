# 03: B2A 与乘法的算术 COT 预处理

**What to build:** 让真实 SCI B2A 和使用 COT 的 MatMul/MulCir 路径，使用输入前生成的随机材料完成输入后的算术相关 OT，保留默认 SCI 的整数结果。

**Blocked by:** 01：输入前材料生命周期与二选一 GOT 闭环。

**Status:** closed

**Resolved:** 2026-10-12; implementation `fb90152`, branch tip `4010dde`, integrated on `integration/sci-sirnn-preprocessed`.

**Evidence:** [真实算术 COT 验收](../../../project/our/results/sci_sirnn_preprocessed_20261012/ticket03_cot/README.md) 与 [预处理接口](../../../project/our/tools/segmented_dcf_runtime/PREPARED_COT.md)。四项真实两方测试覆盖 B2A、原生矩阵与 Hadamard 乘法、混合位宽和 SIMD 尾部；成功模式各执行两个新鲜查询，六种材料错误双方均明确失败。原生 COT 调用在线生成 epoch 为零；输入相关选择及算术修正真实计在线。本票仅交付算术 COT 局部闭环。

- [x] 实现普通和批量算术 COT 的新鲜输入无关材料生成及输入相关在线适配；prepare 在查询输入/隐状态装载前实际完成对应随机 OT 扩展，原本直接执行扩展的 COT 调用使用已生成库存。真实选择和算术相关量的修正保留在线并计费。
- [x] 保持各实际消息位宽下的模二次幂环相关性和双方输出份额；不能用 XOR COT 关系替代算术 COT，也不将真实相关输入带入本次查询的离线材料计划。
- [x] 在真实两方 B2A 及 COT 乘法路径验收，覆盖 MatMul 和逐元素 MulCir 所需的实际协议调用；整数输出与默认 SCI 一致。
- [x] 保留原生乘法算法、累加器宽度、符号处理与后续转换，公开 ScalarMul 保持本地乘法；不将此基线改造成 Beaver 乘法。
- [x] 材料按方向、上下文、位宽、批量分组、有效数量和生成量消费；相关 COT 在线扩展或补充为零，所有适配通信如实计入在线。
- [x] 通过既有两方行为测试覆盖普通/批量模式、混合位宽、尾部、模环边界及连续查询；用实际份额关系和重构整数验收。
- [x] 交付局部正确性、通信和消费证据。未实现预处理的 GOT 或比较步骤可在局部诊断中沿用默认在线路径并明确记录；本票不因此依赖 02，也不宣称完整单步已完成预处理。
