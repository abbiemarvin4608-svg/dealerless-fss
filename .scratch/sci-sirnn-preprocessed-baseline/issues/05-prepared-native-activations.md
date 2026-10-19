# 05: 原生 Sigmoid/TanH 内部预处理闭环

**What to build:** 让原生 SCI Sigmoid/TanH 的完整内部协议链使用输入前准备的材料，在输入后完成与默认 SCI 逐整数相同的激活计算。

**Blocked by:** 04：比较材料与完整整数转换预处理。

**Status:** closed

**Resolved:** 2026-10-12; implementation `8b276ef`, integrated on `integration/sci-sirnn-preprocessed`.

**Evidence:** [原生 Sigmoid/TanH 验收](../../../project/our/results/sci_sirnn_preprocessed_20261012/ticket05_activation/REPORT_CN.md)。两个真实两方测试分别覆盖 13 个尾部元素及 batch=128 对应的 12800 元素；默认/预处理各两个新鲜查询，两种激活输出逐整数匹配。完整内部材料预先生成，12800 元素每查询消费 422400 个安全 bit-triple，输入后原生生成增量为零；诊断重建全部在 output 阶段。

- [x] 依据公开输入数量、环宽和数值参数规划原生 Sigmoid/TanH 内部 LUT、B2A、MUX、hadamard、比较、截断及扩环的实际材料需求。
- [x] 沿用原生 SCI 的近似函数、尺度、舍入和中间环，不替换为 FSS 激活或其他数值算法。
- [x] 在现有真实两方激活入口先实际生成所有所需随机 OT、bit-triple 等库存，双方 material-ready 后再提供查询输入；完整执行两个激活并验证各自输出与默认 SCI 一致。
- [x] 激活内部的 OT 扩展、bit-triple 生成和库存补充在线为零；保留输入相关适配与打开，不将图级激活计时当成内部覆盖证明。
- [x] 材料日志能对应内部子调用、有效数量、消息参数与上下文，记录实际生成/消费以及填充弃置；按新鲜查询一次消费。
- [x] 通过既有两方行为测试覆盖实际单步激活数量、正负输入、临界输入和尾部数量；诊断输出与正式计量分离。
- [x] 交付完整激活整数一致性、阶段通信和内部材料覆盖；本票完成顺序上下文的闭环，多 worker 的库存路由由 07 独立验收。
