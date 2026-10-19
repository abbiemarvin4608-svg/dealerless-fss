# 05: 完整 Sigmoid 替换与单步接入

**What to build:** 通过既有原生激活入口接收 Q7 秘密输入并返回论文 folded Sigmoid 的 Q14 输出；在 FastGRNN 单步中实际替换整个 Sigmoid，TanH 暂保留显式 SCI 对照，以便独立验证这一条路径。

**Blocked by:** 02: 冻结激活整数参考与整图回放；03: 不分段 10-bit LUT 完整查询。

**Status:** closed

**Labels:** completed

- [x] 原生契约为 16-bit/Q7→16-bit/Q14。先安全 signed 扩环 16→32 再本地乘 32；不得本地符号扩展单方份额或在 16 位环内先放大。
- [x] 使用 BitDec 基准执行符号、幅值、裁剪、六位精确索引截断、十位 LUT 与 BitXA 恢复，内核输出 Q12 本地乘 4 并缩环回 16 位。零点为 8192，输出范围 [0,16384]。
- [x] 从完整 Sigmoid 入口分派到新路径，运行中不进入 SCI Sigmoid 的指数、倒数、除法和结果合成链。原生 TanH 保留情况在诊断元数据明确标记，不能称为完整双激活组合。
- [x] 在现有私有构建覆盖、planner 和 worker 入口中连通准备与消费，记录数值算法、LUT family 和上下文；去除被替换的 Sigmoid 旧库存需求，剩余 TanH 及外层调用按真实需要保留。
- [x] 不分段构造在真实两方激活 harness 中穷举全部 65536 编码，和 02 oracle 零差异；覆盖随机分享、最小负数、零、±16、索引边界及两次 fresh query。
- [x] 现有单步入口在 batch=1 默认及非零隐状态下通过“仅替换 Sigmoid”的明文图核验，激活后的尾部按该新整数图比较，不继续要求与 SCI 全图逐位相同。
- [x] 新材料输入前生成并驻留；所有新消息与完整激活/适配/LUT 子阶段接入现有计量，旧 SCI 激活内部 LUT 消费仅属于尚未替换的 TanH。

**Evidence:** `project/our/results/paper_folded_activation_20261013/ticket05/README.md`; native boundary/full-domain random-share logs and both default/nonzero frozen graph rows are retained below that directory.

**Scope boundary:** 单激活替换是明确标记的诊断形态，最终双激活配置及 batch=128 联合驻留由 07 交付。本票不依赖 04 或 06。
