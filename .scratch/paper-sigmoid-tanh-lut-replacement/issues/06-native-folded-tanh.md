# 06: 完整 TanH 替换与单步接入

**What to build:** 通过既有原生激活入口完成论文 TanH 的宽环倍增、folded LUT 与 signed floor 输出缩放；在 FastGRNN 单步中实际替换整个 TanH，Sigmoid 暂保留显式 SCI 对照。

**Blocked by:** 02: 冻结激活整数参考与整图回放；03: 不分段 10-bit LUT 完整查询。

**Status:** closed

**Labels:** completed

- [x] 原生契约为 16-bit/Q7→16-bit/Q7。先安全 signed 扩环至 32 位、乘 32 转 Q12，再在宽环内倍增进入既有 folded Sigmoid 核。
- [x] 计算 Q12 的 2S(2x)−4096，再安全 truncate-reduce 五位，环宽为 32→27→16，得到 signed floor Q7；不得使用向零规则或对本地份额直接右移。
- [x] 所有原生输入及倍增均有宽环余量，最小负数不回绕；输出范围 [−128,128]、零点为 0，负数最后缩放与 02 oracle 一致。
- [x] 从完整 TanH 入口分派，旧 SCI TanH 指数、倒数和后续合成不执行。元数据明确当前为“仅替换 TanH”的诊断形态，保留旧 SCI Sigmoid 对照。
- [x] 私有构建覆盖、planner、worker 与显式上下文完成输入前准备和在线消费；删除实际被替换的 TanH 库存，保留其他图调用真实需求，接口转换成本计入完整激活。
- [x] 不分段构造在真实两方 harness 中穷举全部 65536 编码，与 02 oracle 零差异，覆盖最小负数、近零负值、±8、饱和端点、索引进位、随机分享和两次 fresh query。
- [x] 现有 batch=1 默认与非零隐状态单步通过“仅替换 TanH”的明文图核验；所有新通信和适配/核心/LUT 子阶段可核对，在线材料生成、补充及加载为零。

**Evidence:** `project/our/results/paper_folded_activation_20261013/ticket06/README.md`。
真实随机分享覆盖全部 65536 编码和 25 边界的两次 fresh query；实际默认/非零隐状态各两次单步，15 张量、1351 整数逐次零差异。适配、核心和 LUT 流量守恒通过，旧 SCI TanH OT/triples 消费为零，在线生成和加载为零。保留红测试旧整数差异、完整原始日志、构建与实际资源记录。

**Scope boundary:** 复用已有 folded Sigmoid 内核不构成对 05 原生 Sigmoid 接入票的依赖。两种原生接口和两种 LUT 的最终组合由 07 汇合。
