# 04: 两个 5-bit 段的 LUT 查询

**What to build:** 让同一个公开表查询和完整 Q12 激活入口支持论文的两个五位短段 DPF 构造，生成与不分段版本相同布局的选择份额，并使用同一个在线开值和内积实现。

**Blocked by:** 03: 不分段 10-bit LUT 完整查询。

**Status:** closed

**Labels:** completed

- [x] 具名配置为 `lut_key_family=segmented_boolean_tensor_b2a`、`lut_segments=2`、段宽 (5,5)，与外层 FSS 分段配置分开记录。
- [x] 短段秘密目标与索引 mask 来自同一组 bit。段内求值得到 Boolean equality，经安全张量 AND 合成完整 one-hot，再 B2A 到 32 位输出环；不使用本地份额普通乘法替代安全 AND。
- [x] 平衡张量树逐层批量执行，本配置每元素实际计入 1024 次张量 AND 和 1024 个输出 B2A，以及短段 keygen、mask 转换和全域局部求值。
- [x] 离线只保留与 03 相同格式的 mask 和 1024 项 packed arithmetic selection，去掉全部循环移位输出的 O(ND²) 预计算；临时段向量与工作区在完成后释放。
- [x] 完整复用 03 的减法掩码、开值编码、表布局和在线内积核，在线不新增段间交互、B2A 或全域求值；一次请求只生成本构造所需材料。
- [x] 真实两方逐索引测试、one-hot/索引表示核验、回绕和连续新请求通过；完整 Q12 Sigmoid/TanH 在同一输入下与不分段版本重构一致，份额本身无需一致。
- [x] 复用显式上下文与库存失败行为，按预算 tile 批量生成；交付分段构造的独立离线分项、开值/内积计量与准备 RSS/AS 快照及进程峰值 RSS，工作区估算单列，不能冒用完整 q 树的复杂度标签。

**Evidence:** [双构造完整 q10 和 Q12 回归](../../../project/our/results/paper_folded_activation_20261013/ticket04/README.md)。每构造两个新查询覆盖全部 1024 索引及尾部，验证 one-hot、掩码回绕和精确表输出；14 类在线库存错配及 3 类准备配置在通信前拒绝。分段版准确计入每元素 1024 AND 和 1024 输出 B2A，在线同布局同内积核、零材料生成。完整 Q12 双上下文并发 Sigmoid/TanH 两构造均符合相同边界字面 oracle；四 CPU affinity 和 1.5 GiB AS 限制下通过。

**Scope boundary:** 本票交付可执行的第二种 LUT 构造，不预设它更快。原生接口两种激活的双构造全输入与整步资格在 07 汇合验证。
