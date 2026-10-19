# 03: 不分段 10-bit LUT 完整查询

**What to build:** 通过现有两方查表入口，以完整 10-bit 不分段算术 DPF 完成公开 1024 项表的输入前准备与在线秘密索引查询，并把该生命周期用于已有完整 Q12 激活。

**Blocked by:** 01: 显式上下文下的完整 Q12 激活。

**Status:** closed

**Labels:** completed

- [x] 具名配置为 `lut_key_family=full_q_arithmetic`、`lut_segments=1`、段宽 (10)，输出环为 32 位；保留历史 SCI 小表 DPF 的原标签和行为。
- [x] 每元素生成新鲜秘密 mask-bit，B2A 得到一致的索引环 mask，批量完成完整 q 位 dealerless keygen、单位算术 payload 适配及全域求值，得到双方重构为 one-hot 的算术份额。
- [x] 输入前只驻留 mask 与连续 1024 项 32 位 packed selection，释放不再需要的 key/树工作区。无跨段 AND、无逐叶输出 B2A；mask 转换和 payload 的实际成本仍计入离线。
- [x] 新生命周期固定公开 z=t−r mod 1024，输出按对所有 u 累加 v[u]·T[(u+z) mod 1024] 计算。通过公共偏移或连续区间复用表，本地公开乘法不耗 Beaver。
- [x] 同一阶段索引按批开值，在线不展开 DPF 树、不生成材料；复用一个与 LUT 预处理 family 无关的在线接口，供 04 接入。
- [x] 使用显式上下文，材料绑定 query、算子、worker、元素范围、位宽、表版本与协议 family；错配、重复消费、库存不足明确失败。两个连续 query 各自 prepare 新材料。
- [x] 真实两方测试覆盖全部 1024 个索引、掩码回绕、不同随机分享及批量尾部；重构精确等于公开表，并在已有完整 Q12 激活中回归符号、裁剪和输出恢复。
- [x] 按同深度合批和预算 tile 生成，记录 keygen、全域物化、mask/payload、驻留容量、开值与内积实际成本及通道轮次；工作区峰值标为分配估算，实测 RSS/AS 在 07 完成。本票用现有入口验证小批完整生命周期，实际 batch=128 联合资格在 07 完成。

**Evidence:** [真实两方 exhaustive q10 LUT 和完整 Q12 激活](../../../project/our/results/paper_folded_activation_20261013/ticket03/README.md)。两次新查询覆盖 1024 个索引及三元素尾部，重构 mask/selector 验证完整 one-hot；12 类库存错配及重复消费在通信前拒绝。完整 Q12 Sigmoid/TanH 的两个独立上下文并发回归通过，查询绑定校验已前移到符号阶段通信前，在线材料生成计数为零。

**Scope boundary:** 本票固定公共表查找契约，原生 Q7/Q14 适配由 05/06 承担。共同在线表示与开值约定在此票验收后由分段构造直接复用。
