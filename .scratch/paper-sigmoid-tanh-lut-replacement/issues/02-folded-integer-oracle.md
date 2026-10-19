# 02: 冻结激活整数参考与整图回放

**What to build:** 让验证者可以独立算出新 Sigmoid/TanH 的所有原生整数输出，并在已有明文回放中运行单步或完整序列，作为后续安全执行和模型质量的统一参考。

**Blocked by:** None (can start immediately).

**Status:** closed

**Labels:** completed

- [x] 固定 f=12、q=10 和 1024 项 correction 表，以独立高精度计算逐项验证 floor(4096/(1+exp(j/64)))，记录表版本；expected 不调用安全协议实现生成。
- [x] 对每种激活穷举全部 65536 个原生有符号 16 位输入。输入 Q7 编码乘 32 进入 Q12；Sigmoid Q12 输出乘 4 得 Q14；TanH 按两倍输入复用 Sigmoid，再进行 signed floor 除以 32 得 Q7。
- [x] 明确安全实现应遵守的环宽：输入先 signed 扩环 16→32；索引 32→26→10；TanH 输出 32→27→16。覆盖最小负数、零、±8/±16、索引边界及负数向下舍入；另有 Q12-only 边界参考。
- [x] 现有明文整数图支持显式 SCI 原算法、仅替换 Sigmoid、仅替换 TanH、两者都替换的模式。局部替换只服务诊断，默认 SCI 参考不被改成 folded。
- [x] 相同输入与旧隐状态下可输出既有单步 15 个语义张量；完整 99 步和分类头可运行。激活之外的矩阵乘法、偏置、缩放、缩环及各自舍入规则保持原整数图契约。
- [x] 默认与非零隐状态的回放可复现；激活之前可与 SCI 参考逐位核对，激活及其下游按所选新数值模式生成 expected。独立样本交叉验证既有 folded oracle。
- [x] 记录冻结 checkpoint、量化图、归一化和验证集来源供质量票直接使用；本票无需等安全协议、图并行或完整验证集评估。

**Scope boundary:** 不训练、重选模型或调整量化；不把明文执行计入安全协议性能。

**Resolution:** 2026-10-13 完成。独立 Decimal 80/120 位表逐项一致，两种激活全部原生输入与历史 folded 参考一致；四模式在默认及非零隐状态下各复核 batch=128 的 15 张量/172928 整数。保留 SCI 模式还与历史真实 SCI 三组诊断逐位一致。每种模式以三条冻结真实输入完成 99 步、分类头的标量/向量复核，CLI 亦通过；全量 6798 条质量结果由 10 负责。

**Evidence:** [数值验收](../../../project/our/results/paper_folded_activation_20261013/ticket02/validation.json)、[冻结来源](../../../project/our/results/paper_folded_activation_20261013/ticket02/frozen_sources.json)、[复现说明](../../../project/our/results/paper_folded_activation_20261013/ticket02/README.md)。
