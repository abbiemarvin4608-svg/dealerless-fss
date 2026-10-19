# 04: 比较材料与完整整数转换预处理

**What to build:** 在真实比较、wrap、截断和扩环调用中提前生成所需标准/相关 bit-triple 与 OT 库存，让输入后的完整整数转换保持 SCI 语义并消除这些路径的现场材料生成。

**Blocked by:** 02：多选一 LUT 与批量 GOT 位乘法预处理；03：B2A 与乘法的算术 COT 预处理。

**Status:** closed

**Resolved:** 2026-10-12; integrated branch tip `4eb73a2` (native planner, safe triples, replay and inactive-context fixes) on `integration/sci-sirnn-preprocessed`.

**Evidence:** [真实比较与完整转换验收](../../../project/our/results/sci_sirnn_preprocessed_20261012/ticket04_conversion/README.md)。四项真实规划/三元组测试通过；默认与预处理模式的 42 输出向量、546 整数逐值匹配，每查询生成并消费 1840 个标准/相关安全 bit-triple。输入、在线及输出的原生生成 epoch 为零；unsafe、契约错配、重复/未完成消费和在线生成均明确失败。合并后五项材料收集器检查通过。

- [x] 在 prepare 中、查询输入/隐状态装载前实际运行 SCI 两方安全生成协议，完成普通比较及 comparison-with-equality 消耗的标准或相关 bit-triple 库存；不能只前移其 OT 扩展、仍在输入后生成 bit-triple。禁止使用 Ideal 或预置可被单方知道的联合材料。
- [x] 贯通比较叶节点 OT、比较树的 AND、B2A 与后续修正，完成真实两方比较/wrap、完整截断及扩环的预处理与在线调用闭环。
- [x] 保持 sci_compatible、DIV_RESCALING、符号、负数舍入、低位借位、wrap correction 和输出环；不以数学 floor 诊断替代原生数值契约。
- [x] 在公开调用计划中记录 bit-triple 和各类 OT 的需求与上下文；输入改变但公开配置不变时，材料计划不因秘密值改变。
- [x] 在线执行原有输入相关比较、修正和打开；相关路径在线 OT 扩展、三元组生成和库存补充为零，所有剩余通信计入在线。
- [x] 复用既有真实两方原语与完整转换入口，验证正负数、零、环边界、wrap/借位及不规则数量；结果逐整数匹配默认 SCI。
- [x] 记录生成、消费、公开填充弃置和通信，核验一次性消费与查询结束平衡；每个完整转换都有独立正确性和材料覆盖证据。
