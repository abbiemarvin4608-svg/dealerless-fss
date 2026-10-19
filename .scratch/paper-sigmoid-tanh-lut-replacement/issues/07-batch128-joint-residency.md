# 07: 双激活 Batch=128 联合驻留

**What to build:** 让新具名后端在 batch=128 FastGRNN 完整单步同时使用两种论文激活，并分别以分段或不分段 LUT 完成输入前准备、在线执行、最终状态核验和联合资源验收。

**Blocked by:** 04: 两个 5-bit 段的 LUT 查询；05: 完整 Sigmoid 替换与单步接入；06: 完整 TanH 替换与单步接入。

**Status:** closed

**Labels:** implemented, verified

- [x] 交付 `SCI_Beaver_segmented_PaperLUT` 完整组合，显式标记 `folded_sigma_f12_q10`、LUT family、段数和 BitDec 符号后端；名称中的 segmented 不用于推断 LUT 构造。
- [x] 两种构造分开运行，每次只规划所选 LUT，实际准备两激活合计 25600 元素的选择材料、输入扩环、符号 triples、三次 BitXA、索引截断和 TanH 输出截断，以及外层 Beaver/FSS 材料。
- [x] 新组合完全退出 SCI 两种激活函数体，被替换的 exp/reciprocal 和相关运算库存需求及消费为零；其余图内必要 SCI 调用单独辨识。
- [x] 两构造各自完成两种原生激活的全 65536 编码核验，随后通过默认和非零隐状态的完整 batch=128 单步，15 个语义张量、172928 个整数及新隐状态与新明文图一致。
- [x] 输入和旧隐状态装载之前材料实际生成并驻留；通过两个连续 fresh query 及一次消费核对，在线 keygen、OT/三元组补充、材料加载和等待增量为零。
- [x] 每方 1.5 GiB RLIMIT_AS 内真实运行，四参与者预算有效；FSS prefetch≤512 MiB 并计入总额。100 MiB selector 之外的 OT、Beaver、FSS、triples、线程栈和临时工作区均实测纳入峰值。
- [x] 通过同形状合批与公开 tile 控制 keygen、B2A/AND/payload 和截断准备峰值，完成最终材料后释放临时量，不能用在线装载换取驻留达标。
- [x] 本票以图串行、每激活单 worker 的完整路径交付；不同阶段墙钟、全部通信通道及 LUT/适配子阶段可核对。资格结果、配置与失败证据可复现。

**Scope boundary:** 本票验收完整串行功能与联合驻留，不以单步结果推导 99 步安全全网可驻留。并行配置由 08 验收，正式性能由 11 验收。

**Implementation:** `project/our/results/paper_folded_activation_20261013/ticket07/README.md` 和 `step-final/qualification.json`。四配置各两次 fresh query 均通过整数/驻留审计；最大 VmPeak 761470976 字节。两 LUT 构造各自完整原生 Sigmoid/TanH 全域通过。初始空 SCI 库存 collector 错误和查询内 ordinal 审计错误已保留后修复。
