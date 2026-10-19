# 06: 共享预处理策略贯通四后端完整单步

**What to build:** 在现有两方完整 FastGRNN 单时间步中启用 SCI_preprocessed，并让 SCI_ours 与 Beaver+分段 FSS 的剩余原生 SCI 工作获得同样预处理能力，形成可核验的四后端对照。

**Blocked by:** 05：原生 Sigmoid/TanH 内部预处理闭环。

**Status:** closed

**Resolved:** 2026-10-12; implementation `95d97b4` and qualification `18059c1`, integrated on `integration/sci-sirnn-preprocessed`.

**Evidence:** [四后端完整单步验收](../../../project/our/results/sci_sirnn_preprocessed_20261012/ticket06_step/README.md)。默认及冻结非零隐状态共八行 batch=128 真实诊断，每后端每行两个 fresh query；每查询 15 张量、172928 个整数与新执行默认 SCI 逐值一致。三种预处理后端均取得严格完整材料资格，公开规划纯度、OT/安全 triple 库存和平衡以及 prepare 后 native/bridge 生成增量为零均已核验。缺失 triple owner 回归双方明确失败。此处为正确性验收，正式十查询性能行仍由 08 交付。

- [x] 注册 SCI_preprocessed，保留默认 SCI；SCI_ours 和 SCI_Beaver_segmented 使用同一原生 SCI 预处理策略，实际模式在命令、描述和结果中明确记录。
- [x] 匹配同一冻结模型、单时间步图、batch=128、输入/隐状态、整数语义、共享调度、资源、安全假设及输入前预处理机会。按每个后端实际协议规划材料，数量、内部调用和离线时长不要求相同。
- [x] 按实际后端的公开单步计划生成材料，覆盖原生图级调用与原生激活内部调用；被 FSS/Beaver 替换的在线原生工作不重复预备。
- [x] Beaver/FSS 自身离线生成过程中消耗的原生 OT 如实计入离线，不重复计成待在线消费库存；全部新材料生成接入阶段统计与 epoch 核验。
- [x] prepare 在查询输入/隐状态装载前实际完成本查询随机 OT、标准/相关比较 bit-triple 及其他所需一次性库存，双方 material-ready 后才释放查询输入。完整预处理模式从此至输出结束不得新做 OT 扩展、生成一次性库存或补货；任何未覆盖需求阻止该行取得完整预处理资格。
- [x] 严格库存限制允许消费已完成材料所需的本地 PRG/哈希、FSS 密钥评价、消息与算术相关量适配；这些输入后工作及库存读取、等待、释放全部计在线。在线快照先于 enter_online，input/ready 不隐藏协议成本；不能用允许的本地计算延后完成随机 OT 或 bit-triple 的生成。
- [x] 完整单步保持冻结模型和 sci_compatible 整数语义、原生激活及公开 ScalarMul；不替换默认 SCI 控制组的协议算法。
- [x] 在顺序上下文下完成 batch=128 默认及非零隐状态诊断，每个后端核验现有 15 张量、172928 个整数；正式模式仅保留原有最终输出。
- [x] 库存遵守每方 1.5 GiB 地址空间与四核总预算，使用连续内存或已生成的磁盘材料；输入后的材料读取和等待计入在线及直接总成本。
- [x] 连续查询验证每份材料新鲜且一次消费，留存四后端正确性、完整材料覆盖、阶段和资源日志；对资源失败保留真实状态，不能以小 batch 替代完整单步验收。
