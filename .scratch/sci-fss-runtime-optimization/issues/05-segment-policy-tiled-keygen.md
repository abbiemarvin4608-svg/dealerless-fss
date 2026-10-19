# 05: 按域分段与离线树分块

**What to build:** 完整转换按公开域宽选择 fixed-2、fixed-4 或 depth-cap-8，并使用预算约束的离线树分块控制宽域工作集；以真实两方数据比较段数和 tile 对在线、离线、总成本及内存的影响，尝试在每方 1.5 GiB 下完成宽域两段与 Batch=128 单时间步。

**Blocked by:** 03：连续材料与有界预取。

**Status:** resolved

- [x] 保留 fixed-2/fixed-4 的域宽限制，新增 depth-cap-8：非零域采用最大段宽不超过 8 的最少段数，域宽 1 至 8/9 至 16/17 至 24/25 至 32 分别为 1/2/3/4 段；零位域跳过，按高位优先连续分段，支持不等宽段。
- [x] low/sign/wrap 的策略、实际段数与段宽一致进入离线计划、材料记录及在线契约检查，两方公开计划不匹配在消费前检出。策略仅依赖公开域宽、shape 与显式预算，不依赖秘密输入或评估后的性能结果。
- [x] 使用 03 的稳定连续材料视图，依据公开 batch、段宽、展开节点/hash 缓冲、已驻留材料及总预算选择 tile；估算覆盖中间工作集，完成一块即释放展开树并按确定顺序保留或写出最终材料。分块可在既有标量哈希上完成，不依赖 06 的批量 AES。
- [x] 向 09 的离线准备重叠提供每 tile 工作集估计、内存预留和计算参与者上限，允许按剩余预算控制同时驻留的 tile；保留独立顺序执行能力，本工单不以离线重叠完成为前提。
- [x] 各块根、mask、triples 按原协议新鲜生成，keys 和关联材料跨块顺序一致、每查询一次消费；异常后不可重用已消费材料。分块不改变深度 16 的指数级总展开语义或密码学参数。
- [x] 经现有真实两方 DCF 与完整转换验证域宽 1、8、9、16、17、24、25、31、32、零位域、奇数段、不等宽段、多个 tile 大小及非整 tile 尾部；结果逐值匹配独立 oracle 和 SCI-compatible 整数语义，signed32、shift14、count128 完整转换覆盖 low14/sign31/wrap32、舍入和输出顺序。
- [x] 分段策略与分块分别可配置，保留固定策略、未分块和已验证默认基线；提供策略单独、分块单独及分段加 tile 的必要组合消融。与多域 scan 融合组合时按最深域重算轮数，不预设窄域减少段数必然降低完整转换轮数。
- [x] 按冻结机器、构建、输入、每方四核和 1.5 GiB RLIMIT_AS，实际尝试 31/32 位、count12800、fixed-2 的分块前后对照，fixed-4 作资源参照；每个宽域行预先冻结 600 秒墙钟预算，完成行使用 10 个新鲜查询，保留 OOM/timeout，不能以 count128 替代目标数据。
- [x] 在不整形 localhost 下顺序测量受影响的 16 位 DCF count128/count12800、fixed-2/fixed-4、cGGM 上限 1/4，分别使用 30/10 个新鲜查询；完整 signed32、shift14、count128 转换以 fixed-2/fixed-4/depth-cap-8 各 5 个查询对照，基本原语/转换每行 120 秒。所有正式行无预热，其他开关与匹配基线一致。
- [x] 实际尝试 fixed-2 分块的完整 Batch=128 单时间步，并比较可完成的三种分段策略；每行 600 秒、完成行 10 个新鲜查询且匹配基线重测。默认及非零隐状态的整数诊断和实际调用覆盖通过后才计完成行；正式性能计量关闭中间张量揭示，保留相同四路后端与数值契约。
- [x] 保存实际段宽、tile 大小/数量、参与者、估计与实际峰值内存、驻留材料、原有及额外哈希、离线通信批次/轮数/字节，以及双方对齐的原始 setup/offline/online/直接 query/session total；额外重算与交换计入成本。报告中位数、范围和离散程度，失败注明位置与 null 完成时延，预算内未完成不宣称资源目标通过，不以扩展预算直到有利结果替代原实验；只有通过验收且满足预声明目标的配置才成为推荐。

## Answer

已交付 fixed-2/fixed-4/depth-cap-8 的公开策略、完整 low/sign/wrap 计划与消费前一致性检查，以及覆盖树展开、hash 缓冲、最终材料和上下文预留的预算 tile。现有 DCF/转换测试覆盖 depth-cap-8 边界、零位域、混合段宽、policy mismatch、多个 tile 和尾块；分块保持原树展开与密码学参数。

[wide_confirm](../../../project/our/results/runtime_optimization_20261010/wide_confirm/summary.json)保留 31/32 位 count12800 的 8 个正式资源行：6 个完成行各 10 个新鲜查询，2 个真正未分块 fixed-2 行 OOM，完成成本为 null。预算 auto 的 32 位 fixed-2 行在 1.5 GiB 内完成：offline 中位数约 12.569 s、online 约 0.003042 s、峰值 RSS 约 586 MiB；显式 tile8 同样完成。[policy_confirm](../../../project/our/results/runtime_optimization_20261010/policy_confirm/summary.json)保留 fixed-2/fixed-4/depth-cap-8 的匹配完整转换，每行 5 个新鲜查询；[原语消融](../../../project/our/results/runtime_optimization_20261010/primitive_ablations-recheck/summary.json)覆盖 16 位 count128/count12800、fixed-2/fixed-4 和 worker 上限 1/4。

[fixed-2 tile8 完整 Batch=128 尝试](../../../project/our/results/runtime_optimization_20261010/wide_batch/summary.json)在预声明 600 秒内未完成，状态保留 timeout 和 null 完成时延；不能宣称这一资源目标通过。[depth-cap-8 单步](../../../project/our/results/runtime_optimization_20261010/final_depthcap8/summary.json)完成 10 个新鲜查询，[默认/非零诊断](../../../project/our/results/runtime_optimization_20261010/final_depthcap8_diagnostics/summary.json)通过。可完成策略及失败范围均已交付，推荐只覆盖通过整数与资源验收的配置。历史 SDCF-09 整图和质量验证继续延期。
