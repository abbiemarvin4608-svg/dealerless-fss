# 04: Beaver 本地矩阵乘法与布局优化

**What to build:** 保留 Beaver E/F 的一次批量打开，将服务方可合并的矩阵乘积用分配律合为一次，并通过匹配矩阵布局减少复制，让同一乘法结果使用更少本地工作。

**Blocked by:** 01：补齐计量并冻结匹配基线。

**Status:** resolved

- [x] 服务方将 E 乘本地 v 和 E 乘公开 F 合为 E 乘二者之和，本地 GEMM 从三次变为两次；客户端仍为两次，E/F 打开数量、通信和一次性三元组语义保持一致。
- [x] 所有运算保持模二次幂整数环、符号扩环、累加宽度和后续截断契约，不引入浮点矩阵乘法。
- [x] 使用匹配行列布局的矩阵视图和可复用输出存储减少 Eigen 输入输出复制，验证非方阵、不同环宽、负值、回绕以及输入输出别名情形。
- [x] 通过现有 Beaver 矩阵回归、真实两方重构和 SCI 整数交叉检查；Batch=128 默认/非零隐状态单步诊断保持一致。
- [x] 代数合并、布局改进及其组合均可归因，保留原内核对照；记录本地计算和复制定位成本，不用实现内部调用次数代替协议结果测试。
- [x] 按每方四核和 1.5 GiB 预算测量受影响 MatMul 及单步行，先评估代数/布局收益再决定是否增加参与者；Eigen/OpenMP 显式并行计入同一配额。
- [x] 保存双方离线/在线/直接总成本、通信量和实际参与者；收益、持平和回退都如实报告。

## Answer

已交付服务方 `E(v+F)` 代数合并和匹配 Eigen 行列视图，分别可关闭。矩阵运算保持模二次幂整数环、符号扩环、累加与截断契约；E/F 的批量打开和一次性三元组数量保持一致。[真实两方 Beaver 回归](../../../project/our/results/runtime_optimization_20261010/beaver_all_tiles_correctness/summary.json)及日志核对 278824 个重构整数，覆盖实际单步形状、非方阵、环宽/回绕、tile、布局及输入输出别名。

[四种 kernel 控制及 AES 独立行](../../../project/our/results/runtime_optimization_20261010/kernel_aes/summary.json)共 7 个正式十查询行，[默认/非零隐状态诊断](../../../project/our/results/runtime_optimization_20261010/kernel_aes_diagnostics/summary.json)全部 16 行完成，每查询与 SCI 匹配 15 张量/172928 个整数。[归因报告](../../../project/our/results/runtime_optimization_20261010/kernel_aes/report.md)保留本地 compute、layout-copy、材料覆盖、通信、资源及直接查询分布；mapped GEMM 为单参与者，Eigen 内部并行关闭，最终队列另有实际参与者核验。

代数合并使该队列 online 中位数下降 2.64%，直接 query 中位数上升 1.64%；mapped 布局消除所测复制成本，但总时延没有改善；两者组合的 online 回退 3.31%。因此完成实现与验收，不宣称该局部内核带来端到端加速；保留原内核和独立开关。SDCF-09 继续延期。
