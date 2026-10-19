# 01: 补齐计量并冻结匹配基线

**What to build:** 通过现有两方运行入口，为 SCI、SCI_ours、SCI_Beaver、SCI_Beaver_segmented 输出可核对的真实后端、阶段成本、通信和资源记录，并冻结后续优化使用的匹配基线。

**Blocked by:** None (can start immediately).

**Status:** resolved

- [x] 保持冻结模型、整数语义、公开 ScalarMul 和原生 SCI 激活算法；修正转换来源日志，记录实际比较域、段宽、调用身份、材料消费量、请求与实际 worker 数。
- [x] 覆盖 setup、offline/material-ready、input/ready、online、output、直接 query total 和 session total；转换细分材料装载/反序列化、查询准备、local eval、scan、rounding、B2A 与释放。输入确认后发生的装载和等待计入 online；缺失子项不得当作零。
- [x] 两方按查询对齐后取时延最大值，通信取双方发送字节之和；并行工作时间与墙钟时间分开，保留中位数、min/max、离散程度、首查询和峰值 RSS。
- [x] 冻结 Release 构建、模型输入、服务方 CPU 0,2,4,10、客户端 CPU 6,8,12,16、每方四核和 1.5 GiB RLIMIT_AS；正式测量串行进行，不与编译或重负载并发。
- [x] 固定无预热且每次重新生成一次性材料：16 位 DCF count128 每行 30 次、count12800 每行 10 次；signed div_pow2 输入/输出 32 位、shift14、count128 每行 5 次；Batch=128 单时间步每行 10 次。基本原语/转换每行 120 秒，宽域资源和单步每行 600 秒。
- [x] 复测四路后端基线，转换 fixed-4、activation=1，适用行覆盖 cGGM 上限 1/4；旧五查询样本仅作为有来源的历史背景。主网络为不整形 localhost；通信专项补充 sirnn_lan（RTT 0.8 ms，3016 Mbps）和 wan（RTT 2 ms，100 Mbps）。
- [x] 通过现有原语/完整转换 oracle 及默认、非零隐状态单步整数诊断；正式测量关闭中间张量揭示。保存原始样本、运行配置、命令、构建身份与计量边界。
- [x] 失败行保留 oom/timeout/unsupported/unavailable 和 null 完成时延；资源受限不伪装成功。验收限于现有原语及单时间步，SDCF-09 保持延期。

## Implementation Evidence

- 已实现真实转换来源、每次转换的域宽/段宽/调用身份、材料消费量和实际 cGGM 参与者记录；转换计量增加材料读取、查询准备、本地求值、scan、释放及调用总时延。
- 现有两方汇总保留逐查询对齐后的时延分布和直接 session/query 总计；明示嵌套子任务时间不能相加作为墙钟时间，原生协议未计量子项保持缺失。
- 新增 `runtime-opt-01` 冻结 profile：四路后端、fixed-4、单步十次新鲜查询、对应原语重复数与公开资源预算；默认和非零隐状态诊断均单独运行。
- 本地验证：两方 bridge 的 segmented/full-tree 各 48 个已知整数、8 份一次性材料通过；11 项原语 runner 测试（包含真实两方及冻结计划）通过；14 项 runtime Python 测试通过、1 项仅限原生 binary 的测试跳过；后端可用性检查通过；组合后端 Batch=2 单步计量 smoke 通过。
- 正式基线已完成：六条单时间步行每行 10 个新鲜查询；四路后端在默认及非零隐状态各完成一次诊断，所有候选与 SCI 的 15 张量、172928 个整数逐值一致。
- DCF 的 count128/count12800 全部通过；完整转换 fixed-2/fixed-4、workers1/4 均完成。初次 fixed-4/worker1 行因端口占用未进入计算，原失败保留，独立补测通过。已修复启动前全部原语通道探测与失败原因记录。
- 结果见 runtime_optimization_20261010 的 baseline 和 baseline-port-recheck 记录；基线说明与原始样本由本批结果索引定位。通信优化的补充网络对照在 02/10 验收，01 冻结其参数。
