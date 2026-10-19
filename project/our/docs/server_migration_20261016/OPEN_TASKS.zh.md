# 新服务器迁移后的未完成任务

整理时间：2026-10-19。以下仓库路径均相对仓库根目录。实验状态依据 `project/our/results/precision_motivation_final/results/REPORT.zh.md`、实验原始状态文件和 `project/our/docs/server_migration_20261016/README.zh.md`。本文件区分真实测量、静态资源预检和未运行项目；没有记录的结果保持缺失。

## 状态总览

| 工作 | 状态 | 后续 |
| --- | --- | --- |
| ResNet-50 同环位宽精度 | 16/20/24/28 位完整验证已完成；32/12 复用既有结果 | 无需重跑 |
| ResNet-152 同环位宽精度 | 16/20/24 位校准候选尚未全部解析；28 位校准中断；低位宽完整验证未开始 | 恢复校准、确定尺度、跑完整验证 |
| RNN 混合位宽转换分析 | 已完成现有 batch=128 日志分析 | 无需重跑；按测量边界引用 |
| 分段原语消融 | 复用旧位宽扫描并完成缺失 28 位、batch=128 补测 | 无需重跑；保留单树资源失败记录 |
| CNN 整网成本 | 尚无有效整网性能测量；SCI 预处理的两项资源预检不可用 | 新服务器上测 SCI 与 SCI+Ours；评估预处理所需磁盘 |
| 新服务器复现与清理 | 未完成 | 传输输入、重建依赖、验证运行后再清理旧环境 |
| GitHub 推送 | 本地无 remote | 确认目标仓库 URL 后推送已审阅提交 |

## 实验一：ResNet-152 精度扫描

训练校准样本已冻结为 ImageNet 官方训练集每类两张，共 2000 张；完整验证集为 50000 张。ResNet-50 的 16/5、20/7、24/8、28/9 低位宽结果已完成。ResNet-152 的 32/12 参考结果已复用，但四个低位宽验证结果仍为空。

当前 ResNet-152 候选解析数见 `project/our/results/precision_motivation_final/results/experiment1_accuracy.csv`：16 位 16/16、20 位 20/20、24 位 19/24；28 位 `w28_f11` 校准在 1048/2000 张处中断。对应 `accuracy.json` 标记为 `interrupted`，不是完整校准点。`scan_resume_next_guard.json` 记录了宿主机低内存保护停止，Linux OOM kill 为 0。当前没有发现扫描进程；`width_scan/results.json` 的 `running` 是旧批次状态，不能当作仍在运行的证明。

待办：

1. 在新服务器确认校准数据、权重和断点文件身份，检查设备内存和 cgroup 限制后，从现有 `width_scan` 断点续跑；使用新 guard 输出文件，不覆盖旧日志。
2. 完成 28 位校准候选，依据既定的 2000 张校准集规则确定各位宽小数位；确认 16/20/24 位候选解析和选择记录一致。
3. 对选出的 16、20、24、28 位配置分别完成 50000 张验证集评估，生成 Top-1/Top-5、预测一致率及各类回绕计数和分母。
4. 更新 `project/our/results/precision_motivation_final/results/experiment1_accuracy.csv`、`project/our/results/precision_motivation_final/results/REPORT.zh.md` 和汇总状态。32/12 仍作为复用参考；不要把 20/24/28 位的精度损失全部归因于回绕，现有 ResNet-50 结果也显示分辨率误差的影响。

恢复命令和参数见 `project/our/results/precision_motivation_final/EXECUTION.zh.md`。该文件中较早的状态段已过时，恢复前以最新 checkpoint、guard、进程列表和此处引用的结构化结果交叉核对。

## 实验二：RNN 混合位宽转换分析

此项已完成，无需新服务器重跑。来源为 Google-30/FastGRNN 单时间步、batch=128 的 10 次现有执行记录：每 query 有 6528 次 23 位逻辑比较和 51200 次 31 位逻辑比较；含高位域转换调用占全部转换串行阶段工作量的中位数约 44.85%。

该比例包含相关调用的低位域与转换工作，不是纯高位 DCF 耗时比例或整步墙钟占比。详细算子、分域统计和测量边界见 `project/our/results/precision_motivation_final/experiment2/REPORT.zh.md`；论文中应保留该限定。

## 实验三：原语与 CNN 整网成本

### 原语分段消融

原语部分已完成：复用 ticket10 的 batch=100 扫描记录，另完成输入 28 位（内部比较域 27 位）的 batch=128 补测。2、4、8 段配置完成；单树配置因资源失败不可用。不同 batch 的结果不合并计算加速比，失败配置不填写为零开销。明细见 `project/our/results/precision_motivation_final/primitive_ablation/README.zh.md` 和 `project/our/results/precision_motivation_final/results/primitive_new_batch128.csv`。

### CNN 整网对比

当前固定配置为 32/12、batch=1、优化后的 Ours 构建。SCI_preprocessed 两个模型仅完成公开图资源预检，协议没有启动；另外四项没有合格实测：

| 模型 | SCI | SCI_preprocessed | SCI+Ours（Beaver） |
| --- | --- | --- | --- |
| ResNet-50 | 待测 | 资源预检不可用 | 待测 |
| ResNet-152 | 待测 | 资源预检不可用 | 待测 |

SCI_preprocessed 现有库存格式的线性 COT 静态估算下界为 ResNet-50 双方合计约 2.865 TiB、ResNet-152 约 8.070 TiB；当前实验机 `/tmp` 可用空间约 699 GiB，且原预检另受每方 50 GiB 显式预算限制。这些是线性 COT 的资源下界，不是完整库存需求、协议实测或理论通信下界。新服务器上先核对可用临时磁盘，并按可用空间明确设置每方预算；不满足时保留 `resource_unavailable` 状态。

待办：

1. 将两个固定 32/12 模型导出、验证集输入、SCI 依赖和当前优化版二进制的构建输入迁移或重建。
2. 以 batch=1、`round_trace=off` 和 `optimization-profile=optimized` 分别测 ResNet-50/152 的 SCI 与 SCI+Ours；每个结果核对整数输出，并保存 setup、offline、online、总墙钟、双方通信、CPU、内存和材料磁盘用量。
3. 根据新服务器磁盘容量决定是否运行 SCI_preprocessed。若运行，须使用相同模型、输入和计量口径，并给出已配置材料预算；若仍不可用，只报告资源预检及其边界，不用估算替代性能行。
4. 更新 `project/our/results/precision_motivation_final/results/cnn_performance.csv`、`project/our/results/precision_motivation_final/results/cnn_phases.csv` 和 `project/our/results/precision_motivation_final/results/REPORT.zh.md`。当前预检失败不是协议失败，也不是可用于计算加速比的测量。

CNN ReLU 已完成代码和公开图审计，路径与计数见 `project/our/results/precision_motivation_final/experiment3_relu_audit.zh.md`：每个 ReLU 元素为一次 DReLU 后接一次 bit×x 选择；它是调用链审计，不是端到端性能数据，无需为此重复改造或单独补测。

## 新服务器准备

以下工作在正式测量前完成：

1. 从 `external_revisions.json` 指定的版本重建外部 EzPC/SCI、SiRNN/EdgeML 依赖；本仓库未跟踪的外部工作树补丁需按迁移文档逐项适配，不能直接沿用包含旧机器绝对路径的配置。
2. 检查 cgroup 版本和 `memory_guard.py` 兼容性，设定可复现的整棵进程树内存限制；确认 `/tmp` 容量、CPU 亲和性、GPU/CUDA 组合。
3. 按 `required_inputs.tsv` 迁移数据与固定输入。至少包括完整 ImageNet validation（约 6.4 GiB）、2000 张校准样本（约 1.2 GiB）、ResNet 导出参数（约 1.3 GiB）、RNN 冻结 fixture/模型输入及非零 hidden-state 资格输入。权重、数据、模型参数、断点和构建产物不属于 GitHub 源码包。
4. 重建核心库、SCI bridge、RNN runner、CNN runner 和 CUDA 扩展；执行相应正确性/资格测试后再启动长实验。构建命令和依赖版本依据见 `README.zh.md`、`environment.observed.tsv`。

## 发布与清理

截至整理时，本地分支为 `integration/paper-folded-activation-lut`，尚未配置 Git remote，也没有确认的 GitHub 仓库地址。代码与本任务文档按迁移白名单准备；推送项待取得目标仓库 URL。仓库根目录没有共享 `.gitignore`，新服务器上应继续按白名单管理源码，避免将数据、权重、虚拟环境和构建产物加入 Git；后续可依据迁移清单补一份精确规则，避免宽泛忽略必需 fixture。

`cleanup_candidates.tsv` 只列出待评估的缓存、环境、下载重复件、构建目录和可冷归档轨迹，不是删除指令。只有在新服务器成功构建并核验所需实验输入、结果证据已有独立副本后，才按每条记录注明的条件逐项清理；目前未删除任何路径。
