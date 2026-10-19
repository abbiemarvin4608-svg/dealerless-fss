# ResNet-50 / ResNet-152 数值合同与现有证据

核查日期：2026-10-15。当前配置为完整 torchvision `IMAGENET1K_V1` 模型、原始 1000 类分类头，统一 32 位环、12 位小数；实数值对应整数码除以 4096。精度使用 ImageNet-1K validation 全部 50,000 张图片，性能使用固定的单样本整数输入。

## 逐算子规则

记 `wrap32` 为模 `2^32` 后解释成有符号 32 位整数。输入、权重、偏置及各层输出均使用同一尺度 `2^-12`；图中没有逐层位宽或小数位覆盖。

| 对象或算子 | 当前规则 |
| --- | --- |
| 输入、权重、偏置编码 | 浮点乘 4096，round-to-nearest/ties-to-even，再 `wrap32`。 |
| BatchNorm | 推理前以 FP64 折入卷积权重和偏置，再编码；没有单独的运行时 BN。 |
| 卷积、全连接 | `wrap32(floor(wrap32(sum(x_code*w_code))/4096) + bias_code)`。乘积与累加在同一 32 位环中，先取模并作有符号解释，再向下截断 12 位，最后加偏置。 |
| 残差加法 | 两个同尺度整数码相加并 `wrap32`。 |
| ReLU | 对完整 32 位有符号码取 `max(x,0)`；符号结果为 1 位，选择的数据与输出为 32 位。 |
| MaxPool | 输出仍为 32 位、12 位小数。明文以最小有符号码填边；密态要求直接前驱为 ReLU，以零填边，并通过 `a+ReLU(b-a)` 求最大值，两者在该非负条件下等价。 |
| 全局 AvgPool | 49 个值先求环和，再对有符号和执行 `floor(sum/49)`；输出尺度不变。SCI 使用原生接口；ours 使用精确公开除法。 |

负数截断同样向下取整，例如 `floor(-1/4096)=-1`，不是向零取整。溢出按环回绕；ReLU 的负值归零以外没有饱和裁剪。AvgPool 的 ours 实现内部对小余数使用 16 位比较，最终选择和输出仍为 32 位；这不构成 ReLU16 或张量混合位宽。

明文 GPU 回放执行精确整数运算：卷积/FC 使用 base-128 limb 分解、int8 GEMM 的精确 int32 部分点积与整数重组。`int64` / `__int128` 是实现容器，不改变逻辑上的 32 位环；不会保留完整宽累加的高位再截断。浮点仅用于加载、BN 折叠、入口编码以及独立 FP32 参考。明文 Python/CUDA 与密态 C++ 是遵守同一合同的不同实现。

实现依据：[model.py](../../tools/imagenet_resnet/model.py)、[secure_driver.cpp](../../tools/imagenet_resnet/secure_driver.cpp)、[uniform_integer_gemm.py](../../tools/precision/uniform_integer_gemm.py)、[uniform_fixed_point_math.h](../../tools/precision/native/uniform_fixed_point_math.h)。

## 与既有 RNN 混合位宽的关系

既有 Google-30 / FastGRNN 原生图按张量和算子分配位宽、尺度，并包含扩环、截断和激活近似。现代冻结的 15 调用图中，W1 为 `16/f11 × 8/f5`，使用 24 位乘积、29 位累加，输出 `16/f10`；W2 为 `16/f10 × 16/f12`，使用 32 位乘积、36 位累加，输出 `16/f7`。Sigmoid 为 `16/f7 → 16/f14`，TanH 为 `16/f7 → 16/f7`；循环路径还含 8 位中间张量。这里 `f` 表示小数位数。实际调用见 [build.py](../../tools/segmented_dcf_runtime/build.py)，转换宽度推导见 [run.py](../../tools/segmented_dcf_runtime/run.py)。

本轮 ResNet 是统一 32/12 的同环配置，没有继承 RNN 的混合位宽分配，也没有采用旧 CNN 工单的 Q24.15/ReLU16 配置。当前结果可作为该合同的准确率和成本证据；若另做混合位宽或改用宽累加，需要生成新合同并重新完成对应精度评估。RNN 依据见[多位宽需求](../../REQUIREMENTS_MULTIWIDTH_DEALERLESS_INFERENCE.md)及[04–12 执行报告](../TICKETS_04_12_REPORT_CN.md)。

## 已完成的 50,000 张精度

| 模型 | FP32 Top-1 / Top-5 | 整数 Top-1 / Top-5 | Top-1 变化（百分点） | Top-1 预测一致率 |
| --- | ---: | ---: | ---: | ---: |
| ResNet-50 | 76.146% / 92.872% | 76.128% / 92.878% | -0.018 | 99.610% |
| ResNet-152 | 78.314% / 94.060% | 78.312% / 94.032% | -0.002 | 99.458% |

两份 `accuracy.json` 均为 `complete`、`formal`、`50000/50000`。R50 净多错 9 张（FP32 对而整数错 63，反向 54）；R152 净多错 1 张（81 / 80）。R152 从第 29,600 张恢复，所有样本恰好统计一次。原始记录：[R50](accuracy/resnet50_w32_f12/accuracy.json)、[R152](accuracy/resnet152_w32_f12/accuracy.json)。

本次逐字段比较确认，两模型的 `accuracy/*/export/graph.json` 与 `models/*/graph.json` 在将 batch 及形状首维归一为 1 后完全相同：R50 共 121 个节点，R152 共 359 个节点。已有导出记录中的参数身份一致；本次没有重新读取并散列大型参数二进制。不能将容器类型或 batch 差异视为数值合同变化。

## R50 SCI 已完成的单次密态协议

来源：[party1.json](performance/resnet50_SCI_q1/party1.json)、[party2.json](performance/resnet50_SCI_q1/party2.json)。双方均 `complete`，各执行全部 121 个节点，`queries=1`，`diagnostic=false`。独立读取最终参考 `audit/0121.bin`，核对 party2 的全部 1,000 个输出码，差异为 0。

输入为 `[1,3,224,224]` 固定 dummy 整数张量：种子 20261015，码范围 `[-16384,16383]`，对应约 `[-4,4)`；不是一张有标签的 ImageNet 图片。输入阶段读取预编码整数并交换就绪字节，不包含 JPEG 解码和图像预处理。网络为未整形 localhost。

| 阶段 | 双方墙钟较大值（秒） | 双方进程 CPU 之和（秒） | 双方发送之和（字节） |
| --- | ---: | ---: | ---: |
| setup | 0.574603547 | 0.7180329 | 457,906 |
| offline | 0.000002762 | 0.0000022 | 0 |
| input | 0.001982659 | 0.0029932 | 2 |
| online | 1916.886539841 | 6338.6660960 | 449,284,194,717 |
| output | 0.000843425 | 0.0008654 | 8,000 |
| 上述阶段合计 | 1917.463972234 | 6339.3879897 | 449,284,660,625 |

总通信约 418.428947 GiB。通信只加双方发送量，不再加接收量；各阶段双方发送总量与接收总量相等。墙钟合计采用“各阶段双方较大值之和”，不是外层进程总墙钟。SCI 的材料生成计入 online，offline 近零不表示预处理免费。

旧 `STATUS.json` / `RECOVERY.json` 仍为重启前的 `running`，尚无 `ROUNDS.json`；协议原始结果已完成，轨迹统计收尾被重启打断。本轮已生成 `OUTPUT_AUDIT.json`，全部 1,000 个输出整数零差异，并将经过完整层、阶段和双方字节核验的时间/通信统计恢复为 `COST_MEASUREMENT.json`，已纳入 `METRICS.json` 与 `REPORT.xlsx`。原始计划、状态和轨迹保留；该成本记录为 `complete`、`measurement_mode=time_and_bytes`，实际运行仍标记 `round_trace=on`。这是单次测量，没有重复分布；因果轮数与基于 RTT 的 LAN/WAN 投影均为缺失值。

原先带详细轨迹的批次计划为 `R50/R152 × SCI/SCI_ours_beaver` 共 4 行；协议完成 1/4，因果轨迹收尾完成 0/4，时间/字节已恢复合格 1/4。新的 `*_cost_q1` 批次于 2026-10-15 16:39（Asia/Shanghai）启动，使用相同统一 32/12 数值合同与固定输入，关闭详细轨迹（`round_trace=off`），并于 16:49:47 按用户要求暂停。[批次进度](performance/COST_BATCH_STATUS.json)与[启动记录](memory_protection/COST_LAUNCH_STATUS.json)均为 `paused_by_user`，首行 ResNet-50 / SCI 为 `interrupted`；双方日志最后完成第 48 号节点 `layer2.3.relu1`（49/121 个节点），其余三行尚未启动，中断行不纳入完整性能比较。README 另述的 `SCI_ours` 不是这两个批次的已排定行。

本轮[内存保护记录](memory_protection/cost_guard_20261015T163935_1792053575267100984.json)为 `interrupted`，实验后代已清理；RAM 上限为 2,147,483,648 字节，RAM 加 swap 上限为 2,684,354,560 字节，采样峰值约 504.50 MiB，kernel 峰值约 504.73 MiB，限额触发数和 OOM 增量均为 0。当前等待逐张量/逐算子的混合位宽方案和 SiRNN 对照确定；不再继续本批统一 32/12 性能实验。上述精度表及旧完整协议记录保留为该数值合同的历史证据，新的混合位宽合同需独立记录并重新评估。

建议最小汇报表：精度表保留“模型、冻结数值配置、样本数、FP32 与整数 Top-1/Top-5、agreement”；性能表保留“模型、后端、batch/query 数、输入类型、网络、setup/offline/online/total 墙钟、阶段通信、CPU 时间、输出核对、轮数/状态”。两表按同一模型和数值合同关联，50k 精度复用当前已完成结果。
