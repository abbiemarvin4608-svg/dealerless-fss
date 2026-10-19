# ImageNet ResNet-50 / ResNet-152 实验

本轮接入完整 torchvision ResNet-50 和 ResNet-152，使用 `IMAGENET1K_V1` 官方权重及其 `weights.transforms()`，保留原始 1000 类分类头。性能使用相同模型参数和 `[1,3,224,224]` dummy 输入；精度使用官方 ImageNet-1K validation 全部 50,000 张图片。

## 数值合同

主配置使用统一 32 位算术环和 12 位小数。可选环宽为 8、16、24、32 位，预设小数位分别为 0、4、8、12；小数位不是环宽，不要求为 8 的倍数。导出图的全局 `bits/frac` 供所有模型算子使用，当前没有逐层混合位宽。逐算子规则见 `NUMERIC_CONTRACT.zh.md`。

推理 BatchNorm 先折入卷积浮点权重和偏置，再量化。输入和参数使用 round-to-nearest/ties-to-even 编码，按所选环宽取模。卷积和全连接在该环中累加，解释为带符号整数后向下截断，再加同小数位偏置并取模；残差加法取模，ReLU 和 MaxPool 使用带符号比较。全局平均池化先环求和，再对公开窗口大小执行精确向下除法。

此合同与现有 `precision` 工具默认的宽累加饱和模式不同。两种数值规则的精度不能混用。GPU 回放复用现有 base-128 整数 limb/GEMM；浮点 fake quantization 不能替代精确整数回放。

## 方法和计量

- `SCI`：SCI 原生 OT 算子，材料生成归实际执行阶段。
- `SCI_ours`：相同 SCI 线性层，替换为已有 ours 比较、BitXA ReLU 和截断。该配置作为非线性替换对照。
- `SCI_ours_beaver`：接入已有矩阵 Beaver、ours 比较、BitXA ReLU 和截断的配置；材料生成必须真实执行。

上一轮 sigmoid/tanh 的 LUT、clip22 等优化不直接用于 ResNet 的 ReLU。性能必须实际执行完整网络，不以算子微基准相加代替端到端结果。

记录 setup、输入、离线、在线、输出的进程 CPU 时间、本机墙钟、双方发送字节和因果通信轮数。通信量为两方发送之和，同一消息不重复计算接收量。固定公开图下通信记录不依赖秘密像素；墙钟仍受随机性、系统负载和网络状态影响，需要重复测量。

CPU 时间和本机墙钟包含协议与材料文件读写；`round_trace=on` 的历史运行还包含详细因果轨迹记录。双方采用同一构建、相同线程配置和计量方式。新的 `*_cost_q1` 运行使用 `round_trace=off`，只计量时间和双方发送字节，并核对输出整数，避免生成和后处理数十 GiB 的详细轨迹。该模式不报告因果轮数或 LAN/WAN 延迟投影；两种轨迹模式的墙钟不直接计算方法比值。

LAN/WAN 延用上一轮通信模型：LAN 1000 Mbps/RTT 0.5 ms，WAN 400 Mbps/RTT 4 ms。若没有实际网络整形，明确标记为 `8×bytes/bandwidth + rounds×RTT` 的通信估算，不将其称为完整实测推理延迟。

下载、精度和性能各自保存进度与原始记录；未完成的 Top-1/Top-5 或整网性能使用缺失值，不使用 ImageNette 或缩小图的结果填补。

## 数据与复现

验证集已准备在 `data/val`，共 50,000 张 JPEG、1,000 类，每类 50 张。下载源为公开 Kaggle ImageNet validation 镜像，使用官方 devkit 标签映射至 torchvision 的排序 WNID 类别索引。压缩包 MD5、ZIP CRC、图片解码、全部官方文件编号和标签均已核对；详细来源与校验记录见 `data/MANIFEST.json`。模型使用 torchvision 官方 `IMAGENET1K_V1` 权重，不混用 V2。

以下命令在仓库根目录运行。`evaluate.py` 每 200 张保存断点，再次运行同一命令会自动续跑；不添加 `--limit` 才能生成 50,000 张的正式精度记录。

```bash
project/our/tools/precision/.venv/bin/python -u project/our/tools/imagenet_resnet/evaluate.py \
  --model resnet50 --bits 32 --frac 12 --batch-size 8 --workers 2 --checkpoint-every 200 \
  --val-dir project/our/results/imagenet_resnet_20261015/data/val \
  --output-dir project/our/results/imagenet_resnet_20261015/accuracy/resnet50_w32_f12
```

ResNet-152 将命令中的 `resnet50` 改为 `resnet152`。每个输出目录同时保留 FP32 和整数预测、配置身份与断点，`accuracy.json` 中 `status=complete` 且 `evidence_scope=formal` 表示完成全部验证集。

完整网络性能使用 `models/<model>_w32_f12` 下固定的模型和 dummy 输入。运行目录必须新建，避免覆盖之前的证据。

正式整网运行必须位于实际生效的内存保护内。WSL 每次重启后，先重新建立 2 GiB RAM 加 512 MiB swap 的实验 cgroup（仅此设置步骤需要管理员权限），再以普通用户启动：

```bash
sudo python3 project/our/tools/imagenet_resnet/memory_guard.py configure \
  --uid "$(id -u)" --gid "$(id -g)" --memory-mib 2048 --swap-mib 512

python3 project/our/tools/imagenet_resnet/memory_guard.py run \
  --output project/our/results/imagenet_resnet_20261015/memory_protection/reproduce_cost_guard.json -- \
  python3 project/our/tools/imagenet_resnet/run_secure.py \
  --binary project/our/results/imagenet_resnet_20261015/build_osrng/resnet_secure \
  --graph project/our/results/imagenet_resnet_20261015/models/resnet50_w32_f12/graph.json \
  --parameters project/our/results/imagenet_resnet_20261015/models/resnet50_w32_f12/parameters.bin \
  --input project/our/results/imagenet_resnet_20261015/models/resnet50_w32_f12/audit/input.bin \
  --backend SCI_ours_beaver --queries 1 --timeout 10800 --round-trace off \
  --output project/our/results/imagenet_resnet_20261015/performance/resnet50_SCI_ours_beaver_reproduce

python3 project/our/tools/imagenet_resnet/audit_secure.py \
  --directory project/our/results/imagenet_resnet_20261015/performance/resnet50_SCI_ours_beaver_reproduce \
  --oracle project/our/results/imagenet_resnet_20261015/models/resnet50_w32_f12/audit --final-only

python3 project/our/tools/imagenet_resnet/summarize.py \
  --directory project/our/results/imagenet_resnet_20261015
```

两模型、SCI 与 SCI+ours(Beaver) 各执行一个完整输入，使用相同固定图与输入、统一关闭详细轨迹，并串行运行：

```bash
python3 project/our/tools/imagenet_resnet/memory_guard.py run \
  --output project/our/results/imagenet_resnet_20261015/memory_protection/cost_guard.json -- \
  python3 project/our/tools/imagenet_resnet/run_performance.py \
  --results project/our/results/imagenet_resnet_20261015 \
  --binary project/our/results/imagenet_resnet_20261015/build_osrng/resnet_secure \
  --round-trace off --port 42000 --resume
```

[performance/COST_BATCH_STATUS.json](performance/COST_BATCH_STATUS.json) 保存该批次进度，`*_cost_q1` 保存各行原始计量。每次启动保护器须选择新的报告文件名；已完成行可以复用，未完成的协议不能从中间网络消息恢复。

正式性能运行不写中间秘密份额，完成后比较全部 1,000 个输出整数与同输入的明文整数回放。算子逐层核对使用 `secure_validation` 中独立 Python 大整数参考的小图；`conv1_pilot` 保留完整尺寸首层的额外核对。这些诊断运行不进入正式整网性能表。

`resnet_secure` 与同目录的 `libcnn_ours.so` 共同构成运行版本；计划文件记录两者、图、权重、输入的身份。性能进程分别绑定不相交的 CPU 核；精度回放使用 GPU。统计表中的计算时间为该实际主机配置下的进程 CPU 时间。

原生 SCI 与 ours 桥接统一定义 `EMP_USE_RANDOM_DEVICE=1`，使用 SCI 已有的 `std::random_device` 默认种子分支。此配置消除了 SCI 原始默认分支中未检查 RDSEED 成功状态的问题，不改变定点算术或通信协议。正式性能使用 `build_osrng`；修复前的整网运行单独保存在 `history/pre_osrng`，不进入正式对比。

该主机的 libstdc++ 默认随机源优先使用带成功检查和重试的硬件随机指令，失败时抛出异常；不能将其描述为仅使用操作系统随机源。具体核查见 `IMPLEMENTATION_AUDIT.zh.md`。

本批性能配置各运行一个完整 query，表中 `single` 不表示延迟分布。协议双方使用逻辑 CPU `0..3` 和 `4..7`，分别对应两个物理核；并行 GPU 精度回放的 CPU 亲和性为 `12,14,16,18`，物理核不重叠。计时来自该实际主机负载，未固定 CPU 频率。

整网因果轨迹由 `round_trace_stream.py` 调用经过首层 SCI/Beaver 完整轨迹对照的原生统计器，保持原因果发送轮数定义。统计器使用临时磁盘和有界缓存，默认核验语义作用域但不求逐作用域轮数；离线、在线等完整阶段的轮数均正常统计。`ROUNDS.json` 保存所用统计器版本及其身份，`round_analyzer_validation` 保存核对证据。

`--timeout` 限制整个协议执行；`--offline-budget-seconds` 在 Beaver 材料调用之间检查。`--spool-budget` 分别限制 CNN 非线性与 Beaver 库存，不覆盖既有截断桥接库存；本批使用 0 并监控实际剩余磁盘。诊断图和失败记录不参与正式方法比值。

## 完整验证集精度

两个模型分别使用全部 50,000 张验证图片，32 位环、12 位小数：

| 模型 | 指标 | FP32 实测 | 整数回放实测 | 整数减 FP32（百分点） |
| --- | --- | ---: | ---: | ---: |
| ResNet-50 | Top-1 | 76.146% | 76.128% | -0.018 |
| ResNet-50 | Top-5 | 92.872% | 92.878% | +0.006 |
| ResNet-152 | Top-1 | 78.314% | 78.312% | -0.002 |
| ResNet-152 | Top-5 | 94.060% | 94.032% | -0.028 |

ResNet-50 Top-1 预测一致率为 99.610%；FP32 正确而整数错误 63 张，反向 54 张，净增加 9 张错误。Logit MAE 为 0.017459482，RMSE 为 0.022960876，最大绝对误差为 0.421402931。

ResNet-152 Top-1 预测一致率为 99.458%；FP32 正确而整数错误 81 张，反向 80 张，净增加 1 张错误。Logit MAE 为 0.026044533，RMSE 为 0.033768187，最大绝对误差为 0.534023285。

原始记录为 `accuracy/resnet50_w32_f12/accuracy.json` 和 `accuracy/resnet152_w32_f12/accuracy.json`。ResNet-152 在环境重启后从第 29,600 张断点恢复，全部样本恰好统计一次；记录的评估用时不含停机时间。

## 内存保护与恢复

2026-10-15 宿主机重启后，改用 `memory_guard.py` 将实验进程树限制为 2 GiB RAM 加 512 MiB swap，并监控 Linux 与 Windows 的剩余内存。两模型精度已完整完成；旧 SCI ResNet-50 的时间与字节已从原始 JSON 恢复，并核对全部 1,000 个输出整数，无需重新分析巨量轨迹。限制来源逐行记录，不将恢复统计的限制误记为原协议执行的限制。证据与验证见 `MEMORY_DIAGNOSIS.zh.md`。

新的 4 行成本批次于 2026-10-15 16:39（Asia/Shanghai）启动，并于 16:49:47 按用户要求暂停，等待逐张量/逐算子的混合位宽数值方案及 SiRNN 对照确定。当前[批次进度](performance/COST_BATCH_STATUS.json)及[启动记录](memory_protection/COST_LAUNCH_STATUS.json)均为 `paused_by_user`，首行 ResNet-50 / SCI 为 `interrupted`；后续三行尚未启动。本轮 `round_trace=off`，没有生成详细轨迹。双方日志最后完成节点为第 48 号 `layer2.3.relu1`，即已完成 49/121 个节点；该中断行不作为完整性能测量。

[本轮保护器](memory_protection/cost_guard_20261015T163935_1792053575267100984.json)已为 `interrupted`，实验后代已清理。实际 RAM 上限为 2,147,483,648 字节、RAM 加 swap 上限为 2,684,354,560 字节；采样峰值约 504.50 MiB，kernel 峰值约 504.73 MiB，限额触发数和本轮 OOM 增量均为 0。本次停止源于用户暂停。已完成的两模型 50,000 张统一 32/12 精度和旧 ResNet-50 / SCI 完整协议证据继续保留；混合位宽方案确定后使用新的数值合同和独立结果目录。
