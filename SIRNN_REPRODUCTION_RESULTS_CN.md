# SiRNN Google-30 复现结果与方法

保存日期：2026-10-07。本项目内的结果与方法快照保存在
[`reproduction_results/sirnn_google30_20261007`](reproduction_results/sirnn_google30_20261007/README_ARCHIVE_CN.md)，包括训练权重、量化模型、脚本、FP32 预测和 SCI 30 条两方原始日志。

完整命令、环境、数据校验和日志位于
[`/home/jiang/EzPC/SIRNN/reproduction/README_CN.md`](/home/jiang/EzPC/SIRNN/reproduction/README_CN.md)。

## 复现对象

复现链为：

```text
Speech Commands v0.01
  -> EdgeML/sirnn FastGRNN 训练
  -> SeeDot 混合位宽定点量化
  -> EzPC C++ 生成
  -> SCI 两方推理
```

公开仓库没有找到作者发布的 Google-30 预训练权重或量化模型，所以本次下载官方代码和数据，并使用官方 EdgeML 实现重新训练。生成的权重是本机训练结果，不能称为论文作者原始权重。

## 方法

1. 下载 Speech Commands v0.01，校验 MD5 为 `3cd23799cb2bbdec517f1cc028f8d43c`。
2. 按 EdgeML `sirnn` 分支的 `process_g30.py` 提取 99×32 log-fbank 特征。90 条分层样本与官方脚本逐元素一致，最大误差为 0。
3. 使用官方 `FastGRNNCell` 和 `FastTrainer`：seed=42、hidden=100、W/U rank=16/35、W/U 保留比例=0.2、batch=100、Adam 初始学习率=0.01，第 200 轮衰减为原来的 0.1，训练 300 epoch。官方压缩阶段按 test accuracy 选择 checkpoint，本次选中第 205 轮，训练耗时约 3,422 秒。
4. 使用训练集 mean/std 做归一化，SeeDot 用 512 条覆盖 30 类的训练样本校准，保留全部 6,835 条 test。SeeDot 执行 DDS/VBW 位宽搜索，参数存储使用 8/16 位，累加和输出包含 32 位，输入 `scaleForX=-11`。
5. 由 SeeDot 生成 `predict.ezpc`、`model_fixed.h`、`vars_fixed.h` 和输入 CSV；调用 EzPC 生成 SCI C++，链接 SCI-OT 并编译 `Google-30`。
6. 生成封装针对本机调整 CMake 最低版本为 3.13，以同步编译与返回码检查替代上游异步调用，并限制 SeeDot 并发预测线程以控制内存。对 SCI ArgMax 做了必要的安全中间位宽修正：8 位 logits 先符号扩展到 16 位后比较，类别索引按实际 8 位重构。这避免 logits 约 ±90 时 8 位差分溢出和重构位宽不一致，保留相同模型参数、输入和量化尺度。
7. SCI 两方测试从最终 SeeDot test CSV 中固定选择每类一条真实样本，共 30 条，要求 server/client 返回码为 0，并逐条与 SeeDot 定点预测比较。

## 实测结果

| 层次 | 数据量 | 结果 |
|---|---:|---:|
| TensorFlow FP32 test | 6,835 | 6,177/6,835，90.3731% |
| TensorFlow FP32 validation | 6,798 | 6,073/6,798，89.3351% |
| SeeDot 定点 test | 6,835 | 6,134/6,835，89.7440% |
| SCI 两方安全推理 | 30（覆盖 30 类） | 30/30 与 SeeDot 一致；双方返回码均为 0 |

SCI 30 条子集相对真实标签的准确率为 27/30=90%，与 SeeDot 的预测一致率为 100%。每个样本使用 4 个线程；跨样本中位运行时间约 25.101 秒，中位通信量约 418.381 MiB，典型轮数 52,965；计时范围是 SCI base OT setup 和 sync 之后的协议阶段，wall time 单独保存在 JSON。完整 test 的安全推理尚未运行，因此不能把 30 条结果描述为 6,835 条安全准确率。上述时间是跨样本描述，不是重复性能实验的最终结论。

固定子集的零起始 test 行号如下，顺序对应类别 0–29：

```text
2,4,6,16,1,23,27,24,11,21,19,17,41,50,42,
7,26,20,8,63,144,60,3,43,0,47,5,28,133,92
```

逐条真实标签、SeeDot/SCI 预测、音频路径、返回码和协议指标另存于
[secure_samples.csv](reproduction_results/sirnn_google30_20261007/secure_samples.csv)。

## 固定版本与环境

| 组件 | 本次版本 |
|---|---|
| EzPC | `f24bf3e022dde3493cf0ffe3ebd92dd73936b734` |
| EdgeML `sirnn` | `7e5f88db82b28fabb447d4885c8e88a945b8fd1d` |
| 预处理 | Python 3.8.10，NumPy 1.24.4，SciPy 1.10.1，python_speech_features 0.6 |
| 训练 | Python 3.7.7，TensorFlow 1.15.5，NumPy 1.18.5，SciPy 1.4.1 |
| SeeDot | Python 3.7.7，NumPy 1.19.5，ANTLR runtime 4.7.2，Bokeh 2.1.1，Jinja2 3.0.3 |

训练与 SeeDot 使用不同虚拟环境。完整依赖清单已归档到快照中的
`logs/train_environment.txt`、`requirements_seedot.lock.txt` 和 `requirements_preprocess.txt`。

这里的 8/16/32 位类型描述来自生成代码。SCI 协议还会按运算进行内部扩宽；例如
16×8 位的 W1 点积在 K=32 时使用 29 位累加，再缩放到 16 位输出。
输入位宽、小数精度、各矩阵内部累加位宽和可控制范围，详见
[定点位宽补充说明](SIRNN_FIXED_POINT_EXPLANATION_CN.md)。

数据为 train=51,088、test=6,835、validation=6,798，各行含一个标签和 3,168 个特征。官方下载地址为
<https://storage.googleapis.com/download.tensorflow.org/data/speech_commands_v0.01.tar.gz>；代码来源为
<https://github.com/microsoft/EdgeML/tree/sirnn> 和 <https://github.com/mpc-msri/EzPC/tree/master/SIRNN>。

## 重放命令

```bash
cd /home/jiang/EzPC/SIRNN

# 已有产物时检查完整 FP32 -> SeeDot -> EzPC -> SCI 链
bash reproduction/reproduce_google30.sh inference

# 使用已保存的模型，也可以按阶段执行
bash reproduction/reproduce_google30.sh float
bash reproduction/reproduce_google30.sh quantize
bash reproduction/reproduce_google30.sh build
bash reproduction/reproduce_google30.sh secure
bash reproduction/reproduce_google30.sh summary
```

需要重新下载/预处理、训练并完成推理时，使用新的模型目录：

```bash
cd /home/jiang/EzPC/SIRNN
GOOGLE30_MODEL_DIR=/home/jiang/EzPC/SIRNN/reproduction/models/Google-30-retrain \
  bash reproduction/reproduce_google30.sh all
```

`train` 拒绝覆盖已有模型目录。`inference` 重放 FP32、量化、生成、编译及默认 30 条安全推理；会更新原目录的中间产物和报告。本项目中的快照保存的是此次已通过的结果。

快速重跑三条并写入独立文件：

```bash
cd /home/jiang/EzPC/SIRNN
/usr/bin/python3 reproduction/scripts/run_google30_2pc.py \
  --binary reproduction/FastGRNN/build/bin/Google-30 \
  --predictor-dir reproduction/SeeDot/temp/Predictor \
  --sample-indices 0,1,2 --threads 4 --timeout 600 \
  --output reproduction/google30_2pc_quickcheck.json
```

本次 512 条校准是控制本机内存和运行时间的改动。需要全部训练样本 profiling 时，用 SeeDot 虚拟环境调用 `prepare_seedot.py`，指定 `--calibration-size 51088`，然后重新量化、编译和验证；该结果会与本次校准配置区分记录。

## 证据文件

- [已归档的完整方法](reproduction_results/sirnn_google30_20261007/README_CN.md)
- [已归档的最终结果表](reproduction_results/sirnn_google30_20261007/RESULTS_CN.md)
- [已归档的汇总 JSON](reproduction_results/sirnn_google30_20261007/summary.json)
- [已归档的 SCI 逐样本结果](reproduction_results/sirnn_google30_20261007/google30_2pc_results.json)
- [已归档的训练与数据说明](reproduction_results/sirnn_google30_20261007/DATA_TRAINING.md)
- [原目录的 SCI 可执行文件](/home/jiang/EzPC/SIRNN/reproduction/FastGRNN/build/bin/Google-30)

`summary.json` 当前状态为 `passed`，包含提交版本、数据规模、训练配置、FP32/定点结果、位宽和 SCI 子集统计。

## 复现限制

- 权重是按官方实现本地训练的模型，不是作者未公开的预训练权重。
- FastTrainer 按官方流程使用压缩阶段 test accuracy 选择 checkpoint，因此同时报告 validation，test 不能称为完全独立评估。
- SCI 只验证了每类一条的 30 条真实样本；完整 6,835 条安全推理尚未完成。
- 量化使用 512 条训练样本校准。输入范围保留官方 fixer 由完整 train/test 计算边界的行为；量化配置也不能描述为仅使用训练数据的严格独立评估。
- 旧 `project/our` Google-30 路径使用 v0.02、hidden=128 和不同特征，不能直接与本次 v0.01 FastGRNN 数字合并比较。
