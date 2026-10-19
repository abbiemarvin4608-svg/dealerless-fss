# 三个开源项目复现与 CNN/RNN 实验说明书

本文记录 2026-10-06 至 2026-10-07 在本机完成的复现检查。路径以当前环境为准：
`/home/jiang/EzPC`、`/home/jiang/MP-SPDZ` 和 `/home/jiang/dealerless-lunwen/project/our`。

## 结论先行

| 项目 | 当前状态 | 可复现内容 | 主要限制 |
|---|---|---|---|
| SiRNN + SCI | **Google-30 已跑通** | Speech Commands v0.01、官方 EdgeML FastGRNN 训练、SeeDot 定点量化、EzPC 生成和 SCI 两方推理 | 公开仓库未提供作者预训练权重；安全两方验证覆盖 30 条分层样本，不等同于 6835 条完整 test 的安全推理 |
| CrypTFlow2/SCI | **构建成功** | SCI OT/HE、数学函数、CNN 网络和测试二进制已生成 | SCI 树没有 Google-30 FastGRNN/LSTM 端到端网络，不能据此声称完整 CNN/RNN 论文复现 |
| MP-SPDZ | **两方运行成功** | MASCOT / Semi2k 官方 tutorial、Semi2k 32 位私有输入验证 | CNN/RNN 等价模型与定点规则仍需对齐 |

SCI/CrypTFlow2 已构建，SiRNN Google-30 流程和 MP-SPDZ 官方两方 tutorial 已通过。SiRNN 的可重放产物、命令和限制记录在 `/home/jiang/EzPC/SIRNN/reproduction/README_CN.md`、`DATA_TRAINING.md`、`RESULTS_CN.md` 和 `summary.json`。

当前项目内另保存了 [SiRNN 结果与方法](SIRNN_REPRODUCTION_RESULTS_CN.md) 和
[SiRNN / MP-SPDZ 结果快照](reproduction_results/README_CN.md)，含 SiRNN 训练权重、量化模型、真实两方日志及 MP-SPDZ 本次重放证据。

## 1. SCI/CrypTFlow2

SCI 依赖 g++、CMake、GMP、MPFR、OpenSSL，以及仓库提供的 SEAL/Eigen。旧版 SCI 的 `cmake_minimum_required` 与 CMake 4.4 不兼容，使用仓库已有的配置缓存和构建目录即可继续构建：

```bash
cd /home/jiang/EzPC/SCI
cmake --build build --parallel 4
cmake --build build --target install --parallel 4
```

成功标志：`build/install/lib/libSCI-OT.a`、`libSCI-Math.a` 等静态库，以及 `build/bin/` 下的 `resnet50-OT`、`resnet32_cifar-OT`、`minionn-HE`、`sigmoid-OT`、`tanh-OT` 等二进制。完整构建本次通过；编译中有旧宏和格式字符串 warning，但没有编译错误。

SCI 自带的网络是 CNN/数学函数示例，不是 SiRNN 的 Google-30 模型。使用网络时需要一方先启动 server，再启动 client，例如：

```bash
./build/bin/resnet50-OT r=1 port=32000 < model-file
./build/bin/resnet50-OT r=2 ip=127.0.0.1 port=32000 < image-file
```

具体输入文件格式必须按对应 `networks/main_*.cpp` 检查，不能把任意图像文件直接当作模型输入。

## 2. SiRNN

官方流程是 `Speech Commands v0.01 -> EdgeML/sirnn FastGRNN -> SeeDot -> EzPC -> SCI`。本次已下载并校验数据，使用官方 EdgeML `sirnn` 分支重新训练（300 epoch，hidden=100、rankW=16、rankU=35），再用 SeeDot 生成混合位宽定点模型、`predict.ezpc` 和 `temp/Predictor` 头文件，最后生成并编译 SCI 两方程序。模型参数存储采用 8/16 位，生成代码包含 8/16/32 位整数类型（32 位用于累加/输出）。

实际结果如下：

- FP32 test：6177/6835，90.3731%；FP32 validation：6073/6798，89.3351%。
- SeeDot 定点 test：6134/6835，89.7440%；输入 scale 为 `-11`，校准集为分层 512 条训练样本。
- SCI 两方：30 条真实 test 样本，覆盖 30 个类别，30/30 与 SeeDot 预测一致，双方返回码均为 0。

使用已有模型的分阶段重放入口：

```bash
cd /home/jiang/EzPC/SIRNN
bash reproduction/reproduce_google30.sh float      # 完整 test/valid FP32
bash reproduction/reproduce_google30.sh quantize   # SeeDot 定点模型和预测
bash reproduction/reproduce_google30.sh build      # EzPC 生成、SCI 编译
bash reproduction/reproduce_google30.sh secure     # 30 条真实样本两方推理
bash reproduction/reproduce_google30.sh summary    # 汇总 JSON/结果说明
```

已有产物只验证推理链时可运行 `bash reproduction/reproduce_google30.sh inference`。两方测试默认每类选一条真实 test 样本，也可用 `GOOGLE30_SAMPLE_INDICES=0,1,2` 缩小重跑范围。

重新训练全部流程请使用新目录，`train` 会拒绝覆盖已保存的模型：

```bash
GOOGLE30_MODEL_DIR=/home/jiang/EzPC/SIRNN/reproduction/models/Google-30-retrain \
  bash reproduction/reproduce_google30.sh all
```

结果文件为：

- `reproduction/RESULTS_CN.md`：准确率、样本范围和测量口径；
- `reproduction/summary.json`：训练、量化、位宽、提交版本和两方统计；
- `reproduction/google30_2pc_results.json`：逐样本预测、返回码、通信和日志；
- `reproduction/FastGRNN/build/bin/Google-30`：最终 SCI 两方可执行文件。

公开仓库没有找到作者发布的 Google-30 预训练权重或量化输出，因此模型权重是按官方实现本地训练所得，不能称为论文作者原始权重。FastTrainer 按官方流程在压缩阶段使用 test accuracy 选择 checkpoint；因此同时报告 validation，且不把 test 描述为完全独立评估。SCI 安全验证只覆盖 30 条样本，不能宣称完成 6835 条完整 test 的安全推理。

历史的最小两方 smoke 仍可用于快速检查协议环境：

```bash
cd /home/jiang/dealerless-lunwen
python3 ../external-baselines/sirnn-smoke/run_sirnn_smoke.py
```

本次输出为：

```json
{"accuracy": null, "communication_bytes": 792116,
 "compute_ms": 103.0, "rounds": 209, "runtime_ms": 103.0}
```

该 smoke 只证明协议环境能完成两方交互；完整 Google-30 结果以上述 `reproduction` 产物为准。EzPC 编译器位于 `/home/jiang/EzPC/EzPC/EzPC/ezpc`，运行时如遇 OCaml 动态库错误，先设置：

```bash
export LD_LIBRARY_PATH=/home/jiang/.opam/4.11.2/lib/stublibs:$LD_LIBRARY_PATH
```

不要单独编译 `Library_SIRNN.ezpc`：它是库文件，没有最终的无参数 `main`，会报 `The last definition should the main function`。应使用 SeeDot 生成的 `predict.ezpc` 再调用 `preProcessSIRNN.py` 和 `secureCodegen.py`。

## 3. MP-SPDZ

2026-10-07 已补齐依赖、完成构建。本次在北京时间 14:09 再次执行
`bash reproduction/reproduce.sh`，三项真实两方协议测试全部通过：

| 验证 | 正确性 | 时间（含实时 OT） | 两方总发送量 |
|---|---:|---:|---:|
| MASCOT 官方 tutorial | 25/25 | 1.02904 s | 311.964 MB |
| Semi2k 官方 tutorial，R=64 | 25/25 | 0.0394573 s | 7.18257 MB |
| Semi2k 私有输入验证，R=32 | 7/7 | 0.00772768 s | 0.134451 MB |

每次两方退出码均为 0，打印的预期值与实测值最大绝对误差为 0。上述时间为 Party 0 运行时间（包含实时 OT 预处理）；这些单次 localhost 数字是工具链验证。
完整命令与原始日志见 `/home/jiang/MP-SPDZ/reproduction/README_CN.md` 和
[`latest_replay.json`](/home/jiang/MP-SPDZ/reproduction/latest_replay.json)；首次成功记录保留在 `summary.json`。一键重跑：

```bash
cd /home/jiang/MP-SPDZ
bash reproduction/reproduce.sh
```

网络问题通过 GitHub 归档恢复，依赖版本已记录。本地使用 GCC 11、清空环境
`DEBUG=release`、补齐 libsodium 开发头文件，并用单任务编译控制内存。
32 位验证使用 `f=8/k=15`，尚不是 `our` 的 `ring32/frac16` CNN/RNN 等价实验。

## 4. `project/our` 当前验证

本地结构 baseline smoke：

```bash
cd /home/jiang/dealerless-lunwen/project/our
PYTHONPATH=. python3 -m Baselines.run_baselines \
  --backend our --workload google30_fastgrnn,lstm_2layer --smoke \
  --output Baselines/results/repro_smoke.csv
```

输出 2 行 `status=smoke`。定点模型和 ring 检查：

```bash
python3 -m pytest -q \
  tools/precision/test_fixed_point.py \
  tools/precision/test_ring_model.py \
  Baselines/test_imports.py Baselines/test_reproducibility.py
```

本次共 35 个相关测试通过。这里的 `structure_only`/`smoke` 计时是本地定点结构或算术路径，不是端到端安全协议 latency；外部 backend 未配置命令时必须保留 `unavailable`，不能用本地清文计时替代。

## 5. 建议的 CNN/RNN 实验矩阵

1. **正确性层**：每个模型固定 seed，保存 FP32 参考输出；扫描 `ell={16,20,24,28,32}` 和明确的 `frac_bits`，报告 top-1/任务准确率、最大绝对误差、相对误差和失败样本数。CNN 使用 ImageNette/CIFAR-10，RNN 优先使用 Google Speech Commands；合成 LSTM 只能作为结构回归测试。
2. **协议层**：对 SCI/SiRNN/MP-SPDZ/our 分别记录 offline、online、总耗时、双向通信字节数、轮数、并发线程数和输入 batch。每个配置至少重复 5 次，报告 median 和 p95；server/client 取两方较大时间。
3. **规模层**：CNN 扫描输入分辨率和 batch，RNN 扫描序列长度 `16/32/64/99`、hidden size `64/100/128`。同时记录 ReLU、sigmoid、tanh、rsqrt 实例数，避免只按模型名称比较。
4. **网络层**：在 loopback、带宽/延迟受限的 LAN/WAN 模拟下重复 online 测试；将离线 key/triple 生成与在线交互分开，不把 dealer 分发费用隐藏在 evaluator 结果中。
5. **可审计输出**：每次运行写一行稳定 CSV，附 JSON 元数据（commit、命令、seed、输入哈希、协议、位宽、线程、主机信息）。缺少权重、数据或安全执行器时写 `unavailable` 及原因，不填 0，不声称复现。

推荐比较顺序是：先用本说明书中的 SiRNN Google-30 产物建立清文/定点参考，再实现 `project/our` 的真实 secure executor；MP-SPDZ 已通过官方 tutorial 验证，下一步实现与 manifest 中相同输入形状和实例计数的 CNN/RNN 程序。旧 `project/our` Google-30 结果采用 v0.02、hidden=128 和随机权重，不能直接与本次官方 v0.01 FastGRNN 结果合并比较。
