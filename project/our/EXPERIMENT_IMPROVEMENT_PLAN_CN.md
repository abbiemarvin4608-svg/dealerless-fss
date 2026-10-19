# CNN/RNN 实验改进方案（证据与执行版）

本方案按 `ccf-experiment-designer` 的 `design` 模式编写，目标是把现有
CNN/RNN 结果整理成可比较、可复现的实验包。它只定义数据、模型、协议、指标和
执行顺序，不填入新的实验数字，也不改写已有结果。复现实验结果仍必须来自实际
运行日志和固定产物。

## 先固定证据边界

现有 `Baselines/results/our_32bit_workloads.json` 中的三行 `our` 结果是
`structure_only`：ResNet-50 为 5537.283915 ms，Google-30 FastGRNN 为
139.299176 ms，synthetic 两层 LSTM 为 494.066155 ms。它们来自本地定点结构/算术
路径和确定性输入，没有 secure executor、offline/online 通信或任务准确率，因此
只能回答“模型结构和定点镜像能否运行、算子规模是多少”。这些数字不进入安全
推理 latency 或通信比较表。

现有 `results/nseg_selection_20260927.md` 的 `our` 结果是独立的两方 DPF/DCF
primitive 证据：`nseg=4` 的 offline 36.3243 ms、online 0.2240 ms、offline
277920 B、online 24 B；它可以支撑 primitive 的成本和 nseg 选择，不能外推成
ResNet/FastGRNN/LSTM 的端到端推理。

已有 ImageNette/CIFAR-10 文件中的精度结果属于明文定点精度研究。它们可回答
“Q(ell,s) 对任务精度的影响”，但没有安全通信；应继续保留在 precision 表中。
已有的 SiRNN FSS Google-30 行同时包含不同内部位宽及 dealer 分发开销，必须标记
其通信 scope，不能和统一 Q32.16 的 `our` 清文计时直接合并。MP-SPDZ 的官方
tutorial 只证明工具链和 Semi2k 实时 OT，不能当作 CNN/RNN 模型 baseline。

结果表统一使用下列字段，禁止用空值猜测另一阶段的数字：

| 字段 | 允许值/含义 |
| --- | --- |
| `mode` | `structure_only`、`plaintext_fixed`、`secure_2pc` |
| `phase` | `float_reference`、`quant_eval`、`offline`、`online`、`total` |
| `status` | `measured`、`blocked`、`unavailable` |
| `communication_scope` | `per_party_sent_recv`、`both_party_total`、`server_client_only`、`dealer_included` |
| `model_fingerprint` | 模型参数文件、生成代码和配置的 SHA-256；没有 fingerprint 时不能合并结果 |

`structure_only` 与 `secure_2pc` 不得出现在同一 latency 排名中；`plaintext_fixed`
只和相同定点规则的浮点参考比较。所有未完成项写 `status=blocked/unavailable`，
并写明缺少的数据或执行器。

## 统一工作负载

| 家族 | 主工作负载 | 数据与模型产物 | 定点/协议要求 | 用途 |
| --- | --- | --- | --- | --- |
| CNN 精度 | ImageNette 上的 ImageNet-1K ResNet-50/101/152 | 现有 torchvision V1 权重、ImageNette 3925 张验证图、固定预处理 | `Q(ell,s)` 扫描 `ell={16,20,24,28,32}`，每个 ell 先在 512 张校准图选 s；保留现有严格整数语义 | 量化正确性、位宽选择 |
| CNN 对齐 smoke | ResNet32-CIFAR | SCI 的 `resnet32_cifar-OT` 与相同 CIFAR-10 样本；先检查 `-ell 32` 是否真正生效 | 若 SCI 仍为默认 41-bit，标记不可比；只有输入、权重、截断和环宽都对齐才进入两方表 | 最小端到端 CNN 适配路径 |
| RNN 主模型 | **官方 SiRNN Google-30 FastGRNN** | `/home/jiang/EzPC/SIRNN/reproduction` 的 v0.01、99×32 特征、30 类；`W rank=16`、`U rank=35`、hidden=100；同一 `*.npy`、mean/std、SeeDot 生成文件 | 使用生成的 `datatypes.h`、`model_fixed.h` 和 `X.csv/Y.csv`；记录每个变量的实际 bit/scale，不能假设 Q32.16 | 真实语音精度与两方安全推理主结果 |
| RNN 结构回归 | 两层 synthetic LSTM | 现有 `lstm_2layer`、100 tokens、hidden=128 | 仅 `structure_only` 回归；不宣称任务准确率，不与官方 FastGRNN 行合并 | 检查本地转换器和算子计数 |

官方 Google-30 的产物清单为：

```text
SIRNN_ROOT=/home/jiang/EzPC/SIRNN/reproduction
SIRNN_ROOT/models/Google-30-FastGRNN-r16-u35-h100/{W1,W2,U1,U2,Bg,Bh,zeta,nu,FC,FCbias,mean,std}.npy
SIRNN_ROOT/data/Google-30-processed/{train,test,valid}.npy
SIRNN_ROOT/SeeDot/temp/Predictor/{model_fixed.h,datatypes.h,input/X.csv,input/Y.csv}
```

模型选择使用官方训练规则，test split 曾参与 checkpoint 选择；因此首次报告应
同时给出 `float_evaluation.json` 的 test 和 valid，并明确这一限制。`our` 旧的
v0.02/hidden=128/31-class 测试保留为 legacy 行，不能重命名成官方 Google-30。

## Claim–evidence 矩阵

| 需要支持的结论 | 最小证据 | 不能替代的证据 | 当前状态 |
| --- | --- | --- | --- |
| Q(ell,s) 是否保持 CNN 任务精度 | FP32 与严格定点在同一 ImageNette split 的 top-1、top-5、delta、饱和/回绕计数 | `structure_only` runtime、随机输入、CIFAR 结果冒充 ImageNet | ImageNette/CIFAR precision 文件已有；新增运行只补缺失模型或复核 |
| 分段 DPF/DCF 的成本规律 | 两方 primitive 的 offline/online/bytes/rounds，`nseg` 与 ell 记录 | 将单 primitive 乘以 ReLU 数量后称为端到端结果 | nseg sweep 已测；保留 primitive 表 |
| 我们能否完成安全 CNN 推理 | 同一模型和输入的两方执行、输出重构正确、offline/online 分相计时 | 明文定点镜像、SCI 独立 ResNet、MP-SPDZ tutorial | secure end-to-end executor 尚缺，状态 blocked |
| 官方 SiRNN Google-30 量化是否正确 | SeeDot fixed prediction 与 secure client label 在同一 CSV 行一致；记录样本索引和 wav 路径 | 仅进程 exit=0、随机权重、v0.02、单个 smoke 样本推断全量准确率 | 训练/量化/生成脚本和数据已准备，最终 2PC 结果待运行 |
| RNN 两方效率是否优于 baseline | 同一官方模型、同一输入行、同一线程/网络条件下的 offline、online、total 和 bytes | 将 SiRNN dealer 分发与 evaluator-only、或 our 清文 timing 混在一个列 | 等 official v0.01 生成二进制和 our 对应 executor |

## 实验矩阵与表结构

### A. 明文精度（先做，成本低）

每个 CNN 模型执行 FP32 reference 和严格整数 `Q(ell,s)`。校准集只选 s，验证集
只报告一次。每行至少包含：`model,dataset,ell,s,seed,n_calib,n_eval,top1_fp32,
top1_fixed,delta_pp,top5_fp32,top5_fixed,overflow_count,wrap_count,clip_count,
truncation_mode,weights_hash,status`。RNN 主模型不沿用这套随机模型；其 reference
由官方 `float_evaluation.json` 和 SeeDot fixed log 提供，记录 `test/valid` 分开。

建议最小精度网格：

```text
CNN: ell = 16,20,24,28,32; s = 0..ell-1（仅校准集选择）
Google-30: generated SeeDot scale；若做补充扫描，另建实验名，不覆盖官方量化结果
```

补充的 RNN 位宽扫描必须使用同一 v0.01 特征、同一 rank16/35 权重和同一 split，
并单独命名为 `supplementary_precision`；不得把它描述成 SiRNN 论文复现。

### B. 两方协议与阶段计时

每个 secure 2PC 行必须包含：`backend,model_fingerprint,input_ids,batch,threads,
party_roles,ring_bits_or_variable_scales,offline_ms,online_ms,total_ms,
offline_sent_recv_bytes,online_sent_recv_bytes,rounds,ot_count,peak_rss,
network_profile,repetitions,status`。每个配置预热 1 次，正式运行至少 5 次，报告
median 和 p95；localhost 与形状网络（LAN/WAN）分开。两方时间取较大值；通信同时
报告每方 sent+recv 与两方总和，不能用“某一方发送量”代替总量。

通信 scope 规则：

1. SiRNN evaluator-only 与 dealer/key-generation 必须分列；若存在第三方或离线
   dealer，写 `dealer_included`，不称为纯两方 dealerless 结果。
2. `our` 的 DPF/DCF offline key/COT/Beaver 和 online masked opening 分开；最终
   output reconstruct 不放进 online，或另列 `reconstruct_ms`，所有 backend 同规则。
3. MP-SPDZ Semi2k 的 preprocessing/live OT、online 和总量按 MP-SPDZ 输出分别记录。
   已完成的 tutorial JSON 只放“toolchain validation”附录，不进入模型表。

### C. 端到端主表的准入条件

只有同时满足以下条件才写入主比较表：同一个模型参数 fingerprint、同一输入行或
样本索引、同一输出标签定义、同一 batch、同一 bit/scale 与 truncation 语义、两个
进程正常退出且输出通过 reference check。任一条件不满足，就放入 `blocked` 表，
保留原因和可执行补救步骤。

## 可执行顺序

### P0：官方 Google-30 证据闭环

在 `/home/jiang/EzPC/SIRNN` 执行训练/SeeDot/代码生成后，先不跑大规模 2PC：

```bash
reproduction/venv/bin/python reproduction/scripts/evaluate_google30_official.py \
  --data-dir reproduction/data/Google-30-processed \
  --model-dir reproduction/models/Google-30-FastGRNN-r16-u35-h100 \
  --output-dir reproduction/results/float

reproduction/venv/bin/python reproduction/scripts/run_google30_2pc.py \
  --binary FastGRNN/build/bin/Google-30 \
  --predictor-dir SeeDot/temp/Predictor \
  --sample-indices 0,1,2 --threads 4 --timeout 600 \
  --output reproduction/google30_2pc_results.json
```

先检查选定行的 `X.csv/Y.csv` 与预处理 `jsonl`、SeeDot fixed prediction 一一对应；
再扩展到固定的分层样本（每类至少一行，另加 reference 正确/错误各若干）。小样本
只验证管线，不能填充全量 accuracy；全量 test/valid 仅在资源允许且日志完整时运行。

### P1：让 `our` 能消费官方模型

新增一个独立 adapter（不要覆盖 `Baselines/models.py` 的 legacy random factory）：

1. 从 `model_fixed.h/datatypes.h` 解析变量 bit/scale、矩阵布局和输入编码；
2. 直接读取最终 `X.csv/Y.csv`，禁止从原始 float 重新序列化输入；
3. 先运行 cleartext fixed predictor，核对 SeeDot 预测；
4. 再接入 `our` 两方 executor，输出阶段日志和同一 JSON schema；
5. 记录 `model_fingerprint`、生成命令、git commit、样本索引和输入文件 hash。

在该 adapter 完成前，`our` Google-30 行保持 `blocked`，不得复用旧的 139.299176 ms。

### P2：CNN 端到端最小路径

先以 CIFAR-10 ResNet32 作为工程 smoke。检查 SCI `resnet32_cifar-OT --help` 和
源码的 `ell` 参数；若 `ell=32`、权重、截断和输入格式无法对齐，就只报告 SCI 自身
结果，不做跨 backend latency。`our` 必须接入同一模型后才可比较；现有 ResNet-50
ImageNette 只继续作为精度/结构证据。完整 ImageNet ResNet 不作为本阶段的第一条
安全端到端路径。

### P3：统计与复核

每个可比较配置运行 5 次以上，生成 `results/secure_2pc_<workload>.json/csv`，
保留原始 stdout/stderr。汇总脚本只读取 JSON 中 `status=measured` 且 fingerprint、
phase、communication_scope 齐全的行；遇到 null、混合 mode 或不同 scope 直接报错。
汇总脚本不得把旧结果就地重写，新的 schema 使用新文件名。

## 当前缺口和停止条件

| 缺口 | 影响 | 停止条件/补救 |
| --- | --- | --- |
| `our` 没有安全 CNN/RNN executor | 不能生成端到端安全 latency/bytes | 完成一条两方模型路径并通过 output reconstruction 后再扩展 |
| 旧 FastGRNN 是随机权重、v0.02/hidden128 路径 | 不能代表官方 SiRNN Google-30 | 迁移到官方 v0.01/rank16/35/h100 adapter |
| SCI ResNet32 默认源码 bitlength=41 | 不能与 Q32.16 直接比较 | 明确 `ell=32` 的可运行证据，否则保持 blocked |
| SiRNN generated scale/bit 尚在生成阶段 | 不能预写 accuracy/通信 | 读取最终 `datatypes.h`，记录实际 scale，不填假值 |
| synthetic LSTM 无 checkpoint/任务标签 | 只能回归结构和算子量 | 保留 `structure_only`，不写准确率或 secure claim |
| 结果模式和通信 scope 混杂 | 会产生错误排名 | 采用本方案 schema，旧 JSON 原样保留 |

没有必要为了填满矩阵而运行不匹配的模型。若 P1/P2 在输入格式或安全执行器处
阻塞，应报告 `blocked` 和最小复现日志，同时保留 primitive、明文精度和官方
SiRNN 证据。

## 与现有文件的关系

本方案引用但不修改 `Baselines/manifest.yaml`、`Baselines/results/*.json`、
`results/benchmark_matrix_20260927.md` 和已有精度报告。建议新增配置使用
[`experiment_matrix_v2.yaml`](Baselines/experiment_matrix_v2.yaml)；它明确 canonical
模型、输入产物、阶段和结果 scope，避免把 legacy workload 误合并进主表。

No-fabrication 状态：本文档没有生成 CNN/RNN 新结果，也没有把结构计时、primitive
计时、MP-SPDZ tutorial 或 dealer-assisted FSS 数字转写成端到端安全性能。
