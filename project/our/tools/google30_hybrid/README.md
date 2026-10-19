# SCI + ours：完整 FastGRNN 单时间步

此目录只做单时间步的后端替换验收。`build.py` 从冻结的 `sirnn_fixed.cpp`
直接抽取该图的实际调用和参数，生成隔离的诊断 harness，并在**构建树**副本中
给 `AuxProtocols::MSB` 加 hook；不会修改 `/home/jiang/EzPC`。SCI 原有
MatMul/MatAdd/扩环/截断/舍入/指数和倒数 LUT、门控算法、公开 `zeta=16654` 与
`nu=26006` 均保留。`SEGFSS_SCI_ENABLE=1` 时仅替换 hook 选中的 16-bit MSB；
非 16-bit MSB（包括图内 2/9-bit 调用）、OT LUT 和其他 SCI 原语仍由 SCI 处理。

## 运行

先由工单 05 构建 `libsegfss_sci_bridge.so`，然后：

```sh
python3 build.py --out /tmp/ticket06-build/historical \
  --bridge-lib /tmp/ticket05-build/libsegfss_sci_bridge.so \
  --generated project/our/results/ticket01_native_replay/sources/FastGRNN/Google-30/sirnn_fixed.cpp
python3 run_step.py --binary /tmp/ticket06-build/historical/sci-step-hook \
  --schema /tmp/ticket06-build/historical/schema.json \
  --output project/our/results/ticket06_hybrid_step/historical \
  --model-header project/our/results/ticket01_native_replay/sources/FastGRNN/Google-30/model_fixed.h \
  --input-csv project/our/results/ticket01_native_replay/sci_trace/input/X.csv \
  --variant historical --port 41200
```

`independent` 图使用 `native02_preparation/EzPC/SIRNN/reproduction/FastGRNN/Google-30`
下的 generated/header。每方各持有私密输入，服务方持有矩阵和 bias；诊断模式只把
重构后的整数张量输出给 client 供逐位对照，正式 secure 模式不打印 `VALUE`。持久
连接、fresh mask/key/triples 与单线程设置沿用工单 05；每一个时间步/调用使用新材料。

`result.json` 同时记录 SCI 及 SCI + ours 的 16-bit MSB dispatch、实际 FSS 比较域
`15`、`nseg=4`、替换 tensor 数量、未替换的 SCI 调用和是否发生整数差异。
`step.py` 的 `compare_traces` 在第一个不一致处报告 case、算子、tensor、bits、frac、
index 及两方整数值。此实验不打开正式隐状态，也不作整网性能声明。

## 工单07：整网、批处理与恢复

`build_full.py` 直接复用独立02的15个循环调用和两个分类前调用；只把独立样本
向量化为 `I=batch`，逐行ArgMax仍使用原SCI，并保留8→16符号扩展。两方使用同一
不含模型常量的二进制。服务方读取私有 `server_model.i16`，客户读取私有输入文件；
两个程序均不会打开另一方文件。

```sh
python3 project/our/tools/google30_hybrid/build_full.py \
  --out /tmp/ticket07-build --bridge-lib /tmp/ticket05-build/libsegfss_sci_bridge.so \
  --generated /path/to/independent02/FastGRNN/Google-30/sirnn_fixed.cpp
python3 project/our/tools/google30_hybrid/full.py prepare \
  --output /tmp/ticket07-fixture \
  --quality /home/jiang/dealerless-lunwen/project/our/results/ticket02_independent_quality \
  --generated /path/to/independent02/FastGRNN/Google-30/sirnn_fixed.cpp \
  --model-header /path/to/independent02/FastGRNN/Google-30/model_fixed.h
GOOGLE30_CPUS=10,12 python3 project/our/tools/google30_hybrid/full.py run \
  --binary /tmp/ticket07-build/google30-full --fixture /tmp/ticket07-fixture \
  --output /tmp/google30-full/SCI_ours --backend SCI_ours --batch 32 \
  --oracle /path/to/ticket09/plaintext_quality/predictions.npy \
  --start 0 --count 6798 --port 42800 --timeout 1200
```

SCI使用同一binary、输入、模型、batch和固定顺序，改 `--backend SCI` 与输出目录。
已提交的完整批次自动跳过；失败批次重新启动双方，使用新材料。预测原子写入
`predictions.jsonl`，不允许多个runner同时写同一个output。运行时核对私密输入、
模型文件、生成图、二进制、动态库及SCI源/静态库版本；身份改变时必须新建output，
不覆盖或混入旧结果。`--diagnostic`只用于独立验证，输出每一步的整数张量到client；
其ledger禁止混入正式安全对照。

```sh
python3 project/our/tools/google30_hybrid/compare_full.py \
  --SCI /tmp/google30-full/SCI/predictions.jsonl \
  --hybrid /tmp/google30-full/SCI_ours/predictions.jsonl \
  --contract /tmp/ticket07-fixture/contract.json --output /tmp/secure-comparison.json
```

每个正式ledger逐样本记录SCI整数oracle预测及是否匹配。完整率不到6,798/6,798时，
`full_top1_pct`和完整一致性不能作为已完成结果；`partial_subset_top1_pct`只描述已执行
子集。独立02的SeeDotFLOATEXP预测与SCI近似不同；参考使用经真实SCI验证的整数
oracle，不能用SeeDot预测替代逐位参考。
