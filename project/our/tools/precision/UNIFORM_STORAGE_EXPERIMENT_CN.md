# 全网统一定点存储与宽累加主实验：E0-U-S

## 实验问题与证据范围

对固定预训练网络，测量全网统一 Q(ell,s) 存储、使用经溢出检查的
宽临时累加器时，位宽与分类准确率、FP32 预测一致率之间的关系。
ell 是包含符号位的总位宽，s 是全网统一小数位。解码使用有符号代表元
`[-2^(ell-1), 2^(ell-1)-1]`，实数值为 `code * 2^-s`。

2026-10-08 用户恢复宽临时累加方案，默认配置再次为
`arithmetic_mode=wide, overflow_mode=saturate`。存储宽度为 ell，
乘积和临时累加采用独立的 128 位逻辑寄存器，点积结束缩放一次。

`results/e0_uniform_storage_imagenette` 中已完成的正式结果采用
**wide + saturate**，称为 E0-U-S。此前“24 位接近 FP32”由该规则
支持，不能转述为严格 ell 位环实验的结论。默认恢复不修改已有 CSV。

## 模型和数据

- 完整 torchvision ResNet-50、ResNet-101、ResNet-152，使用
  ImageNet-1K V1 预训练权重和原始 1000 类输出头，不重新训练。
- 现有 `imagenette2-160`：9469 张训练图、3925 张验证图；
  预处理使用对应权重的官方 `weights.transforms()`。
- 从训练 split 取与原 Q-fixed 相同的 512 张类均衡校准图。
  每个位宽的 s 仅由校准集选择；最终使用完整验证集。
- ImageNette 标签映射回 ImageNet 的 1000 类索引。这是完整
  ImageNette 验证集，不是完整 ImageNet-1K 验证集。
- FP32 参考关闭 CUDA matmul 与 cuDNN TF32，设置
  `float32_matmul_precision=highest`。

## 宽累加与饱和存储

定义 ell 位饱和函数：

\[
S_\ell(z)=\min(2^{\ell-1}-1,\max(-2^{\ell-1},z)).
\]

所有输入、卷积与 FC 权重、bias、离线 BN affine 系数、算子输出使用
相同 Q(ell,s)，编码是 `S_ell(floor(x * 2^s))`。

对卷积或线性层，按以下逻辑计算：

```text
acc = bias_code * 2^s
for each dot-product term:
    acc += input_code * weight_code
output_code = S_ell(floor(acc / 2^s))
```

乘积和点积在宽临时寄存器中保留整数值，点积结束后只做一次有符号
floor 缩放。bias 持久存储的尺度为 `2^s`，累加时临时提升到 `2^(2s)`。
标量与整数 limb/GEMM 路径都检查临时范围；发生寄存器溢出时停止，
不把溢出的宿主结果当成网络输出。

BN 离线转换为 `multiplier * input + offset`，两个系数也编码到同一
Q(ell,s)，运行时遵循同样的乘积、offset 提升、累加和一次缩放规则。
残差加法宽加后饱和写回。全局平均池化先宽求和，再 floor 除以窗口
大小并写回。ReLU、maxpool、argmax 对有符号代码进行比较。

`int64` 是存储载体；宽临时量的逻辑范围由 `register_bits` 和运行时
检查约束。该明文实验不证明两方协议已经实现安全扩环、截断与饱和。
MPC 的内部计算环宽需要单独说明，不能称为全程 ell 位环运算。

## 位宽扫描与最终评估

完整重跑方案独立扫描 `ell={12,16,20,24,28,32}`，每个 ell 枚举全部
`s=0,...,ell-1`，每模型 132 个候选，三模型合计 396 个。训练校准
Top-1 最大者为选点；并列时取较小 s。每模型 6 个选点再评估完整
3925 张验证图，合计 18 个最终配置。验证图不参与选点。

wide 模式只有在离线参数、输入、每层输出均无裁剪，存储位宽适合
目标 ell 且临时寄存器通过检查时，才允许同一 s 的跨 ell 精确复用。
复用记录原配置和预测文件。已有同一语义、同一配置的完成结果可续跑。

报告记录 Top-1、相对 FP32 的百分点差、逐样本预测一致率、
`FP32 对/定点错` 与 `定点对/FP32 错` 的计数，以及原始预测和配置。
wide+saturate 记录以下范围指标：

- `offline_clipped`、`input_clipped`：离线系数或输入编码被饱和的元素数。
- `activation_clipped`：算子写回时被饱和的输出数。
- `register_overflows`：宽临时计算超出指定范围的事件数，正式结果要求零。
- `max_storage_bits`：未经饱和的存储值所需位数。
- `max_register_bits` 和 `max_register_bits_kind`：标量实测累加前缀，
  或整数 GEMM 的保守数学上界，二者应按 kind 分别解释。

## 保留的对照

E0-U-R 使用 `ring + wrap`：乘积、累加前缀和池化和先 mod `2^ell`，
再对有符号代表元缩放，逻辑寄存器宽度为 ell。其规则保留在
`STRICT_RING_CN.md`，完整准确率尚未扫描，不能复用 wide 的预测。
原 E0-Q-fixed 每标量乘积立即缩放，逐项点积加法和池化求和均作 ell
位饱和。各数值模型保留独立结果目录。

SiRNN Google-30 使用混合位宽和逐张量尺度，不能将其与本方案的全网
统一 Q16 视为相同设置。此精度评估运行完整明文整数网络，不把运行
时间或 `int64` 容器占用作为两方协议时延和压缩存储成本。

## 完整重跑命令

在 `project/our/tools/precision` 中运行：

```bash
MAX_JOBS=1 python3 run_e0_uniform_storage.py \
  --models resnet50,resnet101,resnet152 \
  --ell-grid 12,16,20,24,28,32 \
  --calibration-samples 512 --test-samples 0 \
  --device cuda --batch-size 16 --workers 2 \
  --integer-backend gemm \
  --register-bits 128 --arithmetic-mode wide --overflow-mode saturate \
  --results-dir results/e0_uniform_storage_imagenette
```

输出包括 `e0_uniform_storage_candidates.csv`、
`e0_uniform_storage_final.csv`、逐模型 selected/status/metadata、
样本 manifest 和 `predictions/`。文件名沿用统一实验接口；判断数值
语义应读取 metadata 中的 arithmetic/overflow mode，而不是文件名。

修改语义、数据或权重时使用新目录。已有 wide 正式证据保留；新增
小样本结果标记 `diagnostic`，不能作为上述正式准确率。严格环可用
`reproduce_uniform_ring.sh` 显式运行，默认目录与 wide 不混用。
