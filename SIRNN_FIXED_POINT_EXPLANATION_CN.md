# SiRNN 权重来源、实际位宽与控制方法

核对日期：2026-10-07。这份说明补充已保存复现报告中的量化和安全运算细节，未修改模型、程序或实测结果。

## 权重来源与两方执行方式

本次没有下载作者的 Google-30 预训练权重。公开 EdgeML/SeeDot 仓库未找到匹配权重，因此下载官方代码和 Speech Commands v0.01 数据，按官方 FastGRNN 配置从随机初始化训练 300 epoch，再导出保存的模型。

训练是本机 TensorFlow 浮点训练。导出的模型经 SeeDot 定点量化后，生成 EzPC/SCI 程序，最后做两方安全推理。server 持有量化模型参数，client 持有输入；模型在这个设置下对 client 保密，输出类别揭示给 client。训练本身没有在两方协议中进行。

## 这次模型的位宽

| 变量或运算 | 实际配置 |
|---|---|
| 输入 X | 有符号逻辑 16 位，`scaleForX=-11`，即刻度 `2^-11` |
| W1、W2、U1、U2、FC | 8 位量化参数 |
| Bg、FCbias | 8 位量化参数 |
| Bh | 16 位量化参数 |
| 隐状态 H 和多数中间激活 | 16 位；U1 投影结果和最终分类分数等部分变量为 8 位 |
| Sigmoid、tanh 接口 | 输入、输出均为 16 位；本次输出小数精度分别为 14、7 位 |
| 16×16 元素乘法 | 32 位乘积，再按尺度缩放到 16 位输出 |
| 分类分数 / ArgMax | 分数为 8 位；安全比较前符号扩展到 16 位；类别索引按 8 位重构 |

输入编码为 `q = trunc(x * 2^11)`，近似恢复为 `x = q * 2^-11`，其中 trunc 为向零截断。因此输入的 16 是总位宽，11 是小数精度；二者不是同一个参数。每个模型张量有自己的尺度，不能将所有中间量都视作与输入相同的小数精度。

### 矩阵乘法内部累加

生成的 SeeDot C++ 中 `int32_t tmp18_32` 等是类型声明。SCI 的 `MatMul_thread` 使用 `bwA+bwB` 调用 LinearOT；LinearOT 在点积长度为 K 时，另增加 `ceil(log2 K)` 个累加位，再截断缩放。实际安全累加位宽不能仅由 C++ 类型或生成代码的 `bwTemp=32` 推断。

| 乘法 | 输入/权重位宽 | 点积长度 K | LinearOT 扩宽后的累加位宽 | 缩放后输出位宽 |
|---|---|---:|---:|---:|
| X × W1 | 16 × 8 | 32 | 29 | 16 |
| W1 投影 × W2 | 16 × 8 | 16 | 28 | 16 |
| H × U1 | 16 × 8 | 100 | 31 | 8 |
| U1 投影 × U2 | 8 × 8 | 35 | 22 | 16 |
| H × FC | 16 × 8 | 100 | 31 | 8 |

这张表描述该模型的矩阵累加阶段。非线性协议也有自身的内部运算和扩宽，接口为 16 位不代表其所有内部变量都为 16 位。

SCI 的输入缓冲区 `MYINT=int32_t`、秘密份额容器 `int64_t`，以及 EzPC 的 `--bitlen 64` 是实现层的存储/生成设置，不能据此把整个模型称为 32 位或 64 位定点推理。

## 如何控制

控制点位于量化与代码生成阶段，当前 `Google-30` 二进制没有运行时的统一位宽参数。

1. SeeDot 当前 `wordLength=16`、`vbwEnabled=True`，从基础 16 位中搜索可降为 8 位的变量，并选择各变量尺度。
2. 可在量化封装中约束 `variableToBitwidthMap`、`demotedVarsList` 和 `demotedVarsOffsets`，例如固定输入与隐状态为 16 位、允许权重降到 8 位，或约束变量保持 16 位。这需要生成对应变体并验证，不是当前命令行已有的一个通用开关。
3. 输入尺度也可指定或约束，但必须同时更新输入编码、参数尺度和计算中的缩放/截断。只修改生成的 `datatypes.h` 会造成不一致。
4. 该版 SeeDot 的 VBW 搜索明确要求基础位宽为 16，降位目标固定为 8。20/24 位等任意配置需要扩展生成器并核对 SCI 后端约束，不能直接修改 `wordLength` 后声称可用。
5. 关闭 `vbwEnabled` 会进入不同的生成路径；当前复现没有验证这种路径下的 EzPC→SCI 全流程。建立主要使用 16 位的实验，可先保持 VBW 路径并约束不降位变量，再实际量化和测试。

修改位宽通常可以复用现有浮点权重，无需重新训练；需要重新量化、生成、编译，并验证定点准确率与两方预测。要比较输入/权重/激活位宽的影响，应保存同一浮点模型的多个量化配置，并分别测量准确率、耗时和通信。

## 对应代码

- [官方训练封装](/home/jiang/EzPC/SIRNN/reproduction/scripts/train_google30_official.py:82)
- [SeeDot 位宽配置](/home/jiang/EzPC/SIRNN/reproduction/SeeDot/seedot/config.py:10)
- [VBW 基础 16 位约束](/home/jiang/EzPC/SIRNN/reproduction/SeeDot/seedot/main.py:455)
- [最终 SCI 运算调用](/home/jiang/EzPC/SIRNN/reproduction/FastGRNN/Google-30/sirnn_fixed.cpp:513)
- [SCI 矩阵乘法实现](/home/jiang/EzPC/SCI/src/library_fixed.cpp:657)
- [LinearOT 累加扩宽实现](/home/jiang/EzPC/SCI/src/LinearOT/linear-ot.cpp:577)
