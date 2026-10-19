# SIGMA 激活与统一定点精度实验核查

## SIGMA 原文与本项目的激活实现

`refer/SIGMA.pdf` 第 17 页 Appendix G 用偶函数

\[
\delta(x)=\frac{1}{1+e^{|x|}}
\]

构造 sigmoid：当 \(x\leq0\) 时返回 \(\delta(x)\)，当 \(x>0\) 时返回
\(1-\delta(x)\)。在输出小数精度 \(f=12\) 下，修正项在
\(|x|\geq16\) 时小于一个输出 ULP。结合原文第 8 页的 LUT 构造，正半轴
以 \(2^{-6}\) 为步长，使用 1024 个表项：

\[
T[j]=\left\lfloor\frac{2^{12}}{1+\exp(j/2^6)}\right\rfloor,
\qquad j=0,\ldots,1023.
\]

本项目 `SigmaActivationLUT` 和 C++ 标量实现先计算绝对值、裁剪到
\(2^{f+4}-1\)，再缩减索引、查表和按符号选择分支。`q=10,f=12` 时，
数值规则对应 SIGMA 的基础折叠近似。它没有采用原文 CPU GeLU 协议中
“先截断、再取绝对值”的重排序；原文说明该优化可能改变索引最低位。

SIGMA 原文没有给出 tanh 协议。本项目使用

\[
\widehat{\tanh}(x)=2\widehat\sigma(2x)-1
\]

派生 tanh，复用 sigmoid 表。因此有效未饱和输入范围约为 \([-8,8]\)。
在输入环中计算 \(2x\) 时，还需要足够的范围以避免回绕。

在 `ell=32,f=12,q=10` 下重算的误差与论文原表一致：

| 函数 | 最大绝对误差 | 最大输出 ULP 误差 |
|---|---:|---:|
| sigmoid | 0.003845139040069445 | 15.749689508124447 |
| 派生 tanh | 0.007568214872941978 | 30.99940811957034 |

这些误差是本项目对数学函数的测量结果，不是 SIGMA 原文报告的 sigmoid/tanh
误差。LUT 查询对指定整数表是精确的；表对真实激活函数的近似误差仍然存在。

旧 E0-B LSTM 准确率表来自使用理想 sigmoid/tanh 后重新编码的扫描，不能
视为已经包含 q10/f12 激活误差的网络准确率。论文已按实际结果来源区分两项
实验。当前 LSTM 的 SIGMA 模式跟随网络小数位 `s`；`s=7` 对应 f7，
不是 f12。与 C++ 路径对齐时要求 `s>=6` 且 `ell>s+4`。

此次修复了 C++ 曾默认使用的未掩码 piecewise 比较路径：该路径直接重构原始
激活，现已禁用。sigmoid/tanh 采用两次 BitDec、三次独立 BitXA、精确截断
和 LUT 查询；批量接口调用同一标量路径，修正了原批量路径的边界索引和
公开常数份额问题。批量接口当前不提供通信轮数加速，旧优化路径的性能数据
不能用于描述修复后的实现。

验证结果：定点/LUT Python 测试 23 项通过，严格 Q-fixed CPU 测试 7 项
通过；实际两方激活检查 t=9、t=14 的双方返回码均为 0，共 84 个检查项通过。
检查包含 19 个标量 sigmoid、15 个 tanh 和 17 个批量 sigmoid 输入的整数
输出一致性，以及本地表、协议模式和禁用路径检查。
两方日志保存在 `project/our/results/sigma_activation_validation/`。

## 当前默认值与显式网络精度

C++ `activation.h` 的标准常量为 `kSigmaDefaultFracBits=12`、
`kSigmaDefaultQ=10`、`kSigmaDefaultTableSize=1024`。省略精度参数的
`sigmaOffline`、`tanhOffline`、`sigmaBatchOffline` 以及明文查表 API
采用该标准；显式指定 `(frac_bits,q)` 的原接口继续支持自定义格式。
`activation_benchmark` 省略 `frac` 和 `q` 时采用 f12/q10。

Python 独立工厂 `tools.precision.sigma_defaults.make_sigma_activation_lut()`
默认创建 Q32.12/q10。`Baselines/models.py` 在没有配置时使用相同工厂，
默认 manifest 的 `precision.rnn.frac_bits` 是 12，CLI 按模型 family
读取该值，CNN 的默认小数位仍是 16。传入显式网络配置或自定义 manifest
时，激活与全网共同采用该 `s`，1024 项 q10 表在输出 Q(ell,s) 下重新编码。

| 路径 | 实际激活 | 小数精度 | 表项数 |
|---|---|---|---:|
| C++ 标准 API/默认 activation benchmark | 安全 SIGMA 折叠表与派生 tanh | f12 | 1024 |
| Baselines 默认 RNN 命令/省略 config 的模型转换 | SIGMA 数值镜像 | s12 | 1024 |
| Baselines 显式 config/主实验 `--lstm-activation sigma` | SIGMA 数值镜像 | caller s，全网统一 | 1024 |
| FP32 网络、显式 `ideal` 模式、旧 `test_google30_fastgrnn.py` | 数学 sigmoid/tanh | FP32 或 caller s 重编码 | 无 SIGMA 表 |

因此“SIGMA 标准默认值为 f12/q10”与“所有运行路径都强制 f12”不同。
对独立 s 的主实验，不能仅把激活固定 f12 而保持其他层采用 s；这样会成为
混合精度实验。若需要独立检验原文 f12 近似，应运行全网 s12 对照并单独
标记，而不是覆盖主实验已选择的 s。旧 ideal 结果保持原有数值及来源。

默认参数的两方复核可在两个终端分别执行：

```bash
project/our/build/test/benchmark/activation_benchmark r=2 p=28651 kind=sigmoid runs=1
project/our/build/test/benchmark/activation_benchmark r=3 p=28651 kind=sigmoid runs=1
```

SERVER CSV 中 `frac_bits=12,q=10`；`kind=tanh` 同样使用该默认表。
默认与显式参数的复核日志、双方退出码和 CLI 结果保存在
`project/our/results/sigma_defaults_validation/`；完整网络 CLI 该次结果是
结构/定点镜像检查，不提供数据集准确率或完整安全网络通信指标。

正确性可通过构建 `SEGFSS_CORRECTNESS_TEST` 后，在两个终端分别运行同一
测试编号的 `r=2` 和 `r=3` 角色复核，例如：

```bash
cd /home/jiang/dealerless-lunwen
cmake --build project/our/build --target SEGFSS_CORRECTNESS_TEST -j2
# 终端一
project/our/build/test/correctness/SEGFSS_CORRECTNESS_TEST r=2 p=35149 t=9
# 终端二
project/our/build/test/correctness/SEGFSS_CORRECTNESS_TEST r=3 p=35149 t=9
```

批量路径改用 `t=14`，两个角色使用相同端口。Python 检查分别是
`python3 -m pytest -q test_sigma_activation_lut.py test_fixed_point.py`
及 `python3 -m pytest -q test_fixed_point_saturated.py -k 'not cuda'`，
在 `project/our/tools/precision` 下执行。

旧 E0-B LSTM 激活语义可以在新目录中复核：

```bash
cd /home/jiang/dealerless-lunwen/project/our/tools/precision
python3 run_e0_2d.py --models lstm --lstm-activation ideal \
  --calibration-samples 512 --results-dir results/e0_b_ideal_replay
```

包含 LUT 误差的独立扫描使用 `--lstm-activation sigma` 和另一个结果目录。
驱动会跳过 C++ 不支持的 `s`，并将模式记录到 CSV；缺少该字段的旧结果目录
禁止写入。本次只验证接口和算子，没有重跑完整网络扫描或替换旧准确率表。

## E0-Q-fixed 的精确定义

这是**全网统一格式**的整数推理实验。固定一对 `(ell,s)` 后：

- 输入、卷积/FC 权重和 bias、离线 BN affine 系数、激活、残差及 logits
  都采用同一个有符号 Q(ell,s) 格式。
- 整数代码范围为 `[-2^(ell-1), 2^(ell-1)-1]`，实际值为 `q * 2^-s`。
- `s` 是小数位数；`ell-s` 包含符号位和整数范围。每个位宽的 `s` 在训练
  校准集上选择，全网没有逐层独立 scale，也没有混合 8/16 位。
- CPU/CUDA 使用 int64 容器，但代码值被限制到逻辑 ell 位。这测量逻辑
  表示宽度与准确率，不是 ell 位紧凑内存布局的实测内存消耗。

它另外强制**每标量乘积立即缩放和饱和，每次加法也立即饱和**：

\[
q_{p_i}=\operatorname{sat}_\ell
\left(\left\lfloor\frac{q_{x_i}q_{w_i}}{2^s}\right\rfloor\right),
\qquad
q_{a_i}=\operatorname{sat}_\ell(q_{a_{i-1}}+q_{p_i}).
\]

这比“所有需要保存的网络张量是 ell 位”更强。全局平均池化也先执行
ell 位饱和求和，再除以窗口大小。对 7×7 非负激活，Q(24,17) 的池化输出
上限约为 `64/49=1.306`。它会改变特征，不能把较高分类准确率解读为数值
恢复或与 FP32 逐样本输出一致。

逐乘积缩放还会引入额外误差。例如 Q(16,11) 下，`0.01` 编码为 20，
单项乘积 `floor(20*20/2048)=0`；1000 项逐乘积缩放后累加仍为 0。
若先宽累加再缩放，则得到 `floor(1000*400/2048)=195`，解码约为
0.0952。两者保存的输入和输出都可以使用相同的 16 位格式，区别在运算规则。

已有完整 ImageNette 验证结果来自 3925 张图及完整的预训练 ResNet
1000 类输出头，没有重新训练：

| 网络 | FP32 | 严格 Q20 | 严格 Q24 |
|---|---:|---:|---:|
| ResNet50 | 75.4140% | 71.8471% (s15) | 79.6943% (s17) |
| ResNet101 | 78.1146% | 69.9363% (s15) | 81.3758% (s17) |
| ResNet152 | 77.4777% | 62.5478% (s15) | 79.7197% (s17) |

原始结果：`results/e0_q_fixed_strict_imagenette_all/e0_q_fixed_final.csv`。
这些数据支持“上述统一格式与严格算术组合中，20 位配置出现明显精度损失”。
它们尚未单独测出存储位宽的必要下界。

## 对“高位宽存储需求”的实验要求

应将存储位宽与累加器位宽拆开控制。主实验保持输入、参数、BN affine
系数和所有层输出均为统一 Q(ell,s)，但用足够宽的整数临时寄存器形成完整
点积，结束后一次缩放并返回 ell 位。平均池化同样先宽求和、再除法，最后
返回 ell 位。宽累加器须检查溢出；不能假设 int64 对所有候选总是足够。

对同一浮点权重扫描 `ell={12,16,20,24,28,32}` 和各合法 `s`，在训练集
独立校准后固定配置，在完整验证集记录 top-1、与 FP32 的预测一致率、
参数/输入/激活饱和次数及临时累加器溢出。将现有严格 E0-Q-fixed 作为
运算规则对照，才能分辨存储量化和有限累加的影响。

当前其他实验也没有完全隔离这个目标：E0-R 耦合存储与寄存器宽度，且
bias/BN offset 用 2s 编码；E0-Q-all 使用浮点算子和未量化 BN 运行统计。
后者完整验证中 ResNet152 的 20 位结果为 77.4268%，FP32 为 77.4777%，
说明解除严格运算约束可能改变位宽需求，但不能替代上述完整整数主实验。

可检验的命题应是“给定模型、权重、数据集、统一定点语义和允许精度损失，
16 位是否足够，以及更高位宽达到什么精度”。实验不能提前保证
“所有大型神经网络必须高位宽”，也不能据这些结果推出每个 DCF/ReLU
输入都必须是 32 位。本文 32 位配置表示协议支持的统一工作域，最小精度
需求需要按具体网络测量。
