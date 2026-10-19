# 严格 `2^ell` 环算术说明

## 显式严格环对照的规则

2026-10-08 用户恢复宽临时累加方案；当前默认为
`arithmetic_mode=wide, overflow_mode=saturate`。本文件保留之前要求的
严格环规则，运行该对照时显式指定 `ring + wrap`。

输入、参数与算子输出使用统一 Q(ell,s)，同时**乘积、每次点积累加、
bias 提升、BN affine、残差加法与池化求和也在同一 `2^ell` 环内**。
点积结束后，对最终环元素的有符号代表元做一次 floor 缩放。

严格环对照组合为 `arithmetic_mode=ring, overflow_mode=wrap`，逻辑
寄存器宽度等于 ell。定义

\[
W_\ell(z)=((z+2^{\ell-1})\bmod 2^\ell)-2^{\ell-1}.
\]

一次操作结果超出 `[-2^(ell-1), 2^(ell-1)-1]` 时按此式回绕，
不钳位到上下边界。参数/输入编码使用 `W_ell(floor(x * 2^s))`。

## 点积与一次缩放

```text
acc = W_ell(bias_code * 2^s)
for i:
    term = W_ell(x_code[i] * weight_code[i])
    acc = W_ell(acc + term)
y_code = W_ell(floor(acc / 2^s))
```

对负数的 floor 除法等价于有符号算术右移，不是向零截断。bias 代码
存储尺度为 `2^s`，在原始乘积尺度下提升并先取模。BN 的 multiplier
与 offset 按相同规则执行。

例如 ell=8、s=2，单项代码乘积 `10 * 20 = 200`：

| 数值模型 | 缩放前值 | 输出代码 |
|---|---:|---:|
| 严格 ell 位环 | `W_8(200) = -56` | `floor(-56/4) = -14` |
| 宽点积后缩放 | `200` | `floor(200/4) = 50` |

缩放之后的输出都在 8 位范围内，但它们已经不同。因此，仅检查最终
层输出的范围不足以证明点积遵守 ell 位环；必须在缩放之前丢弃环外
高位。

## 残差、池化与比较

残差输出为 `W_ell(left_code + right_code)`。平均池化逐项执行
`sum = W_ell(sum + code)`，再用有符号环和 floor 除以窗口大小。
例如 ell=8，4 个输入代码均为 100：环和为 `W_8(400)=-112`，平均
输出为 `-112 // 4 = -28`；宽和求平均得到 100，语义不同。

ReLU、maxpool 和 argmax 使用环代码的有符号代表元比较。公共除法与
缩放按照有符号 floor 定义，不把它们替换为环中的乘法逆元。

## 为什么仍然可以使用 int64 与整数 GEMM

`int64` 仅提供宿主容器。局部乘法或整数 GEMM 的 limb 点积可以采用
更宽的实现临时量，但网络逻辑值遵守 ell 位模约束。
对整数而言：

\[
W_\ell\left(\sum_i W_\ell(x_iw_i)\right)
=W_\ell\left(\sum_i x_iw_i\right).
\]

因此整数 limb 重组可通过等价的模乘加生成最终 acc。缩放必须读取
`W_ell(acc)` 的有符号值，而不是利用宽精确点积的高位。实现层次上
并不要求每个 CPU/GPU 寄存器都物理压缩成 ell 位；这是一项数值语义
测试，物理内存大小和两方协议成本需要单独测量。

## 回绕指标如何解释

`offline_wrapped` 与 `input_wrapped` 是编码越界的元素数。
`activation_wrapped` 是后端计算发生模回绕的输出数，每个输出至多
计一次；不是所有原始乘积/累加前缀的事件总数。

`activation_overflow_event_scope=scalar_mac` 对应标量 MAC，
`limb_reconstruction_or_scalar_mac` 对应整数 GEMM 的 limb 重组加其余
标量算子（BN、池化和残差）。数值环代码等价
不意味着两种 scope 的回绕计数相等，计数不能直接跨后端比较。
`max_register_bits_kind=logical_ring_width` 只表示已经回绕的逻辑值
位宽，不超过 ell，不可用作未经回绕点积最小所需位宽的证据。

## 旧结果与新的准确率实验

旧 `e0_uniform_storage_imagenette` 使用 128 位宽累加和饱和编码/写回，
属于 E0-U-S 对照；其“24 位接近 FP32”结论不适用于上述严格环规则。
旧 E0-Q-fixed 逐项缩放与饱和累加也不是严格环。

正式 ring 准确率实验预定输出到 `e0_uniform_ring_imagenette`，按同一
训练校准集扫描 ell 与 s，再用完整验证集评估训练选点。本轮只完成
实现修正与正确性验证，未生成新的完整 ring 准确率表。真实网络小
batch 的 scalar/GEMM 一致性属于诊断证据，不代替正式准确率。

本轮验证结果：28 项严格环测试、18 项历史对照测试、10 项调度测试通过。
完整预训练 ResNet50/152 的 Q(24,15) 小批量（各2张）逐算子范围与
scalar/GEMM logits 完全一致检查通过。命令、日志和源码快照位于
`../../results/uniform_ring_validation/README_CN.md`。
