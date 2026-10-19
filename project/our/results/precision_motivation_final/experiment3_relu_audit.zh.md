# 实验三 CNN ReLU 调用链审计

本文件审计实验三当前 CNN 实现、生成后的 SCI 源码和 ResNet-50/152 的 batch1 公开图。三个正式后端为 `SCI`、`SCI_preprocessed` 和 `SCI_ours_beaver`。这是代码与公开图审计，不是新增性能实测；本次没有启动协议或 GPU 测试，也没有修改协议代码。

## 结论与共同入口

三个后端都按 `ReLU(x) = [x >= 0] × x` 实现。每个 ReLU 元素只计算一次符号判断（DReLU），再执行一次秘密 Boolean bit 与算术环元素的选择乘法；同一个 ReLU 没有重复计算输入符号。零点采用非负判断，乘积仍为零。

共同入口见 [secure_driver.cpp:310](../../tools/imagenet_resnet/secure_driver.cpp:310)：

```cpp
auto signs = msb(x, bits);
if (options.role == 1) for (auto& s : signs) s ^= 1;
return select(signs, x, bits);
```

`msb` 返回输入符号的秘密 XOR 分享；仅由一方异或 1，得到非负判断的秘密分享，再传给 `select`。图节点派发见 [secure_driver.cpp:370](../../tools/imagenet_resnet/secure_driver.cpp:370)。虽然 setup 在第 248 行创建了 `ReLURingProtocol` 对象，当前 ReLU 节点没有调用其完整 ReLU helper，而是走上述共同分解。

## 三个正式后端的实际路径

| 后端 | 符号判断 | `bit × x` 选择 | 材料来源 |
|---|---|---|---|
| `SCI` | `AuxProtocols::MSB` | `AuxProtocols::multiplexer` | 原生 SCI 材料在线生成 |
| `SCI_preprocessed` | 同一 `AuxProtocols::MSB` 算法 | 同一 `AuxProtocols::multiplexer` 算法 | 原生 OT 和 bit-triple 材料在输入前生成，在线从库存消费 |
| `SCI_ours_beaver` | CNN MSB 包装调用 `segmentedDcfOnlineBatchGroup` | `bitxaOnline` | 输入前准备分段 DCF 和 BitXA 材料，在线逐批消费 |

后端分支见 [secure_driver.cpp:258](../../tools/imagenet_resnet/secure_driver.cpp:258) 的 `msb` 和 [secure_driver.cpp:282](../../tools/imagenet_resnet/secure_driver.cpp:282) 的 `select`。`SCI_ours_beaver` 名称中的 Beaver 指线性层实现；其 ReLU 的选择步骤仍调用专用 BitXA。

原生 SCI 的生成源码中，[AuxProtocols::MSB:180](../imagenet_resnet_20261015/final_migration/build/BuildingBlocks_aux-protocols.cpp.cpp:180) 提取低位、执行一次 `mill->compare(..., bw_x-1, true)`，再与本地最高位异或。其 [multiplexer:61](../imagenet_resnet_20261015/final_migration/build/BuildingBlocks_aux-protocols.cpp.cpp:61) 用两个并行方向的 COT 实现秘密选择，每方一发一收；MUX 内部没有再调用 MSB 或 compare。一次逻辑比较不代表只有一个通信轮，两个 COT 方向也不代表重复 DReLU。

Ours 的 [cnn_bridge.cpp:69](../../tools/imagenet_resnet/cnn_bridge.cpp:69) 先打开掩码输入，再调用一次批量分段 DCF，最后合成符号分享；[cnn_bridge.cpp:157](../../tools/imagenet_resnet/cnn_bridge.cpp:157) 的选择分支调用 `bitxaOnline`。其在线实现见 [relu.cpp:295](../../src/relu.cpp:295)：环值与 Boolean bit 合并打开，余下乘积展开在本地完成。BitXA 离线阶段用 Beaver 乘法构造相关掩码，但这不等于在线选择时再次执行一个通用环乘法协议。

`SCI_preprocessed` 保持上述原生 ReLU 分解，改变的是底层材料的生成时机和消费方式。原生公开计划与预处理入口见 [secure_driver.cpp:473](../../tools/imagenet_resnet/secure_driver.cpp:473)；生成的 [prepared_cot.h:122](../imagenet_resnet_20261015/final_migration/build/prepared_cot.h:122) 从库存恢复随机 pads，在线交换 choice correction 和 payload，未改变 ReLU 的代数语义。

## 32 位输入与 31 位比较域

两个正式 CNN 图采用 `bits=32, frac=12`。ReLU 的输入、输出以及 BitXA/multiplexer 的算术环均为 32 位；`msb(x, bits)` 的参数也是 32。

内部比较域是 31 位：

- 原生 SCI 在 [AuxProtocols::MSB:195](../imagenet_resnet_20261015/final_migration/build/BuildingBlocks_aux-protocols.cpp.cpp:195) 比较 `bw_x-1` 个低位，再结合本地最高位恢复符号。
- Ours 在 [cnn_bridge.cpp:60](../../tools/imagenet_resnet/cnn_bridge.cpp:60) 生成 `width-1` 位阈值的 DCF 材料，在线第 73 行设置 `material.bits=call.bits-1`。第 84–96 行仍先处理完整 32 位掩码输入，再取低 31 位作为 DCF query，并结合掩码符号和打开值最高位恢复结果。

因此，31 位是 DReLU 内部比较的域宽，不是将网络输入量化成 31 位。Ours 的 `depth-cap-8` 策略对 31 位域使用四段，段宽为 8、8、8、7；计数和布局见 [segment.cpp:29](../../src/segment.cpp:29) 与 [segment.cpp:42](../../src/segment.cpp:42)。四段构成一次完整 DReLU 的内部计算，不是四次独立 DReLU，也没有将完整比较域缩为 8 位。

## 4096 分块与显式 ReLU 图统计

`msb` 和 `select` 分别在第 260 行和第 284 行按最多 4096 个元素分块。因此包含 N 个元素的 ReLU 节点执行 `ceil(N/4096)` 次批量 MSB 和相同次数的批量选择。driver 先完成该节点全部 MSB 批次，再完成全部选择批次；目前没有跨节点合并这些调用，也没有把 MSB 和 BitXA 合成同一个调用。

静态统计输入为 [ResNet-50 图](../imagenet_resnet_20261015/models/resnet50_w32_f12/graph.json) 和 [ResNet-152 图](../imagenet_resnet_20261015/models/resnet152_w32_f12/graph.json)。对每个 `op=relu` 节点，将 shape 各维相乘得到元素数，再逐节点累加 `ceil(elements/4096)`：

| batch1 公开图 | 显式 ReLU 节点数 | ReLU 元素总数 | 批量 MSB 次数 | 批量选择次数 |
|---|---:|---:|---:|---:|
| ResNet-50 | 49 | 9,608,704 | 2,364 | 2,364 |
| ResNet-152 | 151 | 21,048,832 | 5,206 | 5,206 |

表中两个调用次数列是每方、每次推理的批量 API 调用数。逐元素上，每个显式 ReLU 元素各对应一次 DReLU 和一次选择；这些数字不能解释为通信轮数。三后端共用图级调度，因此这些图级次数相同，底层协议成本不同。

## MaxPool 的额外调用

[secure_driver.cpp:382](../../tools/imagenet_resnet/secure_driver.cpp:382) 要求 MaxPool 的直接输入节点为 ReLU。第 389–397 行遍历窗口：第一个候选直接作为初值，后续候选执行 `out += ReLU(candidate-out)`。因此一个 3×3 MaxPool 对每个输出元素额外执行八次 DReLU 和八次选择。

两张公开图各有一个输出 shape 为 `[1,64,56,56]` 的 3×3 MaxPool，共 200,704 个输出元素。每次候选比较产生 `ceil(200704/4096)=49` 个批次，八次比较合计：

| 图 | MaxPool 候选比较次数/输出元素 | 额外逐元素 DReLU 数 | 额外批量 MSB 数 | 额外批量选择数 |
|---|---:|---:|---:|---:|
| ResNet-50 | 8 | 1,605,632 | 392 | 392 |
| ResNet-152 | 8 | 1,605,632 | 392 | 392 |

这些调用不包含在上一张显式 ReLU 表中。全图还有 AvgPool 和截断所需的比较或转换，不能将显式 ReLU 加 MaxPool 的次数当作全部非线性协议调用数。

## 已迁移优化及其适用范围

当前构建使用已有 RNN 优化构建的核心库；链接记录见 [build.json:369](../imagenet_resnet_20261015/final_migration/build/build.json:369)。optimized profile 的开关见 [secure_driver.cpp:167](../../tools/imagenet_resnet/secure_driver.cpp:167)。

Ours ReLU 的 MSB 路径已进入当前批量分段 DCF 实现，使用 depth-cap-8、批量 AES、equality/leaf-hash 剪枝、leaf correction-word 合并、生成 workspace 复用、direct COT、紧凑 key 序列化、打包通信和在线 workspace 复用。CNN 在 [cnn_bridge.cpp:60](../../tools/imagenet_resnet/cnn_bridge.cpp:60) 请求无需最终 equality 输出的离线材料，在第 93 行把持久 `msb_workspace` 传入批量在线接口。DCF 内部扫描按层合并 Boolean AND 打开，见 [segmented_dcf.cpp:447](../../src/segmented_dcf.cpp:447)；这属于同一次 DReLU 内部的优化。

BitXA 使用已有的批量合并打开与环值/bit 打包实现，见 [relu.cpp:71](../../src/relu.cpp:71)。CNN 转换路径也继承域融合、sign/wrap 融合等优化，但这些属于截断/转换内部，不能据此声称当前 ReLU 已与前一转换融合或复用了其符号。CNN 当前仍单独计算 ReLU 的 MSB。

RNN 的四项专用在线优化见 [CHECKS.zh.md:5](../online_optimization_20261014/CHECKS.zh.md:5)：`--folded-reuse-input-sign` 复用 signed16→32 扩位符号，`--folded-clip-bits 22` 缩小特定 clip 比较域，`--lut-skip-zero` 和 `--lut-simd` 优化 LUT。它们针对已有 Sigmoid/TanH folded 路径及其数值范围。当前 CNN 图没有 Sigmoid、TanH 或 LUT，没有采用该输入符号复用，也没有把这些专用优化套到通用 32 位 ReLU；RNN 时间步/图并行调度亦未迁入顺序 CNN 图。

## 证据边界

本审计确认实际调用语义、内部域宽、分块方式和公开图静态调用数。代码路径进入优化实现不等于已经实测了每项优化的收益。本文件没有新增端到端时间、通信量、精度或吞吐数据，不替代正式协议资格或性能结果。
