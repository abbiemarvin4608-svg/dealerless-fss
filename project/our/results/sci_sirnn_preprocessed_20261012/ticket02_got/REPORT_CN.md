# 工单 02：KKOT LUT 与批量 GOT 验收

本目录保存两方真实 SCI 协议的局部验收日志。未运行完整 FastGRNN 单步，
不构成完整 SCI_preprocessed 结果，也不用于最终 batch=128 加速比。
COT、比较 triple 和激活整体闭环由后续工单实现。

## 实现与阶段划分

私有 SCI header overlay 保留默认路径，向 KKOT 四种 typed 入口和
IKNP send/recv_batched_got 接入查询库存。每个查询在装载选择和消息前：

- KKOT 执行真实扩展、抽取随机选择，并用 CCRF 完成全部分支的随机掩码；
- 批量 GOT 执行真实 IKNP 扩展、随机选择和全部消息长度对应的 CRH 哈希；
  143、187、192 bit 的聚合消息已展开到最终掩码字节；
- 完成库存后释放扩展矩阵，双方确认 MATERIAL_READY，随后输入才进入。

在线保留 KKOT 的模 N 选择修正，或批量 GOT 的一位选择修正，以及真实输入
消息的打包、掩码和结果重建。消费契约覆盖查询、方向、数量、每条消息位宽和
每次 OT 的消息分组。库存不足、契约不同或重复调用直接失败，在线不补货。
普通 IKNP GOT 的既有预处理行为经回归测试保持一致。

## 验收入口

测试调用实际 AuxProtocols::lookup_table，覆盖 N=2/4/8/16/32/64/128/256，
uint8_t 位宽 1/7/8 和 uint64_t 位宽 13/64。每个 radix 连续消费两组
不规则数量（11 和 5），然后在下一查询重新生成库存。

位乘法调用实际 LinearOT::matmul_multiplexer：

- 3x3 bit 矩阵乘 3x11 算术矩阵，9 个 OT、13 bit、每个 OT 11 条消息；
- 11x3 算术矩阵乘 3x3 bit 矩阵，9 个 OT、17 bit、每个 OT 11 条消息；
- 两种布局均消费 straight/reversed OT，重建与独立明文结果相同；
- 另覆盖 7 个 OT 的 64 bit x 3 消息，以及 5 个 OT 的 5 bit x 1 消息。

每组执行 prepared 和 default 模式，均连续执行两个不同输入查询。
负例改变分组契约、重复消费，双方退出码均为 2，并输出明确错误。
所有正例的输入、ready、online、output 阶段原生生成 epoch 增量均为 0。

## 通信证据

以下值为两方发送字节之和，每个查询均相同；包括各阶段实际同步字节及尾部填充。

| 局部入口 | 离线发送字节 | 在线发送字节 | 库存 |
| --- | ---: | ---: | --- |
| 单个 N=8 LUT，11 个 13-bit 消息 | 8194 | 151 | 请求/消费 11，生成 256，填充丢弃 245 |
| 全部 radix 两组 LUT | 131074 | 29305 | 每 radix 请求/消费 16，生成 512，丢弃 496 |
| 位矩阵及混合长消息 | 32770 | 3274 | 每方向请求/消费 30，生成 1024，丢弃 994 |

位矩阵在线日志额外包含测试核验所需的明文重建（每方 528 byte）；
这些局部秒数和重建通信不会作为完整单步协议性能。
KKOT 的选择修正按 count * log2(N) 记载，实际发送按字节取整；
批量 GOT 每个随机 OT 发送一位选择修正，消息掩码按分组总长度逐 OT 取整。
生成数量保留 SCI 扩展的 SIMD 取整，填充材料不会进入下一查询。

## 命令与结果

在 runtime 工具目录执行：

    flock /tmp/sci-preprocessed-build.lock env SDCF_LUT_GOT_EVIDENCE_DIR=../../results/sci_sirnn_preprocessed_20261012/ticket02_got python3 -m unittest test_prepared_lut_got test_prepared_got -v

6 项测试全部通过。编译时使用 O2、真实 SCI 源码与 SCI-OT/GC 静态库；
本工单协议测试每方限制最多 4 CPU 和 1.5 GiB 地址空间，OpenMP 为 2。
子目录保存每个模式的两个 party 日志及退出码。
