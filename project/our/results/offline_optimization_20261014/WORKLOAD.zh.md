# 离线优化减少了哪些工作量

本说明比较资格检查的 `ours_full10_zero` 与 `ours_optimized_zero`：batch 128、单个 recurrent timestep、两次全新查询，使用相同的 full q10 LUT。这里报告整数工作计数及生成时的内存估算；资格检查中的时间不作为性能结果。七组正式实验的通信、轮数和时间由 `formal/REPORT.zh.md` 单独报告。

数据来自 [full10 parties.json](qualification/ours_full10_zero/parties.json)、[optimized parties.json](qualification/ours_optimized_zero/parties.json) 和两组的 `party1.log` / `party2.log`。下表均按**每方、每次查询**计数。两方的工作计数一致；两个查询的计数也一致。DCF 日志没有 query 字段，因此先核对其前后两半的 `(bits,count,nseg,hashes,products,OT counts)` 序列完全一致，再按两次查询拆分。

## 1. 融合 sign 与 wrap，删除第二套比较树

有符号转换原先分别准备低位借位、符号比较和完整输入位宽的 wrap 比较。融合后保留低位和符号比较，用一个 Boolean AND 重建联合校正。源码中的恒等式是：令 `a = r_MSB`、`l = [z_low < r_low]`、`b = z_MSB`，则

```text
wrap - sign = (1 - 2*b) * ((a XOR b) AND (l XOR b))
```

该 AND 使用新鲜的离线材料；因此融合既减少比较树，也引入每个融合元素一个 AND。实现见 [sci_truncation.cpp](../../src/sci_truncation.cpp) 的域规划及联合校正。

图中的 19 个转换调用具有如下工作量。`count × bits` 对应比较树逐层 bit×block 产品数，`count × nseg` 对应分段键数。

| 转换比较域 | full10 调用数 | optimized 调用数 | full10 元素数 | optimized 元素数 | full10 count × bits | optimized count × bits |
|---|---:|---:|---:|---:|---:|---:|
| low / 借位 | 12 | 12 | 115,456 | 115,456 | 843,008 | 843,008 |
| sign | 15 | 15 | 170,752 | 170,752 | 3,396,864 | 3,396,864 |
| wrap | 15 | 0 | 170,752 | 0 | 3,567,616 | 0 |

例如 `MatMul_01` 的 slot 0 为 4,096 个 16→29 bit 扩展。原先 sign 的比较位宽为 15，wrap 为 16，各分 4 段；优化后 sign 仍为 `[4,4,4,3]`，wrap 位宽和段数均为 0。纯缩位转换没有独立 sign/wrap 域。

此外，四个 folded activation worker 各处理 6,400 个元素；其 sign/wrap 融合再删除 25,600 个 16-bit wrap 比较实例。图转换和激活合计删除 **196,352 个 wrap 实例、785,408 个分段键、3,977,216 个 bit×block 产品**，并准备 **196,352 个**额外 Boolean AND，每个融合元素一个。

| 全部 DCF 生成工作（包括转换与 folded activation） | full10 | optimized | 减少 |
|---|---:|---:|---:|
| 生成调用数 | 52 | 33 | 19 |
| 比较实例数 | 520,960 | 324,608 | 196,352 |
| 分段键数 `sum(count × nseg)` | 2,026,496 | 1,241,088 | 785,408 |
| 内部节点 PRF | 58,009,472 | 27,416,448 | 30,593,024 |
| 叶节点 PRF | 120,071,936 | 57,315,072 | 62,756,864 |
| PRF 合计 | 178,081,408 | 84,731,520 | **52.42%** |
| bit×block 产品 | 8,665,088 | 4,687,872 | **45.90%** |

这些是 `dcf_generation` 的实际整数计数；分段键数由每条记录的 count 和 nseg 相乘获得。每条记录的 extra hashes、临时产品 triple 和产品 E/F opening 均为 0。

## 2. 相关 OT 使用 native COT

bit×block 产品的两组消息具有已知 XOR 差值，可以直接使用 COT。它保留每个元素独立的差值，并由 OT 扩展生成新鲜的随机发送基值。实现见 [secure_ops.cpp](../../src/dfss/mpc/secure_ops.cpp) 的 `direct_cot` 分支与 [comms.cpp](../../src/dfss/mpc/comms.cpp) 的 per-element block COT。

| DCF 内 OT 账本 | full10 | optimized |
|---|---:|---:|
| general 128-bit OT | 17,330,176 | 0 |
| native 128-bit COT | 0 | 9,375,744 |

账本在每方均记录发送、接收两个方向，因此 OT/COT 计数都是 `2 × bit_block_products`。两方的账本记录的是相同的逻辑交换，不能再把两方账本相加当作额外的实例数。

对于**相同的产品数和 OT 分批方式**，native COT 的 correction payload 每方向由两个 128-bit 块变成一个；两个方向合计每个 bit×block 产品少 32 字节。以优化后的 4,687,872 个产品推导，payload 少 150,011,904 字节（143.0625 MiB）/查询。这个推导只解释 block COT 的消息成本，端到端通信还包括 OT 扩展矩阵、开值和其他算子，应读取正式实验的两方 sent 总量。

ring COT 同样把普通 OT 的两条模环消息改为一条相关消息，适用于 GroupElement multiplexer/B2A 路径，按真实位宽 1–64 bit 序列化。资格检查没有输出 `native_ring_cot_values`，因此这里不报告它的运行次数或单独的通信收益；正式实验的 `fused_blockcot → fused_bothcot` 消融量化其整体效果。

## 3. full q10 LUT 每棵树只展开一次

原 fullQ 路径先展开树求 payload 校正，再展开同一棵树生成 selection shares。缓存本 tile 的叶节点后，第二步直接读取缓存。LUT 域为 1,024，每个 selector 的一次展开包含 511 次内部 PRF 和 1,024 次叶 PRF，见 [dpf_lut.cpp](../../src/dpf_lut.cpp) 的 `fullQBatchOffline`。

| 本地 LUT 展开工作（不含 DCF 与 cGGM 交互式 keygen） | full10 | optimized |
|---|---:|---:|
| selector 数 | 25,600 | 25,600 |
| 树展开次数 | 51,200 | 25,600 |
| 内部节点 PRF | 26,163,200 | 13,081,600 |
| 叶节点 PRF | 52,428,800 | 26,214,400 |
| PRF 合计 | 78,592,000 | 39,296,000 |

optimized 数值来自每方 8 条 `LUT_LEAF_CACHE` 日志：4 条/查询，每条 6,400 selectors。full10 未输出本地展开计数；其数值由相同 selector 数和源码的 `expansions = n × (cache ? 1 : 2)` 推导，明确区分于运行日志计数。缓存让这一部分 PRF 次数减少 **50%**，并减少重复的树节点生成。

每个 worker 的 6,400 selectors 分成 7 个 tile：6×1,024 加尾部 256。叶缓存峰值为 `1,024 × 1,024 × 16 = 16,777,216` 字节，即 **16 MiB/worker**。缓存随 tile 销毁，不进入常驻查询材料。folded worker 的临时 working estimate 从 38,509,056 增到 55,286,272 字节，恰好多 16 MiB；缓存通过 1.5 GiB budget、512 MiB reserve 和常驻材料估算选择 tile 大小。

DCF 仍使用预算内分块与可复用工作区。资格检查中单个 DCF 生成调用的最大 working estimate 分别为 318,520,960 和 318,851,840 字节；减少总工作量不会必然减少单块峰值。最大工作区 capacity 分别为 74,458,800 和 70,883,928 字节。这里的 working/capacity 是算子分配估算，进程 RSS/VmPeak 由正式报告给出。

## 4. 核对计算任务与新鲜材料数量

两组每次查询均执行 19 个图转换调用、6 个 Beaver 生成调用、4 个 folded activation worker，并生成和消耗 25,600 个一次性 activation selectors；余额为 0，online_generated 为 0。

Beaver 算术工作量一致：4 个矩阵 triple、25,600 个逐元素 triple、141,464 个 material elements、32 个 cross-term batches 与 16 个 local products。上述 DCF/COT/cache 优化的工作量收益来自校正融合、消息表示和消除重复展开；正式结果需要用七组消融分别量化这些改动及从 segmented LUT 切换到 full10 LUT 的影响。
