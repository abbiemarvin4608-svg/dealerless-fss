# SCI+ours 在线优化候选

优先实施**复用输入适配器符号、收窄 clip 比较、LUT 跳零项与 SIMD**。前两项删除重复或不必要的比较工作，第三项降低本地查表计算；都可以保留当前整数算法。以下是现有代码与测量的分析，候选尚未实现或复测。

当前口径为 batch128、单个 recurrent timestep、每组10次新鲜材料。优化版在线墙钟中位数为0.145469 s，双端发送总量5,516,706 B（5.261 MiB），完整因果轮次139。WAN通信投影0.666334 s，其中139轮×4 ms=0.556 s，约83.4%来自往返依赖。

完整 folded 激活占2,883,392 B，即全在线通信的52.3%。阶段数据来自[原始双方记录](formal/ours_optimized/parties.json)，整理在[ONLINE_HOTSPOTS.json](ONLINE_HOTSPOTS.json)。

| 激活步骤 | 每查询双方发送量 | 占全在线通信 | 每个 worker 的阶段轮次 |
|---|---:|---:|---:|
| sign | 793,600 B | 14.39% | 31 |
| clip 比较 | 793,600 B | 14.39% | 31 |
| abs、clip、restore 三次 BitXA | 633,600 B | 11.49% | 各1 |
| index truncation | 256,000 B | 4.64% | 2 |
| LUT opening | 102,400 B | 1.86% | 1 |
| TanH output adapter | 131,264 B | 2.38% | 4 |
| LUT 本地内积 | 0 B | 0% | 0 |

阶段轮次属于各 worker 路径，不能将并行 worker 的轮次相加替代139轮整图记录。LUT本地内积每方累计工作中位数约26.86 ms，亦不能当作并行墙钟。

## 1. 复用输入适配器已经计算的符号

[folded_native.cpp](../../src/folded_native.cpp) 当前先执行 signed16→32 扩位，再乘32得到Q12；TanH随后再乘2。[sci_truncation.cpp](../../src/sci_truncation.cpp) 在扩位时已经计算出原输入符号的XOR分享：

```text
sign = sign_compare.L XOR mask_sign_share XOR public(z_MSB)
```

原始int16输入乘32的范围为[-1,048,576,1,048,544]，乘64为[-2,097,152,2,097,088]，均不溢出signed32，所以符号保持。将这一分享交给激活核心，转成nonnegative分享，可跳过当前31层 `signBitsOnline`。

可删除的现有阶段对应793,600 B/query（全在线通信14.39%）及每条激活路径的31层依赖；还可删除1,587,200个Boolean sign triples/query。上述数值是被删除阶段的计数，净墙钟与完整因果轮次仍需复测。

实现需要让输入扩位API返回辅助符号分享，为激活新增显式准备模式，并调整当前对sign/clip同长度库存的验证。符号分享只用于同一query、context、call和输入；准备时就不生成被跳过的sign库存。

## 2. 只把 clip 比较从32位缩到22位

当前激活核心固定ring32，但clip只判定 `abs(u)-65,536` 的符号。覆盖全部int16输入时，该值在Sigmoid路径为[-65,536,983,040]，在TanH路径为[-65,536,2,031,616]；两者均在signed22范围[-2,097,152,2,097,151]内。

因此可以保留ring32的BitXA、LUT和输出适配器，只将clip比较输入局部归约到ring22，增加独立的 `comparison_bits` 和匹配的Boolean库存。串行进位从31层变21层；按当前打包布局推导，可再删除256,000 B/query（全在线通信4.64%）和512,000个Boolean triples。

这比同时改全部算术工作环更容易核验。全路径收窄也可探索：保守采用Sigmoid ring22、TanH ring23，需同步修改表输出环、输入扩位和TanH输出截断合同。

## 3. LUT 精确跳过零系数，并做 SIMD

当前公开表为 `delta(j)=floor(4096/(1+exp(j/64)))`，j=533…1023的491项均为0。[dpf_lut.cpp](../../src/dpf_lut.cpp) 的 `shiftedDotProduct32` 仍遍历1024项。可按公开非零区间和已公开的旋转量，只扫描对应的selector连续片段。

每方每查询乘加数可从26,214,400降为13,644,800，减少47.95%；静态selector读取量从100 MiB降为约52.05 MiB。它保持整数结果不变，通信量不变，乘加降幅不代表整个在线墙钟同幅下降。

当前内积已经消除了逐项旋转取模和表复制，但仍逐项解码字节并执行标量乘加。可继续使用AVX2低32位乘加、多个累加器和分块访存；以模2^32语义保持结果一致。

## 4. 剩余 clip 比较改为 FSS 或并行进位树

已有 `--folded-sign-backend segmented_fss` 开关同时替换sign和clip；fixed-4下private MSB由masked opening与两层scan组成，适合检查WAN轮次收益。它增加离线比较材料和在线本地FSS评估，历史版本曾出现在线本机墙钟退化，当前构建需要重新消融。

在复用adapter sign之后，更合理的是仅替换剩余clip，避免准备不再需要的sign树。另一个选择是将串行进位改为只求最高位的平衡进位树，降低依赖深度，但需评估新增AND数量与字节量。

## 其他较小或需要新协议的路径

- 合并图内“截断后本地reduce”：六处调用、57,728个元素可删除冗余fused sign/wrap AND和一项B2A校正。按现有位打包推导约减少43,296 B/query，离线材料也减少；toward-zero AND仍在原批次，阶段轮次不变。修改必须表达新的目标环合同。
- 将联合AND结果直接转换为算术校正：使用离线mixed Boolean/arithmetic材料，在打开AND掩码的同轮完成B2A，适用转换可由5轮降至4轮；增加离线准备成本。
- Sigmoid先完成后继续执行仅依赖它的节点10/11/12，与TanH剩余工作重叠，减少当前join等待；完整收益需因果轨迹验证。
- 把TanH最终Q7舍入并入双输出LUT，保留负分支floor/正分支ceil的区别，再用ring16恢复符号；可省输出截断，但增加本地表乘加，需要与前面的LUT优化一起评估。

当前packing、域合并、配对扩位、图并行、尾部并行均已开启。`graph-openings batched`主要合并MatAdd5/7，而本次model-bias precompute已使它们成为本地运算，因此该开关对当前配置收益有限。后续应继续同时报告在线、离线与两者合计，尤其检验FSS和mixed材料是否只是将成本转移到预处理。
