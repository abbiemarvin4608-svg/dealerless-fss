# 在线优化实现与整数核对

本轮保持 frozen f12/q10 folded 整数映射。算术环、查表域和输出舍入保持原数值合同。

已实现四个独立选项：

- `--folded-reuse-input-sign`：复用 signed16→32 输入扩位已经产生的秘密 XOR 符号分享；不生成核心重复 sign 库存。辅助分享绑定 channel、query/call binding 与 generation epoch，一次消费。
- `--folded-clip-bits 22`：只缩小 `abs(u)-65536` 的比较环和匹配库存；BitXA、abs、LUT 与输出继续使用 ring32。
- `--lut-skip-zero`：按公开非零区间跳过精确的零乘积，保留 1024 项定义域与旋转对齐。
- `--lut-simd`：运行时检测 AVX2，使用低 32 位乘加和标量尾部；无 AVX2 时回退标量。

完整 native int16 范围保证符号保持：Sigmoid 为 u=32n，TanH 为 u=64n。clip 比较输入分别在 [-65536,983040] 和 [-65536,2031616]，均可用 signed22。默认 generic activation 自动沿用当前环宽；native bridge 显式限定 32/22，通用 32 位任意输入不采用这一范围优化。

## 基础检查

| 检查 | 覆盖 | 结果 |
|---|---|---|
| LUT 本地核 | 22 张表 × 1024 旋转 × 4 配置，90112 次两分享检查；零表、分散非零区间、尾部、模溢出与真实 533 项支持 | passed；AVX2 available=true |
| 输入辅助符号 | 7 个边界 × fusion0/1，无额外通信/轮数；辅助输出符号与重构输入一致 | 双方 passed |
| native sign/clip | 19 边界，两激活；sign/LUT 两后端 × reuse0/1 × clip32/22；完整 65536 输入本地范围核对、库存消费及绑定 | 双方 passed |
| 旧 API 兼容 | 同 context generic16→paper32、FSS segments4→2 | 双方 passed |

日志和机器可读状态在 [checks](checks/PRIMITIVES.json)。运行时配置、计量、阶段通信和库存的 29 项轻量测试执行成功，其中 1 项依赖外部 integration binary 的检查跳过；实际新构建由整步资格覆盖，见 `checks/runtime-tests.log`。

## 整步独立参考

每配置：batch128、单个 timestep、零/非零初始隐藏状态各两次新鲜查询。每 query 核对 15 张量及最终隐藏状态，共 172928 个整数。

| 配置 | query 数 | 检查整数 | 差异 |
|---|---:|---:|---:|
| ours_online_before | 4 | 691712 | 0 |
| ours_online_optimized | 4 | 691712 | 0 |
| SCI_SiRNN_preprocessed | 4 | 691712 | 0 |
| SCI_SiRNN_unprocessed | 4 | 691712 | 0 |

ours 逐整数核对同一独立 folded 参考；SCI 两配置核对 native SCI 参考。方法之间的激活数值算法不同，不能以两者张量相等作为跨算法要求。未预处理 SCI 使用明确的输入后生成政策，其 prepared stock 和零 online generation 为 N/A，原始审计结果保留。

当前安全执行及成本覆盖单时间步。完整 6798 条 × 99 步加分类头的已有质量是同映射明文回放：folded 88.996764%，FP32 89.335099%，相差 0.338335 个百分点；本轮没有改变模型、量化或表。

首个资格批次因新构建脚本遗漏 Beaver/转换挂钩而在查询材料生成前失败，归档于 `qualification-invalid-build-hooks/`。构建入口已改为现有完整 `sci_conversion/build.py`，最终资格与正式批次使用同一新构建；失败批次不参与性能统计。
