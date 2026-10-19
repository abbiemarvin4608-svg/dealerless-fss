# 工单 05：原生 Sigmoid/TanH 完整预处理

本票在真实两方 MathFunctions 入口完成原生 SCI Sigmoid/TanH 内部材料的输入前生成，
保持原近似函数、尺度、舍入和中间环。测试使用冻结单步的参数：
Sigmoid 输入/输出环宽 16/16，尺度 7/14；TanH 环宽 16/16，尺度 7/7。
没有替换激活数值算法。

## 完整链路

公共规划执行真实 MathFunctions 链路，涵盖实际比较、LUT、MUX、B2A、hadamard、
截断与扩环调用。独立 scratch 输入全零和全 65535 两次规划的规范请求逐字相同；
规划通信、原生材料生成、bit-triple 生成和 PRG block 推进均为零。
真实输入只在材料 seal 和双方 MATERIAL_READY 之后装载。

新增共享 prepare_query_materials 接口按公开上下文顺序回放所有 OT 和安全 bit-triple
请求，生成最终随机 OT 掩码及比较 triple。输入后的在线执行保留选择、算术相关量、
掩码消息和比较打开，只消费新鲜库存，禁止扩展、triple 生成或自动补货。

两次不同查询分别完整执行两种激活，每次都重新生成库存。日志包括公共子调用的
协议名、方向、位宽、消息分组和数量，并通过消费游标与实际库存平衡核验全部调用。
各激活的材料计数如下，12800 元素的两个查询计数相同：

| 原生激活 | 原生 OT 消费 | bit-triple 消费 | 在线材料生成增量 |
| --- | ---: | ---: | ---: |
| Sigmoid | 1913600 | 281600 | 0 |
| TanH | 1222400 | 140800 | 0 |

bit-triple 每查询合计请求/生成/消费均为 422400，库存剩余为零。
OT 的 SIMD 取整和填充丢弃单独记录；比较应用数量的取整保留实际有效请求。
下一个查询重新生成材料，不复用上一查询随机 OT 或 triple。

## 整数与资源验收

实际执行 13 元素尾部和 batch=128 对应的 12800 元素两组测试；每组在默认与
预处理模式各运行两次查询，两种激活的所有输出逐整数相同。
输入覆盖正值、负值、零与 ±1，尺度边界附近，16-bit 最大/最小值及其邻值。
12800 元素测试含全部 25 种边界输入，第二查询轮换输入并改变两方共享。

构建为 O3、DNDEBUG、真实 SCI 源码与 SCI-OT/GC 静态库，限制每方最多 4 CPU、
1.5 GiB 地址空间，OpenMP 2，Eigen 不额外开线程。
预处理大向量实测峰值 RSS：party1 328852 KiB，party2 67304 KiB。
本票为单上下文验收，多 worker 与两分支库存由工单 07 验证。

## 阶段通信

12800 元素每个查询双方发送字节总和：

| 阶段 | 两方发送字节 |
| --- | ---: |
| offline，包括公共计划核对及真实材料生成 | 80093302 |
| input | 0 |
| ready | 2 |
| online，完整原生激活 | 31481602 |
| output，包括两种激活的诊断重建 | 204802 |

输入、ready、online 和 output 阶段的原生生成 epoch 增量均为零。
所有重建和完整输出日志在 output 阶段，在线计量不含诊断重建。
日志中的时间为局部链路验收，不是四后端完整单步加速比；正式公平结果由工单 08 重测。

## 复现

在 segmented_dcf_runtime 工具目录执行：

    env SDCF_ACTIVATION_EVIDENCE_DIR=../../results/sci_sirnn_preprocessed_20261012/ticket05_activation python3 -m unittest test_prepared_activations -v

编译自动持有 /tmp/sci-preprocessed-build.lock。
2 项测试通过，保存的 default_13、prepared_13、default_12800、prepared_12800
目录包含两个 party 的阶段日志、诊断输出、退出码和资源上限。
