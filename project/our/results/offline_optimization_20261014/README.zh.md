# SCI+ours 离线优化与 SiRNN 复测

四项优化已实现，七组正式运行全部完成。相对优化前 5+5 LUT，离线通信减少 61.83%，本机离线墙钟中位数从 11.8113 s 降至 5.1630 s，减少 56.29%。

完整数值、累计消融、median/p95 与资源峰值见 [正式报告](formal/REPORT.zh.md)；逐 query 数据见 [METRICS.json](formal/METRICS.json)。

| 方法 | 离线通信 (MiB) | 在线通信 (MiB) | 本机离线 / 在线墙钟中位数 (s) | LAN 离线 / 在线投影 (s) | WAN 离线 / 在线投影 (s) |
|---|---:|---:|---:|---:|---:|
| SCI/SiRNN | 176.97 | 87.19 | 1.6668 / 0.4653 | 1.5196 / 0.8239 | 3.9914 / 2.5686 |
| SCI+ours 优化版 | 1069.48 | 5.26 | 5.1630 / 0.1455 | 9.3849 / 0.1136 | 25.7366 / 0.6663 |

LAN 为 1000 Mbps、0.5 ms RTT，WAN 为 400 Mbps、4 ms RTT。协议在未限速 localhost 运行，网络数值由实测双端发送字节和核验后的因果轮次投影，只计通信时间。每组为 batch128、单个 recurrent timestep、零初始隐藏状态、10 次 fresh 材料、无 warmup、每方4逻辑CPU和1.5GiB地址空间；这不是完整99步网络推理成本。

优化版在线投影分别为 SiRNN 的13.79%（LAN）和25.94%（WAN），离线分别为6.176倍和6.448倍。每次重新生成材料后的离线＋在线投影分别为9.4986 / 26.4029 s，SiRNN为2.3435 / 6.5600 s；合计仍高约4倍。两方法保留各自原生SCI/folded激活算法。

改动与证据：

- full10 算术 DPF 替换 5+5 Boolean tensor LUT。
- 有符号 sign/wrap 比较融合，适用调用删除 wrap DCF；在线 AND 增加5轮整步依赖。
- `direct_cot` 与 native ring COT 压缩相关OT消息。
- 当前 tile 叶缓存消除重复树展开，缓存受生成预算约束。
- [正确性检查](CHECKS.zh.md)：COT、融合、缓存、序列化/预取以及零/非零隐藏状态独立整数核验均通过。
- [算法工作量](WORKLOAD.zh.md)：区分实际计数和源码推导。
- [固定构建](build/FINAL_IDENTITY.json)、[源文件指纹](build/final-source-hashes.json)、[正式计划](formal/PLAN.json)、[执行状态](formal/STATUS.json)。

完整开关例子见 [runtime README](../../tools/segmented_dcf_runtime/README.md)。固定构建的完整重复运行命令为：

```bash
python3 project/our/results/offline_optimization_20261014/run_experiments.py formal \
  --output project/our/results/offline_optimization_20261014/formal-repeat
python3 project/our/results/offline_optimization_20261014/summarize.py \
  --directory project/our/results/offline_optimization_20261014/formal-repeat
```

重复运行需要可绑定本机端口，并使用新的输出目录。正式脚本会验证已通过资格检查的runtime与动态桥接库标识，按计划逐组执行。每个子目录保留双方日志、通信轨迹、配置、命令、结果和库存记录。原有用户修改已保留；`final-worktree.diff` 是整个未提交工作区的快照，包含本轮之前的改动。
