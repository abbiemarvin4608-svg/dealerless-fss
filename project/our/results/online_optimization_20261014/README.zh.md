# SCI+ours 在线优化实验

四组正式测试已完成，每组 10 次新鲜查询。结果文件：[Excel](formal/REPORT.xlsx)、[完整报告](formal/REPORT.zh.md)、[原始汇总](formal/METRICS.json)、[最终核对](formal/ARTIFACT_QA.json)。Excel 保留在线优化前、优化后、预处理 SiRNN 和未预处理 SiRNN 四组，分别列出 LAN/WAN 的离线、在线及合计，中位数和 p95；计算时间同时提供进程 CPU 与本机墙钟。

优化后在线通信为 4,467,106 B，较优化前减少 19.03%；因果轮数 139→98，减少 29.50%；单方最大 CPU 时间中位数 0.363321→0.286456 秒，减少 21.16%。在线本机墙钟中位数 0.149078→0.124187 秒；但 p95 为 0.932727 秒，高于优化前的 0.164528 秒，本轮没有证明尾延迟改善。

SCI+ours 在线通信低于预处理和未预处理 SiRNN，但离线仍需 1,085,775,374 B、827 轮，离线与在线合计开销仍高于两组 SiRNN。LAN/WAN 是以下通信模型的估算，并未在对应限速网络测量墙钟。

四配置分别在零/非零隐藏状态下核对 691,712 个整数，差异均为 0；本轮保持原 folded 整数映射。实现与检查见 [CHECKS](CHECKS.zh.md)。

统一构建路径为 `build/core/` 和 `build/runtime/`。脚本默认加载其中的 bridge/runtime，并沿用上一轮相同 fixture。每方4个逻辑CPU、1.5GiB地址空间上限、同一分配器配置、batch128、单个 recurrent timestep、warmup0。本轮结果写入新目录，不混入历史正式样本。

默认四组：

| 配置 | 后端与改动 |
|---|---|
| ours_online_before | 上一轮完整offline优化：full10、sign/wrap融合、block/ring COT、叶缓存 |
| ours_online_optimized | 在上一行增加输入符号复用、22-bit clip比较、跳过LUT零系数、SIMD |
| SCI_SiRNN_preprocessed | 原生SCI/SiRNN，query native材料在input前准备 |
| SCI_SiRNN_unprocessed | 同一个runtime，`--backend SCI`，不启用`--native-preprocessing`，协议生成归online |

`--include-ablations`增加输入符号复用、再加clip22、再加LUT零系数三行；最终行再加入SIMD。各行都有资格检查：默认/非零隐藏状态各2次全新查询，使用独立整数oracle核对所有15个张量。未预处理SCI的零online生成与prepared库存检查标为N/A，并要求online出现真实native生成；原auditor检查结果原样保留。

```bash
python3 project/our/results/online_optimization_20261014/run_experiments.py qualify \
  --output project/our/results/online_optimization_20261014/qualification

python3 project/our/results/online_optimization_20261014/run_experiments.py formal \
  --output project/our/results/online_optimization_20261014/formal

python3 project/our/results/online_optimization_20261014/summarize.py \
  --directory project/our/results/online_optimization_20261014/formal
```

需要消融时在前两个命令均加入`--include-ablations`。可用`--binary`、`--bridge`、`--fixture`、`--hidden`、`--qualification`指定路径；`--dry-run`只写PLAN/STATUS，不运行party。formal要求所选行在同一构建上的两种隐藏状态资格检查都通过。runner通过`LD_LIBRARY_PATH`及`ldd`核对实际加载的bridge。

汇总生成`METRICS.json`、`REPORT.zh.md`和`REPORT.xlsx`。Excel包含LAN/WAN、逐query、Setup、资源库存、比较、说明sheet。

CPU时间采用各阶段`CLOCK_PROCESS_CPUTIME_ID`，累计进程全部线程的user+sys时间，包含trace及syscalls，可能超过墙钟。报告单方最大CPU、两方总CPU和两方最大本机墙钟；本机墙钟包含通信等待，不能叫纯计算。

LAN默认为1000Mbps/RTT0.5ms，WAN为400Mbps/RTT4ms。通信投影为`8×两方sent bytes/带宽 + RTT×因果轮数`，分别列传输、RTT与合计；没有测量对应网络墙钟，也不把本机wall与投影相加。未预处理SCI测得的offline原样保留，setup/model_prepare独立呈现。

均报告10query的median及p95；p95采用排序索引`0.95×(n−1)`的线性插值，无置信区间。离线＋在线在逐query求和后统计，排除setup/input/ready/output。SCI采用原生激活，ours采用folded f12/q10，跨方法比较包含该整数算法差异。
