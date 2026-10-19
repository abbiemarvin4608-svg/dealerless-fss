# ticket13 同配置 LAN/WAN 在线与离线复测

PaperLUT 的在线通信模型相对 SCI+SiRNN 在 LAN 降低 86.18%（7.2350x），在 WAN 降低 74.57%（3.9323x）。
其离线通信模型为 SCI+SiRNN 的 LAN 15.7830 倍、WAN 15.6798 倍。

## 配置与口径

两套方案使用同一持久构建和 fixture，batch=128、steps=1、parallel-a2、10 个 fresh query、warmup=0。每方四个 CPU、1.5 GiB 地址空间上限，generation budget 显式为 1610612736 B，reserve 与 prefetch 各为 536870912 B。
两套使用统一的 allocator 环境：移除继承的 MALLOC_* 和 GLIBC_TUNABLES 后固定 MALLOC_ARENA_MAX=2、MALLOC_TRIM_THRESHOLD_=1、MALLOC_MMAP_THRESHOLD_=131072；控制器也设置 1.5 GiB 地址空间上限。
PaperLUT 使用 folded_activations=both、segmented_boolean_tensor_b2a（5+5 bits）、folded_sign_backend=bitdec，Sigmoid/TanH 都采用完整 folded LUT；SCI+SiRNN 对照为 SCI_preprocessed，保留 SCI 原生激活。这里比较两套方法的完整单步成本。两套执行顺序进行。

LAN 为 1 Gbps / 0.5 ms RTT，WAN 为 400 Mbps / 4 ms RTT。协议在 localhost 执行；LAN/WAN 数字按用户给定模型从实测字节和因果轮次计算。

```text
D = 两方 sent bytes 之和（每条消息计一次）
传输时延 = 8 × D / 带宽(bit/s)
传播时延 = logical_rounds × RTT(s)
通信时延 = 传输时延 + 传播时延
```

轮次取两方因果 send-flight 的关键路径，保留并行分支的 fork/join 依赖。每 query 先算各项，随后取 10 个 query 的中位数。offline 为输入前的新材料生成与就绪工作；online 为 input/ready 之后的完整协议执行。两阶段表不含 setup、model_prepare、input、ready、output 和 teardown。

## 复测结果

| 方法 | 阶段 | 发送 bytes | 因果轮次 | localhost 墙钟(s) |
| --- | --- | ---: | ---: | ---: |
| SCI+SiRNN（SCI_preprocessed） | offline | 185,571,244 | 70 | 1.680032026 |
| SCI+SiRNN（SCI_preprocessed） | online | 91,429,766 | 185 | 0.483954080 |
| Segmented DPF PaperLUT | offline | 2,938,230,430 | 955 | 11.606024504 |
| Segmented DPF PaperLUT | online | 5,860,322 | 134 | 0.202816914 |

### LAN

| 方法 | 阶段 | 传输(s) | 传播(s) | 通信合计(s) |
| --- | --- | ---: | ---: | ---: |
| SCI+SiRNN（SCI_preprocessed） | offline | 1.484569952 | 0.035000000 | 1.519569952 |
| SCI+SiRNN（SCI_preprocessed） | online | 0.731438128 | 0.092500000 | 0.823938128 |
| Segmented DPF PaperLUT | offline | 23.505843440 | 0.477500000 | 23.983343440 |
| Segmented DPF PaperLUT | online | 0.046882576 | 0.067000000 | 0.113882576 |

### WAN

| 方法 | 阶段 | 传输(s) | 传播(s) | 通信合计(s) |
| --- | --- | ---: | ---: | ---: |
| SCI+SiRNN（SCI_preprocessed） | offline | 3.711424880 | 0.280000000 | 3.991424880 |
| SCI+SiRNN（SCI_preprocessed） | online | 1.828595320 | 0.740000000 | 2.568595320 |
| Segmented DPF PaperLUT | offline | 58.764608600 | 3.820000000 | 62.584608600 |
| Segmented DPF PaperLUT | online | 0.117206440 | 0.536000000 | 0.653206440 |

| 方法 | offline+online LAN 通信(s) | offline+online WAN 通信(s) |
| --- | ---: | ---: |
| SCI+SiRNN（SCI_preprocessed） | 2.343508080 | 6.560020200 |
| Segmented DPF PaperLUT | 24.097226016 | 63.237815040 |

将每次新材料生成也计入 offline+online 后，PaperLUT 的通信模型为 SCI+SiRNN 的 LAN 10.2825 倍、WAN 9.6399 倍。在线收益伴随更高的离线成本。

localhost 墙钟单独列出，包含本地计算、协议等待与计量开销；LAN/WAN 通信模型只含传输与传播。两者相加不能直接视为网络环境中的实测总时延。

## 接口统计

这次两套新执行的 result.json 均直接写入以下字段：

- `measurements.queries[*].offline/online.logical_rounds`、`party_sent_bytes`、`party_received_bytes`。
- `measurements.queries[*].offline/online.network_projection.LAN/WAN` 中的 `transmission_seconds`、`propagation_seconds`、`communication_seconds`。
- `measurements.network_projection.networks.LAN/WAN` 中的公式、每 query 明细、阶段中位数与 `offline_plus_online_seconds_median`。

命令使用 `--round-trace --network localhost --network-projection both`。汇总脚本逐 query 核对接口字段与 rounds.json，并重算传输、传播、合计和中位数；全部通过。接口投影覆盖 offline/online，offline+online 仅为这两个阶段之和。

接口与通道配置聚焦测试共 18 个：17 个通过，1 个需指定编译二进制的环境测试跳过。

## 原始证据

- SCI+SiRNN（SCI_preprocessed）：[result.json](SCI_preprocessed/result.json)、[rounds.json](SCI_preprocessed/rounds.json)、[commands.json](SCI_preprocessed/commands.json)、[实际两方环境](SCI_preprocessed/EFFECTIVE_ENVIRONMENT.json)。
- Segmented DPF PaperLUT：[result.json](SCI_Beaver_segmented_PaperLUT/result.json)、[rounds.json](SCI_Beaver_segmented_PaperLUT/rounds.json)、[commands.json](SCI_Beaver_segmented_PaperLUT/commands.json)、[实际两方环境](SCI_Beaver_segmented_PaperLUT/EFFECTIVE_ENVIRONMENT.json)。
- [NETWORK_MODEL.json](NETWORK_MODEL.json) 保存逐 query 复核后的统计和对比比值。
- [OUTER_COMMANDS.json](OUTER_COMMANDS.json) 保存完整启动命令、执行顺序、环境净化和固定 allocator；[RUN_STATUS.json](RUN_STATUS.json) 保存两行执行状态；[run_matched.py](run_matched.py) 可复现启动过程。
- [summarize.py](summarize.py) 对照冻结命令、环境、共同配置及每 query 接口字段后生成本报告。

旧 ticket13-network-retest 的协议数据仍保留；接口尚未回填的旧结果、预算为 0 的 ticket13-interface-projection-final-v3、未冻结 allocator 的 ticket13-paperlut-budget-1610612736-v1、relay 失败或中断目录均未进入本报告的样本。
