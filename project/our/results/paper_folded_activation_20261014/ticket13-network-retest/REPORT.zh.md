# ticket13 LAN/WAN 通信模型复测

本页保留接口回填前的历史批次。当前复测入口为[统一预算与 allocator 的新接口报告](../ticket13-interface-projection-matched/REPORT.zh.md)，其两套新执行直接在 `result.json` 写入逐 query 轮次及 LAN/WAN 三项时间。

本报告使用同一冻结构建、batch=128、单时间步、10 个 fresh query 的两方 localhost 实测。
localhost 只用于取得协议字节和因果轮次；LAN/WAN 数字是通信模型投影，不是网络整形实测总时延。

## 计算规则

```text
传输时间 = 8 × 双方 sent bytes 之和 / 带宽(bit/s)
传播时间 = 因果轮次 × RTT
通信时间 = 传输时间 + 传播时间
```

LAN 为 1 Gbps、0.5 ms RTT；WAN 为 400 Mbps、4 ms RTT。每个 query 先计算，表中取 10 个 query 的中位数。

## 离线与在线

| 方法 | 阶段 | 实测墙钟(s) | bytes | 轮次 | LAN 传输(s) | LAN 传播(s) | LAN 通信(s) | WAN 传输(s) | WAN 传播(s) | WAN 通信(s) |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| SCI+SiRNN (SCI_preprocessed) | offline | 1.514550864 | 185571244 | 70 | 1.484569952 | 0.035000000 | 1.519569952 | 3.711424880 | 0.280000000 | 3.991424880 |
| SCI+SiRNN (SCI_preprocessed) | online | 0.401160082 | 91429766 | 185 | 0.731438128 | 0.092500000 | 0.823938128 | 1.828595320 | 0.740000000 | 2.568595320 |
| SCI_Beaver_segmented_PaperLUT | offline | 11.513973570 | 2938230430 | 955 | 23.505843440 | 0.477500000 | 23.983343440 | 58.764608600 | 3.820000000 | 62.584608600 |
| SCI_Beaver_segmented_PaperLUT | online | 0.206597823 | 5860322 | 134 | 0.046882576 | 0.067000000 | 0.113882576 | 0.117206440 | 0.536000000 | 0.653206440 |

在线模型相对：SCI+SiRNN / PaperLUT 为 LAN 7.234980x、WAN 3.932287x；离线为 LAN 0.063359x、WAN 0.063776x。

## 查询阶段合计

下表包含 offline、input、ready、online、output；setup、model_prepare 和 teardown 不计入。

| 方法 | bytes | 轮次 | LAN 通信(s) | WAN 通信(s) |
| --- | ---: | ---: | ---: | ---: |
| SCI+SiRNN (SCI_preprocessed) | 277103414 | 257 | 2.345327312 | 6.570068280 |
| SCI_Beaver_segmented_PaperLUT | 2944193156 | 1091 | 24.099045248 | 63.247863120 |

## 接口统计核对

当前 `result.json` 的 `measurements` 已保存 `sent_bytes` 和墙钟；其各阶段 `logical_rounds` 仍为 `null`。
因果轮次只写入同目录 `rounds.json`，`transmission_seconds`、`propagation_seconds` 和通信总时间也没有回写到 `result.json`。
运行时随后已加入 `measurements.network_projection` 写入逻辑；本批 `result.json` 生成早于该改动，因此本报告仍以独立投影文件为准。
本目录的 `NETWORK_MODEL.json` 固化了这次复测的逐 query 投影及来源哈希，因此本报告没有把 localhost 墙钟当作 LAN/WAN 时延。

## 解释

PaperLUT 的在线通信量和轮次都显著较小；其离线通信量约 15.83 倍于 SCI+SiRNN，导致离线 LAN/WAN 通信模型明显更高。
这些结果是通信量与因果关键路径轮次的确定性投影，不包含本地计算、材料生成/加载和 relay 排队。

