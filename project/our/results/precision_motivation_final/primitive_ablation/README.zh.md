# 实验三原语：复用结果与唯一缺项

`reused_batch100.csv` 保留原 ticket10 的 batch=100、一次预热和三次正式重复，原始结果未改写。
50 个复用配置中，44 个各完成三次正式重复；6 个单树配置（24/25/32 位、两种网络）在预热时真实 bad_alloc，随后 18 次正式重复未运行。它们是资源不可用记录，不是 50 个全部成功的测量点。全部汇总中位数已与保留的原始重复记录交叉核对。
复用输入位宽为 16/20/24/25/32；对应内部比较域分别为 15/19/23/24/31。
因此 23/31 位 FSS 比较域已有实测，不追加名为“输入23/31”的冗余测试。

只补缺失的 MSB 输入28位（内部比较域27位）：batch=128，SCI 与 ours nseg=1/2/4/8，LAN 0ms/1Gbps、受控 WAN 2ms/100Mbps，一次预热、五次新材料正式重复。
新 batch128 与旧 batch100 分栏报告，不跨 batch 计算加速比或拼成相同实验配置的曲线。该 batch 是原语标量数，不是完整 CNN 图像批量。

每方继承旧 runner 的 768MiB 地址空间上限；外层同时使用既有 memory_guard 限制整个实验树 2GiB RAM+512MiB swap。128个公开边界输入单独核对，不进入性能统计。单树失败保留原始 OOM/超时/退出码与预算，剩余重复记 not_run，不强行重试。

网络口径为本地真实 TCP 协议经应用负载 relay 的受控链路模型。WAN 每向加入 1ms 传播延迟，两方向分别共享 100Mbps 序列化时钟，所有 SCI/dFSS 通道共用相应方向限速。LAN 模型为零附加传播延迟、每向 1Gbps。测量包含真实协议执行与 relay 等待，但不是物理跨地域 WAN、tc 整形或完整 TCP 拥塞仿真；不将名义 RTT 当作实际测得的全链路 RTT。

旧 runner 的聚合只识别 repeat1..3，因此补测使用本工具独立聚合 repeat1..5，避免漏算。setup/offline/online/总时延、各阶段通信和内存均分开报告；没有因果轮数测量时保持缺失。

## 当前结果

三个构建产物已完成并记录身份。LAN 和 WAN 下 SCI 与 2/4/8 段配置均各完成五次正式重复，共40次；LAN 四组独立边界核对均为零错误。两网络的单树配置均在预热时真实 `std::bad_alloc`，后续共十次正式重复未运行，单树边界核对未完成。最终汇总状态为 `complete_with_resource_failures`。

首次运行时，WAN 8 段的第一次正式重复在产生指标前被外层 guard 终止。停止原因是 Windows 可用虚拟内存持续低于保护阈值；不是内核 OOM。原始 guard 为 `/tmp/primitive28-memory-guard.json`，中断记录保留在 `missing_width_runs/RESOURCE_INTERRUPTION.json`，当时的 partial 汇总保留在 `missing_width_runs/resume_wan_n8/summary_before_resume.json`。暂停下载并确认主机内存恢复后，缺失五次重复已全部完成；恢复 guard 为 `/tmp/primitive28-resume-memory-guard.json`，退出码0、无新增OOM。

原来的未完成 attempt 目录保持原样。窄恢复入口只向 `missing_width_runs/resume_wan_n8/` 写入这五次缺项，沿用原保留输入，并在启动前复核冻结配置、构建、原始结果及输入身份。原计划未记录输入哈希；恢复计划如实标注输入哈希是在中断后对现存私有文件首次记录。

## 已执行的恢复命令

以下窄恢复已完成，入口拒绝重复覆盖：

```bash
python3 project/our/tools/imagenet_resnet/memory_guard.py run --output /tmp/primitive28-resume-memory-guard.json -- python3 project/our/tools/imagenet_resnet/resume_primitive_ablation.py --execute
```

不重新执行已经完成的测量，不放宽资源上限，也不把未运行项当作测得的零开销。

## 结果范围

输入28位、batch128的五次正式重复中位数如下，单位ms：

| 网络 | 后端/段数 | setup | offline | online | total |
| --- | --- | ---: | ---: | ---: | ---: |
| LAN | SCI | 220.046 | 不适用 | 4.831 | 225.215 |
| LAN | ours/2 | 1344.175 | 110.370 | 3.099 | 1459.482 |
| LAN | ours/4 | 1338.039 | 48.696 | 5.132 | 1390.593 |
| LAN | ours/8 | 1340.698 | 40.769 | 6.909 | 1388.367 |
| WAN | SCI | 252.907 | 不适用 | 17.638 | 270.385 |
| WAN | ours/2 | 1375.546 | 222.071 | 7.928 | 1621.689 |
| WAN | ours/4 | 1375.503 | 132.219 | 13.013 | 1517.411 |
| WAN | ours/8 | 1351.407 | 130.973 | 17.812 | 1494.352 |

各阶段独立取中位数，因此表中阶段中位数之和不必等于total中位数。SCI在online内生成原生材料，未提供预处理SCI基线。结果支持单树在该资源预算下失败而分段可运行，以及分段数对离线和在线成本的权衡；不能据此宣称端到端普遍快于SCI。两方online发送字节总和分别为SCI 63,808 B、2段3,072 B、4段5,120 B、8段9,216 B。
