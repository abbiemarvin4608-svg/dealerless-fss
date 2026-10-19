# Google-30 SCI 真实计量

这一工具编译工单01冻结的SCI混合位宽程序，双方通过真实TCP转发器完成99步及分类头。它不生成明文或结构模拟性能。工单02的新量化图须另外接入，不能借用这里的历史模型测量作为独立质量结果。

```bash
python3 project/our/tools/google30_benchmark/build.py \
  --sources project/our/results/ticket01_native_replay/sources \
  --sci /home/jiang/EzPC/SCI --output /tmp/google30-metered-build --jobs 2
python3 project/our/tools/google30_benchmark/calibrate.py --output /tmp/link-calibration.json
# 在复制的frozen_config.json中设置绝对build/input路径；采集前冻结。
python3 project/our/tools/google30_benchmark/run.py \
  --config project/our/results/ticket04_sci_measurement/frozen_config.json \
  --output /tmp/google30-measurement
python3 -m unittest discover -s project/our/tools/google30_benchmark/tests -v
```

`run.run_pair(build, input_dir, output, network, ..., queries=1)` 是后续同语义后端可复用的实际双方执行接口。可执行文件需提供相同CLI和`GOOGLE30_METRICS`阶段JSON；使用`GOOGLE30_PROXY_OFFSET=1000`的客户网络端口映射。`--profiles`可选择冻结的网络组；`--smoke`只做功能检查，不计入正式统计。结果、日志与双方退出状态均落盘；失败/超时停止并回收双方和代理。每个进程地址空间预算1GiB。

## 隐私输入与计数

服务端程序包含原生模型；客户端编译时将所有模型数组替换为零，只保留原契约的公开学习标量zeta/nu。客户端才读取X/Y，服务端不打开客户CSV。保持原生holder/zero-share输入表示和私密矩阵协议；没有改成公开权重乘法。这个输入分享阶段无需网络字节，零通信是实测的合法原生表示。

所有SCI `NetIO::send_data`/`recv_data`入口计数，两个方向分别记录；每个阶段校验发送字节等于对方接收字节。链路总字节只加双方发送，包含初始化、输出和原库finalize统计交换。字节为应用协议载荷，不含TCP/IP包头或重传。I/O调用数及原库方向切换数与协议依赖轮数分开；后者无法从库计数可靠恢复，填null。

## 时间边界

- `application_io`：参数解析完成后，到初始化前的客户CSV解析和本地准备。
- `initialization_base_ot`：全部IOPack、OTPack、协议对象构造、连接和原生同步，包含base OT。
- `input_sharing`：初始化结束，到首个时间步前的holder/zero-share编码及数组分配。
- `online_compute`：全部99时间步、分类头、ArgMax。OT extension/预处理、扩环、截断、位宽转换位于这一范围内。没有将这些成本扣掉，也没有声称它们已被独立剥离；独立`offline_seconds=null/not_separable`。
- `output_reconstruction`：原生输出揭示，仅客户得到预测。
- `teardown`：数组释放、原库finalize及其统计字节交换和协议对象释放。

每方以steady_clock记录阶段时间；`total_seconds`为两方进程内部完整生命周期的较大值，包括初始化，另保存启动/等待代理和双方的`observed_wall_seconds`。各阶段取两方最大值，因此各阶段最大值之和不必严格等于total。峰值RSS为每方getrusage值，不把双方峰值之和冒充同一时刻内存。

主重复组N=1，每次启动新进程、重新base OT并生成/消耗新扩展材料。附加LAN/2ms各一次N=2持久会话实测：连接/base OT只初始化一次，两个99步查询各自继续消耗新的OT extension材料，末尾只finalize一次。每方JSON保存`queries`的真实时间/发送/接收delta；汇总给出整个会话总成本/2及第二次查询时间。仅复用长期base OT种子与连接，绝不重置OT扩展计数器或复用掩码/消息。附加会话只有一次观测，不与3重复主组混合。仍保存`total − initialization`以供分量核算，但它不是独立实测的稳态延迟。batch=1的逐序列吞吐为查询数/完整会话时间，没有测并行batch吞吐。

## 网络模型

无CAP_NET_ADMIN和免密sudo，不能设置tc。`link_proxy.py`在真实TCP连接上进行每方向共享序列化带宽和单程传播延迟；三个SCI套接字共享同一链路预算。源密码协议照常交互，转发器不参与秘密计算。它是应用载荷链路模型，不模拟TCP拥塞控制、丢包、IP包头或真实互联网路由。

LAN配置为0ms/1Gbps；低延迟WAN模型为2ms/100Mbps，另做20ms/100Mbps的预定600秒资源尝试。实测RTT高于目标值是调度/转发开销，见每组echo和bulk校准。不能将2ms结果无条件推广到典型跨地域WAN。20ms超时是删失观察，不是完成运行，也不参加成功运行中位数。

SCI也是CryptFlow2的基础库；这里执行的Google-30路径来自SiRNN。不能将同一binary的同一组数据重复标记为另一份独立CryptFlow2 baseline。独立CryptFlow2 CNN对比属于后续CNN工单。

正式v2驱动在进入初始化前一次性读取客户CSV，后续q>1复用同一冻结输入（模型重复测量），模型协议材料独立生成/推进，`query_begin/end`只用于公开计量。单次查询total含末尾finalize；N>1的first_query为application_io+initialization+query0，不含最后会话关闭，与session total/N分开。
