# 08: 四参与者图并行与激活分片

**What to build:** 在相同四核与内存预算内，让两种 LUT 构造都支持线性分支、gate/candidate 激活及尾部更新的图并行，并把每种激活分给两个独立 worker，获得可与串行配置公平比较的完整单步。

**Blocked by:** 07: 双激活 Batch=128 联合驻留。

**Status:** closed

**Labels:** closed

- [x] 保留图串行/每激活单 worker、双分支/每激活单 worker、双分支/每激活双 worker 三种显式调度，均支持两种 LUT family；参数和实际参与者记入结果。
- [x] 双 worker 配置每种激活各处理 6400 个元素，两种激活合计四个同时计算参与者；每 worker 独占上下文、OT、通道、材料游标和工作区。
- [x] 输入矩阵链与状态矩阵链、Sigmoid 与 TanH、两条尾部更新链分别按真实依赖 fork/join，定点运算顺序保持原图。广播常量与适用模型预计算按既有契约共享。
- [x] 同一阶段跨元素合批，LUT 内积按 worker 的连续元素范围执行，轮转采用公共偏移/连续区间。两种 LUT 使用同一在线核，不新增嵌套内积线程池。
- [x] 本地 cGGM/AES、OT 与图任务共享四参与者额度，Eigen 内部并行关闭。多 activation worker 每个 OT 一个参与者；独立单 worker 模式可按既有规则使用两个，实际配置据实记录。
- [x] 输入前 keygen、张量 AND/B2A 或 payload 适配按同公开形状合批，复用有界工作区；网络由上下文所属调用线程处理，不把当前串行化生成池标成多 slot 完全并发。
- [x] 两种 LUT 的三种调度在默认/非零状态下结果与同一 oracle 一致，覆盖 batch=1、batch=128、非整分片和连续 fresh query；材料输入前就绪与一次消费规则不因并行改变。
- [x] 每种调度实测联合内存和计算参与者，所有新通道进入被动通信轨迹；轮次从 fork/join 因果关键路径计算，不能把 worker 时间或轮次相加当作整体墙钟。
- [x] 交付完整单步局部匹配计量及复现配置；正式十查询性能与推荐准入留给 11。历史实验性 overlap、跨分支开值融合或额外资源不自动加入。

**Scope boundary:** 本票固定 BitDec 符号后端以隔离调度变化，09 的分段符号候选在最终交付时组合验证。

## 验收

最终 owner-thread transport 版本两种 LUT、三种调度、两种隐状态的 batch=1 与 batch=128 各 12 行均通过，每行两次 fresh query；每 query 分别比较 1351 与 172928 个完整单步整数，全部零差异。259 元素在原生激活接口上按 130/129 分片，两构造两次 fresh query 同值，实际 hashing/transport 均为每 worker 一个参与者。43 个聚焦测试和真实 cGGM scoped-cap 回归通过。

四 CPU affinity、1.5 GiB RLIMIT_AS、512 MiB inclusive prefetch、完整上下文前缀、材料输入前就绪及一次消费均通过公共审计；最大 party VmPeak 936857600 字节、RSS 563761152 字节。整激活 join 与 all-folded 直接准备计时、阶段 work 归因、同一完整 DAG 上的 scope 字节和因果轮次保留于 [08 证据](../../../project/our/results/paper_folded_activation_20261013/ticket08/README.md)，两查询局部匹配表见该目录 `batch128-owner/local-matched.md`。历史失败、superseded 行及压缩后的原始轨迹均保留；正式效率结论留给 11。
