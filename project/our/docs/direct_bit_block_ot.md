# 即时 bit × block 乘积的四个匹配配置

`run.py --bit-block-mode` 把下面四种配置显式传给两方进程；默认是 `legacy`。

| 配置 | 乘积协议 | 每方向消息编码 | e/f 开值 |
| --- | --- | --- | --- |
| `legacy` | 临时 Boolean × block Beaver 三元组 | 两次 64-bit general OT | 两次 |
| `block_ot` | 相同临时三元组 | 一次 128-bit general OT | 两次 |
| `direct_ot_legacy` | 直接双向 OT | 两次 64-bit general OT | 无 |
| `direct_ot` | 直接双向 OT | 一次 128-bit general OT | 无 |

每元素相关量不同，因此编码使用 chosen-message general OT；固定全批 delta 的 COT 不能代替它。`direct_ot_legacy` 与 `direct_ot` 共用代数，只改变同一选择 bit 对应的 block 是分成两个 limb 还是一次传输。两个方向顺序执行，均使用已有各方向的有状态 OT 上下文。

令 XOR 分享为 `a = a0 XOR a1`、`B = B0 XOR B1`。每调用、每元素，参与者 `p` 重新独立采样 128-bit `rp`，作为发送者输入 `(rp, rp XOR Bp)`；对方以自己的规范化低 bit `a_(1-p)` 选择消息。参与者 `p` 接收

```
qp = r_(1-p) XOR (ap * B_(1-p))
zp = rp XOR qp XOR (ap * Bp)
```

所以 `z0 XOR z1 = (a0 XOR a1) * (B0 XOR B1)`。这里乘法表示 bit 为 1 时取 block，否则取 0；所有 block 运算是 XOR。随机 mask 不跨调用或请求复用。

在半诚实 chosen-message OT 理想功能下，发送者不知道对方选择 bit；接收者只得到对方 fresh mask 覆盖的一条消息，消息独立均匀，不能恢复对方 block。对任一参与者，从自己的输入和输出份额可模拟接收消息 `qp = zp XOR rp XOR ap*Bp`，且输出份额被对方随机 mask 均匀掩盖。两个 OT 方向的顺序组合不额外公开输入信息。安全论证依赖现有 OT 实现及其半诚实安全假设，不包含主动攻击安全。

只修改即时 `bitBlockMultiply`。`prepareDpfBitBlockMulMaterial` 与 consume 接口继续保持原生命周期和一次消费限制；已经准备的材料不会因为环境模式改变而匹配另一协议。

`Peer::bit_block_ledger` 在一次即时乘积成功后记录产品数量、临时三元组生成/消费数量、两方向 general OT 的逻辑数量与 e/f opening 次数。产品和三元组数量是共同逻辑库存，两方记录应该相同，不能把两方相加当作两倍库存。64/128-bit OT 数量包含本方发送和接收参与的两方向，实际扩展 padding、base OT 与通信字节从真实 Peer/round trace 单独读取。cGGM 的 `CggmGenerationStats` 给出当前调用的差值；DCF generation 日志汇总 tile 的差值，避免把即时删除的 e/f 再算作叶 CW 合并收益。既有 `cot_batches` 仅表示相关生成阶段的两个 OT 方向，不表示 64-bit limb 编码的实际消息次数；准确编码库存使用新增 general-OT 数量。

`--merge-leaf-cw` 在叶部 high-block 和两个 low-byte 已同时就绪后按原顺序连接为一个字节载荷，保持总载荷 `18 * key_count` bytes/方向。内部层 CW 仍逐层打开，不能合并依赖上一层结果的载荷。

`--reuse-generation-workspace` 为树节点、内部哈希与公开 offsets/counts 使用连续 buffer；同形状 tile 复用 capacity。对象在单次 DCF 生成返回前释放，生成预算在其整个存活期内持续计入全部 capacity。现有预算保留原保守上界，不据此减少 tile，也不声称释放 vector 等于分配器已归还地址空间。RSS/地址空间及 RLIMIT_AS 的最终资格仍由完整 runner 测量确定。

独立协议审查于 2026-10-13 完成：DPF 批量实现代理只读检查直接路径和以上代数，确认四个交叉项相加、fresh-mask 隐私及两 limb 的联合均匀性，并确认有状态 OT 的顺序组合符合本轮半诚实模型。实际两方 case 15 覆盖四配置 × `7,129,7` 个逐元素不同 block，校验重构值和完整逻辑库存；case 2 覆盖四个乘积配置、单独 CW 合批、单独连续 workspace 及组合，全部域上的 less/equality 结果一致，同形状第二次调用没有 capacity 增长。

局部归因在真实 `cggmGen2pcBatch` 边界运行同一 binary 的四种乘积配置，CW 合批和 workspace 分别以 `direct_ot` 为基准，另测两者组合。固定生成 `128 × 100 = 12800` 个深度 8 的独立段密钥，以 1024 个为 tile，每配置三个 fresh 两方会话，按正序、逆序、正序交替；每方四个不重叠逻辑 CPU、1.5 GiB RLIMIT_AS。密钥压入紧凑结构常驻内存、生成 workspace 销毁之后才构造公开查询，在线只做本地 PRG 求值，less/equality 重构校验记为独立 output 阶段。该实验衡量 cGGM 原语，不代表整图 a2 加速或完整模型性能；完整 FastGRNN 的默认/非零隐状态逐位诊断和两阶段正式准入由工单 13 单独记录。

2026-10-13 的 21 个匹配会话均通过独立比较 oracle、库存和资源约束，原始日志与无损压缩 send/receive trace 在 `results/sci_ours_joint_dpf_20261013/cggm-generation-isolated/`。构造 Peer 时 setup 的 Base-OT 通信为双方合计 457906 bytes、12 因果轮次；不执行 priming，首次调用中惰性 OT 成本留在 offline。下表字节是双方真实发送量之和，轮次来自匹配收到的字节区间并保留 actor 依赖的关键路径，墙钟是三个会话每次两方最大值的中位数。

| 配置 | 离线 bytes | 离线因果轮次 | 离线墙钟 ms |
| --- | ---: | ---: | ---: |
| legacy | 19916800 | 248 | 113.306 |
| block_ot | 16640000 | 196 | 82.741 |
| direct_ot_legacy | 16435200 | 196 | 112.284 |
| direct_ot | 13158400 | 144 | 74.821 |
| direct_ot + CW 合批 | 13158400 | 131 | 74.355 |
| direct_ot + workspace | 13158400 | 144 | 65.290 |
| direct_ot + 两者 | 13158400 | 131 | 66.856 |

因此，128-bit 合并相对相同代数的两个 limb 编码节约 3276800 bytes、52 轮；直接协议相对相同编码的临时三元组路径节约 3481600 bytes、52 轮；CW 合批只减少 13 轮而不减少载荷，workspace 不改变通信。每方驻留紧凑密钥 2048000 bytes，复用 workspace 在首 tile 后没有 capacity 增长。在线 cGGM 本地求值均为零 bytes、零轮，中位墙钟 0.72–1.04 ms；这些短时测量不能单独证明组合在完整模型中满足在线墙钟不退化。网络模型由代码按 `8 × 双方发送 bytes / 带宽 + RTT × 因果轮次` 计算；LAN 为 1 Gbps/0.5 ms RTT，WAN 为 400 Mbps/4 ms RTT。两个 OT 方向数、XOR opening 数及 `Peer::rounds` 均不代替因果轮次。三个开关保持候选，默认仍使用 legacy，直至工单 13 的完整两阶段判定。
