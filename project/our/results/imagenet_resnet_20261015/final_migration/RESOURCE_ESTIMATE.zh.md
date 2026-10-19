# 完整 CNN 的公开图资源预估

这些数值是公开图和当前序列化格式的静态推导，不是完整网络实测。适用配置：batch1、ring32/f12、tile64、optimized profile、depth-cap-8、当前 x86_64 构建。脚本为 `project/our/tools/imagenet_resnet/estimate_runtime_resources.py`，输出见本目录两个 `*_resource_estimate.json`。

## 参数与规划内存

`secure_driver.cpp::readWords` 直接读入一个 `vector<uint64_t>`，通过返回值优化或移动赋给 `Runtime::parameters`，没有第二个整份参数vector。只有role1读取参数；role2不读取。ResNet-152参数文件为480,936,768字节，约458.657 MiB。文件缓存可能另外占用相同规模的可回收页；不能把这一份参数误算为双方各一份。

两个batch1图的最大张量都是6,422,528字节（6.125 MiB），按 `evaluate` 的使用计数释放规则，存活张量加下一输出的最大合计都是19,267,584字节（18.375 MiB）。这不含节点内部scratch、协议状态、分配器和缓存，因此不是总峰值承诺。矩阵单次公开临时上界：A为8 MiB、B为1,806,336字节、输出为131,072字节。

初始 `plan()` 不做矩阵乘法，但仍分配/清零公开shape的tensor与分块。仅矩阵块累计清零量约为R50的881,655,616字节、R152的2,372,140,864字节，属于顺序工作量，不是同时驻留内存。现有证据不要求为这个加载/规划阶段改成mmap。

`SCI_preprocessed` 在offline还有第二次原生计划：拦截发生在OT调用层，LinearOT中的相关值构造与累加循环仍执行，CPU工作可达约32倍标量MAC数量。应先执行下述磁盘预估再启动该计划。

## 原生预处理的磁盘下界

对一个矩阵调用，令A为 `rows*inner`，B为 `inner*cols`，C为 `rows*cols`。原生 `Alice_has_A` 路径使用 `32*inner*cols` 个COT，每个COT带 `rows` 个消息。因此每个标量MAC总计32个pad位置。

当前COT序列化为：发送方每位置保存zero/one两个uint64，接收方保存一个uint64，并且每个COT保存一个uint8随机选择。各方每个记录还包含三个uint64长度字段。

双方线性COT文件总量为 `MAC*32*24 + COT_count + COT_calls*48`。这已经是完整SCI预处理文件的下界，因为尚未计入非线性GOT、bit triples和其他材料。

| 公开图量 | ResNet-50 | ResNet-152 |
| --- | ---: | ---: |
| 标量MAC | 4,089,184,256 | 11,513,626,624 |
| 矩阵调用 | 4,271 | 11,515 |
| 线性COT实例 | 9,707,675,648 | 30,619,426,816 |
| 线性COT记录 | 135,872 | 366,912 |
| 双方线性spool下界（字节） | 3,150,207,706,112 | 8,873,102,285,824 |
| 双方线性spool下界（TiB，约） | 2.865 | 8.070 |

核查时 `/tmp` 与workspace位于同一文件系统，可用约700 GiB。两个原生全量预处理配置的下界均超过可用空间，不能用不设上限的方式等待磁盘写满。资源预检应记录 `resource_unavailable` 或相应不可运行状态，并明确这是当前材料存储实现和预算下不可行，而不是该协议的理论空间下界。它不能填入实测延迟、通信或OOM值。

## Ours 的磁盘推导

矩阵Beaver每个调用保存U/V/W三个uint64向量，字节为 `24+8*(rows*inner+inner*cols+rows*cols)`。同一个B块在不同输出行块的独立材料中仍分别保存，因此不能把实际Beaver文件量替换成规划时只分配一次B的清零量。

非线性序列化按每个MSB、BitXA、AND的实际公开调用数计算。32位MSB的31位域在depth-cap-8下有4个segment，每行材料为645字节；每条记录头为76字节。16位MSB每行为316字节，同样有76字节记录头。BitXA每行为25字节、每调用头12字节；AND每行为3字节、每调用头36字节。所有标量批次上限为4096。

32到32的floor右移12转换在当前fused sign/wrap配置下使用12位与31位两个DCF域，每行材料933字节，每调用头1056字节。该公式包括所有掩码、比较key、Boolean triples和结构化metrics。ABI尺寸固定为 `CggmKeyRecord=48`、`CnnCall=12`、`SegmentedDcfMetrics=320`、`SciTruncationMetrics=88` 字节，结构或序列化变化后须更新估算。

| 单查询材料字节 | ResNet-50 | ResNet-152 |
| --- | ---: | ---: |
| 每方Beaver spool | 3,134,148,520 | 9,639,196,360 |
| 每方非线性spool | 7,518,026,080 | 15,183,161,936 |
| 每方conversion spool | 10,373,166,120 | 21,049,806,696 |
| 双方三类spool合计 | 42,050,681,440 | 91,744,329,984 |
| 双方合计GiB（约） | 39.16 | 85.44 |

这两个Ours配置在上述磁盘容量下可容纳材料，仍须为日志、原模型、运行期间其他文件和文件系统开销留余量。该结论不预言运行时间或RAM峰值。

三类公式已与既有小图两次查询的实测序列化逐项核对：每方Beaver 52,416字节、非线性928,264字节、conversion 738,249字节，全部精确吻合。没有为估算重新启动任何协议。

## 已确认的实现问题与修复状态

COT在线consume将payload恢复到长期存在的 `Stock.calls` 条目。原send/receive末尾仅调用 `clear()`，保留了每个已经消费的向量容量，造成在线RAM随累计消费增长。CNN生成overlay已改用 `cnn_spool::release` 的空vector交换；独立spool测试中消费后capacity为0的断言已通过。当前二进制重建和单路 `SCI_preprocessed` 资格复验均已完成，记录在 `qualification/SCI_preprocessed_capacity`：两次新查询共核对3,938个整数值，差异为0；每方每次查询生成和消费均为57条原生库存，在线原生生成epoch增量为0。

本次重建、独立测试编译/运行和协议复验的四份 `*capacity*guard.json` 均为 `complete`，退出码为0，沿用2 GiB RAM加512 MiB swap守护。原三后端资格记录仍保留各自的旧二进制身份；此次只复验受payload释放修改影响的原生预处理路径，后续正式运行使用当前重建二进制。

生成header与helper header都在现有构建checksum依赖中，重建不会误用旧对象。

`run_performance.py` 已增加 `SCI_preprocessed` 的公开图资源预检：启动协议前检查每方材料预算及 `/tmp` 双方磁盘下界，不可行时单列 `resource_unavailable` 并继续其他配置；`--resume` 核对并保留该记录。所有可运行行完成且包含资源不可用行时，批次状态为 `complete_with_resource_failures`。protocol非零、输出audit失败或resume验证失败仍会结束批次，不能将尚未执行的配置计为完成，也不能将资源不可用行当作实测结果。
