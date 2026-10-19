# Batch=128：SCI+SiRNN 真实预处理公平对照

在双方都预处理剩余原生 SCI 工作后，Beaver+分段 FSS 组合仍降低在线时延和通信；仅替换图级转换的 SCI_ours 本轮在线时延增加。预处理显著增加每查询离线成本，所有预处理方案的直接查询总时延均高于默认 SCI。

测试使用两方 FastGRNN 单时间步、batch=128、同一冻结模型与输入，localhost，无网络整形。每行 10 个新鲜查询、warmup=0、600 秒预算；服务方 CPU 0,2,4,10，客户端 CPU 6,8,12,16，每方四核及 1.5 GiB RLIMIT_AS。a4-sequential 顺序执行图任务，激活最多四个 worker；a2-parallel 两分支各最多两个计算参与者。八行串行测量。

默认 SCI 仅使用持久 setup，不启用每查询原生预处理；SCI_preprocessed、SCI_ours、SCI_Beaver_segmented 都在输入与隐状态装载前实际生成本查询所需随机 OT 和安全标准/相关 bit-triple。SCI_ours 仅替换图级截断/扩环，组合后端还替换图级乘法。Sigmoid/TanH 数值算法保持原生 SCI。

16 个默认/非零隐状态诊断分别与新 SCI 控制和冻结 SCI 输出逐整数一致；每查询 15 张量、172928 个整数。八个正式行关闭中间张量揭示。原始日志重新收集后与已保存的材料、计量和配置记录一致，全部 24 行通过资格核对，其中 8 行进入主表。

时间采用同查询双方最大墙钟时间；通信采用双方发送字节之和。表内时间单位为秒，通信单位为 MiB（2^20 bytes）。query total、首请求和会话总时间均直接测量；阶段中位数不相加。会话总通信按 application_io、setup、各 query total、teardown 的实际计数差累计，覆盖加载模型、上下文初始化、所有查询及结束握手，不重复计入 first_query 或细分阶段；双方发送与对方接收计数交叉一致。每阶段 10 个样本、标准差及范围保存在 COMPARISON.json。

## a4-sequential

| 后端 | offline | online 中位数 [min,max] | 直接 query total | setup | 首请求 | session total | RSS 双方 MiB |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| SCI | 0.000088 | 1.277002 [1.245177,1.399133] | 1.278204 | 0.634877 | 2.037395 | 13.521529 | 358.77 / 226.64 |
| SCI_preprocessed | 1.533476 | 0.525713 [0.505341,0.598461] | 2.080642 | 0.634715 | 2.884993 | 21.785091 | 833.07 / 518.29 |
| SCI_ours | 5.513640 | 0.567141 [0.548523,0.585339] | 6.089556 | 1.930958 | 8.206896 | 63.154691 | 817.19 / 653.75 |
| SCI_Beaver_segmented | 6.390551 | 0.281187 [0.277514,0.302800] | 6.675237 | 1.928583 | 8.762603 | 68.643186 | 474.88 / 327.26 |

| 后端 | input 时间 | ready 时间 | output 时间 | offline 通信 | online 通信 | query total 通信 | setup 通信 | 首请求通信 | session total 通信 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| SCI | 0.000615 | 0.000460 | 0.000078 | 0.000002 | 273.157763 | 273.255424 | 1.746813 | 275.002237 | 2734.301065 |
| SCI_preprocessed | 0.000672 | 0.000498 | 0.000065 | 185.926855 | 87.932745 | 273.957260 | 1.746813 | 275.704073 | 2741.319422 |
| SCI_ours | 0.000611 | 0.000466 | 0.000102 | 1633.058275 | 85.747396 | 1718.903332 | 2.183506 | 1721.086838 | 17191.216831 |
| SCI_Beaver_segmented | 0.000670 | 0.000522 | 0.000083 | 2126.692591 | 36.440401 | 2163.230652 | 2.183506 | 2165.414158 | 21634.490032 |

| 后端 | input / ready / output 通信 MiB |
| --- | --- |
| SCI | 0.000000 / 0.000002 / 0.097658 |
| SCI_preprocessed | 0.000000 / 0.000002 / 0.097658 |
| SCI_ours | 0.000000 / 0.000002 / 0.097658 |
| SCI_Beaver_segmented | 0.000000 / 0.000002 / 0.097658 |

SCI → SCI_preprocessed：在线倍数 2.429x，在线时延下降 58.83%，在线通信下降 67.81%。每查询离线时延变化 +1.533388 秒，直接 query total 变化 +0.802437 秒（+62.78%）；离线通信变化 +185.926853 MiB，直接 query total 通信变化 +0.701836 MiB。
本轮在线样本范围分离；这描述本次 10 查询观测，不作为显著性检验。

SCI_preprocessed → SCI_ours：在线倍数 0.927x，在线时延增加 7.88%，在线通信下降 2.49%。每查询离线时延变化 +3.980164 秒，直接 query total 变化 +4.008914 秒（+192.68%）；离线通信变化 +1447.131420 MiB，直接 query total 通信变化 +1444.946072 MiB。
这两行在线样本范围重叠；本轮中位数差异不足以给出稳定排序，也未进行显著性检验。

SCI_preprocessed → SCI_Beaver_segmented：在线倍数 1.870x，在线时延下降 46.51%，在线通信下降 58.56%。每查询离线时延变化 +4.857075 秒，直接 query total 变化 +4.594596 秒（+220.83%）；离线通信变化 +1940.765736 MiB，直接 query total 通信变化 +1889.273392 MiB。
本轮在线样本范围分离；这描述本次 10 查询观测，不作为显著性检验。

## a2-parallel

| 后端 | offline | online 中位数 [min,max] | 直接 query total | setup | 首请求 | session total | RSS 双方 MiB |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| SCI | 0.000065 | 1.132586 [1.073890,1.210928] | 1.133537 | 0.643147 | 1.856988 | 12.076716 | 362.18 / 230.55 |
| SCI_preprocessed | 1.491372 | 0.414972 [0.402432,0.422863] | 1.907757 | 0.639712 | 2.761682 | 19.875620 | 877.67 / 528.83 |
| SCI_ours | 5.610748 | 0.498325 [0.487069,0.653577] | 6.101866 | 3.221334 | 9.602751 | 64.705470 | 867.42 / 735.98 |
| SCI_Beaver_segmented | 6.619960 | 0.286066 [0.278047,0.348084] | 6.913272 | 3.230041 | 10.278354 | 72.646302 | 526.07 / 322.43 |

| 后端 | input 时间 | ready 时间 | output 时间 | offline 通信 | online 通信 | query total 通信 | setup 通信 | 首请求通信 | session total 通信 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| SCI | 0.000649 | 0.000507 | 0.000063 | 0.000002 | 273.720263 | 273.817924 | 1.746813 | 275.564737 | 2739.926065 |
| SCI_preprocessed | 0.000623 | 0.000476 | 0.000071 | 186.425039 | 87.932745 | 274.455444 | 1.746813 | 276.202257 | 2746.301264 |
| SCI_ours | 0.000617 | 0.000476 | 0.000110 | 1633.603527 | 85.747396 | 1719.448584 | 2.620199 | 1722.068783 | 17197.106043 |
| SCI_Beaver_segmented | 0.000715 | 0.000544 | 0.000103 | 2127.237852 | 36.440401 | 2163.775913 | 2.620199 | 2166.396112 | 21640.379339 |

| 后端 | input / ready / output 通信 MiB |
| --- | --- |
| SCI | 0.000000 / 0.000002 / 0.097658 |
| SCI_preprocessed | 0.000000 / 0.000002 / 0.097658 |
| SCI_ours | 0.000000 / 0.000002 / 0.097658 |
| SCI_Beaver_segmented | 0.000000 / 0.000002 / 0.097658 |

SCI → SCI_preprocessed：在线倍数 2.729x，在线时延下降 63.36%，在线通信下降 67.87%。每查询离线时延变化 +1.491307 秒，直接 query total 变化 +0.774220 秒（+68.30%）；离线通信变化 +186.425037 MiB，直接 query total 通信变化 +0.637520 MiB。
本轮在线样本范围分离；这描述本次 10 查询观测，不作为显著性检验。

SCI_preprocessed → SCI_ours：在线倍数 0.833x，在线时延增加 20.09%，在线通信下降 2.49%。每查询离线时延变化 +4.119376 秒，直接 query total 变化 +4.194109 秒（+219.84%）；离线通信变化 +1447.178488 MiB，直接 query total 通信变化 +1444.993139 MiB。
本轮在线样本范围分离；这描述本次 10 查询观测，不作为显著性检验。

SCI_preprocessed → SCI_Beaver_segmented：在线倍数 1.451x，在线时延下降 31.06%，在线通信下降 58.56%。每查询离线时延变化 +5.128588 秒，直接 query total 变化 +5.005515 秒（+262.38%）；离线通信变化 +1940.812813 MiB，直接 query total 通信变化 +1889.320469 MiB。
本轮在线样本范围分离；这描述本次 10 查询观测，不作为显著性检验。

## 24 行资格

| 类型 | 行 | 状态 | 合格 |
| --- | --- | --- | --- |
| formal | SCI-a4-sequential | complete | True |
| formal | SCI_preprocessed-a4-sequential | complete | True |
| formal | SCI_ours-a4-sequential | complete | True |
| formal | SCI_Beaver_segmented-a4-sequential | complete | True |
| formal | SCI-a2-parallel | complete | True |
| formal | SCI_preprocessed-a2-parallel | complete | True |
| formal | SCI_ours-a2-parallel | complete | True |
| formal | SCI_Beaver_segmented-a2-parallel | complete | True |
| diagnostic | default-a4-sequential-SCI | complete | True |
| diagnostic | default-a4-sequential-SCI_preprocessed | complete | True |
| diagnostic | default-a4-sequential-SCI_ours | complete | True |
| diagnostic | default-a4-sequential-SCI_Beaver_segmented | complete | True |
| diagnostic | default-a2-parallel-SCI | complete | True |
| diagnostic | default-a2-parallel-SCI_preprocessed | complete | True |
| diagnostic | default-a2-parallel-SCI_ours | complete | True |
| diagnostic | default-a2-parallel-SCI_Beaver_segmented | complete | True |
| diagnostic | nonzero-a4-sequential-SCI | complete | True |
| diagnostic | nonzero-a4-sequential-SCI_preprocessed | complete | True |
| diagnostic | nonzero-a4-sequential-SCI_ours | complete | True |
| diagnostic | nonzero-a4-sequential-SCI_Beaver_segmented | complete | True |
| diagnostic | nonzero-a2-parallel-SCI | complete | True |
| diagnostic | nonzero-a2-parallel-SCI_preprocessed | complete | True |
| diagnostic | nonzero-a2-parallel-SCI_ours | complete | True |
| diagnostic | nonzero-a2-parallel-SCI_Beaver_segmented | complete | True |

## 每查询材料与在线边界

| cohort | 后端 | 每方有效随机 OT | 每方生成随机 OT（含 padding） | 每方安全 bit-triple | 标准 / 相关 triple 调用 |
| --- | --- | ---: | ---: | ---: | --- |
| a4-sequential | SCI | 0 | 0 | 0 | 0 / 0 |
| a4-sequential | SCI_preprocessed | 6237216 | 7002624 | 1902336 | 78 / 18 |
| a4-sequential | SCI_ours | 4529056 | 5082112 | 773632 | 60 / 8 |
| a4-sequential | SCI_Beaver_segmented | 3136000 | 3549184 | 422400 | 52 / 8 |
| a2-parallel | SCI | 0 | 0 | 0 | 0 / 0 |
| a2-parallel | SCI_preprocessed | 6237216 | 6959104 | 1902336 | 52 / 14 |
| a2-parallel | SCI_ours | 4529056 | 5041664 | 773632 | 34 / 4 |
| a2-parallel | SCI_Beaver_segmented | 3136000 | 3508736 | 422400 | 26 / 4 |

上表是剩余原生 SCI 库存，不包含已经替换的调用；组合后端的 Beaver 与 FSS 材料另有原始生成/消费记录。所有查询的两方库存逐项核验，随机 OT padding 在查询结束丢弃，剩余有效库存为零；bit-triple 由实际安全协议生成。四个原生上下文的计划与消费完整覆盖，严格模式 input、ready、online、output 的原生和 bridge 生成 epoch 增量均为零。

Base OT 与持久上下文创建计 setup；每查询随机 OT 扩展、bit-triple 库存、Beaver/FSS 材料生成及 material-ready 工作计 offline。输入相关选择修正、算术相关量/消息适配、比较 AND 的打开与修正留在 online；输入后的材料读取、等待、掩码与密钥评价同样留在 online。input 仅装载查询数据，ready 仅同步，表中通信证明没有把协议通信隐藏在这两个阶段。最终公开输出重构计 output，直接 query total 包含全部这些成本。

预处理机会相同，材料数量与工作量随协议替换而不同。本轮 Beaver+分段 FSS 的在线收益伴随更大的离线和直接总时间；SCI_ours 未取得在线时延优势。SDCF-09 的 99 步整图、分类头及质量验收继续延期，本报告不更新整网性能或 Top-1 结论。

复核入口：ELIGIBILITY.json 保存全部资格检查，COMPARISON.json 保存逐行分布与六组比例；formal/、diagnostics/ 保存声明、实际命令、配置与两方日志，runner/ 保存冻结运行器源码。
