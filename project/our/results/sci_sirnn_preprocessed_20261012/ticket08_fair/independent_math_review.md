# Ticket 08 中文报告独立数字核对

结论：当前八行时间表、六组比例与资格状态全部通过只读复核。未修改候选文件，未重跑性能。

从 `formal/*/parties.json` 独立按查询读取双方 phase，先取同查询双方 seconds 最大值，再对十个查询求中位数；通信读取双方 sent_bytes 之和。重新算出的 offline、input、ready、online、output、直接 query_total 的全部十样本、中位数、范围和总体标准差均与 `COMPARISON.json` 一致。setup、first_query 和直接 session_total 秒数也一致；中文表格六位小数舍入正确。

| 行 | online 中位数（秒） | 直接 query_total 中位数（秒） | offline 中位数（秒） |
|---|---:|---:|---:|
| SCI a4-sequential | 1.277002463 | 1.278204458 | 0.000088027 |
| SCI_preprocessed a4-sequential | 0.525712963 | 2.080641723 | 1.533476048 |
| SCI_ours a4-sequential | 0.567140697 | 6.089555902 | 5.513640108 |
| SCI_Beaver_segmented a4-sequential | 0.281186950 | 6.675237386 | 6.390550929 |
| SCI a2-parallel | 1.132585741 | 1.133537293 | 0.000065021 |
| SCI_preprocessed a2-parallel | 0.414972129 | 1.907757333 | 1.491371970 |
| SCI_ours a2-parallel | 0.498325222 | 6.101866107 | 5.610747580 |
| SCI_Beaver_segmented a2-parallel | 0.286065865 | 6.913272475 | 6.619960332 |

| cohort | 对比 | 在线倍数 | 在线时延变化 | 在线通信降幅 |
|---|---|---:|---:|---:|
| a4-sequential | SCI → SCI_preprocessed | 2.429086883 | -58.832267% | 67.808806% |
| a4-sequential | SCI_preprocessed → SCI_ours | 0.926953339 | +7.880295% | 2.485250% |
| a4-sequential | SCI_preprocessed → SCI_Beaver_segmented | 1.869620777 | -46.513217% | 58.558781% |
| a2-parallel | SCI → SCI_preprocessed | 2.729305566 | -63.360643% | 67.874960% |
| a2-parallel | SCI_preprocessed → SCI_ours | 0.832733546 | +20.086432% | 2.485250% |
| a2-parallel | SCI_preprocessed → SCI_Beaver_segmented | 1.450617427 | -31.063837% | 58.558781% |

六组 offline/query_total 时间与通信的绝对变化、比例、降幅同样核对一致。报告明确展示较大的离线和直接总成本，没有把在线收益写成总成本收益。a4 SCI_preprocessed 与 SCI_ours 在线样本范围相交，报告明确“不足以给出稳定排序”；其余五组范围分离也仅描述本轮十查询观测。

`ELIGIBILITY.json` 的全局 `eligible=true`、`complete=true`；正式八行、诊断十六行与六个 cohort 均齐全且 eligible=true。逐行与双方全部 checks 为 true； artifact checks、冻结二进制/bridge、唯一声明、无多余输出及 public scalar contract 均通过。`COMPARISON.json` 有八行和六组对比，全部 eligible=true。

核对时 session_total_bytes 尚待执行代理补入。独立用不重叠 application_io + setup + 每查询直接 query_total + teardown 的双方实测 sent_bytes 得到：a4 的 SCI/SCI_preprocessed/SCI_ours/SCI_Beaver_segmented 分别为 2867122474、2874481754、18026297380、22685407020 bytes；a2 分别为 2873020714、2879705594、18032472666、22691582406 bytes。已告知 root，代理正补该字段并重生成报告。
