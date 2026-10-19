# 工单 11：完整激活及单步实测

状态：`complete`；补测状态：`exhausted_budget`。每组包含 10 次 fresh 查询，warmup=0；全部已声明的组及失败记录均保留。数值为中位数，内存为各组最大峰值。

## 完整查询边界

| 冻结行 | 状态 | 组数 | Offline (s) | Online (s) | Query total (s) | Sigmoid online (s) | TanH online (s) |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| sequential-a1-segmented-bitdec | qualified | 3 | 11.609379 | 0.281986 | 11.897886 | 0.063779 | 0.064778 |
| sequential-a1-full-bitdec | qualified | 3 | 7.662239 | 0.283932 | 7.944549 | 0.065021 | 0.067477 |
| parallel-a2-segmented-bitdec | qualified | 3 | 11.731217 | 0.207982 | 11.944162 | 0.089768 | 0.089408 |
| parallel-a2-full-bitdec | qualified | 3 | 7.539369 | 0.210444 | 7.745670 | 0.090128 | 0.090120 |
| parallel-a1-full-bitdec | qualified | 1 | 8.722264 | 0.286085 | 9.008916 | 0.140319 | 0.143298 |
| parallel-a2-SCI_preprocessed | qualified | 1 | 1.905504 | 0.578907 | 2.488718 | 0.210572 | 0.196832 |
| parallel-a2-SCI_Beaver_segmented | qualified | 1 | 6.316493 | 0.298845 | 6.609690 | 0.156991 | 0.150512 |
| parallel-a2-SCI_Beaver_segmented_DPF_LUT | qualified | 1 | 7.182176 | 0.234775 | 7.419993 | 0.092591 | 0.075703 |
| sequential-a1-full-segmented-sign | qualified | 1 | 9.424263 | 0.318564 | 9.792776 | 0.068117 | 0.070962 |
| parallel-a2-full-segmented-sign | qualified | 1 | 9.362083 | 0.233304 | 9.626882 | 0.084657 | 0.093000 |

## 会话准备及资源

| 冻结行 | Setup (s) | 公开表准备 work (s) | Owner model (s) | First request (s) | Session total (s) | 两方 RSS 峰值 (MiB) | 两方 VM 峰值 (MiB) |
| --- | ---: | ---: | ---: | ---: | ---: | --- | --- |
| sequential-a1-segmented-bitdec | 2.764103 | 0.000122 | 0.000027 | 15.235555 | 121.688730 | 459.56, 446.46 | 588.65, 570.59 |
| sequential-a1-full-bitdec | 2.768466 | 0.000070 | 0.000041 | 11.175506 | 81.916579 | 480.27, 452.43 | 586.93, 569.66 |
| parallel-a2-segmented-bitdec | 8.447005 | 0.000275 | 0.000041 | 21.410980 | 128.008388 | 478.90, 449.55 | 640.09, 607.85 |
| parallel-a2-full-bitdec | 8.444042 | 0.000252 | 0.000029 | 16.857636 | 85.820855 | 481.74, 497.16 | 631.93, 645.33 |
| parallel-a1-full-bitdec | 8.595690 | 0.000439 | 0.000030 | 18.065046 | 98.747151 | 473.98, 475.23 | 660.98, 644.28 |
| parallel-a2-SCI_preprocessed | 0.723682 | unavailable | 0.000034 | 3.557375 | 25.699316 | 710.59, 493.54 | 892.38, 608.89 |
| parallel-a2-SCI_Beaver_segmented | 3.376732 | unavailable | 0.000042 | 10.317144 | 69.383430 | 553.09, 298.71 | 644.63, 421.30 |
| parallel-a2-SCI_Beaver_segmented_DPF_LUT | 8.629554 | unavailable | 0.000041 | 16.198621 | 83.313547 | 385.27, 354.09 | 536.39, 503.54 |
| sequential-a1-full-segmented-sign | 2.810898 | 0.000071 | 0.000035 | 13.109287 | 101.667490 | 542.10, 545.43 | 645.48, 647.59 |
| parallel-a2-full-segmented-sign | 8.618223 | 0.000398 | 0.000044 | 19.218428 | 105.655917 | 509.22, 516.05 | 657.82, 660.64 |

公开表准备已计入 setup，不能重复加到 setup 或 first request。每组的 setup、owner model、first request 和 session total 是独立直接边界；完整原值在 `groups[].data`。每方及外部解析控制进程的地址空间上限均为 1.5 GiB；控制进程的资源与归档耗时单独记录，不进入协议延迟。

## 配对比较

| 比较 | Step online 加速比 [95% BCa] | Online 判定 | Offline 判定 | 两阶段准入 | 补测状态 |
| --- | --- | --- | --- | --- | --- |
| sequential-a1-lut-construction | 0.9931 [0.9635, 1.0430] | unresolved | improved | unresolved | blocked_by_group_budget |
| sequential-a1-sign | 0.8913 [0.8239, 0.9300] | regressed | regressed | experimental_regression | not_needed |
| parallel-a2-lut-construction | 0.9883 [0.9124, 1.0299] | unresolved | improved | unresolved | blocked_by_group_budget |
| parallel-a2-sign | 0.9020 [0.8397, 0.9833] | regressed | regressed | experimental_regression | not_needed |
| segmented-schedule | 1.3558 [1.3208, 1.4168] | improved | unresolved | experimental_regression | not_needed |
| full-schedule | 1.3492 [1.2417, 1.4143] | improved | unresolved | experimental_regression | not_needed |
| full-branches | 0.9925 [0.9523, 1.0476] | unresolved | regressed | experimental_regression | not_needed |
| full-sharding | 1.3594 [1.2423, 1.4365] | improved | improved | experimental_regression | not_needed |
| method-vs-SCI_preprocessed | 2.7509 [2.5271, 2.9320] | improved | regressed | experimental_regression | not_needed |
| method-vs-SCI_Beaver_segmented | 1.4201 [1.2993, 1.5187] | improved | regressed | experimental_regression | not_needed |
| method-vs-SCI_Beaver_segmented_DPF_LUT | 1.1156 [1.0239, 1.1731] | improved | regressed | experimental_regression | not_needed |

加速比为 baseline/candidate；LUT 构造为 segmented/full。Wall 使用冻结的 10000 次、seed20261013 BCa，在每个测量组内独立抽样，再计算合并中位数之比。网络模型沿用确定性中位数判定，相等为 unchanged。准入与 LUT online 的预定相关指标触发 candidate→baseline 逆序补测，每行最多 3 组；预算用尽后的 unresolved 保留为实验性结论。

## LUT online 边界

| 冻结行 | 激活 | LUT work (s) | 开启窗口 (s) | 本地乘积窗口 (s) | LUT 联合窗口 (s) | 完整激活 (s) |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| sequential-a1-segmented-bitdec | sigmoid | 0.005368 | 0.000321 | 0.005085 | 0.005370 | 0.063779 |
| sequential-a1-segmented-bitdec | tanh | 0.005254 | 0.000319 | 0.004985 | 0.005257 | 0.064778 |
| sequential-a1-full-bitdec | sigmoid | 0.005163 | 0.000407 | 0.004817 | 0.005165 | 0.065021 |
| sequential-a1-full-bitdec | tanh | 0.005019 | 0.000286 | 0.004760 | 0.005021 | 0.067477 |
| parallel-a2-segmented-bitdec | sigmoid | 0.014695 | 0.005568 | 0.012599 | 0.013017 | 0.089768 |
| parallel-a2-segmented-bitdec | tanh | 0.015566 | 0.005304 | 0.012299 | 0.013229 | 0.089408 |
| parallel-a2-full-bitdec | sigmoid | 0.015095 | 0.005663 | 0.012864 | 0.013434 | 0.090128 |
| parallel-a2-full-bitdec | tanh | 0.015059 | 0.004585 | 0.011605 | 0.012094 | 0.090120 |
| parallel-a1-full-bitdec | sigmoid | 0.010844 | 0.000527 | 0.010338 | 0.010847 | 0.140319 |
| parallel-a1-full-bitdec | tanh | 0.009854 | 0.000550 | 0.009455 | 0.009857 | 0.143298 |
| sequential-a1-full-segmented-sign | sigmoid | 0.006952 | 0.000434 | 0.006518 | 0.006953 | 0.068117 |
| sequential-a1-full-segmented-sign | tanh | 0.006628 | 0.000376 | 0.006346 | 0.006631 | 0.070962 |
| parallel-a2-full-segmented-sign | sigmoid | 0.016954 | 0.002144 | 0.009382 | 0.009647 | 0.084657 |
| parallel-a2-full-segmented-sign | tanh | 0.016364 | 0.001545 | 0.009713 | 0.009917 | 0.093000 |

LUT 联合窗口使用每方最早开启到最晚本地乘积结束，再取两方最大值，包含调度间隙。阶段 work 是各 worker 的工作量归因，不能作为并行 elapsed。每种激活独立的 offline wall 未单独计时，全部 folded preparation 的直接计时保留在 JSON。

## 通信与因果轮次

| 冻结行 | 边界 | 双方 sent bytes | 因果轮次 | LAN 模型 (s) | WAN 模型 (s) |
| --- | --- | ---: | ---: | ---: | ---: |
| sequential-a1-segmented-bitdec | step/offline | 2936755540 | 870 | 23.929044 | 62.215111 |
| sequential-a1-segmented-bitdec | step/online | 5860226 | 233 | 0.163382 | 1.049205 |
| sequential-a1-segmented-bitdec | sigmoid/online | 1398432 | 69 | 0.045687 | 0.303969 |
| sequential-a1-segmented-bitdec | sigmoid/lut/online | 51200 | 1 | 0.000910 | 0.005024 |
| sequential-a1-segmented-bitdec | tanh/online | 1529664 | 73 | 0.048737 | 0.322593 |
| sequential-a1-segmented-bitdec | tanh/lut/online | 51200 | 1 | 0.000910 | 0.005024 |
| sequential-a1-full-bitdec | step/offline | 1850189140 | 1000 | 15.301513 | 41.003783 |
| sequential-a1-full-bitdec | step/online | 5860226 | 233 | 0.163382 | 1.049205 |
| sequential-a1-full-bitdec | sigmoid/online | 1398432 | 69 | 0.045687 | 0.303969 |
| sequential-a1-full-bitdec | sigmoid/lut/online | 51200 | 1 | 0.000910 | 0.005024 |
| sequential-a1-full-bitdec | tanh/online | 1529664 | 73 | 0.048737 | 0.322593 |
| sequential-a1-full-bitdec | tanh/lut/online | 51200 | 1 | 0.000910 | 0.005024 |
| parallel-a2-segmented-bitdec | step/offline | 2938230430 | 955 | 23.983343 | 62.584609 |
| parallel-a2-segmented-bitdec | step/online | 5860322 | 134 | 0.113883 | 0.653206 |
| parallel-a2-segmented-bitdec | sigmoid/online | 1398464 | 72 | 0.047188 | 0.315969 |
| parallel-a2-segmented-bitdec | sigmoid/lut/online | 51200 | 1 | 0.000910 | 0.005024 |
| parallel-a2-segmented-bitdec | tanh/online | 1529728 | 76 | 0.050238 | 0.334595 |
| parallel-a2-segmented-bitdec | tanh/lut/online | 51200 | 1 | 0.000910 | 0.005024 |
| parallel-a2-full-bitdec | step/offline | 1851664030 | 1095 | 15.360812 | 41.413281 |
| parallel-a2-full-bitdec | step/online | 5860322 | 134 | 0.113883 | 0.653206 |
| parallel-a2-full-bitdec | sigmoid/online | 1398464 | 72 | 0.047188 | 0.315969 |
| parallel-a2-full-bitdec | sigmoid/lut/online | 51200 | 1 | 0.000910 | 0.005024 |
| parallel-a2-full-bitdec | tanh/online | 1529728 | 76 | 0.050238 | 0.334595 |
| parallel-a2-full-bitdec | tanh/lut/online | 51200 | 1 | 0.000910 | 0.005024 |
| parallel-a1-full-bitdec | step/offline | 1850189332 | 1001 | 15.302015 | 41.007787 |
| parallel-a1-full-bitdec | step/online | 5860226 | 131 | 0.112382 | 0.641205 |
| parallel-a1-full-bitdec | sigmoid/online | 1398432 | 69 | 0.045687 | 0.303969 |
| parallel-a1-full-bitdec | sigmoid/lut/online | 51200 | 1 | 0.000910 | 0.005024 |
| parallel-a1-full-bitdec | tanh/online | 1529664 | 73 | 0.048737 | 0.322593 |
| parallel-a1-full-bitdec | tanh/lut/online | 51200 | 1 | 0.000910 | 0.005024 |
| parallel-a2-SCI_preprocessed | step/offline | 185571244 | 70 | 1.519570 | 3.991425 |
| parallel-a2-SCI_preprocessed | step/online | 91429766 | 185 | 0.823938 | 2.568595 |
| parallel-a2-SCI_preprocessed | sigmoid/online | 21324800 | 52 | 0.196598 | 0.634496 |
| parallel-a2-SCI_preprocessed | tanh/online | 10156800 | 39 | 0.100754 | 0.359136 |
| parallel-a2-SCI_Beaver_segmented | step/offline | 1484987486 | 605 | 12.182400 | 32.119750 |
| parallel-a2-SCI_Beaver_segmented | step/online | 34413730 | 109 | 0.329810 | 1.124275 |
| parallel-a2-SCI_Beaver_segmented | sigmoid/online | 21324800 | 52 | 0.196598 | 0.634496 |
| parallel-a2-SCI_Beaver_segmented | tanh/online | 10156800 | 39 | 0.100754 | 0.359136 |
| parallel-a2-SCI_Beaver_segmented_DPF_LUT | step/offline | 1564189438 | 1475 | 13.251016 | 37.183789 |
| parallel-a2-SCI_Beaver_segmented_DPF_LUT | step/online | 11256930 | 107 | 0.143555 | 0.653139 |
| parallel-a2-SCI_Beaver_segmented_DPF_LUT | sigmoid/online | 5699200 | 50 | 0.070594 | 0.313984 |
| parallel-a2-SCI_Beaver_segmented_DPF_LUT | tanh/online | 2625600 | 37 | 0.039505 | 0.200512 |
| sequential-a1-full-segmented-sign | step/offline | 2049271124 | 1046 | 16.917169 | 45.169422 |
| sequential-a1-full-segmented-sign | step/online | 5245826 | 121 | 0.102467 | 0.588917 |
| sequential-a1-full-segmented-sign | sigmoid/online | 1091232 | 13 | 0.015230 | 0.073825 |
| sequential-a1-full-segmented-sign | sigmoid/lut/online | 51200 | 1 | 0.000910 | 0.005024 |
| sequential-a1-full-segmented-sign | tanh/online | 1222464 | 17 | 0.018280 | 0.092449 |
| sequential-a1-full-segmented-sign | tanh/lut/online | 51200 | 1 | 0.000910 | 0.005024 |
| parallel-a2-full-segmented-sign | step/offline | 2052843166 | 1187 | 17.016245 | 45.804863 |
| parallel-a2-full-segmented-sign | step/online | 5245922 | 78 | 0.080967 | 0.416918 |
| parallel-a2-full-segmented-sign | sigmoid/online | 1091264 | 16 | 0.016730 | 0.085825 |
| parallel-a2-full-segmented-sign | sigmoid/lut/online | 51200 | 1 | 0.000910 | 0.005024 |
| parallel-a2-full-segmented-sign | tanh/online | 1222528 | 20 | 0.019780 | 0.104451 |
| parallel-a2-full-segmented-sign | tanh/lut/online | 51200 | 1 | 0.000910 | 0.005024 |

模型仅计算通信：`8 × 双方 sent bytes / bandwidth + RTT × causal rounds`。LAN 为 1 Gbit/s、0.5 ms RTT；WAN 为 400 Mbit/s、4 ms RTT。不添加 localhost compute。LUT 因果轮次在完整 phase 依赖 DAG 上计算所选阶段联合发送深度，保留外部依赖，不能累加 worker 轮次。

## 每阶段归因

| 冻结行 | 激活 | 阶段 | Offline work (s) | Online work (s) | Offline sent bytes | Online sent bytes |
| --- | --- | --- | ---: | ---: | ---: | ---: |
| sequential-a1-segmented-bitdec | sigmoid | input_adapter | 0.162177 | 0.007294 | 60633088 | 108832 |
| sequential-a1-segmented-bitdec | sigmoid | shared_sign_prepare | 0.205103 | 0.000000 | 27277312 | 0 |
| sequential-a1-segmented-bitdec | sigmoid | shared_bitxa_prepare | 0.379916 | 0.000000 | 92160000 | 0 |
| sequential-a1-segmented-bitdec | sigmoid | sign | 0.000000 | 0.015629 | 0 | 396800 |
| sequential-a1-segmented-bitdec | sigmoid | abs_bitxa | 0.000000 | 0.001337 | 0 | 105600 |
| sequential-a1-segmented-bitdec | sigmoid | clip_compare | 0.000000 | 0.014694 | 0 | 396800 |
| sequential-a1-segmented-bitdec | sigmoid | clip_bitxa | 0.000000 | 0.001217 | 0 | 105600 |
| sequential-a1-segmented-bitdec | sigmoid | index_truncation | 0.035940 | 0.003394 | 12220416 | 128000 |
| sequential-a1-segmented-bitdec | sigmoid | lut_mask | 0.008846 | 0.000000 | 2417152 | 0 |
| sequential-a1-segmented-bitdec | sigmoid | lut_keygen | 0.076602 | 0.000000 | 16486400 | 0 |
| sequential-a1-segmented-bitdec | sigmoid | lut_materialize | 0.059928 | 0.000000 | 0 | 0 |
| sequential-a1-segmented-bitdec | sigmoid | lut_tensor_and | 1.644575 | 0.000000 | 229376000 | 0 |
| sequential-a1-segmented-bitdec | sigmoid | lut_leaf_b2a | 0.894593 | 0.000000 | 314572800 | 0 |
| sequential-a1-segmented-bitdec | sigmoid | lut_payload | 0.000000 | 0.000000 | 0 | 0 |
| sequential-a1-segmented-bitdec | sigmoid | lut_open | 0.000000 | 0.000321 | 0 | 51200 |
| sequential-a1-segmented-bitdec | sigmoid | lut_product | 0.000000 | 0.005084 | 0 | 0 |
| sequential-a1-segmented-bitdec | sigmoid | restore_bitxa | 0.000000 | 0.001645 | 0 | 105600 |
| sequential-a1-segmented-bitdec | sigmoid | output_adapter | 0.000000 | 0.000003 | 0 | 0 |
| sequential-a1-segmented-bitdec | tanh | input_adapter | 0.165090 | 0.008110 | 60633088 | 108832 |
| sequential-a1-segmented-bitdec | tanh | shared_sign_prepare | 0.211376 | 0.000000 | 27277312 | 0 |
| sequential-a1-segmented-bitdec | tanh | shared_bitxa_prepare | 0.394321 | 0.000000 | 92160000 | 0 |
| sequential-a1-segmented-bitdec | tanh | sign | 0.000000 | 0.014452 | 0 | 396800 |
| sequential-a1-segmented-bitdec | tanh | abs_bitxa | 0.000000 | 0.001283 | 0 | 105600 |
| sequential-a1-segmented-bitdec | tanh | clip_compare | 0.000000 | 0.013931 | 0 | 396800 |
| sequential-a1-segmented-bitdec | tanh | clip_bitxa | 0.000000 | 0.001205 | 0 | 105600 |
| sequential-a1-segmented-bitdec | tanh | index_truncation | 0.036525 | 0.003508 | 12220416 | 128000 |
| sequential-a1-segmented-bitdec | tanh | lut_mask | 0.008975 | 0.000000 | 2417152 | 0 |
| sequential-a1-segmented-bitdec | tanh | lut_keygen | 0.079071 | 0.000000 | 16486400 | 0 |
| sequential-a1-segmented-bitdec | tanh | lut_materialize | 0.062900 | 0.000000 | 0 | 0 |
| sequential-a1-segmented-bitdec | tanh | lut_tensor_and | 1.660942 | 0.000000 | 229376000 | 0 |
| sequential-a1-segmented-bitdec | tanh | lut_leaf_b2a | 0.910868 | 0.000000 | 314572800 | 0 |
| sequential-a1-segmented-bitdec | tanh | lut_payload | 0.000000 | 0.000000 | 0 | 0 |
| sequential-a1-segmented-bitdec | tanh | lut_open | 0.000000 | 0.000319 | 0 | 51200 |
| sequential-a1-segmented-bitdec | tanh | lut_product | 0.000000 | 0.004984 | 0 | 0 |
| sequential-a1-segmented-bitdec | tanh | restore_bitxa | 0.000000 | 0.001795 | 0 | 105600 |
| sequential-a1-segmented-bitdec | tanh | output_adapter | 0.070114 | 0.003354 | 20150272 | 131232 |
| sequential-a1-full-bitdec | sigmoid | input_adapter | 0.159776 | 0.007160 | 60633088 | 108832 |
| sequential-a1-full-bitdec | sigmoid | shared_sign_prepare | 0.219172 | 0.000000 | 27277312 | 0 |
| sequential-a1-full-bitdec | sigmoid | shared_bitxa_prepare | 0.383910 | 0.000000 | 92160000 | 0 |
| sequential-a1-full-bitdec | sigmoid | sign | 0.000000 | 0.015110 | 0 | 396800 |
| sequential-a1-full-bitdec | sigmoid | abs_bitxa | 0.000000 | 0.001299 | 0 | 105600 |
| sequential-a1-full-bitdec | sigmoid | clip_compare | 0.000000 | 0.015326 | 0 | 396800 |
| sequential-a1-full-bitdec | sigmoid | clip_bitxa | 0.000000 | 0.001241 | 0 | 105600 |
| sequential-a1-full-bitdec | sigmoid | index_truncation | 0.036493 | 0.003541 | 12220416 | 128000 |
| sequential-a1-full-bitdec | sigmoid | lut_mask | 0.008493 | 0.000000 | 2417152 | 0 |
| sequential-a1-full-bitdec | sigmoid | lut_keygen | 0.159962 | 0.000000 | 16435200 | 0 |
| sequential-a1-full-bitdec | sigmoid | lut_materialize | 0.497438 | 0.000000 | 0 | 0 |
| sequential-a1-full-bitdec | sigmoid | lut_tensor_and | 0.000000 | 0.000000 | 0 | 0 |
| sequential-a1-full-bitdec | sigmoid | lut_leaf_b2a | 0.000000 | 0.000000 | 0 | 0 |
| sequential-a1-full-bitdec | sigmoid | lut_payload | 0.017136 | 0.000000 | 716800 | 0 |
| sequential-a1-full-bitdec | sigmoid | lut_open | 0.000000 | 0.000406 | 0 | 51200 |
| sequential-a1-full-bitdec | sigmoid | lut_product | 0.000000 | 0.004816 | 0 | 0 |
| sequential-a1-full-bitdec | sigmoid | restore_bitxa | 0.000000 | 0.001623 | 0 | 105600 |
| sequential-a1-full-bitdec | sigmoid | output_adapter | 0.000000 | 0.000004 | 0 | 0 |
| sequential-a1-full-bitdec | tanh | input_adapter | 0.162790 | 0.008244 | 60633088 | 108832 |
| sequential-a1-full-bitdec | tanh | shared_sign_prepare | 0.222776 | 0.000000 | 27277312 | 0 |
| sequential-a1-full-bitdec | tanh | shared_bitxa_prepare | 0.395479 | 0.000000 | 92160000 | 0 |
| sequential-a1-full-bitdec | tanh | sign | 0.000000 | 0.014855 | 0 | 396800 |
| sequential-a1-full-bitdec | tanh | abs_bitxa | 0.000000 | 0.001387 | 0 | 105600 |
| sequential-a1-full-bitdec | tanh | clip_compare | 0.000000 | 0.014553 | 0 | 396800 |
| sequential-a1-full-bitdec | tanh | clip_bitxa | 0.000000 | 0.001237 | 0 | 105600 |
| sequential-a1-full-bitdec | tanh | index_truncation | 0.038908 | 0.003651 | 12220416 | 128000 |
| sequential-a1-full-bitdec | tanh | lut_mask | 0.008804 | 0.000000 | 2417152 | 0 |
| sequential-a1-full-bitdec | tanh | lut_keygen | 0.164758 | 0.000000 | 16435200 | 0 |
| sequential-a1-full-bitdec | tanh | lut_materialize | 0.499262 | 0.000000 | 0 | 0 |
| sequential-a1-full-bitdec | tanh | lut_tensor_and | 0.000000 | 0.000000 | 0 | 0 |
| sequential-a1-full-bitdec | tanh | lut_leaf_b2a | 0.000000 | 0.000000 | 0 | 0 |
| sequential-a1-full-bitdec | tanh | lut_payload | 0.019107 | 0.000000 | 716800 | 0 |
| sequential-a1-full-bitdec | tanh | lut_open | 0.000000 | 0.000285 | 0 | 51200 |
| sequential-a1-full-bitdec | tanh | lut_product | 0.000000 | 0.004759 | 0 | 0 |
| sequential-a1-full-bitdec | tanh | restore_bitxa | 0.000000 | 0.001721 | 0 | 105600 |
| sequential-a1-full-bitdec | tanh | output_adapter | 0.069212 | 0.003175 | 20150272 | 131232 |
| parallel-a2-segmented-bitdec | sigmoid | input_adapter | 0.153548 | 0.016283 | 60960768 | 108864 |
| parallel-a2-segmented-bitdec | sigmoid | shared_sign_prepare | 0.190481 | 0.000000 | 27801600 | 0 |
| parallel-a2-segmented-bitdec | sigmoid | shared_bitxa_prepare | 0.394024 | 0.000000 | 92160000 | 0 |
| parallel-a2-segmented-bitdec | sigmoid | sign | 0.000000 | 0.031047 | 0 | 396800 |
| parallel-a2-segmented-bitdec | sigmoid | abs_bitxa | 0.000000 | 0.001724 | 0 | 105600 |
| parallel-a2-segmented-bitdec | sigmoid | clip_compare | 0.000000 | 0.030800 | 0 | 396800 |
| parallel-a2-segmented-bitdec | sigmoid | clip_bitxa | 0.000000 | 0.001931 | 0 | 105600 |
| parallel-a2-segmented-bitdec | sigmoid | index_truncation | 0.034653 | 0.007636 | 12220416 | 128000 |
| parallel-a2-segmented-bitdec | sigmoid | lut_mask | 0.009027 | 0.000000 | 2417152 | 0 |
| parallel-a2-segmented-bitdec | sigmoid | lut_keygen | 0.080985 | 0.000000 | 16486400 | 0 |
| parallel-a2-segmented-bitdec | sigmoid | lut_materialize | 0.059924 | 0.000000 | 0 | 0 |
| parallel-a2-segmented-bitdec | sigmoid | lut_tensor_and | 1.660271 | 0.000000 | 229376000 | 0 |
| parallel-a2-segmented-bitdec | sigmoid | lut_leaf_b2a | 0.913207 | 0.000000 | 314572800 | 0 |
| parallel-a2-segmented-bitdec | sigmoid | lut_payload | 0.000000 | 0.000000 | 0 | 0 |
| parallel-a2-segmented-bitdec | sigmoid | lut_open | 0.000000 | 0.001009 | 0 | 51200 |
| parallel-a2-segmented-bitdec | sigmoid | lut_product | 0.000000 | 0.013955 | 0 | 0 |
| parallel-a2-segmented-bitdec | sigmoid | restore_bitxa | 0.000000 | 0.004637 | 0 | 105600 |
| parallel-a2-segmented-bitdec | sigmoid | output_adapter | 0.000000 | 0.000008 | 0 | 0 |
| parallel-a2-segmented-bitdec | tanh | input_adapter | 0.153386 | 0.014581 | 60960768 | 108864 |
| parallel-a2-segmented-bitdec | tanh | shared_sign_prepare | 0.187338 | 0.000000 | 27801600 | 0 |
| parallel-a2-segmented-bitdec | tanh | shared_bitxa_prepare | 0.392374 | 0.000000 | 92160000 | 0 |
| parallel-a2-segmented-bitdec | tanh | sign | 0.000000 | 0.029801 | 0 | 396800 |
| parallel-a2-segmented-bitdec | tanh | abs_bitxa | 0.000000 | 0.002025 | 0 | 105600 |
| parallel-a2-segmented-bitdec | tanh | clip_compare | 0.000000 | 0.031085 | 0 | 396800 |
| parallel-a2-segmented-bitdec | tanh | clip_bitxa | 0.000000 | 0.002311 | 0 | 105600 |
| parallel-a2-segmented-bitdec | tanh | index_truncation | 0.035894 | 0.007017 | 12220416 | 128000 |
| parallel-a2-segmented-bitdec | tanh | lut_mask | 0.008809 | 0.000000 | 2417152 | 0 |
| parallel-a2-segmented-bitdec | tanh | lut_keygen | 0.079878 | 0.000000 | 16486400 | 0 |
| parallel-a2-segmented-bitdec | tanh | lut_materialize | 0.062436 | 0.000000 | 0 | 0 |
| parallel-a2-segmented-bitdec | tanh | lut_tensor_and | 1.665316 | 0.000000 | 229376000 | 0 |
| parallel-a2-segmented-bitdec | tanh | lut_leaf_b2a | 0.923738 | 0.000000 | 314572800 | 0 |
| parallel-a2-segmented-bitdec | tanh | lut_payload | 0.000000 | 0.000000 | 0 | 0 |
| parallel-a2-segmented-bitdec | tanh | lut_open | 0.000000 | 0.001226 | 0 | 51200 |
| parallel-a2-segmented-bitdec | tanh | lut_product | 0.000000 | 0.013417 | 0 | 0 |
| parallel-a2-segmented-bitdec | tanh | restore_bitxa | 0.000000 | 0.004304 | 0 | 105600 |
| parallel-a2-segmented-bitdec | tanh | output_adapter | 0.062376 | 0.006360 | 19920896 | 131264 |
| parallel-a2-full-bitdec | sigmoid | input_adapter | 0.153978 | 0.017096 | 60960768 | 108864 |
| parallel-a2-full-bitdec | sigmoid | shared_sign_prepare | 0.194376 | 0.000000 | 27801600 | 0 |
| parallel-a2-full-bitdec | sigmoid | shared_bitxa_prepare | 0.394074 | 0.000000 | 92160000 | 0 |
| parallel-a2-full-bitdec | sigmoid | sign | 0.000000 | 0.030658 | 0 | 396800 |
| parallel-a2-full-bitdec | sigmoid | abs_bitxa | 0.000000 | 0.001752 | 0 | 105600 |
| parallel-a2-full-bitdec | sigmoid | clip_compare | 0.000000 | 0.032245 | 0 | 396800 |
| parallel-a2-full-bitdec | sigmoid | clip_bitxa | 0.000000 | 0.002024 | 0 | 105600 |
| parallel-a2-full-bitdec | sigmoid | index_truncation | 0.035653 | 0.007862 | 12220416 | 128000 |
| parallel-a2-full-bitdec | sigmoid | lut_mask | 0.009229 | 0.000000 | 2417152 | 0 |
| parallel-a2-full-bitdec | sigmoid | lut_keygen | 0.158419 | 0.000000 | 16435200 | 0 |
| parallel-a2-full-bitdec | sigmoid | lut_materialize | 0.491861 | 0.000000 | 0 | 0 |
| parallel-a2-full-bitdec | sigmoid | lut_tensor_and | 0.000000 | 0.000000 | 0 | 0 |
| parallel-a2-full-bitdec | sigmoid | lut_leaf_b2a | 0.000000 | 0.000000 | 0 | 0 |
| parallel-a2-full-bitdec | sigmoid | lut_payload | 0.019497 | 0.000000 | 716800 | 0 |
| parallel-a2-full-bitdec | sigmoid | lut_open | 0.000000 | 0.001144 | 0 | 51200 |
| parallel-a2-full-bitdec | sigmoid | lut_product | 0.000000 | 0.014226 | 0 | 0 |
| parallel-a2-full-bitdec | sigmoid | restore_bitxa | 0.000000 | 0.004424 | 0 | 105600 |
| parallel-a2-full-bitdec | sigmoid | output_adapter | 0.000000 | 0.000007 | 0 | 0 |
| parallel-a2-full-bitdec | tanh | input_adapter | 0.156466 | 0.014124 | 60960768 | 108864 |
| parallel-a2-full-bitdec | tanh | shared_sign_prepare | 0.191013 | 0.000000 | 27801600 | 0 |
| parallel-a2-full-bitdec | tanh | shared_bitxa_prepare | 0.405980 | 0.000000 | 92160000 | 0 |
| parallel-a2-full-bitdec | tanh | sign | 0.000000 | 0.031144 | 0 | 396800 |
| parallel-a2-full-bitdec | tanh | abs_bitxa | 0.000000 | 0.002058 | 0 | 105600 |
| parallel-a2-full-bitdec | tanh | clip_compare | 0.000000 | 0.030498 | 0 | 396800 |
| parallel-a2-full-bitdec | tanh | clip_bitxa | 0.000000 | 0.002306 | 0 | 105600 |
| parallel-a2-full-bitdec | tanh | index_truncation | 0.038280 | 0.007049 | 12220416 | 128000 |
| parallel-a2-full-bitdec | tanh | lut_mask | 0.008899 | 0.000000 | 2417152 | 0 |
| parallel-a2-full-bitdec | tanh | lut_keygen | 0.162874 | 0.000000 | 16435200 | 0 |
| parallel-a2-full-bitdec | tanh | lut_materialize | 0.490910 | 0.000000 | 0 | 0 |
| parallel-a2-full-bitdec | tanh | lut_tensor_and | 0.000000 | 0.000000 | 0 | 0 |
| parallel-a2-full-bitdec | tanh | lut_leaf_b2a | 0.000000 | 0.000000 | 0 | 0 |
| parallel-a2-full-bitdec | tanh | lut_payload | 0.018416 | 0.000000 | 716800 | 0 |
| parallel-a2-full-bitdec | tanh | lut_open | 0.000000 | 0.001301 | 0 | 51200 |
| parallel-a2-full-bitdec | tanh | lut_product | 0.000000 | 0.014118 | 0 | 0 |
| parallel-a2-full-bitdec | tanh | restore_bitxa | 0.000000 | 0.004803 | 0 | 105600 |
| parallel-a2-full-bitdec | tanh | output_adapter | 0.066581 | 0.006573 | 19920896 | 131264 |
| parallel-a1-full-bitdec | sigmoid | input_adapter | 0.169205 | 0.013229 | 60633088 | 108832 |
| parallel-a1-full-bitdec | sigmoid | shared_sign_prepare | 0.228519 | 0.000000 | 27277312 | 0 |
| parallel-a1-full-bitdec | sigmoid | shared_bitxa_prepare | 0.380818 | 0.000000 | 92160000 | 0 |
| parallel-a1-full-bitdec | sigmoid | sign | 0.000000 | 0.031469 | 0 | 396800 |
| parallel-a1-full-bitdec | sigmoid | abs_bitxa | 0.000000 | 0.001947 | 0 | 105600 |
| parallel-a1-full-bitdec | sigmoid | clip_compare | 0.000000 | 0.034642 | 0 | 396800 |
| parallel-a1-full-bitdec | sigmoid | clip_bitxa | 0.000000 | 0.002005 | 0 | 105600 |
| parallel-a1-full-bitdec | sigmoid | index_truncation | 0.038754 | 0.005866 | 12220416 | 128000 |
| parallel-a1-full-bitdec | sigmoid | lut_mask | 0.008187 | 0.000000 | 2417152 | 0 |
| parallel-a1-full-bitdec | sigmoid | lut_keygen | 0.167523 | 0.000000 | 16435200 | 0 |
| parallel-a1-full-bitdec | sigmoid | lut_materialize | 0.546420 | 0.000000 | 0 | 0 |
| parallel-a1-full-bitdec | sigmoid | lut_tensor_and | 0.000000 | 0.000000 | 0 | 0 |
| parallel-a1-full-bitdec | sigmoid | lut_leaf_b2a | 0.000000 | 0.000000 | 0 | 0 |
| parallel-a1-full-bitdec | sigmoid | lut_payload | 0.019645 | 0.000000 | 716800 | 0 |
| parallel-a1-full-bitdec | sigmoid | lut_open | 0.000000 | 0.000526 | 0 | 51200 |
| parallel-a1-full-bitdec | sigmoid | lut_product | 0.000000 | 0.010337 | 0 | 0 |
| parallel-a1-full-bitdec | sigmoid | restore_bitxa | 0.000000 | 0.003456 | 0 | 105600 |
| parallel-a1-full-bitdec | sigmoid | output_adapter | 0.000000 | 0.000007 | 0 | 0 |
| parallel-a1-full-bitdec | tanh | input_adapter | 0.167542 | 0.012748 | 60633088 | 108832 |
| parallel-a1-full-bitdec | tanh | shared_sign_prepare | 0.242827 | 0.000000 | 27277312 | 0 |
| parallel-a1-full-bitdec | tanh | shared_bitxa_prepare | 0.401911 | 0.000000 | 92160000 | 0 |
| parallel-a1-full-bitdec | tanh | sign | 0.000000 | 0.029225 | 0 | 396800 |
| parallel-a1-full-bitdec | tanh | abs_bitxa | 0.000000 | 0.002229 | 0 | 105600 |
| parallel-a1-full-bitdec | tanh | clip_compare | 0.000000 | 0.032745 | 0 | 396800 |
| parallel-a1-full-bitdec | tanh | clip_bitxa | 0.000000 | 0.002273 | 0 | 105600 |
| parallel-a1-full-bitdec | tanh | index_truncation | 0.042972 | 0.004593 | 12220416 | 128000 |
| parallel-a1-full-bitdec | tanh | lut_mask | 0.008931 | 0.000000 | 2417152 | 0 |
| parallel-a1-full-bitdec | tanh | lut_keygen | 0.170644 | 0.000000 | 16435200 | 0 |
| parallel-a1-full-bitdec | tanh | lut_materialize | 0.542832 | 0.000000 | 0 | 0 |
| parallel-a1-full-bitdec | tanh | lut_tensor_and | 0.000000 | 0.000000 | 0 | 0 |
| parallel-a1-full-bitdec | tanh | lut_leaf_b2a | 0.000000 | 0.000000 | 0 | 0 |
| parallel-a1-full-bitdec | tanh | lut_payload | 0.020117 | 0.000000 | 716800 | 0 |
| parallel-a1-full-bitdec | tanh | lut_open | 0.000000 | 0.000549 | 0 | 51200 |
| parallel-a1-full-bitdec | tanh | lut_product | 0.000000 | 0.009455 | 0 | 0 |
| parallel-a1-full-bitdec | tanh | restore_bitxa | 0.000000 | 0.003484 | 0 | 105600 |
| parallel-a1-full-bitdec | tanh | output_adapter | 0.073947 | 0.005202 | 20150272 | 131232 |
| sequential-a1-full-segmented-sign | sigmoid | input_adapter | 0.162561 | 0.008733 | 60633088 | 108832 |
| sequential-a1-full-segmented-sign | sigmoid | shared_sign_prepare | 0.000000 | 0.000000 | 0 | 0 |
| sequential-a1-full-segmented-sign | sigmoid | shared_bitxa_prepare | 0.392406 | 0.000000 | 92160000 | 0 |
| sequential-a1-full-segmented-sign | sigmoid | sign | 0.369125 | 0.008148 | 63409152 | 243200 |
| sequential-a1-full-segmented-sign | sigmoid | abs_bitxa | 0.000000 | 0.001672 | 0 | 105600 |
| sequential-a1-full-segmented-sign | sigmoid | clip_compare | 0.317971 | 0.008006 | 63409152 | 243200 |
| sequential-a1-full-segmented-sign | sigmoid | clip_bitxa | 0.000000 | 0.001740 | 0 | 105600 |
| sequential-a1-full-segmented-sign | sigmoid | index_truncation | 0.037151 | 0.003551 | 12220416 | 128000 |
| sequential-a1-full-segmented-sign | sigmoid | lut_mask | 0.009143 | 0.000000 | 2417152 | 0 |
| sequential-a1-full-segmented-sign | sigmoid | lut_keygen | 0.168296 | 0.000000 | 16435200 | 0 |
| sequential-a1-full-segmented-sign | sigmoid | lut_materialize | 0.557178 | 0.000000 | 0 | 0 |
| sequential-a1-full-segmented-sign | sigmoid | lut_tensor_and | 0.000000 | 0.000000 | 0 | 0 |
| sequential-a1-full-segmented-sign | sigmoid | lut_leaf_b2a | 0.000000 | 0.000000 | 0 | 0 |
| sequential-a1-full-segmented-sign | sigmoid | lut_payload | 0.021259 | 0.000000 | 716800 | 0 |
| sequential-a1-full-segmented-sign | sigmoid | lut_open | 0.000000 | 0.000433 | 0 | 51200 |
| sequential-a1-full-segmented-sign | sigmoid | lut_product | 0.000000 | 0.006518 | 0 | 0 |
| sequential-a1-full-segmented-sign | sigmoid | restore_bitxa | 0.000000 | 0.002110 | 0 | 105600 |
| sequential-a1-full-segmented-sign | sigmoid | output_adapter | 0.000000 | 0.000004 | 0 | 0 |
| sequential-a1-full-segmented-sign | tanh | input_adapter | 0.168727 | 0.008910 | 60633088 | 108832 |
| sequential-a1-full-segmented-sign | tanh | shared_sign_prepare | 0.000000 | 0.000000 | 0 | 0 |
| sequential-a1-full-segmented-sign | tanh | shared_bitxa_prepare | 0.399271 | 0.000000 | 92160000 | 0 |
| sequential-a1-full-segmented-sign | tanh | sign | 0.363772 | 0.008401 | 63409152 | 243200 |
| sequential-a1-full-segmented-sign | tanh | abs_bitxa | 0.000000 | 0.001610 | 0 | 105600 |
| sequential-a1-full-segmented-sign | tanh | clip_compare | 0.323321 | 0.008311 | 63409152 | 243200 |
| sequential-a1-full-segmented-sign | tanh | clip_bitxa | 0.000000 | 0.001661 | 0 | 105600 |
| sequential-a1-full-segmented-sign | tanh | index_truncation | 0.042553 | 0.003461 | 12220416 | 128000 |
| sequential-a1-full-segmented-sign | tanh | lut_mask | 0.008331 | 0.000000 | 2417152 | 0 |
| sequential-a1-full-segmented-sign | tanh | lut_keygen | 0.194182 | 0.000000 | 16435200 | 0 |
| sequential-a1-full-segmented-sign | tanh | lut_materialize | 0.554092 | 0.000000 | 0 | 0 |
| sequential-a1-full-segmented-sign | tanh | lut_tensor_and | 0.000000 | 0.000000 | 0 | 0 |
| sequential-a1-full-segmented-sign | tanh | lut_leaf_b2a | 0.000000 | 0.000000 | 0 | 0 |
| sequential-a1-full-segmented-sign | tanh | lut_payload | 0.020219 | 0.000000 | 716800 | 0 |
| sequential-a1-full-segmented-sign | tanh | lut_open | 0.000000 | 0.000375 | 0 | 51200 |
| sequential-a1-full-segmented-sign | tanh | lut_product | 0.000000 | 0.006346 | 0 | 0 |
| sequential-a1-full-segmented-sign | tanh | restore_bitxa | 0.000000 | 0.001902 | 0 | 105600 |
| sequential-a1-full-segmented-sign | tanh | output_adapter | 0.071595 | 0.003566 | 20150272 | 131232 |
| parallel-a2-full-segmented-sign | sigmoid | input_adapter | 0.161873 | 0.016457 | 60960768 | 108864 |
| parallel-a2-full-segmented-sign | sigmoid | shared_sign_prepare | 0.000000 | 0.000000 | 0 | 0 |
| parallel-a2-full-segmented-sign | sigmoid | shared_bitxa_prepare | 0.394378 | 0.000000 | 92160000 | 0 |
| parallel-a2-full-segmented-sign | sigmoid | sign | 0.328887 | 0.021812 | 64195584 | 243200 |
| parallel-a2-full-segmented-sign | sigmoid | abs_bitxa | 0.000000 | 0.002538 | 0 | 105600 |
| parallel-a2-full-segmented-sign | sigmoid | clip_compare | 0.317325 | 0.017913 | 64195584 | 243200 |
| parallel-a2-full-segmented-sign | sigmoid | clip_bitxa | 0.000000 | 0.002660 | 0 | 105600 |
| parallel-a2-full-segmented-sign | sigmoid | index_truncation | 0.037174 | 0.009236 | 12220416 | 128000 |
| parallel-a2-full-segmented-sign | sigmoid | lut_mask | 0.009517 | 0.000000 | 2417152 | 0 |
| parallel-a2-full-segmented-sign | sigmoid | lut_keygen | 0.169208 | 0.000000 | 16435200 | 0 |
| parallel-a2-full-segmented-sign | sigmoid | lut_materialize | 0.550210 | 0.000000 | 0 | 0 |
| parallel-a2-full-segmented-sign | sigmoid | lut_tensor_and | 0.000000 | 0.000000 | 0 | 0 |
| parallel-a2-full-segmented-sign | sigmoid | lut_leaf_b2a | 0.000000 | 0.000000 | 0 | 0 |
| parallel-a2-full-segmented-sign | sigmoid | lut_payload | 0.022227 | 0.000000 | 716800 | 0 |
| parallel-a2-full-segmented-sign | sigmoid | lut_open | 0.000000 | 0.001205 | 0 | 51200 |
| parallel-a2-full-segmented-sign | sigmoid | lut_product | 0.000000 | 0.015534 | 0 | 0 |
| parallel-a2-full-segmented-sign | sigmoid | restore_bitxa | 0.000000 | 0.004739 | 0 | 105600 |
| parallel-a2-full-segmented-sign | sigmoid | output_adapter | 0.000000 | 0.000008 | 0 | 0 |
| parallel-a2-full-segmented-sign | tanh | input_adapter | 0.163991 | 0.016051 | 60960768 | 108864 |
| parallel-a2-full-segmented-sign | tanh | shared_sign_prepare | 0.000000 | 0.000000 | 0 | 0 |
| parallel-a2-full-segmented-sign | tanh | shared_bitxa_prepare | 0.398355 | 0.000000 | 92160000 | 0 |
| parallel-a2-full-segmented-sign | tanh | sign | 0.333944 | 0.020446 | 64195584 | 243200 |
| parallel-a2-full-segmented-sign | tanh | abs_bitxa | 0.000000 | 0.002378 | 0 | 105600 |
| parallel-a2-full-segmented-sign | tanh | clip_compare | 0.321165 | 0.020969 | 64195584 | 243200 |
| parallel-a2-full-segmented-sign | tanh | clip_bitxa | 0.000000 | 0.002645 | 0 | 105600 |
| parallel-a2-full-segmented-sign | tanh | index_truncation | 0.040035 | 0.007417 | 12220416 | 128000 |
| parallel-a2-full-segmented-sign | tanh | lut_mask | 0.009286 | 0.000000 | 2417152 | 0 |
| parallel-a2-full-segmented-sign | tanh | lut_keygen | 0.184792 | 0.000000 | 16435200 | 0 |
| parallel-a2-full-segmented-sign | tanh | lut_materialize | 0.560099 | 0.000000 | 0 | 0 |
| parallel-a2-full-segmented-sign | tanh | lut_tensor_and | 0.000000 | 0.000000 | 0 | 0 |
| parallel-a2-full-segmented-sign | tanh | lut_leaf_b2a | 0.000000 | 0.000000 | 0 | 0 |
| parallel-a2-full-segmented-sign | tanh | lut_payload | 0.021595 | 0.000000 | 716800 | 0 |
| parallel-a2-full-segmented-sign | tanh | lut_open | 0.000000 | 0.001102 | 0 | 51200 |
| parallel-a2-full-segmented-sign | tanh | lut_product | 0.000000 | 0.014970 | 0 | 0 |
| parallel-a2-full-segmented-sign | tanh | restore_bitxa | 0.000000 | 0.006067 | 0 | 105600 |
| parallel-a2-full-segmented-sign | tanh | output_adapter | 0.068459 | 0.007539 | 19920896 | 131264 |

q10 索引当前以 2 字节编码，两个激活的双方 mask-index payload 共 102400 字节；64000 字节的紧凑 bit 编码尚未实现，不计为收益。LUT 构造同时改变树分段和 arithmetic-payload/B2A；SCI 对照还改变数值激活算法。最终方法及质量建议由工单 12 汇总。
