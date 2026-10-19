# 工单 11：完整激活及单步实测

状态：`incomplete`；补测状态：`pending_extra_groups`。每组包含 10 次 fresh 查询，warmup=0；全部已声明的组及失败记录均保留。数值为中位数，内存为各组最大峰值。

## 完整查询边界

| 冻结行 | 状态 | 组数 | Offline (s) | Online (s) | Query total (s) | Sigmoid online (s) | TanH online (s) |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| sequential-a1-segmented-bitdec | qualified | 1 | 11.247355 | 0.275360 | 11.522610 | 0.062959 | 0.064824 |
| sequential-a1-full-bitdec | qualified | 1 | 7.353355 | 0.284366 | 7.648165 | 0.066995 | 0.067806 |
| parallel-a2-segmented-bitdec | qualified | 1 | 11.119874 | 0.205534 | 11.335896 | 0.086473 | 0.088375 |
| parallel-a2-full-bitdec | oom | 1 | unavailable | unavailable | unavailable | unavailable | unavailable |
| parallel-a1-full-bitdec | qualified | 1 | 7.237498 | 0.222955 | 7.466329 | 0.101339 | 0.104261 |
| parallel-a2-SCI_preprocessed | qualified | 1 | 1.498957 | 0.378935 | 1.884982 | 0.090133 | 0.079188 |
| parallel-a2-SCI_Beaver_segmented | qualified | 1 | 4.924013 | 0.203116 | 5.131829 | 0.087687 | 0.071252 |
| parallel-a2-SCI_Beaver_segmented_DPF_LUT | qualified | 1 | 5.815211 | 0.179386 | 5.994645 | 0.061731 | 0.045855 |
| sequential-a1-full-segmented-sign | qualified | 1 | 7.990248 | 0.258295 | 8.264656 | 0.055334 | 0.055081 |
| parallel-a2-full-segmented-sign | qualified | 1 | 8.022305 | 0.187379 | 8.236822 | 0.070984 | 0.071056 |

## 会话准备及资源

| 冻结行 | Setup (s) | 公开表准备 work (s) | Owner model (s) | First request (s) | Session total (s) | 两方 RSS 峰值 (MiB) | 两方 VM 峰值 (MiB) |
| --- | ---: | ---: | ---: | ---: | ---: | --- | --- |
| sequential-a1-segmented-bitdec | 2.764686 | 0.000087 | 0.000026 | 14.465444 | 117.936316 | 495.04, 517.41 | 654.55, 675.43 |
| sequential-a1-full-bitdec | 2.781111 | 0.000124 | 0.000030 | 10.987257 | 80.409916 | 553.21, 549.41 | 710.69, 707.35 |
| parallel-a2-segmented-bitdec | 8.473024 | 0.000255 | 0.000030 | 20.112819 | 122.177759 | 519.99, 526.67 | 908.53, 908.55 |
| parallel-a2-full-bitdec | unavailable | unavailable | unavailable | unavailable | unavailable | unavailable | unavailable |
| parallel-a1-full-bitdec | 8.487815 | 0.000244 | 0.000029 | 16.315064 | 83.799701 | 739.16, 782.96 | 1075.81, 1115.96 |
| parallel-a2-SCI_preprocessed | 0.660011 | unavailable | 0.000035 | 2.788443 | 19.759570 | 861.11, 547.41 | 1113.95, 873.75 |
| parallel-a2-SCI_Beaver_segmented | 3.246456 | unavailable | 0.000026 | 8.672652 | 54.924566 | 621.85, 369.49 | 770.70, 657.82 |
| parallel-a2-SCI_Beaver_segmented_DPF_LUT | 8.457164 | unavailable | 0.000027 | 14.679792 | 68.860964 | 480.23, 427.70 | 869.18, 816.72 |
| sequential-a1-full-segmented-sign | 2.755075 | 0.000075 | 0.000021 | 11.259584 | 85.829668 | 570.79, 567.66 | 730.09, 725.88 |
| parallel-a2-full-segmented-sign | 8.451489 | 0.000240 | 0.000028 | 16.808895 | 90.780791 | 552.42, 544.57 | 939.86, 927.13 |

公开表准备已计入 setup，不能重复加到 setup 或 first request。每组的 setup、owner model、first request 和 session total 是独立直接边界；完整原值在 `groups[].data`。每方及外部解析控制进程的地址空间上限均为 1.5 GiB；控制进程的资源与归档耗时单独记录，不进入协议延迟。

## 配对比较

| 比较 | Step online 加速比 [95% BCa] | Online 判定 | Offline 判定 | 两阶段准入 | 补测状态 |
| --- | --- | --- | --- | --- | --- |
| sequential-a1-lut-construction | 0.9683 [0.9314, 1.0045] | unresolved | improved | unresolved | required |
| sequential-a1-sign | 1.1009 [1.0572, 1.1463] | improved | regressed | experimental_regression | not_needed |
| parallel-a2-lut-construction | unavailable | unavailable | unavailable | unavailable | unavailable |
| parallel-a2-sign | unavailable | unavailable | unavailable | unavailable | unavailable |
| segmented-schedule | 1.3397 [1.3149, 1.3726] | improved | unresolved | experimental_regression | not_needed |
| full-schedule | unavailable | unavailable | unavailable | unavailable | unavailable |
| full-branches | 1.2754 [1.2298, 1.3312] | improved | unresolved | experimental_regression | not_needed |
| full-sharding | unavailable | unavailable | unavailable | unavailable | unavailable |
| method-vs-SCI_preprocessed | unavailable | unavailable | unavailable | unavailable | unavailable |
| method-vs-SCI_Beaver_segmented | unavailable | unavailable | unavailable | unavailable | unavailable |
| method-vs-SCI_Beaver_segmented_DPF_LUT | unavailable | unavailable | unavailable | unavailable | unavailable |

加速比为 baseline/candidate；LUT 构造为 segmented/full。Wall 使用冻结的 10000 次、seed20261013 BCa，在每个测量组内独立抽样，再计算合并中位数之比。网络模型沿用确定性中位数判定，相等为 unchanged。准入与 LUT online 的预定相关指标触发 candidate→baseline 逆序补测，每行最多 3 组；预算用尽后的 unresolved 保留为实验性结论。

## LUT online 边界

| 冻结行 | 激活 | LUT work (s) | 开启窗口 (s) | 本地乘积窗口 (s) | LUT 联合窗口 (s) | 完整激活 (s) |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| sequential-a1-segmented-bitdec | sigmoid | 0.004953 | 0.000442 | 0.004696 | 0.004954 | 0.062959 |
| sequential-a1-segmented-bitdec | tanh | 0.004674 | 0.000241 | 0.004471 | 0.004675 | 0.064824 |
| sequential-a1-full-bitdec | sigmoid | 0.005159 | 0.000260 | 0.004839 | 0.005162 | 0.066995 |
| sequential-a1-full-bitdec | tanh | 0.005364 | 0.000409 | 0.004505 | 0.005364 | 0.067806 |
| parallel-a2-segmented-bitdec | sigmoid | 0.013714 | 0.006230 | 0.012525 | 0.013165 | 0.086473 |
| parallel-a2-segmented-bitdec | tanh | 0.015047 | 0.006397 | 0.010832 | 0.011746 | 0.088375 |
| parallel-a1-full-bitdec | sigmoid | 0.008578 | 0.000626 | 0.008254 | 0.008581 | 0.101339 |
| parallel-a1-full-bitdec | tanh | 0.008499 | 0.000696 | 0.008245 | 0.008502 | 0.104261 |
| sequential-a1-full-segmented-sign | sigmoid | 0.005242 | 0.000289 | 0.004714 | 0.005245 | 0.055334 |
| sequential-a1-full-segmented-sign | tanh | 0.004403 | 0.000273 | 0.004150 | 0.004405 | 0.055081 |
| parallel-a2-full-segmented-sign | sigmoid | 0.015411 | 0.004688 | 0.012580 | 0.012764 | 0.070984 |
| parallel-a2-full-segmented-sign | tanh | 0.014285 | 0.002751 | 0.011283 | 0.011599 | 0.071056 |

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
| sequential-a1-segmented-bitdec | sigmoid | input_adapter | 0.149879 | 0.006657 | 60633088 | 108832 |
| sequential-a1-segmented-bitdec | sigmoid | shared_sign_prepare | 0.198194 | 0.000000 | 27277312 | 0 |
| sequential-a1-segmented-bitdec | sigmoid | shared_bitxa_prepare | 0.353244 | 0.000000 | 92160000 | 0 |
| sequential-a1-segmented-bitdec | sigmoid | sign | 0.000000 | 0.014556 | 0 | 396800 |
| sequential-a1-segmented-bitdec | sigmoid | abs_bitxa | 0.000000 | 0.001259 | 0 | 105600 |
| sequential-a1-segmented-bitdec | sigmoid | clip_compare | 0.000000 | 0.014863 | 0 | 396800 |
| sequential-a1-segmented-bitdec | sigmoid | clip_bitxa | 0.000000 | 0.001204 | 0 | 105600 |
| sequential-a1-segmented-bitdec | sigmoid | index_truncation | 0.032053 | 0.003657 | 12220416 | 128000 |
| sequential-a1-segmented-bitdec | sigmoid | lut_mask | 0.008397 | 0.000000 | 2417152 | 0 |
| sequential-a1-segmented-bitdec | sigmoid | lut_keygen | 0.073380 | 0.000000 | 16486400 | 0 |
| sequential-a1-segmented-bitdec | sigmoid | lut_materialize | 0.059656 | 0.000000 | 0 | 0 |
| sequential-a1-segmented-bitdec | sigmoid | lut_tensor_and | 1.598188 | 0.000000 | 229376000 | 0 |
| sequential-a1-segmented-bitdec | sigmoid | lut_leaf_b2a | 0.837335 | 0.000000 | 314572800 | 0 |
| sequential-a1-segmented-bitdec | sigmoid | lut_payload | 0.000000 | 0.000000 | 0 | 0 |
| sequential-a1-segmented-bitdec | sigmoid | lut_open | 0.000000 | 0.000441 | 0 | 51200 |
| sequential-a1-segmented-bitdec | sigmoid | lut_product | 0.000000 | 0.004696 | 0 | 0 |
| sequential-a1-segmented-bitdec | sigmoid | restore_bitxa | 0.000000 | 0.001520 | 0 | 105600 |
| sequential-a1-segmented-bitdec | sigmoid | output_adapter | 0.000000 | 0.000003 | 0 | 0 |
| sequential-a1-segmented-bitdec | tanh | input_adapter | 0.146672 | 0.007807 | 60633088 | 108832 |
| sequential-a1-segmented-bitdec | tanh | shared_sign_prepare | 0.196422 | 0.000000 | 27277312 | 0 |
| sequential-a1-segmented-bitdec | tanh | shared_bitxa_prepare | 0.391456 | 0.000000 | 92160000 | 0 |
| sequential-a1-segmented-bitdec | tanh | sign | 0.000000 | 0.014191 | 0 | 396800 |
| sequential-a1-segmented-bitdec | tanh | abs_bitxa | 0.000000 | 0.001301 | 0 | 105600 |
| sequential-a1-segmented-bitdec | tanh | clip_compare | 0.000000 | 0.014247 | 0 | 396800 |
| sequential-a1-segmented-bitdec | tanh | clip_bitxa | 0.000000 | 0.001185 | 0 | 105600 |
| sequential-a1-segmented-bitdec | tanh | index_truncation | 0.032654 | 0.003603 | 12220416 | 128000 |
| sequential-a1-segmented-bitdec | tanh | lut_mask | 0.008613 | 0.000000 | 2417152 | 0 |
| sequential-a1-segmented-bitdec | tanh | lut_keygen | 0.066725 | 0.000000 | 16486400 | 0 |
| sequential-a1-segmented-bitdec | tanh | lut_materialize | 0.059064 | 0.000000 | 0 | 0 |
| sequential-a1-segmented-bitdec | tanh | lut_tensor_and | 1.600051 | 0.000000 | 229376000 | 0 |
| sequential-a1-segmented-bitdec | tanh | lut_leaf_b2a | 0.827441 | 0.000000 | 314572800 | 0 |
| sequential-a1-segmented-bitdec | tanh | lut_payload | 0.000000 | 0.000000 | 0 | 0 |
| sequential-a1-segmented-bitdec | tanh | lut_open | 0.000000 | 0.000240 | 0 | 51200 |
| sequential-a1-segmented-bitdec | tanh | lut_product | 0.000000 | 0.004471 | 0 | 0 |
| sequential-a1-segmented-bitdec | tanh | restore_bitxa | 0.000000 | 0.001518 | 0 | 105600 |
| sequential-a1-segmented-bitdec | tanh | output_adapter | 0.055489 | 0.002921 | 20150272 | 131232 |
| sequential-a1-full-bitdec | sigmoid | input_adapter | 0.154166 | 0.007089 | 60633088 | 108832 |
| sequential-a1-full-bitdec | sigmoid | shared_sign_prepare | 0.202660 | 0.000000 | 27277312 | 0 |
| sequential-a1-full-bitdec | sigmoid | shared_bitxa_prepare | 0.383626 | 0.000000 | 92160000 | 0 |
| sequential-a1-full-bitdec | sigmoid | sign | 0.000000 | 0.014957 | 0 | 396800 |
| sequential-a1-full-bitdec | sigmoid | abs_bitxa | 0.000000 | 0.001240 | 0 | 105600 |
| sequential-a1-full-bitdec | sigmoid | clip_compare | 0.000000 | 0.015342 | 0 | 396800 |
| sequential-a1-full-bitdec | sigmoid | clip_bitxa | 0.000000 | 0.001226 | 0 | 105600 |
| sequential-a1-full-bitdec | sigmoid | index_truncation | 0.032218 | 0.002889 | 12220416 | 128000 |
| sequential-a1-full-bitdec | sigmoid | lut_mask | 0.008365 | 0.000000 | 2417152 | 0 |
| sequential-a1-full-bitdec | sigmoid | lut_keygen | 0.141474 | 0.000000 | 16435200 | 0 |
| sequential-a1-full-bitdec | sigmoid | lut_materialize | 0.490491 | 0.000000 | 0 | 0 |
| sequential-a1-full-bitdec | sigmoid | lut_tensor_and | 0.000000 | 0.000000 | 0 | 0 |
| sequential-a1-full-bitdec | sigmoid | lut_leaf_b2a | 0.000000 | 0.000000 | 0 | 0 |
| sequential-a1-full-bitdec | sigmoid | lut_payload | 0.017033 | 0.000000 | 716800 | 0 |
| sequential-a1-full-bitdec | sigmoid | lut_open | 0.000000 | 0.000259 | 0 | 51200 |
| sequential-a1-full-bitdec | sigmoid | lut_product | 0.000000 | 0.004839 | 0 | 0 |
| sequential-a1-full-bitdec | sigmoid | restore_bitxa | 0.000000 | 0.001656 | 0 | 105600 |
| sequential-a1-full-bitdec | sigmoid | output_adapter | 0.000000 | 0.000003 | 0 | 0 |
| sequential-a1-full-bitdec | tanh | input_adapter | 0.148542 | 0.008185 | 60633088 | 108832 |
| sequential-a1-full-bitdec | tanh | shared_sign_prepare | 0.199425 | 0.000000 | 27277312 | 0 |
| sequential-a1-full-bitdec | tanh | shared_bitxa_prepare | 0.390414 | 0.000000 | 92160000 | 0 |
| sequential-a1-full-bitdec | tanh | sign | 0.000000 | 0.015050 | 0 | 396800 |
| sequential-a1-full-bitdec | tanh | abs_bitxa | 0.000000 | 0.001303 | 0 | 105600 |
| sequential-a1-full-bitdec | tanh | clip_compare | 0.000000 | 0.014821 | 0 | 396800 |
| sequential-a1-full-bitdec | tanh | clip_bitxa | 0.000000 | 0.001210 | 0 | 105600 |
| sequential-a1-full-bitdec | tanh | index_truncation | 0.031990 | 0.003572 | 12220416 | 128000 |
| sequential-a1-full-bitdec | tanh | lut_mask | 0.008625 | 0.000000 | 2417152 | 0 |
| sequential-a1-full-bitdec | tanh | lut_keygen | 0.140879 | 0.000000 | 16435200 | 0 |
| sequential-a1-full-bitdec | tanh | lut_materialize | 0.491621 | 0.000000 | 0 | 0 |
| sequential-a1-full-bitdec | tanh | lut_tensor_and | 0.000000 | 0.000000 | 0 | 0 |
| sequential-a1-full-bitdec | tanh | lut_leaf_b2a | 0.000000 | 0.000000 | 0 | 0 |
| sequential-a1-full-bitdec | tanh | lut_payload | 0.016495 | 0.000000 | 716800 | 0 |
| sequential-a1-full-bitdec | tanh | lut_open | 0.000000 | 0.000409 | 0 | 51200 |
| sequential-a1-full-bitdec | tanh | lut_product | 0.000000 | 0.004504 | 0 | 0 |
| sequential-a1-full-bitdec | tanh | restore_bitxa | 0.000000 | 0.001942 | 0 | 105600 |
| sequential-a1-full-bitdec | tanh | output_adapter | 0.056656 | 0.003075 | 20150272 | 131232 |
| parallel-a2-segmented-bitdec | sigmoid | input_adapter | 0.141128 | 0.014364 | 60960768 | 108864 |
| parallel-a2-segmented-bitdec | sigmoid | shared_sign_prepare | 0.180401 | 0.000000 | 27801600 | 0 |
| parallel-a2-segmented-bitdec | sigmoid | shared_bitxa_prepare | 0.360245 | 0.000000 | 92160000 | 0 |
| parallel-a2-segmented-bitdec | sigmoid | sign | 0.000000 | 0.032829 | 0 | 396800 |
| parallel-a2-segmented-bitdec | sigmoid | abs_bitxa | 0.000000 | 0.001796 | 0 | 105600 |
| parallel-a2-segmented-bitdec | sigmoid | clip_compare | 0.000000 | 0.031471 | 0 | 396800 |
| parallel-a2-segmented-bitdec | sigmoid | clip_bitxa | 0.000000 | 0.001754 | 0 | 105600 |
| parallel-a2-segmented-bitdec | sigmoid | index_truncation | 0.031176 | 0.007434 | 12220416 | 128000 |
| parallel-a2-segmented-bitdec | sigmoid | lut_mask | 0.008621 | 0.000000 | 2417152 | 0 |
| parallel-a2-segmented-bitdec | sigmoid | lut_keygen | 0.073098 | 0.000000 | 16486400 | 0 |
| parallel-a2-segmented-bitdec | sigmoid | lut_materialize | 0.061932 | 0.000000 | 0 | 0 |
| parallel-a2-segmented-bitdec | sigmoid | lut_tensor_and | 1.600745 | 0.000000 | 229376000 | 0 |
| parallel-a2-segmented-bitdec | sigmoid | lut_leaf_b2a | 0.832142 | 0.000000 | 314572800 | 0 |
| parallel-a2-segmented-bitdec | sigmoid | lut_payload | 0.000000 | 0.000000 | 0 | 0 |
| parallel-a2-segmented-bitdec | sigmoid | lut_open | 0.000000 | 0.000984 | 0 | 51200 |
| parallel-a2-segmented-bitdec | sigmoid | lut_product | 0.000000 | 0.013281 | 0 | 0 |
| parallel-a2-segmented-bitdec | sigmoid | restore_bitxa | 0.000000 | 0.005082 | 0 | 105600 |
| parallel-a2-segmented-bitdec | sigmoid | output_adapter | 0.000000 | 0.000005 | 0 | 0 |
| parallel-a2-segmented-bitdec | tanh | input_adapter | 0.140965 | 0.014339 | 60960768 | 108864 |
| parallel-a2-segmented-bitdec | tanh | shared_sign_prepare | 0.175733 | 0.000000 | 27801600 | 0 |
| parallel-a2-segmented-bitdec | tanh | shared_bitxa_prepare | 0.351460 | 0.000000 | 92160000 | 0 |
| parallel-a2-segmented-bitdec | tanh | sign | 0.000000 | 0.035126 | 0 | 396800 |
| parallel-a2-segmented-bitdec | tanh | abs_bitxa | 0.000000 | 0.001931 | 0 | 105600 |
| parallel-a2-segmented-bitdec | tanh | clip_compare | 0.000000 | 0.030959 | 0 | 396800 |
| parallel-a2-segmented-bitdec | tanh | clip_bitxa | 0.000000 | 0.001976 | 0 | 105600 |
| parallel-a2-segmented-bitdec | tanh | index_truncation | 0.030721 | 0.007993 | 12220416 | 128000 |
| parallel-a2-segmented-bitdec | tanh | lut_mask | 0.008076 | 0.000000 | 2417152 | 0 |
| parallel-a2-segmented-bitdec | tanh | lut_keygen | 0.074539 | 0.000000 | 16486400 | 0 |
| parallel-a2-segmented-bitdec | tanh | lut_materialize | 0.060029 | 0.000000 | 0 | 0 |
| parallel-a2-segmented-bitdec | tanh | lut_tensor_and | 1.586206 | 0.000000 | 229376000 | 0 |
| parallel-a2-segmented-bitdec | tanh | lut_leaf_b2a | 0.835915 | 0.000000 | 314572800 | 0 |
| parallel-a2-segmented-bitdec | tanh | lut_payload | 0.000000 | 0.000000 | 0 | 0 |
| parallel-a2-segmented-bitdec | tanh | lut_open | 0.000000 | 0.001239 | 0 | 51200 |
| parallel-a2-segmented-bitdec | tanh | lut_product | 0.000000 | 0.013614 | 0 | 0 |
| parallel-a2-segmented-bitdec | tanh | restore_bitxa | 0.000000 | 0.003619 | 0 | 105600 |
| parallel-a2-segmented-bitdec | tanh | output_adapter | 0.054955 | 0.005888 | 19920896 | 131264 |
| parallel-a1-full-bitdec | sigmoid | input_adapter | 0.147786 | 0.010146 | 60633088 | 108832 |
| parallel-a1-full-bitdec | sigmoid | shared_sign_prepare | 0.200646 | 0.000000 | 27277312 | 0 |
| parallel-a1-full-bitdec | sigmoid | shared_bitxa_prepare | 0.362466 | 0.000000 | 92160000 | 0 |
| parallel-a1-full-bitdec | sigmoid | sign | 0.000000 | 0.024317 | 0 | 396800 |
| parallel-a1-full-bitdec | sigmoid | abs_bitxa | 0.000000 | 0.001524 | 0 | 105600 |
| parallel-a1-full-bitdec | sigmoid | clip_compare | 0.000000 | 0.024425 | 0 | 396800 |
| parallel-a1-full-bitdec | sigmoid | clip_bitxa | 0.000000 | 0.001450 | 0 | 105600 |
| parallel-a1-full-bitdec | sigmoid | index_truncation | 0.032072 | 0.005386 | 12220416 | 128000 |
| parallel-a1-full-bitdec | sigmoid | lut_mask | 0.008530 | 0.000000 | 2417152 | 0 |
| parallel-a1-full-bitdec | sigmoid | lut_keygen | 0.141572 | 0.000000 | 16435200 | 0 |
| parallel-a1-full-bitdec | sigmoid | lut_materialize | 0.488168 | 0.000000 | 0 | 0 |
| parallel-a1-full-bitdec | sigmoid | lut_tensor_and | 0.000000 | 0.000000 | 0 | 0 |
| parallel-a1-full-bitdec | sigmoid | lut_leaf_b2a | 0.000000 | 0.000000 | 0 | 0 |
| parallel-a1-full-bitdec | sigmoid | lut_payload | 0.015820 | 0.000000 | 716800 | 0 |
| parallel-a1-full-bitdec | sigmoid | lut_open | 0.000000 | 0.000626 | 0 | 51200 |
| parallel-a1-full-bitdec | sigmoid | lut_product | 0.000000 | 0.008253 | 0 | 0 |
| parallel-a1-full-bitdec | sigmoid | restore_bitxa | 0.000000 | 0.002541 | 0 | 105600 |
| parallel-a1-full-bitdec | sigmoid | output_adapter | 0.000000 | 0.000003 | 0 | 0 |
| parallel-a1-full-bitdec | tanh | input_adapter | 0.146166 | 0.010986 | 60633088 | 108832 |
| parallel-a1-full-bitdec | tanh | shared_sign_prepare | 0.200170 | 0.000000 | 27277312 | 0 |
| parallel-a1-full-bitdec | tanh | shared_bitxa_prepare | 0.357499 | 0.000000 | 92160000 | 0 |
| parallel-a1-full-bitdec | tanh | sign | 0.000000 | 0.023573 | 0 | 396800 |
| parallel-a1-full-bitdec | tanh | abs_bitxa | 0.000000 | 0.001571 | 0 | 105600 |
| parallel-a1-full-bitdec | tanh | clip_compare | 0.000000 | 0.023838 | 0 | 396800 |
| parallel-a1-full-bitdec | tanh | clip_bitxa | 0.000000 | 0.001590 | 0 | 105600 |
| parallel-a1-full-bitdec | tanh | index_truncation | 0.032084 | 0.005422 | 12220416 | 128000 |
| parallel-a1-full-bitdec | tanh | lut_mask | 0.008122 | 0.000000 | 2417152 | 0 |
| parallel-a1-full-bitdec | tanh | lut_keygen | 0.139927 | 0.000000 | 16435200 | 0 |
| parallel-a1-full-bitdec | tanh | lut_materialize | 0.486387 | 0.000000 | 0 | 0 |
| parallel-a1-full-bitdec | tanh | lut_tensor_and | 0.000000 | 0.000000 | 0 | 0 |
| parallel-a1-full-bitdec | tanh | lut_leaf_b2a | 0.000000 | 0.000000 | 0 | 0 |
| parallel-a1-full-bitdec | tanh | lut_payload | 0.016109 | 0.000000 | 716800 | 0 |
| parallel-a1-full-bitdec | tanh | lut_open | 0.000000 | 0.000695 | 0 | 51200 |
| parallel-a1-full-bitdec | tanh | lut_product | 0.000000 | 0.008245 | 0 | 0 |
| parallel-a1-full-bitdec | tanh | restore_bitxa | 0.000000 | 0.002994 | 0 | 105600 |
| parallel-a1-full-bitdec | tanh | output_adapter | 0.055055 | 0.004046 | 20150272 | 131232 |
| sequential-a1-full-segmented-sign | sigmoid | input_adapter | 0.151676 | 0.007023 | 60633088 | 108832 |
| sequential-a1-full-segmented-sign | sigmoid | shared_sign_prepare | 0.000000 | 0.000000 | 0 | 0 |
| sequential-a1-full-segmented-sign | sigmoid | shared_bitxa_prepare | 0.366383 | 0.000000 | 92160000 | 0 |
| sequential-a1-full-segmented-sign | sigmoid | sign | 0.281143 | 0.006791 | 63409152 | 243200 |
| sequential-a1-full-segmented-sign | sigmoid | abs_bitxa | 0.000000 | 0.001316 | 0 | 105600 |
| sequential-a1-full-segmented-sign | sigmoid | clip_compare | 0.258976 | 0.006731 | 63409152 | 243200 |
| sequential-a1-full-segmented-sign | sigmoid | clip_bitxa | 0.000000 | 0.001229 | 0 | 105600 |
| sequential-a1-full-segmented-sign | sigmoid | index_truncation | 0.031826 | 0.003283 | 12220416 | 128000 |
| sequential-a1-full-segmented-sign | sigmoid | lut_mask | 0.008008 | 0.000000 | 2417152 | 0 |
| sequential-a1-full-segmented-sign | sigmoid | lut_keygen | 0.142877 | 0.000000 | 16435200 | 0 |
| sequential-a1-full-segmented-sign | sigmoid | lut_materialize | 0.496197 | 0.000000 | 0 | 0 |
| sequential-a1-full-segmented-sign | sigmoid | lut_tensor_and | 0.000000 | 0.000000 | 0 | 0 |
| sequential-a1-full-segmented-sign | sigmoid | lut_leaf_b2a | 0.000000 | 0.000000 | 0 | 0 |
| sequential-a1-full-segmented-sign | sigmoid | lut_payload | 0.016303 | 0.000000 | 716800 | 0 |
| sequential-a1-full-segmented-sign | sigmoid | lut_open | 0.000000 | 0.000289 | 0 | 51200 |
| sequential-a1-full-segmented-sign | sigmoid | lut_product | 0.000000 | 0.004714 | 0 | 0 |
| sequential-a1-full-segmented-sign | sigmoid | restore_bitxa | 0.000000 | 0.001490 | 0 | 105600 |
| sequential-a1-full-segmented-sign | sigmoid | output_adapter | 0.000000 | 0.000003 | 0 | 0 |
| sequential-a1-full-segmented-sign | tanh | input_adapter | 0.149407 | 0.007796 | 60633088 | 108832 |
| sequential-a1-full-segmented-sign | tanh | shared_sign_prepare | 0.000000 | 0.000000 | 0 | 0 |
| sequential-a1-full-segmented-sign | tanh | shared_bitxa_prepare | 0.377467 | 0.000000 | 92160000 | 0 |
| sequential-a1-full-segmented-sign | tanh | sign | 0.290809 | 0.006556 | 63409152 | 243200 |
| sequential-a1-full-segmented-sign | tanh | abs_bitxa | 0.000000 | 0.001296 | 0 | 105600 |
| sequential-a1-full-segmented-sign | tanh | clip_compare | 0.261693 | 0.006462 | 63409152 | 243200 |
| sequential-a1-full-segmented-sign | tanh | clip_bitxa | 0.000000 | 0.001370 | 0 | 105600 |
| sequential-a1-full-segmented-sign | tanh | index_truncation | 0.032295 | 0.003935 | 12220416 | 128000 |
| sequential-a1-full-segmented-sign | tanh | lut_mask | 0.007916 | 0.000000 | 2417152 | 0 |
| sequential-a1-full-segmented-sign | tanh | lut_keygen | 0.143006 | 0.000000 | 16435200 | 0 |
| sequential-a1-full-segmented-sign | tanh | lut_materialize | 0.487402 | 0.000000 | 0 | 0 |
| sequential-a1-full-segmented-sign | tanh | lut_tensor_and | 0.000000 | 0.000000 | 0 | 0 |
| sequential-a1-full-segmented-sign | tanh | lut_leaf_b2a | 0.000000 | 0.000000 | 0 | 0 |
| sequential-a1-full-segmented-sign | tanh | lut_payload | 0.016193 | 0.000000 | 716800 | 0 |
| sequential-a1-full-segmented-sign | tanh | lut_open | 0.000000 | 0.000272 | 0 | 51200 |
| sequential-a1-full-segmented-sign | tanh | lut_product | 0.000000 | 0.004150 | 0 | 0 |
| sequential-a1-full-segmented-sign | tanh | restore_bitxa | 0.000000 | 0.001544 | 0 | 105600 |
| sequential-a1-full-segmented-sign | tanh | output_adapter | 0.057181 | 0.003043 | 20150272 | 131232 |
| parallel-a2-full-segmented-sign | sigmoid | input_adapter | 0.140534 | 0.014693 | 60960768 | 108864 |
| parallel-a2-full-segmented-sign | sigmoid | shared_sign_prepare | 0.000000 | 0.000000 | 0 | 0 |
| parallel-a2-full-segmented-sign | sigmoid | shared_bitxa_prepare | 0.376063 | 0.000000 | 92160000 | 0 |
| parallel-a2-full-segmented-sign | sigmoid | sign | 0.267336 | 0.014435 | 64195584 | 243200 |
| parallel-a2-full-segmented-sign | sigmoid | abs_bitxa | 0.000000 | 0.001805 | 0 | 105600 |
| parallel-a2-full-segmented-sign | sigmoid | clip_compare | 0.255222 | 0.012942 | 64195584 | 243200 |
| parallel-a2-full-segmented-sign | sigmoid | clip_bitxa | 0.000000 | 0.001783 | 0 | 105600 |
| parallel-a2-full-segmented-sign | sigmoid | index_truncation | 0.031199 | 0.007506 | 12220416 | 128000 |
| parallel-a2-full-segmented-sign | sigmoid | lut_mask | 0.008477 | 0.000000 | 2417152 | 0 |
| parallel-a2-full-segmented-sign | sigmoid | lut_keygen | 0.139541 | 0.000000 | 16435200 | 0 |
| parallel-a2-full-segmented-sign | sigmoid | lut_materialize | 0.490563 | 0.000000 | 0 | 0 |
| parallel-a2-full-segmented-sign | sigmoid | lut_tensor_and | 0.000000 | 0.000000 | 0 | 0 |
| parallel-a2-full-segmented-sign | sigmoid | lut_leaf_b2a | 0.000000 | 0.000000 | 0 | 0 |
| parallel-a2-full-segmented-sign | sigmoid | lut_payload | 0.017588 | 0.000000 | 716800 | 0 |
| parallel-a2-full-segmented-sign | sigmoid | lut_open | 0.000000 | 0.001038 | 0 | 51200 |
| parallel-a2-full-segmented-sign | sigmoid | lut_product | 0.000000 | 0.014346 | 0 | 0 |
| parallel-a2-full-segmented-sign | sigmoid | restore_bitxa | 0.000000 | 0.003816 | 0 | 105600 |
| parallel-a2-full-segmented-sign | sigmoid | output_adapter | 0.000000 | 0.000005 | 0 | 0 |
| parallel-a2-full-segmented-sign | tanh | input_adapter | 0.138010 | 0.015397 | 60960768 | 108864 |
| parallel-a2-full-segmented-sign | tanh | shared_sign_prepare | 0.000000 | 0.000000 | 0 | 0 |
| parallel-a2-full-segmented-sign | tanh | shared_bitxa_prepare | 0.378524 | 0.000000 | 92160000 | 0 |
| parallel-a2-full-segmented-sign | tanh | sign | 0.267049 | 0.014871 | 64195584 | 243200 |
| parallel-a2-full-segmented-sign | tanh | abs_bitxa | 0.000000 | 0.001832 | 0 | 105600 |
| parallel-a2-full-segmented-sign | tanh | clip_compare | 0.260672 | 0.012817 | 64195584 | 243200 |
| parallel-a2-full-segmented-sign | tanh | clip_bitxa | 0.000000 | 0.001586 | 0 | 105600 |
| parallel-a2-full-segmented-sign | tanh | index_truncation | 0.030506 | 0.006635 | 12220416 | 128000 |
| parallel-a2-full-segmented-sign | tanh | lut_mask | 0.009154 | 0.000000 | 2417152 | 0 |
| parallel-a2-full-segmented-sign | tanh | lut_keygen | 0.143258 | 0.000000 | 16435200 | 0 |
| parallel-a2-full-segmented-sign | tanh | lut_materialize | 0.488918 | 0.000000 | 0 | 0 |
| parallel-a2-full-segmented-sign | tanh | lut_tensor_and | 0.000000 | 0.000000 | 0 | 0 |
| parallel-a2-full-segmented-sign | tanh | lut_leaf_b2a | 0.000000 | 0.000000 | 0 | 0 |
| parallel-a2-full-segmented-sign | tanh | lut_payload | 0.017503 | 0.000000 | 716800 | 0 |
| parallel-a2-full-segmented-sign | tanh | lut_open | 0.000000 | 0.000911 | 0 | 51200 |
| parallel-a2-full-segmented-sign | tanh | lut_product | 0.000000 | 0.013512 | 0 | 0 |
| parallel-a2-full-segmented-sign | tanh | restore_bitxa | 0.000000 | 0.004494 | 0 | 105600 |
| parallel-a2-full-segmented-sign | tanh | output_adapter | 0.056715 | 0.006422 | 19920896 | 131264 |

q10 索引当前以 2 字节编码，两个激活的双方 mask-index payload 共 102400 字节；64000 字节的紧凑 bit 编码尚未实现，不计为收益。LUT 构造同时改变树分段和 arithmetic-payload/B2A；SCI 对照还改变数值激活算法。最终方法及质量建议由工单 12 汇总。

## 缺失或失败证据

- `parallel-a2-full-bitdec`: allocation failure under declared address-space cap
