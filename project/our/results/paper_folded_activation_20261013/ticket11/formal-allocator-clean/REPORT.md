# 工单 11：完整激活及单步实测

状态：`incomplete`；补测状态：`pending_extra_groups`。每组包含 10 次 fresh 查询，warmup=0；全部已声明的组及失败记录均保留。数值为中位数，内存为各组最大峰值。

## 完整查询边界

| 冻结行 | 状态 | 组数 | Offline (s) | Online (s) | Query total (s) | Sigmoid online (s) | TanH online (s) |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| sequential-a1-segmented-bitdec | failed | 2 | unavailable | unavailable | unavailable | unavailable | unavailable |
| sequential-a1-full-bitdec | failed | 2 | unavailable | unavailable | unavailable | unavailable | unavailable |
| parallel-a2-segmented-bitdec | qualified | 1 | 11.609835 | 0.206263 | 11.819511 | 0.083196 | 0.086839 |
| parallel-a2-full-bitdec | qualified | 1 | 7.432160 | 0.206937 | 7.653964 | 0.088390 | 0.089780 |
| parallel-a1-full-bitdec | qualified | 1 | 9.203311 | 0.329276 | 9.562397 | 0.117566 | 0.139443 |
| parallel-a2-SCI_preprocessed | qualified | 1 | 1.921489 | 0.668370 | 2.698334 | 0.206664 | 0.192250 |
| parallel-a2-SCI_Beaver_segmented | qualified | 1 | 7.101357 | 0.430736 | 7.559295 | 0.255241 | 0.239637 |
| parallel-a2-SCI_Beaver_segmented_DPF_LUT | qualified | 1 | 7.520556 | 0.267406 | 7.836331 | 0.098242 | 0.080758 |
| sequential-a1-full-segmented-sign | qualified | 1 | 9.738003 | 0.327012 | 10.051447 | 0.065101 | 0.071887 |
| parallel-a2-full-segmented-sign | qualified | 1 | 9.461762 | 0.219250 | 9.887429 | 0.083410 | 0.086325 |

## 会话准备及资源

| 冻结行 | Setup (s) | 公开表准备 work (s) | Owner model (s) | First request (s) | Session total (s) | 两方 RSS 峰值 (MiB) | 两方 VM 峰值 (MiB) |
| --- | ---: | ---: | ---: | ---: | ---: | --- | --- |
| sequential-a1-segmented-bitdec | unavailable | unavailable | unavailable | unavailable | unavailable | unavailable | unavailable |
| sequential-a1-full-bitdec | unavailable | unavailable | unavailable | unavailable | unavailable | unavailable | unavailable |
| parallel-a2-segmented-bitdec | 8.433058 | 0.000251 | 0.000037 | 21.004978 | 127.131984 | 472.52, 445.46 | 636.06, 605.10 |
| parallel-a2-full-bitdec | 8.415807 | 0.000283 | 0.000027 | 16.687731 | 85.779188 | 483.34, 485.82 | 635.05, 633.70 |
| parallel-a1-full-bitdec | 9.131375 | 0.000808 | 0.000038 | 18.603214 | 105.830105 | 494.20, 469.06 | 661.61, 644.25 |
| parallel-a2-SCI_preprocessed | 0.681629 | unavailable | 0.000031 | 3.082282 | 28.792260 | 727.07, 493.85 | 906.32, 663.58 |
| parallel-a2-SCI_Beaver_segmented | 3.295473 | unavailable | 0.000028 | 10.064553 | 79.664618 | 490.20, 293.61 | 593.34, 420.60 |
| parallel-a2-SCI_Beaver_segmented_DPF_LUT | 8.472661 | unavailable | 0.000025 | 17.625712 | 90.782254 | 381.68, 355.29 | 536.30, 505.44 |
| sequential-a1-full-segmented-sign | 2.781404 | 0.000600 | 0.000035 | 12.113716 | 105.694247 | 537.77, 542.60 | 645.49, 652.66 |
| parallel-a2-full-segmented-sign | 8.476924 | 0.000704 | 0.000023 | 18.057111 | 109.102781 | 509.75, 522.04 | 665.07, 672.62 |

公开表准备已计入 setup，不能重复加到 setup 或 first request。每组的 setup、owner model、first request 和 session total 是独立直接边界；完整原值在 `groups[].data`。每方及外部解析控制进程的地址空间上限均为 1.5 GiB；控制进程的资源与归档耗时单独记录，不进入协议延迟。

## 配对比较

| 比较 | Step online 加速比 [95% BCa] | Online 判定 | Offline 判定 | 两阶段准入 | 补测状态 |
| --- | --- | --- | --- | --- | --- |
| sequential-a1-lut-construction | unavailable | unavailable | unavailable | unavailable | unavailable |
| sequential-a1-sign | unavailable | unavailable | unavailable | unavailable | unavailable |
| parallel-a2-lut-construction | 0.9967 [0.9114, 1.0297] | unresolved | improved | unresolved | required |
| parallel-a2-sign | 0.9438 [0.4550, 1.0574] | unresolved | regressed | experimental_regression | not_needed |
| segmented-schedule | unavailable | unavailable | unavailable | unavailable | unavailable |
| full-schedule | unavailable | unavailable | unavailable | unavailable | unavailable |
| full-branches | unavailable | unavailable | unavailable | unavailable | unavailable |
| full-sharding | 1.5912 [1.1256, 2.4917] | improved | improved | experimental_regression | not_needed |
| method-vs-SCI_preprocessed | 3.2298 [2.4972, 4.3289] | improved | regressed | experimental_regression | not_needed |
| method-vs-SCI_Beaver_segmented | 2.0815 [1.8472, 2.3772] | improved | unresolved | experimental_regression | not_needed |
| method-vs-SCI_Beaver_segmented_DPF_LUT | 1.2922 [1.0997, 1.6089] | improved | unresolved | experimental_regression | not_needed |

加速比为 baseline/candidate；LUT 构造为 segmented/full。Wall 使用冻结的 10000 次、seed20261013 BCa，在每个测量组内独立抽样，再计算合并中位数之比。网络模型沿用确定性中位数判定，相等为 unchanged。准入与 LUT online 的预定相关指标触发 candidate→baseline 逆序补测，每行最多 3 组；预算用尽后的 unresolved 保留为实验性结论。

## LUT online 边界

| 冻结行 | 激活 | LUT work (s) | 开启窗口 (s) | 本地乘积窗口 (s) | LUT 联合窗口 (s) | 完整激活 (s) |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| parallel-a2-segmented-bitdec | sigmoid | 0.013909 | 0.004387 | 0.011366 | 0.011433 | 0.083196 |
| parallel-a2-segmented-bitdec | tanh | 0.014626 | 0.004868 | 0.011371 | 0.011886 | 0.086839 |
| parallel-a2-full-bitdec | sigmoid | 0.015344 | 0.004701 | 0.012686 | 0.012862 | 0.088390 |
| parallel-a2-full-bitdec | tanh | 0.014838 | 0.005517 | 0.012380 | 0.012946 | 0.089780 |
| parallel-a1-full-bitdec | sigmoid | 0.010625 | 0.000592 | 0.010305 | 0.010627 | 0.117566 |
| parallel-a1-full-bitdec | tanh | 0.011307 | 0.001061 | 0.010455 | 0.011311 | 0.139443 |
| sequential-a1-full-segmented-sign | sigmoid | 0.005886 | 0.000338 | 0.005604 | 0.005899 | 0.065101 |
| sequential-a1-full-segmented-sign | tanh | 0.006659 | 0.000410 | 0.006356 | 0.006661 | 0.071887 |
| parallel-a2-full-segmented-sign | sigmoid | 0.015762 | 0.003203 | 0.011384 | 0.012150 | 0.083410 |
| parallel-a2-full-segmented-sign | tanh | 0.013789 | 0.005318 | 0.011370 | 0.011778 | 0.086325 |

LUT 联合窗口使用每方最早开启到最晚本地乘积结束，再取两方最大值，包含调度间隙。阶段 work 是各 worker 的工作量归因，不能作为并行 elapsed。每种激活独立的 offline wall 未单独计时，全部 folded preparation 的直接计时保留在 JSON。

## 通信与因果轮次

| 冻结行 | 边界 | 双方 sent bytes | 因果轮次 | LAN 模型 (s) | WAN 模型 (s) |
| --- | --- | ---: | ---: | ---: | ---: |
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
| parallel-a2-segmented-bitdec | sigmoid | input_adapter | 0.146640 | 0.015093 | 60960768 | 108864 |
| parallel-a2-segmented-bitdec | sigmoid | shared_sign_prepare | 0.183831 | 0.000000 | 27801600 | 0 |
| parallel-a2-segmented-bitdec | sigmoid | shared_bitxa_prepare | 0.392304 | 0.000000 | 92160000 | 0 |
| parallel-a2-segmented-bitdec | sigmoid | sign | 0.000000 | 0.029184 | 0 | 396800 |
| parallel-a2-segmented-bitdec | sigmoid | abs_bitxa | 0.000000 | 0.001659 | 0 | 105600 |
| parallel-a2-segmented-bitdec | sigmoid | clip_compare | 0.000000 | 0.028871 | 0 | 396800 |
| parallel-a2-segmented-bitdec | sigmoid | clip_bitxa | 0.000000 | 0.001639 | 0 | 105600 |
| parallel-a2-segmented-bitdec | sigmoid | index_truncation | 0.033795 | 0.007377 | 12220416 | 128000 |
| parallel-a2-segmented-bitdec | sigmoid | lut_mask | 0.009012 | 0.000000 | 2417152 | 0 |
| parallel-a2-segmented-bitdec | sigmoid | lut_keygen | 0.078691 | 0.000000 | 16486400 | 0 |
| parallel-a2-segmented-bitdec | sigmoid | lut_materialize | 0.060753 | 0.000000 | 0 | 0 |
| parallel-a2-segmented-bitdec | sigmoid | lut_tensor_and | 1.629826 | 0.000000 | 229376000 | 0 |
| parallel-a2-segmented-bitdec | sigmoid | lut_leaf_b2a | 0.896923 | 0.000000 | 314572800 | 0 |
| parallel-a2-segmented-bitdec | sigmoid | lut_payload | 0.000000 | 0.000000 | 0 | 0 |
| parallel-a2-segmented-bitdec | sigmoid | lut_open | 0.000000 | 0.000959 | 0 | 51200 |
| parallel-a2-segmented-bitdec | sigmoid | lut_product | 0.000000 | 0.013012 | 0 | 0 |
| parallel-a2-segmented-bitdec | sigmoid | restore_bitxa | 0.000000 | 0.003554 | 0 | 105600 |
| parallel-a2-segmented-bitdec | sigmoid | output_adapter | 0.000000 | 0.000005 | 0 | 0 |
| parallel-a2-segmented-bitdec | tanh | input_adapter | 0.147292 | 0.014275 | 60960768 | 108864 |
| parallel-a2-segmented-bitdec | tanh | shared_sign_prepare | 0.185725 | 0.000000 | 27801600 | 0 |
| parallel-a2-segmented-bitdec | tanh | shared_bitxa_prepare | 0.400151 | 0.000000 | 92160000 | 0 |
| parallel-a2-segmented-bitdec | tanh | sign | 0.000000 | 0.029291 | 0 | 396800 |
| parallel-a2-segmented-bitdec | tanh | abs_bitxa | 0.000000 | 0.001691 | 0 | 105600 |
| parallel-a2-segmented-bitdec | tanh | clip_compare | 0.000000 | 0.028978 | 0 | 396800 |
| parallel-a2-segmented-bitdec | tanh | clip_bitxa | 0.000000 | 0.002197 | 0 | 105600 |
| parallel-a2-segmented-bitdec | tanh | index_truncation | 0.035411 | 0.007051 | 12220416 | 128000 |
| parallel-a2-segmented-bitdec | tanh | lut_mask | 0.008578 | 0.000000 | 2417152 | 0 |
| parallel-a2-segmented-bitdec | tanh | lut_keygen | 0.076606 | 0.000000 | 16486400 | 0 |
| parallel-a2-segmented-bitdec | tanh | lut_materialize | 0.060073 | 0.000000 | 0 | 0 |
| parallel-a2-segmented-bitdec | tanh | lut_tensor_and | 1.636158 | 0.000000 | 229376000 | 0 |
| parallel-a2-segmented-bitdec | tanh | lut_leaf_b2a | 0.904082 | 0.000000 | 314572800 | 0 |
| parallel-a2-segmented-bitdec | tanh | lut_payload | 0.000000 | 0.000000 | 0 | 0 |
| parallel-a2-segmented-bitdec | tanh | lut_open | 0.000000 | 0.001438 | 0 | 51200 |
| parallel-a2-segmented-bitdec | tanh | lut_product | 0.000000 | 0.013036 | 0 | 0 |
| parallel-a2-segmented-bitdec | tanh | restore_bitxa | 0.000000 | 0.003989 | 0 | 105600 |
| parallel-a2-segmented-bitdec | tanh | output_adapter | 0.060644 | 0.005600 | 19920896 | 131264 |
| parallel-a2-full-bitdec | sigmoid | input_adapter | 0.144946 | 0.015659 | 60960768 | 108864 |
| parallel-a2-full-bitdec | sigmoid | shared_sign_prepare | 0.185608 | 0.000000 | 27801600 | 0 |
| parallel-a2-full-bitdec | sigmoid | shared_bitxa_prepare | 0.389710 | 0.000000 | 92160000 | 0 |
| parallel-a2-full-bitdec | sigmoid | sign | 0.000000 | 0.030487 | 0 | 396800 |
| parallel-a2-full-bitdec | sigmoid | abs_bitxa | 0.000000 | 0.001662 | 0 | 105600 |
| parallel-a2-full-bitdec | sigmoid | clip_compare | 0.000000 | 0.030366 | 0 | 396800 |
| parallel-a2-full-bitdec | sigmoid | clip_bitxa | 0.000000 | 0.001847 | 0 | 105600 |
| parallel-a2-full-bitdec | sigmoid | index_truncation | 0.034347 | 0.008110 | 12220416 | 128000 |
| parallel-a2-full-bitdec | sigmoid | lut_mask | 0.009278 | 0.000000 | 2417152 | 0 |
| parallel-a2-full-bitdec | sigmoid | lut_keygen | 0.145763 | 0.000000 | 16435200 | 0 |
| parallel-a2-full-bitdec | sigmoid | lut_materialize | 0.489869 | 0.000000 | 0 | 0 |
| parallel-a2-full-bitdec | sigmoid | lut_tensor_and | 0.000000 | 0.000000 | 0 | 0 |
| parallel-a2-full-bitdec | sigmoid | lut_leaf_b2a | 0.000000 | 0.000000 | 0 | 0 |
| parallel-a2-full-bitdec | sigmoid | lut_payload | 0.018958 | 0.000000 | 716800 | 0 |
| parallel-a2-full-bitdec | sigmoid | lut_open | 0.000000 | 0.001066 | 0 | 51200 |
| parallel-a2-full-bitdec | sigmoid | lut_product | 0.000000 | 0.014448 | 0 | 0 |
| parallel-a2-full-bitdec | sigmoid | restore_bitxa | 0.000000 | 0.003857 | 0 | 105600 |
| parallel-a2-full-bitdec | sigmoid | output_adapter | 0.000000 | 0.000007 | 0 | 0 |
| parallel-a2-full-bitdec | tanh | input_adapter | 0.142524 | 0.013969 | 60960768 | 108864 |
| parallel-a2-full-bitdec | tanh | shared_sign_prepare | 0.191073 | 0.000000 | 27801600 | 0 |
| parallel-a2-full-bitdec | tanh | shared_bitxa_prepare | 0.396271 | 0.000000 | 92160000 | 0 |
| parallel-a2-full-bitdec | tanh | sign | 0.000000 | 0.029476 | 0 | 396800 |
| parallel-a2-full-bitdec | tanh | abs_bitxa | 0.000000 | 0.001654 | 0 | 105600 |
| parallel-a2-full-bitdec | tanh | clip_compare | 0.000000 | 0.030298 | 0 | 396800 |
| parallel-a2-full-bitdec | tanh | clip_bitxa | 0.000000 | 0.001937 | 0 | 105600 |
| parallel-a2-full-bitdec | tanh | index_truncation | 0.037535 | 0.007169 | 12220416 | 128000 |
| parallel-a2-full-bitdec | tanh | lut_mask | 0.008348 | 0.000000 | 2417152 | 0 |
| parallel-a2-full-bitdec | tanh | lut_keygen | 0.159600 | 0.000000 | 16435200 | 0 |
| parallel-a2-full-bitdec | tanh | lut_materialize | 0.489697 | 0.000000 | 0 | 0 |
| parallel-a2-full-bitdec | tanh | lut_tensor_and | 0.000000 | 0.000000 | 0 | 0 |
| parallel-a2-full-bitdec | tanh | lut_leaf_b2a | 0.000000 | 0.000000 | 0 | 0 |
| parallel-a2-full-bitdec | tanh | lut_payload | 0.018150 | 0.000000 | 716800 | 0 |
| parallel-a2-full-bitdec | tanh | lut_open | 0.000000 | 0.001508 | 0 | 51200 |
| parallel-a2-full-bitdec | tanh | lut_product | 0.000000 | 0.014212 | 0 | 0 |
| parallel-a2-full-bitdec | tanh | restore_bitxa | 0.000000 | 0.004772 | 0 | 105600 |
| parallel-a2-full-bitdec | tanh | output_adapter | 0.066536 | 0.006387 | 19920896 | 131264 |
| parallel-a1-full-bitdec | sigmoid | input_adapter | 0.173177 | 0.011690 | 60633088 | 108832 |
| parallel-a1-full-bitdec | sigmoid | shared_sign_prepare | 0.236109 | 0.000000 | 27277312 | 0 |
| parallel-a1-full-bitdec | sigmoid | shared_bitxa_prepare | 0.416702 | 0.000000 | 92160000 | 0 |
| parallel-a1-full-bitdec | sigmoid | sign | 0.000000 | 0.031602 | 0 | 396800 |
| parallel-a1-full-bitdec | sigmoid | abs_bitxa | 0.000000 | 0.001867 | 0 | 105600 |
| parallel-a1-full-bitdec | sigmoid | clip_compare | 0.000000 | 0.029831 | 0 | 396800 |
| parallel-a1-full-bitdec | sigmoid | clip_bitxa | 0.000000 | 0.001727 | 0 | 105600 |
| parallel-a1-full-bitdec | sigmoid | index_truncation | 0.038714 | 0.004570 | 12220416 | 128000 |
| parallel-a1-full-bitdec | sigmoid | lut_mask | 0.008699 | 0.000000 | 2417152 | 0 |
| parallel-a1-full-bitdec | sigmoid | lut_keygen | 0.184112 | 0.000000 | 16435200 | 0 |
| parallel-a1-full-bitdec | sigmoid | lut_materialize | 0.576296 | 0.000000 | 0 | 0 |
| parallel-a1-full-bitdec | sigmoid | lut_tensor_and | 0.000000 | 0.000000 | 0 | 0 |
| parallel-a1-full-bitdec | sigmoid | lut_leaf_b2a | 0.000000 | 0.000000 | 0 | 0 |
| parallel-a1-full-bitdec | sigmoid | lut_payload | 0.023751 | 0.000000 | 716800 | 0 |
| parallel-a1-full-bitdec | sigmoid | lut_open | 0.000000 | 0.000592 | 0 | 51200 |
| parallel-a1-full-bitdec | sigmoid | lut_product | 0.000000 | 0.010305 | 0 | 0 |
| parallel-a1-full-bitdec | sigmoid | restore_bitxa | 0.000000 | 0.002776 | 0 | 105600 |
| parallel-a1-full-bitdec | sigmoid | output_adapter | 0.000000 | 0.000007 | 0 | 0 |
| parallel-a1-full-bitdec | tanh | input_adapter | 0.175894 | 0.015669 | 60633088 | 108832 |
| parallel-a1-full-bitdec | tanh | shared_sign_prepare | 0.245678 | 0.000000 | 27277312 | 0 |
| parallel-a1-full-bitdec | tanh | shared_bitxa_prepare | 0.412012 | 0.000000 | 92160000 | 0 |
| parallel-a1-full-bitdec | tanh | sign | 0.000000 | 0.033630 | 0 | 396800 |
| parallel-a1-full-bitdec | tanh | abs_bitxa | 0.000000 | 0.002484 | 0 | 105600 |
| parallel-a1-full-bitdec | tanh | clip_compare | 0.000000 | 0.032559 | 0 | 396800 |
| parallel-a1-full-bitdec | tanh | clip_bitxa | 0.000000 | 0.002101 | 0 | 105600 |
| parallel-a1-full-bitdec | tanh | index_truncation | 0.042526 | 0.006804 | 12220416 | 128000 |
| parallel-a1-full-bitdec | tanh | lut_mask | 0.008757 | 0.000000 | 2417152 | 0 |
| parallel-a1-full-bitdec | tanh | lut_keygen | 0.190751 | 0.000000 | 16435200 | 0 |
| parallel-a1-full-bitdec | tanh | lut_materialize | 0.610116 | 0.000000 | 0 | 0 |
| parallel-a1-full-bitdec | tanh | lut_tensor_and | 0.000000 | 0.000000 | 0 | 0 |
| parallel-a1-full-bitdec | tanh | lut_leaf_b2a | 0.000000 | 0.000000 | 0 | 0 |
| parallel-a1-full-bitdec | tanh | lut_payload | 0.028374 | 0.000000 | 716800 | 0 |
| parallel-a1-full-bitdec | tanh | lut_open | 0.000000 | 0.001061 | 0 | 51200 |
| parallel-a1-full-bitdec | tanh | lut_product | 0.000000 | 0.010454 | 0 | 0 |
| parallel-a1-full-bitdec | tanh | restore_bitxa | 0.000000 | 0.002949 | 0 | 105600 |
| parallel-a1-full-bitdec | tanh | output_adapter | 0.076836 | 0.004239 | 20150272 | 131232 |
| sequential-a1-full-segmented-sign | sigmoid | input_adapter | 0.171032 | 0.008435 | 60633088 | 108832 |
| sequential-a1-full-segmented-sign | sigmoid | shared_sign_prepare | 0.000000 | 0.000000 | 0 | 0 |
| sequential-a1-full-segmented-sign | sigmoid | shared_bitxa_prepare | 0.401752 | 0.000000 | 92160000 | 0 |
| sequential-a1-full-segmented-sign | sigmoid | sign | 0.356662 | 0.007890 | 63409152 | 243200 |
| sequential-a1-full-segmented-sign | sigmoid | abs_bitxa | 0.000000 | 0.001623 | 0 | 105600 |
| sequential-a1-full-segmented-sign | sigmoid | clip_compare | 0.303675 | 0.007985 | 63409152 | 243200 |
| sequential-a1-full-segmented-sign | sigmoid | clip_bitxa | 0.000000 | 0.001432 | 0 | 105600 |
| sequential-a1-full-segmented-sign | sigmoid | index_truncation | 0.038668 | 0.004001 | 12220416 | 128000 |
| sequential-a1-full-segmented-sign | sigmoid | lut_mask | 0.009308 | 0.000000 | 2417152 | 0 |
| sequential-a1-full-segmented-sign | sigmoid | lut_keygen | 0.177993 | 0.000000 | 16435200 | 0 |
| sequential-a1-full-segmented-sign | sigmoid | lut_materialize | 0.635186 | 0.000000 | 0 | 0 |
| sequential-a1-full-segmented-sign | sigmoid | lut_tensor_and | 0.000000 | 0.000000 | 0 | 0 |
| sequential-a1-full-segmented-sign | sigmoid | lut_leaf_b2a | 0.000000 | 0.000000 | 0 | 0 |
| sequential-a1-full-segmented-sign | sigmoid | lut_payload | 0.023914 | 0.000000 | 716800 | 0 |
| sequential-a1-full-segmented-sign | sigmoid | lut_open | 0.000000 | 0.000337 | 0 | 51200 |
| sequential-a1-full-segmented-sign | sigmoid | lut_product | 0.000000 | 0.005603 | 0 | 0 |
| sequential-a1-full-segmented-sign | sigmoid | restore_bitxa | 0.000000 | 0.001714 | 0 | 105600 |
| sequential-a1-full-segmented-sign | sigmoid | output_adapter | 0.000000 | 0.000005 | 0 | 0 |
| sequential-a1-full-segmented-sign | tanh | input_adapter | 0.168090 | 0.010025 | 60633088 | 108832 |
| sequential-a1-full-segmented-sign | tanh | shared_sign_prepare | 0.000000 | 0.000000 | 0 | 0 |
| sequential-a1-full-segmented-sign | tanh | shared_bitxa_prepare | 0.424979 | 0.000000 | 92160000 | 0 |
| sequential-a1-full-segmented-sign | tanh | sign | 0.340485 | 0.008273 | 63409152 | 243200 |
| sequential-a1-full-segmented-sign | tanh | abs_bitxa | 0.000000 | 0.001561 | 0 | 105600 |
| sequential-a1-full-segmented-sign | tanh | clip_compare | 0.317055 | 0.008743 | 63409152 | 243200 |
| sequential-a1-full-segmented-sign | tanh | clip_bitxa | 0.000000 | 0.001628 | 0 | 105600 |
| sequential-a1-full-segmented-sign | tanh | index_truncation | 0.042743 | 0.003736 | 12220416 | 128000 |
| sequential-a1-full-segmented-sign | tanh | lut_mask | 0.008776 | 0.000000 | 2417152 | 0 |
| sequential-a1-full-segmented-sign | tanh | lut_keygen | 0.196269 | 0.000000 | 16435200 | 0 |
| sequential-a1-full-segmented-sign | tanh | lut_materialize | 0.599572 | 0.000000 | 0 | 0 |
| sequential-a1-full-segmented-sign | tanh | lut_tensor_and | 0.000000 | 0.000000 | 0 | 0 |
| sequential-a1-full-segmented-sign | tanh | lut_leaf_b2a | 0.000000 | 0.000000 | 0 | 0 |
| sequential-a1-full-segmented-sign | tanh | lut_payload | 0.023952 | 0.000000 | 716800 | 0 |
| sequential-a1-full-segmented-sign | tanh | lut_open | 0.000000 | 0.000409 | 0 | 51200 |
| sequential-a1-full-segmented-sign | tanh | lut_product | 0.000000 | 0.006356 | 0 | 0 |
| sequential-a1-full-segmented-sign | tanh | restore_bitxa | 0.000000 | 0.002260 | 0 | 105600 |
| sequential-a1-full-segmented-sign | tanh | output_adapter | 0.071511 | 0.003962 | 20150272 | 131232 |
| parallel-a2-full-segmented-sign | sigmoid | input_adapter | 0.168044 | 0.016724 | 60960768 | 108864 |
| parallel-a2-full-segmented-sign | sigmoid | shared_sign_prepare | 0.000000 | 0.000000 | 0 | 0 |
| parallel-a2-full-segmented-sign | sigmoid | shared_bitxa_prepare | 0.402644 | 0.000000 | 92160000 | 0 |
| parallel-a2-full-segmented-sign | sigmoid | sign | 0.313972 | 0.019015 | 64195584 | 243200 |
| parallel-a2-full-segmented-sign | sigmoid | abs_bitxa | 0.000000 | 0.002080 | 0 | 105600 |
| parallel-a2-full-segmented-sign | sigmoid | clip_compare | 0.301673 | 0.016019 | 64195584 | 243200 |
| parallel-a2-full-segmented-sign | sigmoid | clip_bitxa | 0.000000 | 0.002463 | 0 | 105600 |
| parallel-a2-full-segmented-sign | sigmoid | index_truncation | 0.038819 | 0.007661 | 12220416 | 128000 |
| parallel-a2-full-segmented-sign | sigmoid | lut_mask | 0.009924 | 0.000000 | 2417152 | 0 |
| parallel-a2-full-segmented-sign | sigmoid | lut_keygen | 0.186450 | 0.000000 | 16435200 | 0 |
| parallel-a2-full-segmented-sign | sigmoid | lut_materialize | 0.594369 | 0.000000 | 0 | 0 |
| parallel-a2-full-segmented-sign | sigmoid | lut_tensor_and | 0.000000 | 0.000000 | 0 | 0 |
| parallel-a2-full-segmented-sign | sigmoid | lut_leaf_b2a | 0.000000 | 0.000000 | 0 | 0 |
| parallel-a2-full-segmented-sign | sigmoid | lut_payload | 0.023858 | 0.000000 | 716800 | 0 |
| parallel-a2-full-segmented-sign | sigmoid | lut_open | 0.000000 | 0.000749 | 0 | 51200 |
| parallel-a2-full-segmented-sign | sigmoid | lut_product | 0.000000 | 0.014816 | 0 | 0 |
| parallel-a2-full-segmented-sign | sigmoid | restore_bitxa | 0.000000 | 0.005561 | 0 | 105600 |
| parallel-a2-full-segmented-sign | sigmoid | output_adapter | 0.000000 | 0.000012 | 0 | 0 |
| parallel-a2-full-segmented-sign | tanh | input_adapter | 0.173999 | 0.015178 | 60960768 | 108864 |
| parallel-a2-full-segmented-sign | tanh | shared_sign_prepare | 0.000000 | 0.000000 | 0 | 0 |
| parallel-a2-full-segmented-sign | tanh | shared_bitxa_prepare | 0.412809 | 0.000000 | 92160000 | 0 |
| parallel-a2-full-segmented-sign | tanh | sign | 0.326290 | 0.015828 | 64195584 | 243200 |
| parallel-a2-full-segmented-sign | tanh | abs_bitxa | 0.000000 | 0.002393 | 0 | 105600 |
| parallel-a2-full-segmented-sign | tanh | clip_compare | 0.317747 | 0.018506 | 64195584 | 243200 |
| parallel-a2-full-segmented-sign | tanh | clip_bitxa | 0.000000 | 0.002427 | 0 | 105600 |
| parallel-a2-full-segmented-sign | tanh | index_truncation | 0.040379 | 0.007201 | 12220416 | 128000 |
| parallel-a2-full-segmented-sign | tanh | lut_mask | 0.009171 | 0.000000 | 2417152 | 0 |
| parallel-a2-full-segmented-sign | tanh | lut_keygen | 0.182314 | 0.000000 | 16435200 | 0 |
| parallel-a2-full-segmented-sign | tanh | lut_materialize | 0.580918 | 0.000000 | 0 | 0 |
| parallel-a2-full-segmented-sign | tanh | lut_tensor_and | 0.000000 | 0.000000 | 0 | 0 |
| parallel-a2-full-segmented-sign | tanh | lut_leaf_b2a | 0.000000 | 0.000000 | 0 | 0 |
| parallel-a2-full-segmented-sign | tanh | lut_payload | 0.024082 | 0.000000 | 716800 | 0 |
| parallel-a2-full-segmented-sign | tanh | lut_open | 0.000000 | 0.001257 | 0 | 51200 |
| parallel-a2-full-segmented-sign | tanh | lut_product | 0.000000 | 0.013135 | 0 | 0 |
| parallel-a2-full-segmented-sign | tanh | restore_bitxa | 0.000000 | 0.004287 | 0 | 105600 |
| parallel-a2-full-segmented-sign | tanh | output_adapter | 0.067828 | 0.007124 | 19920896 | 131264 |

q10 索引当前以 2 字节编码，两个激活的双方 mask-index payload 共 102400 字节；64000 字节的紧凑 bit 编码尚未实现，不计为收益。LUT 构造同时改变树分段和 arithmetic-payload/B2A；SCI 对照还改变数值激活算法。最终方法及质量建议由工单 12 汇总。

## 缺失或失败证据

- `sequential-a1-segmented-bitdec`: runner produced no result.json
- `sequential-a1-full-bitdec`: runner produced no result.json
