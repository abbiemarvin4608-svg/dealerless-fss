# 实现与正确性证据

四条优化路径均实现为可独立选择的运行配置：

- `--lut-key-family full_q_arithmetic` 使用 full10 算术 DPF，替换 5+5 Boolean tensor 构造。
- `--fuse-sign-wrap` 在有符号扩展中复用 sign 比较，删除 wrap DCF 和一项 correction mask；与 rounding 共用一批 AND，无 rounding 的适用调用增加一轮在线 AND。
- `--bit-block-mode direct_cot --native-ring-cot` 使用原生 block/arithmetic COT；两方向使用独立 OT 实例及新鲜随机 sender base。
- `--cache-lut-leaves` 复用当前 tile 的 full-Q 叶子，减少一次本地树展开；预算包含最终 selectors、工作区及叶缓存，必要时缩小 tile。

基础验证：

| 检查 | 结果 | 证据 |
|---|---|---|
| Block/ring COT 的模式、位宽、尾部长度与交替库存 | 两端通过 | [ot-alternating-final](checks/ot-alternating-final/result.json) |
| Sign/wrap：101 配置、1165 输入值 | 两端通过 | [sign-wrap-results](checks/sign-wrap-results.md) |
| 缓存相同分享、预算缩 tile、fresh 材料与拒绝 reuse | 四项两端通过 | [leaf-cache](checks/leaf-cache/README.md) |
| Fusion spool、预取与多个 context | 两端通过 | [bridge-final](checks/bridge-final/result.json) |

整步独立整数核对；每行两次新鲜材料，零或非零初始隐藏状态：

| 配置 | 结果 | query 数 | 检查值数 | 误差数 |
|---|---|---:|---:|---:|
| ours_full10_zero | passed | 2 | 345856 | 0 |
| ours_full10_nonzero | passed | 2 | 345856 | 0 |
| ours_optimized_zero | passed | 2 | 345856 | 0 |
| ours_optimized_nonzero | passed | 2 | 345856 | 0 |
| SCI_SiRNN_zero | passed | 2 | 345856 | 0 |
| SCI_SiRNN_nonzero | passed | 2 | 345856 | 0 |

原生 SCI 与 folded 激活分别核对各自整数算法。优化版与 full10 使用同一 folded 参考；跨 SCI 比较保留激活算法的区别。完整证据在 [qualification](qualification/STATUS.json)。

正式实验复用通过核对的同一构建，每行 10 次新鲜材料，batch128、单个 recurrent timestep、每方四核与 1.5 GiB 地址空间。LAN/WAN 使用字节和因果轮次通信投影，实际墙钟来自未限速 localhost，二者分别报告。

最终动态桥接库加载位置及指纹在 [FINAL_IDENTITY](build/FINAL_IDENTITY.json)。runtime 编译侧 manifest 早于最后一次共享库重建；实验 PLAN/STATUS 冻结并逐行校验实际库指纹，共同 RUNPATH 指向该库。
