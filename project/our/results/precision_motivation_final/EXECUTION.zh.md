# 执行状态与恢复

本轮已完成RNN转换日志分析、缺失28位原语补测、CNN适用优化迁移与整数路径核验。实验一ResNet-50的四个低位宽完整验证已完成，ResNet-152仍在运行；完整CNN成本尚未测完，不能从诊断小图推算或填入。

## 当前资源限制

Linux cgroup已恢复为整个实验进程树2 GiB RAM加512 MiB swap。此前训练图下载在782类、1564张处被Windows宿主机低虚拟内存保护停止；释放宿主机内存后续传完成，`data/download_resume32_guard.json` 和 `data/DOWNLOAD_STATUS.json` 均为 complete，1000类共2000张。历史停止记录保留，不覆盖。

训练校准清单已冻结为 `experiment1/calibration_manifest.json`，每类两张，与验证集文件身份无重叠。图片来自官方训练压缩包每类最先出现的两张JPEG；清单中的固定种子只在已取得的每类两张子集内选择，不代表从完整训练集随机抽样。

首次正式扫描在ResNet-50的24位校准末尾被宿主机低虚拟内存保护停止，见 `experiment1/scan_next_guard.json`。恢复后的 `experiment1/scan_background_guard.json` 完成ResNet-50全部校准与四个完整验证点，随后在ResNet-152的16位校准期间再次触发宿主机低虚拟内存停止。两次Linux OOM计数均为0。

ResNet-152此前的中断断点 `w16_f13` 已在守护下核验，见 `experiment1/interrupted_checkpoint_guard.json`。随后 `experiment1/scan_resume_guard.json` 完成16、20位校准，并在24位的 `w24_f19` 处理80张后触发宿主机低虚拟内存保护停止，Linux OOM计数仍为0。最新断点的预测长度、logit元素数、按清单重算的正确数和agreement均一致，metrics/accuracy亦匹配。

当前ResNet-152校准已处理55/88个候选：16位16/16（选s=6）、20位20/20（选s=7）、24位19/24（暂最佳s=8）、28位0/28；包括完整评估和有证书的准确率上界剪枝。四个低位宽完整验证尚未开始。宿主机内存恢复后已按原参数续跑，当前持久任务守护记录为 `experiment1/scan_resume_next_guard.json`、日志为 `experiment1/scan_resume_next_guard.log`；结果与断点仍在 `experiment1/width_scan`。后台任务仍由同一cgroup和宿主机阈值守护，不能绕过保护自动反复重启。宿主机虚拟内存保护阈值仍为1 GiB，不调整既定保护阈值。

ResNet-50在2000张训练校准图上选择16/5、20/7、24/8、28/9；相应50000张完整验证Top-1为4.252%、74.260%、75.530%、75.844%。复用32/12结果76.128%，同一FP32结果76.146%。20/7点积发生134次回绕，分母555749200000；其他所选低位宽配置的已测回绕类别计数均为0。因此这些精度差异不能全部归因于回绕，必须同时考虑定点分辨率；现有结果本身不证明32位是唯一满足任意精度容忍度的配置。

为支持长时运行，守护器增加 `run --background`，保留日志和PID，仍执行原有run路径。扫描停止信号只设置标记，在完整batch提交后退出并保存；重复SIGTERM不会中断保存。聚焦CPU检查10 passed、1 skipped，包含停止后恢复不重复计数测试；后台守护入口另以立即退出命令核验成功。

实验三 ReLU 的代码与公开图审计已完成，见 `experiment3_relu_audit.zh.md`。三个正式后端均按每元素一次DReLU加一次bit×x选择实现；Ours的32位输入对应31位分段比较域，选择调用BitXA。该审计不包含新增端到端性能实测。

完整预处理SCI的当前材料存储实现仅线性COT就需要双方合计约2.865/8.070 TiB，超过当前约699 GiB可用空间。两行已保存公开图预检，协议未启动；它们是资源不可用记录，不是协议理论空间下界，也不含实测延迟、通信或OOM结果。其余CNN后端尚未运行。

## 恢复顺序

所有命令在仓库根目录执行。每次守护输出须选择新文件名，已有证据不能覆盖。整个流程一次只启动一个内存守护任务。

训练校准样本已下载完成。仅在数据未完成时使用下述续传命令，并选择新的守护输出名：

```bash
python3 project/our/tools/imagenet_resnet/memory_guard.py run \
  --output project/our/results/precision_motivation_final/data/download_next_guard.json -- \
  env MALLOC_ARENA_MAX=2 MALLOC_TRIM_THRESHOLD_=1 \
  project/our/tools/precision/.venv/bin/python \
  project/our/tools/imagenet_resnet/download_calibration.py \
  --class-map project/our/results/imagenet_resnet_20261015/data/class_to_idx.json \
  --output project/our/results/precision_motivation_final/data --workers 8
```

2000张清单已冻结。下述命令重复执行仅核验一致性：

```bash
project/our/tools/precision/.venv/bin/python \
  project/our/tools/imagenet_resnet/calibration.py \
  --train-dir project/our/results/precision_motivation_final/data/train \
  --val-dir project/our/results/imagenet_resnet_20261015/data/val \
  --output project/our/results/precision_motivation_final/experiment1/calibration_manifest.json
```

当前扫描正在运行，不要重复启动。需要恢复时，先确认cgroup无实验进程，再将下述守护输出名改为新文件名；输出目录保持不变以复用断点：

```bash
python3 project/our/tools/imagenet_resnet/memory_guard.py run \
  --background \
  --output project/our/results/precision_motivation_final/experiment1/scan_resume_after_next_guard.json -- \
  env MALLOC_ARENA_MAX=2 MALLOC_TRIM_THRESHOLD_=1 \
  project/our/tools/precision/.venv/bin/python \
  project/our/tools/imagenet_resnet/width_scan.py \
  --calibration-manifest project/our/results/precision_motivation_final/experiment1/calibration_manifest.json \
  --val-dir project/our/results/imagenet_resnet_20261015/data/val \
  --output-dir project/our/results/precision_motivation_final/experiment1/width_scan \
  --anchor-dir project/our/results/imagenet_resnet_20261015/accuracy \
  --device cuda --batch-size 8 --workers 0 --checkpoint-every 256
```

最后执行固定32/12、batch1完整CNN成本。预处理SCI行会复核已有预检并保留不可用状态，继续其余后端；不以单图成本乘128代替实测。两项材料预算均为每方50 GiB，其他运行参数相同。

```bash
python3 project/our/tools/imagenet_resnet/memory_guard.py run \
  --output project/our/results/precision_motivation_final/performance_next_guard.json -- \
  /usr/bin/python3 project/our/tools/imagenet_resnet/run_performance.py \
  --results project/our/results/precision_motivation_final \
  --binary project/our/results/imagenet_resnet_20261015/final_migration/build/resnet_secure \
  --models-root project/our/results/imagenet_resnet_20261015/models \
  --round-trace off --optimization-profile optimized \
  --spool-budget 53687091200 --native-spool-budget 53687091200 --resume
```

汇总读取所有已落盘的真实记录：

```bash
/usr/bin/python3 project/our/tools/imagenet_resnet/summarize_final.py
```

已有32/12两条完整精度、实验二和原语消融无需重跑。当前状态见 `results/REPORT.zh.md`；迁移范围、已完成的核验和静态资源推导见 `../imagenet_resnet_20261015/final_migration`。
