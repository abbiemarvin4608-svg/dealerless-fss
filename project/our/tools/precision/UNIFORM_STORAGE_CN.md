# E0-U-S：统一 ell 位存储与宽累加实验使用说明

2026-10-08 用户恢复宽临时累加方案。当前默认是
`--arithmetic-mode wide --overflow-mode saturate`：输入、权重、bias、
BN 推理系数和层输出统一 Q(ell,s)；乘积与点积在经检查的 128 位宽
临时寄存器中计算，点积结束后 floor 缩放一次并饱和写回 ell 位。
平均池化先宽求和再除法，残差宽加后写回。该模式是明文精度模拟器；
两方宽环协议的共享转换、截断和饱和必须另外实现与验证。

实验协议见 [UNIFORM_STORAGE_EXPERIMENT_CN.md](UNIFORM_STORAGE_EXPERIMENT_CN.md)。
显式严格环对照的数值规则见 [STRICT_RING_CN.md](STRICT_RING_CN.md)。

## 正式方案与当前证据

完整 torchvision ResNet-50/101/152，`IMAGENET1K_V1` 权重、官方
`weights.transforms()`、原始 1000 类输出头，不重新训练。训练 split
选 512 张类均衡图校准，完整 ImageNette validation 为 3925 张；
FP32 matmul 与 cuDNN 的 TF32 关闭。

每个 `ell ∈ {12,16,20,24,28,32}` 遍历全部 `s=0..ell-1`。按校准
Top-1 选择 s，并列取较小 s；验证集不参与选点。每模型 132 个校准
配置和 6 个完整验证配置，三模型为 396+18。wide 模式只在逐项数值
边界证明结果完全相同时复用跨 ell 的预测，并保留测量来源。

已完成完整准确率 CSV 位于 `results/e0_uniform_storage_imagenette`，
属于 wide+saturate 的 E0-U-S；此前 24 位接近 FP32 的数字适用于该
规则。**未完成正式 strict ring 准确率扫描。** 恢复默认不改写已有
CSV、预测或源码快照。

## 完整运行

```bash
cd /home/jiang/dealerless-lunwen/project/our/tools/precision
MAX_JOBS=1 python3 run_e0_uniform_storage.py \
  --models resnet50,resnet101,resnet152 \
  --ell-grid 12,16,20,24,28,32 \
  --calibration-samples 512 --test-samples 0 \
  --device cuda --batch-size 16 --workers 2 \
  --integer-backend gemm \
  --register-bits 128 --arithmetic-mode wide --overflow-mode saturate \
  --results-dir results/e0_uniform_storage_imagenette
```

显存不足时调小 batch。中断后使用同一命令继续；按模型调度时修改
`--models`，避免同时写一个目录。`--phase scan` 仅做校准，
`--phase validate` 使用已经完成的全部校准候选。

`--integer-backend scalar` 使用标量路径，`gemm` 使用整数 limb/GEMM
路径。二者采用相同宽累加和饱和写回规则。更换后端须用新目录；
寄存器统计应区分标量路径的实测前缀与 GEMM 的已证明上界。

诊断命令应另设目录：

```bash
python3 run_e0_uniform_storage.py \
  --models resnet50 --ell-grid 16,32 \
  --calibration-samples 16 --test-samples 16 \
  --device cuda --batch-size 8 --integer-backend gemm \
  --arithmetic-mode wide --overflow-mode saturate \
  --results-dir results/e0_uniform_storage_diagnostic
```

缩小样本的结果标记 `evidence_scope=diagnostic`。正式证据要求训练
校准 512 张、完整验证 3925 张；只将完成的配置写入结果表，不为未
运行的配置填零。明文耗时不是安全推理性能。

## 结果文件与复核

- `e0_uniform_storage_candidates.csv`：全部训练校准候选。
- `e0_uniform_storage_final.csv`：训练选点后的完整验证结果。
- `uniform_storage_<model>_runs.csv`、`_selected.csv`、`_status.json`、
  `_metadata.json`：配置、选择记录、检查点与完成状态。
- `*_manifest.csv`：样本路径、顺序及 ImageNet 标签映射。
- `predictions/`：FP32 与定点逐样本预测。
- `UNIFORM_STORAGE_REPORT_CN.md`：当前语义、规模与已完成结果。

报告 Top-1、FP32 一致率、预测变化配对计数、参数/输入/层输出的
饱和裁剪计数、临时寄存器溢出及最大位宽。零寄存器溢出表示宽临时
计算通过范围检查；零裁剪不代表没有量化误差。

## 保留的显式严格环对照

`reproduce_uniform_storage.sh` 显式设置宽累加和饱和，复现已有结果。
`reproduce_uniform_ring.sh` 显式设置 `ring + wrap`；若要运行严格环，
使用其独立目录：

```bash
python3 run_e0_uniform_storage.py \
  --models resnet50,resnet101,resnet152 \
  --ell-grid 12,16,20,24,28,32 \
  --calibration-samples 512 --test-samples 0 \
  --device cuda --batch-size 16 --workers 2 --integer-backend gemm \
  --arithmetic-mode ring --overflow-mode wrap \
  --results-dir results/e0_uniform_ring_imagenette
```

历史 CSV 和源代码快照保留原样，不能因默认值变更把其重命名为 ring
结果。严格 ring、宽累加饱和、原逐项饱和 E0-Q-fixed 与 E0-B 均为
独立数值模型。
