# 原语独立构建

本目录与 CNN 构建目录互不共享输出。CMake 单线程构建及 sweep 构建均已完成，三个目标产物已用于正式 input28 补测。构建身份记录在 `missing_width_runs/plan.json`。

已执行的构建命令及对应保护日志：

```bash
python3 project/our/tools/imagenet_resnet/memory_guard.py run --output /tmp/primitive28-build-core-guard.json -- cmake --build project/our/results/precision_motivation_final/primitive_ablation/build --target segfss_sci_bridge -j 1
python3 project/our/tools/imagenet_resnet/memory_guard.py run --output /tmp/primitive28-build-sweep-guard.json -- python3 project/our/tools/sci_sweep/build.py --build project/our/results/precision_motivation_final/primitive_ablation/build
```

第一条构建 `libsegfss_sci_bridge.so`，第二条生成 `SCI_MSB_SWEEP` 与 `connect_offset.so`。若保护日志已存在，应选择新的日志路径，保留之前的资源记录。原构建目录的内容不会被清空。

初次配置命令为：

```bash
cmake -S project/our -B project/our/results/precision_motivation_final/primitive_ablation/build -DDEALERLESS_FSS_ROOT=/home/jiang/dealerless-lunwen/project/dealerless-FSS_public -DEZPC_ROOT=/home/jiang/EzPC -DCMAKE_BUILD_TYPE=Release
```

这两次构建均成功退出。新构建身份与旧 batch100 分开记录，不声称二进制相同。正式补测在一次主机内存保护中断后已完成窄范围恢复，结果和保留的资源失败记录见 `README.zh.md`。
