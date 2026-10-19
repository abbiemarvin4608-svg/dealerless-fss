# CNN 运行时迁移与资格验证

构建入口为 `project/our/tools/imagenet_resnet/build_secure.py`，合格二进制为本目录 `build/resnet_secure`，桥接库为 `build/libcnn_ours.so`。使用既有 `online_optimization_20261014/build/core` 中重新构建的协议库。CNN 数值配置保持统一 ring32/f12。

## 已迁移的适用优化

| 部分 | 本次实现 |
| --- | --- |
| CNN MSB | 随机掩码与 B2A 后调用当前 `segmentedDcfOfflineBatch` / `segmentedDcfOnlineBatchGroup`；使用 depth-cap-8、无需顶层 equality 的扫描、紧凑 key 序列化和复用在线 workspace。 |
| cGGM / DCF | 继承现有 AES 离线/在线、叶哈希与 equality 剪枝、leaf CW 合并、direct COT 和生成 workspace 复用。 |
| 打包与转换 | 继承位/环打包、连续材料布局、转换内部域融合及 sign/wrap 融合；调用边界仍按公开 CNN 图调度。 |
| Beaver 线性层 | 保留已有 mapped/packed 矩阵材料与代数，增加在线 MatmulWorkspace 复用；一次性材料存入私有临时文件。 |
| SCI 真实预处理 | 复用 RNN 原生 OT/bit-triple 计划与生成逻辑，以公开零数据 dry-run 获得完整调用计划，双方核对后生成材料；随机 pads/triples 分记录写入私有临时文件并在在线调用时恢复。 |
| 普通 GOT | 用现有 grouped prepared GOT 的 group=1 接口承接普通 IKNP GOT，避免整图单次巨量预分配；保留普通调用的公开签名和消费计数。 |
| 生命周期 | 每个查询重新生成材料；严格库存拒绝在线生成、库存耗尽与剩余库存。 |

CNN 没有 Sigmoid、TanH 或 LUT 节点，相关专用优化不适用。RNN 的时间步/图并行调度不直接移入顺序 CNN 图。`baseline` profile 关闭对应优化开关，但不是旧二进制的逐指令复现，且本轮未新增 baseline 性能实验。

## AvgPool 的比较口径

旧 SCI AvgPool helper 使用未纳入原生库存钩子的 OT 路径。三个主后端统一使用 driver 已有的精确公开除法分解，分别调用对应后端的 MSB、AND 和 selection，JSON 标识为 `shared_exact_public_division_MSB_AND_selection`。

数值语义保持不变：49 项首先在 32 位环中求和，再对有符号代表元执行 floor(sum/49)。因此 `SCI` 行表示使用原生 SCI 原语的共同 CNN 图，不能把该 AvgPool 组合声称为未经更改的旧 SCI helper。

## 资格验证

输入为既有 `contract_fixtures/w32` 独立 Python-bigint oracle；图含 Conv、ReLU、MaxPool、残差加法、AvgPool/49 和 FC，包含有符号负数与环回绕。每个后端执行两次新查询，逐层重建分享并核对。

| 结果目录 | 查询数 | 核对整数码 | 差异 | 在线原生生成 epoch 增量/方/查询 |
| --- | ---: | ---: | ---: | ---: |
| `qualification/SCI` | 2 | 3,938 | 0 | 70 |
| `qualification/SCI_preprocessed` | 2 | 3,938 | 0 | 0 |
| `qualification/SCI_ours_beaver_retry` | 2 | 3,938 | 0 | 0 |
| `qualification/SCI_preprocessed_capacity`（容量释放修复后） | 2 | 3,938 | 0 | 0 |

每个查询公开计划包含 20 个 MSB/AND/selection 调用、2 个截断调用与5个矩阵调用。`SCI_preprocessed` 每方每查询写入和消费均为57条原生库存；party1材料为13,240,303字节，party2为5,814,721字节。两次查询分别生成和消费，不复用一次性材料。

Ours 每方每查询的非线性临时文件为928,264字节，conversion返回的材料计数为738,249字节，Beaver临时文件为52,416字节。conversion字段由写入前后的 `ftell` 差值累计得到，是序列化材料字节数，不包含文件系统分配开销。

容量修复将原生COT在线消费后的 `clear()` 改为 `cnn_spool::release`，释放恢复payload所占的vector容量。当前 `build/resnet_secure` 已完成重建；`qualification/SCI_preprocessed_capacity` 的两次新查询通过逐层整数审计，每方每次查询的 `native_spool_records` 和 `native_spool_consumed` 均为57，`native_generation_epoch_delta` 均为0。该修改仅涉及原生COT payload释放，因此本次仅复验 `SCI_preprocessed`。

原三后端资格记录的 `PLAN.json` 保留各自运行时的二进制身份，不能解释为对当前重建二进制的三后端重测。新增容量修复记录对应当前构建，后续正式运行使用该构建。

`test_native_spool` 验证了交错owner的恢复顺序、精确payload、保存后释放vector容量、单次消费及预算触发；容量修复后的独立测试还断言消费后vector capacity为0，结果为PASS。`rebuild_capacity_guard.json`、`native_spool_capacity_compile_guard.json`、`native_spool_capacity_run_guard.json` 和 `qualification_SCI_preprocessed_capacity_guard.json` 均为 `complete`，退出码为0。所有构建和协议检查均在既有2 GiB RAM加512 MiB swap守护内完成。上述检查属于诊断资格验证，不能作为完整ResNet精度或性能结果。

## 计量与保留记录

driver报告 `pre_setup.load_seconds`、`pre_setup.plan_seconds`；原生SCI公开计划的生成与双方核对计入offline，并单列 `native_public_plan_seconds`。材料分别通过 `native_spool_bytes`、`ours_nonlinear_spool_bytes`、`conversion_material_bytes`、`beaver_spool_bytes` 报告。runner保留外层 `protocol_process_wall_seconds` 与 `launcher_wall_seconds`；阶段双方墙钟最大值之和仍是另一个口径。

首次build因header重复定义失败；重试因CNN桥接缺少 `mpc/secure_ops.h` 中B2A声明失败，均已修复。Ours首次资格运行因segment policy使用下划线拼写而失败，已改为库接受的 `depth-cap-8` / `fixed-4`，成功记录使用 `_retry` 目录。此前失败守护和日志均保留。

未启动完整ResNet-50/152运行，也未启动CNN batch128。完整网络的库存规模、磁盘需求、资源限制和端到端性能仍由正式运行测量。
