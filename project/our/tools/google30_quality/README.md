# Google-30 独立质量产物

本工具冻结模型、校准清单和选择规则，再生成训练侧 SeeDot 混合位宽图，最后评价官方 `valid` 的全部 6,798 个样本。默认产物与已完成结论见 [工单 02 报告](../../results/ticket02_independent_quality/REPORT_CN.md)。

`valid` 未参与已审计的模型/量化选择，但历史上曾被单独评价。因此这是 **selection-independent holdout**，不称为首次观察的盲测。历史官方 `test` 曾用于 checkpoint 选择，不能作为新的独立质量评估集。

## 重建一个新版本

在仓库根目录执行，输出目录必须不存在：

```bash
project/our/tools/google30_quality/run.sh \
  /home/jiang/EzPC/SIRNN/reproduction \
  project/our/results/ticket02_independent_quality_replay
```

需要本地已完成官方训练与数据预处理的 source bundle，包含 `EdgeML-sirnn`、`reproduction/data`、`reproduction/models`，以及已安装依赖的 `seedot-venv`（Python 3.7）和 `venv`（TensorFlow 1.15）。这两个环境由不同 SeeDot/TF 依赖要求保留。外部 source bundle 只读；所有修改在新输出目录的复制品中进行。

流程：

1. `prepare.py` 审计三份样本清单的样本和说话者交集，按来源固定 checkpoint，保存模型/预处理/清单与 `freeze_policy.json`；只打开 `train.npy`。校准为种子 42、每类首个样本加随机补齐至 512 个训练样本；训练侧副本同时填充 SeeDot 的 `train.npy`、`test.npy` 两个接口。
2. `quantize.py` 执行一次原生 DDS/VBW 搜索，保留原生 `red_disagree` 排序和 8/16/32 位候选，校准损失预算预先固定为 1 pp；保存全部候选日志、最终位宽/尺度和生成的 C++/EzPC 程序。不会向量化器提供 heldout。
3. 量化冻结后，`evaluate_fp32.py` 用官方 TensorFlow FastGRNNCell 评价 FP32；`evaluate_fixed.py` 运行实际生成的 C++ 图，只增加最终 ArgMax 前 logits 捕获。先与未插桩 SeeDot Predictor 的 512 条校准预测逐条比对，再评价 heldout。固定点输入采用 SeeDot 的六位小数 CSV 和向零截断编码，FP32 采用原生 normalized float32。
4. `export_contract.py` 从新图导出 18 个调用点的命名参数、位宽、尺度、累加/截断语义；`finalize.py` 保存逐样本预测、数值误差及统一 JSON。

本流程不根据最终结果重试选点。未通过门槛的结果仍保存。`run.sh` 不覆盖已有冻结版本；阶段入口可用于恢复被中断的评价，不能用来更改被冻结模型和量化配置。

## 校验

测试 seam 为 CLI 的数据划分拒绝行为、冻结阶段对最终特征的隔离、公开质量报告；没有模拟安全 executor。

```bash
GOOGLE30_REPRODUCTION=/home/jiang/EzPC/SIRNN/reproduction \
OPENBLAS_NUM_THREADS=1 \
/home/jiang/EzPC/SIRNN/reproduction/seedot-venv/bin/python \
  -m unittest discover -s project/our/tools/google30_quality -v
```

未设置 `GOOGLE30_REPRODUCTION` 时仅跳过需要真实 source bundle 的冻结集成测试。该测试提供故意损坏的官方 test/valid 特征文件，确认准备/选点输入仍能建立；另验证评估说话者出现在模型选择集时拒绝运行。

这是明文质量链。SeeDot 的 `FLOATEXP` 激活与 SCI 安全实现不等价；后续真实 SCI executor 必须对这个新版本重新做整数与预测验证，不能把历史 30 样本一致性移植过来。
