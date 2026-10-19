# 11: 完成 Google-30 双后端性能对照

**What to build:** 在同一 Google-30 原生混合位宽图上完成 SCI/SiRNN OT 与 `SCI + ours` 两个后端的端到端安全对照，使用独立质量产物和 #10 的真实 sweep 结果形成第一版可审计性能表。

**Blocked by:** #04 打通 SCI 真实推理的完整计量；#07 完成 SCI + ours 整网一致性；#10 测量比较位宽与分段数成本曲线

**Status:** completed

**Report:** [ticket 11 report](../../../project/our/results/ticket11_comparison/REPORT_CN.md)

- [x] 单独报告相对约定参考的数值误差，注明比较张量、尺度、单位和统计方式；数值误差与 Top-1、agreement 分开，不以分类指标替代。
- [x] 同图轨固定量化权重、输入顺序、尺度、累加、截断、舍入和激活近似，仅替换安全比较/MSB；系统轨固定 FP32 checkpoint、数据和可比校准规则，允许各后端选择自身达标配置，两轨分别出表。
- [x] 同图配置中 SCI + ours 的逐算子整数输出和完整定点预测与 SCI 一致；系统轨不同数值配置分别与其冻结参考核验。比较重构值，不要求随机秘密份额字面相同。
- [x] 质量评估使用 #02 的独立评估产物，同时报告 FP32 Top-1、loss_pp、prediction agreement 和未通过 1.0 pp 门槛的配置。
- [x] 性能只包含 SCI/SiRNN OT 与 `SCI + ours` 两个后端；MP-SPDZ、ABY-64 和 dealer-assisted FSS 不得提前混入本双后端结果或主排名。
- [x] SCI 保留数值配置，SCI + ours 的分段选择有 #10 的测量依据，不通过原语 sweep 改动同图数值语义；报告 offline、online、total 延迟、通信、依赖轮数和峰值内存。
- [x] 计量明确包含密钥生成、输入转换、扩环、截断、位宽转换和输出重构，并分别提供首查询与跨查询摊销成本。
- [x] 在相同硬件、线程、batch、LAN/WAN 条件和重复次数下运行，保存双方原始日志、版本、配置和资源失败信息。
- [x] 输出一份独立质量与性能结果产物，明确该阶段只有两个后端，禁止据此声称已完成三后端主表或普遍优于所有 baseline。
