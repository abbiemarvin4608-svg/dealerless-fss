# SiRNN 标准模型测试基准 —— 复现方案总结

**目的**：SiRNN (S&P 2021, "SIRNN: A Math Library for Secure RNN Inference") 是 OT-based
dealerless 安全 RNN 推理的 SOTA，是我们 RNN 精度论证和效率对比的主要参照。本文档从
`refer/SiRnn.pdf` 提取其**标准测试模型**，并明确我们需要复现哪些实验。

---

## 一、SiRNN 论文的完整 Benchmark 清单

SiRNN 的评估分四层（论文 Section VI）：

### Layer 1 — Microbenchmarks（数学函数，Table I & II）
单个数学函数的精度/通信/延迟，对比 MP-SPDZ / ABY / EMP：
- **Exponentiation**（指数，用于 RBF kernel）
- **Sigmoid**（LSTM/GRU 门控）
- **Tanh**（LSTM/GRU 门控）
- **Reciprocal Square Root**（L2Normalize）

指标：ULP error（units in the last place，数值精度）、每实例通信 KB、10²–10⁵ 实例总时间。
SiRNN 比 MP-SPDZ 好 7–423×，比 ABY/EMP 好 90×（通信）/ 97×（运行时）。

### Layer 2 — Prior-work DNNs（Table III）
复现前人 benchmark，证明协议通用性：
| Benchmark | 架构 | 数学函数负载 | 任务 |
|-----------|------|-------------|------|
| **MiniONN LSTM** | 2 层 LSTM | 每层 800 sigmoid + 200 tanh | 文本分类 |
| **DeepSecure B4** | 2 层 tanh | 2000 + 500 tanh 实例 | 传感器数据 → 19 类物理活动 |

### Layer 3 — Case studies（核心贡献，Table IV）★ 最重要
三个真实领域模型，前人无法处理：
| 模型 | 架构 | 数据集 | 任务 | 数学函数规模 |
|------|------|--------|------|-------------|
| **Industrial-72** | GRU RNN | 传感器时序 [72] | 板球击球质量反馈 | 7 sigmoid + 7 tanh 层，各 64 实例 |
| **Google-30** | RNN (FastGRNN) | Google Speech Commands [112] | 关键词识别（"Yes"/"No"/数字/方向） | 99 sigmoid + 99 tanh 层，各 100 实例 |
| **Heads** | CNN + RNN 混合 | SCUT Head [96] | 图像人头检测 | 128+128 层 ×18096 实例 + rsqrt；**~300 万次 sigmoid/tanh** |

关键实测结果（SiRNN，batch=1）：
- Industrial-72: 3.7s, 23.8 MB
- Google-30: 49.6s, 0.45 GB
- Heads: 409.7s (~7min), 85.5 GB

### Layer 4 — Ablation（Table IV）
同样的定点代码用 ABY (uniform 64-bit) 跑作 baseline → SiRNN 快 3 个数量级。

---

## 二、SiRNN 的定点精度设定（关键参照）

SiRNN 用**非均匀位宽**（8/16/32 bit 混用，16 bit 用于数学函数 I/O），这是它的核心创新。
各 benchmark 的 scale（小数位）配置：
- Industrial-72: sigmoid 输入 scale 8 / 输出 scale 14；tanh 两端 scale 8
- Google-30: sigmoid 输入 scale 6 / 输出 scale 14；tanh 两端 scale 6
- Heads: sigmoid 输入 scale 11/13 / 输出 scale 14；tanh scale 11/13；rsqrt scale 9–12

**对我们的意义**：SiRNN 声称"无精度损失"（matches/exceeds fixed-point model accuracy）。
我们的 dealerless FSS 方案是**均匀 32-bit**，需要证明在同样模型上也能达到无损精度，
且效率有竞争力。

---

## 三、当前项目状态 vs SiRNN benchmark（缺口分析）

| SiRNN 要素 | 我们现状 | 缺口 |
|-----------|---------|------|
| CNN 真实数据集精度 | ✅ ImageNette + CIFAR-10 实测（E0-EXT） | 无 |
| RNN 真实数据集精度 | ⚠️ **仅 synthetic 随机序列**（`test_deep_lstm.py` 用 `torch.randint`） | **缺真实 benchmark** |
| SiRNN 协议效率对比 | 📋 已规划（E5，`libSCI-Math.a` 已编译） | 待运行 |
| 数学函数 microbenchmark | ❌ 未做 | sigmoid/tanh/exp/rsqrt 的 FSS 实现精度对比 |
| Google-30 / Industrial / Heads | ❌ 未做 | 标准 RNN benchmark 模型 |

**核心缺口**：我们的 RNN 精度结论（"LSTM 在 ℓ=20 保持 90-92%"）建立在**合成数据**上，
存在和 CNN 完全相同的科学性隐患——正如 CIFAR-10 低估了 ImageNet 的精度需求，
synthetic 序列可能也低估了真实语音/时序任务的精度需求。

---

## 四、建议复现的实验方案（按优先级）

### 【P0 - 必做】RNN 真实数据集精度测试（补 E0 的 RNN 短板）
**动机**：把 CNN 的 ImageNette 严谨性对齐到 RNN 侧。

**方案 A（推荐，可行性最高）— Google Speech Commands (Google-30)**
- 数据集：`torchaudio.datasets.SPEECHCOMMANDS`（公开，~2.3GB，35 类命令词）
- 模型：FastGRNN 或标准 LSTM/GRU（SiRNN 用 FastGRNN [74]）
- 测试：ℓ ∈ {16, 20, 24, 28, 32}，MPC 截断语义，实测 test-set 准确率
- 产出：验证 RNN 真实语音任务的精度悬崖，回答"synthetic 是否低估精度需求"
- 预计工作量：训练 + 量化评估，1–2 天

**方案 B（更快）— 序列 MNIST / IMDB**
- Sequential-MNIST（把 28×28 图按行当序列）或 IMDB 情感分类
- torchvision/torchtext 内置，无需额外下载
- 比 synthetic 真实，但不如语音贴合 SiRNN 场景

### 【P1 - 重要】数学函数精度 microbenchmark（对齐 SiRNN Table I）
**动机**：SiRNN 的核心是 sigmoid/tanh/exp/rsqrt 的高精度实现。我们用 FSS/Grotto
做同样的函数，需要证明精度（ULP error）和通信有竞争力。
- 实现：用我们的 dealerless FSS 评估 sigmoid/tanh/exp/rsqrt
- 对比：ULP error vs SiRNN 报告的值（sigmoid/tanh 精确，exp ULP 3，rsqrt ULP 4）
- 指标：ULP error、每实例通信、10²–10⁵ 实例延迟
- 产出：Table I 风格对比表

### 【P2 - 效率对比】SiRNN 协议 head-to-head（已在 EXPERIMENT_PLAN E5 规划）
**动机**：dealerless 场景下和 SOTA 直接比时间/通信。
- Build：`cd /home/jiang/EzPC/SCI/build && cmake -DBUILD_NETWORKS=ON .. && cmake --build . --target install`
- 目标网络：`relevance32-OT`、`hinet4`、`ffnn128`
- 对比：SiRNN OT-based vs 我们的 FSS-based，相同 RNN workload、32-bit
- 已就绪：`libSCI-Math.a` 已编译（见 EXPERIMENT_PLAN.md:16）

### 【P3 - 受限】完整 case study 复现（Industrial / Heads）

**2026-09-03 实际核查结果**（已下载 EdgeML 源码 `/home/jiang/EdgeML` 逐一验证）：

| 模型 | 架构代码 | 预训练权重 | 数据集 | 可复现性 |
|------|---------|-----------|--------|---------|
| **Google-30 (FastGRNN)** | ✅ EdgeML `graph/rnn.py` 官方 `FastGRNNCell` | ❌ 无 | ✅ Speech Commands v0.02 已下载 | **可做**（自训练） |
| **Industrial-72** | ❌ Shiftry 仓库不存在（`git ls-remote` 失败） | ❌ 无 | ❌ 私有工业传感器数据，从未公开 | **不可复现** |
| **Heads (RNNPool)** | ✅ EdgeML `Face_Detection/` 完整 | ❌ 无（README 要求自训 300 epochs） | ⚠️ WIDER FACE ~3GB；SiRNN 用的 SCUT Head 是另一数据集 | 需 300 epochs 训练 |

**关键事实**：EdgeML 全仓库 `find -name "*.pt" -o -name "*.pth"` 返回空 —— 微软未发布任何预训练权重。

**Industrial-72 不可复现的原因**（论文需如实说明）：
该模型使用私有的板球运动传感器数据 [72]，数据集从未公开；生成其定点代码的 Shiftry
编译器仓库亦不可访问。SiRNN 作者能够评估该模型，是因为 Shiftry 与 SiRNN 出自同一研究组。

**Heads 的限制**：本机为 CPU-only PyTorch（`torch.cuda.is_available() == False`），
而 RPool_Face 官方训练配置为多 GPU、300 epochs，CPU 上训练需数天至数周。已与用户确认
**暂不复现 Heads**。

**重要澄清 —— SiRNN 论文本身没有做位宽扫描实验**：
SiRNN 的三个 case study 报告的是**协议效率**（延迟、通信量）与数学函数的 ULP 误差，
**不包含** ℓ∈{16,20,24,28} 的准确率曲线。其位宽由 Shiftry 编译器为嵌入式部署固定给出
（8/16/32 非均匀位宽），论文贡献是"如何在 MPC 下高效计算这些非均匀位宽的数学函数"，
而非"需要多少位"。因此 Industrial-72 / Heads 不存在可照搬的"定点位数测试"。
本项目的位宽扫描是对 SiRNN 的**补充**，而非复现。

---

## 五、和论文叙事的衔接

E0-EXT 已证明 **CNN 需要 ℓ≥24（ImageNette 实测）**。补上 RNN 真实数据后，论文可形成完整论证：

> "无论 CNN（ImageNette，ℓ≥24）还是 RNN（Google-30 语音，ℓ≥?），真实任务都需要高精度。
> 单一协议须无损服务所有模型 → ℓ=32 是安全设计点。在此精度下 monolithic FSS 生成不可行，
> segment decomposition 成为必需。SiRNN 是 dealerless RNN 的 SOTA baseline，
> 我们在相同 workload 上达到无损精度且效率有竞争力。"

**关键**：P0（RNN 真实数据）直接补齐当前最大的科学性短板——现有 RNN 结论建立在
synthetic 数据上，和被推翻的"CIFAR-10 足够"是同一类风险。

---

## 参考

- SiRNN 论文：`refer/SiRnn.pdf`（S&P 2021, eprint 2021/459）
- SiRNN 实现：https://github.com/mpc-msri/EzPC（已编译进 `libSCI-Math.a`）
- FastGRNN [74]：Kusupati et al., NeurIPS 2018
- Google Speech Commands [112]：Warden, arXiv 2018
- RNNPool/Heads [104]：Saha et al., NeurIPS 2020
- Shiftry 编译器 [72]：Kumar et al., OOPSLA 2020（生成定点 C++）
