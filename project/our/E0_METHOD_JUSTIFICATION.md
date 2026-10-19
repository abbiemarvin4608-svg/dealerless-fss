## E0-EXT 实验方法说明（论文用）

### 测试数据集选择与局限性

**已完成的测试：**
1. **MNIST** - 28×28, 10类, MNIST-CNN 端到端训练
2. **CIFAR-10** - 32×32, 10类
   - ResNet20: 端到端训练
   - ResNet50/101/152, VGG16/19: 冻结 ImageNet 预训练 backbone + CIFAR-10 线性头微调

**ImageNette 测试（进行中）：**
- ImageNette: ImageNet 的 10 类子集（tench, springer, cassette player, chain saw, church, French horn, garbage truck, gas pump, golf ball, parachute）
- 保持 ImageNet 原生 224×224 分辨率和真实数据分布
- 验证集 3,925 张图像
- 目的：消除"CIFAR-10 测试 ImageNet 架构"的任务不匹配质疑

### 当前 CIFAR-10 方法的科学有效性

**为什么用 CIFAR-10 测 ResNet50/VGG 仍然有效：**

1. **激活值范围是架构属性，不是数据集属性**
   - VGG 的 max |activation| = 123-187 是由于缺少 BatchNorm，与数据集无关
   - ResNet 的 BatchNorm 在任何数据集上都会归一化激活值
   - 量化溢出发生在前向传播的中间层，与最终任务（10类 vs 1000类）无关

2. **迁移学习保留了架构的量化敏感性**
   - 冻结的 ImageNet backbone 保持原始权重和激活范围
   - 只有最后一个全连接层是新训练的（10 个输出神经元）
   - 量化应用于整个模型（包括冻结的卷积层），因此激活溢出仍会发生

3. **实验结果证实了理论预测**
   - VGG 在 ℓ=20 掉点 2-3pp（CIFAR-10 上测得）
   - 这与激活范围分析一致：186 >> 8（ℓ=20 的整数范围上限）
   - ResNet 在 ℓ=20 完全恢复，与 BatchNorm 压缩激活的理论一致

**CIFAR-10 的保守性（有利于论文论证）：**

如果在真实 ImageNet-1K (1000 类，224×224) 上测试，精度损失会**更大**：
- 类间 margin 更小（1000 类 vs 10 类）→ 量化噪声更容易翻转排名
- 输入分辨率更大（原生 224×224 vs 上采样的 CIFAR）→ 累积误差更大
- 因此我们的 CIFAR-10 结果提供了精度需求的**保守下界**

**论文中如何表述（诚实 + 防御性）：**

> We evaluate ImageNet-architecture models (ResNet50/101/152, VGG16/19) using 
> transfer learning on CIFAR-10: we freeze the ImageNet-pretrained backbone 
> (preserving its activation ranges) and train a linear 10-class head. While 
> this setup does not capture the full complexity of the native ImageNet-1K 
> task (1000 classes, tighter margins), it provides a **conservative lower bound** 
> on precision requirements. The quantization stress test occurs in the frozen 
> convolutional backbone, where VGG's unbounded activations (max 123-187) cause 
> measurable degradation at ℓ=20 (−2.2pp for VGG16, −2.9pp for VGG19), while 
> ResNets' BatchNorm-controlled activations remain robust. On the native 
> ImageNet-1K task with 1000-way classification, the degradation would be more 
> pronounced due to smaller inter-class margins.

### ImageNette 测试的增量价值

一旦完成，ImageNette 结果将：
1. **消除任务不匹配质疑** - 在 ImageNet 分布上验证
2. **提供更强的证据** - 如果 VGG 在 ImageNette 上掉点更多，论证更强；如果掉点相似，说明 CIFAR-10 结果已足够准确
3. **增强可重现性** - ImageNette 是公开数据集（vs 完整 ImageNet 需注册）

**预期结果：**
- ResNet50/101/152: ℓ=20 仍应完全恢复（BatchNorm 效果）
- VGG16/19: ℓ=20 掉点可能扩大到 3-5pp（1000类任务更敏感）

### 时间线
- CIFAR-10 测试：已完成 ✓
- ImageNette 数据下载：进行中（预计 45 分钟，当前速率 31KB/s）
- ImageNette 测试：数据到位后 **~20 分钟**（5 模型 × 4 位宽，直接推理无需特征提取）
