# Phase 2: Sigmoid/Tanh 批量化优化实施计划

## 目标
将多个 Sigmoid/Tanh 操作批量化，减少通信轮次从 O(N) 到 O(1)。

## 当前状态
- ✅ Phase 1 完成：单个 Sigmoid 从 64 轮优化到 ~6 轮
- ✅ 11/12 测试通过
- ❌ 已知问题：sigmoid(-1) 边界情况（容差问题）

## 批量化优化策略

### 1. 批量接口设计

```cpp
// Batch offline: 生成 batch_size 个 Sigmoid 密钥
std::vector<SigmaActivationKey> sigmaBatchOffline(
    int party_id, int batch_size, int bits, int frac_bits, int q, Peer* player);

// Batch online: 批量计算 Sigmoid
std::vector<GroupElement> sigmaBatchOnline(
    int party_id, 
    const std::vector<GroupElement>& inputs,
    std::vector<SigmaActivationKey>& keys, 
    Peer* player);
```

### 2. 批量化关键点

#### A. 批量打开输入 (Batch Reconstruct)
**当前**: 每个输入单独打开 → N 轮通信
```cpp
for (int i = 0; i < N; i++) {
    reconstruct(&x[i]);  // 1 round each
}
```

**优化**: 一次打开所有输入 → 1 轮通信
```cpp
reconstruct(N, x_array, bits);  // 1 round total
```

#### B. 批量 DPF 评估
**当前**: 串行评估每个输入的 DPF
```cpp
for (int i = 0; i < N; i++) {
    evalAllSegments(party_id, keys[i].segment_keys, xhat_segments[i]);
}
```

**优化**: 并行评估所有 DPF（本地操作，无需优化）

#### C. 批量 inferComparison
**当前**: 每个比较需要 ~3 轮（AND 操作）
```cpp
for (int i = 0; i < N; i++) {
    inferComparisonPreprocessed(...);  // ~3 rounds each
}
```

**优化**: 批量 AND 操作 → 合并为 1 轮
- 收集所有需要的 AND 对
- 一次性执行所有 AND 操作
- 分发结果

#### D. 批量 BitXA
**当前**: 每个 bitxa 单独通信
```cpp
for (int i = 0; i < N; i++) {
    bitxaOnline(party_id, input[i], bit[i], key[i], player);
}
```

**优化**: 检查 bitxaOnline 是否支持批量模式

### 3. 预期轮次分析

| 阶段 | 当前 (N 个 Sigmoid) | 批量化 | 改进 |
|------|---------------------|--------|------|
| 打开输入 | N 轮 | 1 轮 | N× |
| 符号判断 | N × 3 轮 | 3 轮 | N× |
| 裁剪判断 | N × 3 轮 | 3 轮 | N× |
| BitXA (3×) | N × 3 轮 | 3 轮 (if batched) | N× |
| **总计** | **~10N 轮** | **~10 轮** | **N× 加速** |

对于 N=100 (Google-30 RNN):
- 当前: ~1000 轮
- 批量: ~10 轮
- **加速: 100×**

### 4. 实施步骤

#### Step 1: 实现批量打开 (最高优先级)
```cpp
// 使用现有的 reconstruct(size, arr, bw) API
void batchReconstruct(std::vector<GroupElement>& inputs) {
    reconstruct(inputs.size(), inputs.data(), inputs[0].bitsize);
}
```

#### Step 2: 批量 inferComparison
```cpp
std::vector<u8> batchInferComparison(
    int party_id,
    const std::vector<std::vector<SegmentEvalResult>>& all_segment_results,
    std::vector<BooleanAndPreprocessing>& preprocessings,
    Peer* player);
```

关键优化：
- 收集所有 AND 操作对
- 调用批量 AND: `booleanAndOnline(party_id, all_lhs, all_rhs, all_prep, player)`
- 分发结果到各个比较

#### Step 3: 实现 sigmaBatchOffline
```cpp
std::vector<SigmaActivationKey> sigmaBatchOffline(
    int party_id, int batch_size, int bits, int frac_bits, int q, Peer* player) {
    
    std::vector<SigmaActivationKey> keys(batch_size);
    
    for (int i = 0; i < batch_size; i++) {
        keys[i] = sigmaOffline(party_id, bits, frac_bits, q, player);
    }
    
    return keys;
}
```

#### Step 4: 实现 sigmaBatchOnline
```cpp
std::vector<GroupElement> sigmaBatchOnline(
    int party_id,
    const std::vector<GroupElement>& inputs,
    std::vector<SigmaActivationKey>& keys,
    Peer* player) {
    
    const int batch_size = inputs.size();
    
    // 1. 批量打开所有输入 (1 轮)
    std::vector<GroupElement> inputs_opened = inputs;
    reconstruct(batch_size, inputs_opened.data(), inputs[0].bitsize);
    
    // 2. 批量符号判断 (3 轮)
    std::vector<u8> is_negative_vec = batchPiecewiseCompare(
        party_id, inputs_opened, {keys[i].sign_cmp for all i}, player);
    
    // 3. 批量计算 magnitude
    std::vector<GroupElement> magnitudes(batch_size);
    for (int i = 0; i < batch_size; i++) {
        u8 nonnegative = is_negative_vec[i] ^ public_one;
        GroupElement relu = bitxaOnline(party_id, inputs[i], nonnegative, keys[i].relu_mul, player);
        magnitudes[i] = relu * 2 - inputs[i];
    }
    
    // 4. 批量裁剪判断 (3 轮)
    std::vector<u8> below_bound_vec = batchPiecewiseCompare(
        party_id, magnitudes, {keys[i].clip_cmp for all i}, player);
    
    // 5. 后续计算...
    std::vector<GroupElement> results(batch_size);
    // ... 实现剩余逻辑
    
    return results;
}
```

### 5. 测试策略

#### 单元测试
```cpp
void test_batch_reconstruct() {
    // 验证批量打开 == 多次单独打开
}

void test_batch_infer_comparison() {
    // 验证批量比较 == 多次单独比较
}
```

#### 正确性测试
```cpp
void test_sigmaBatch_correctness() {
    // 对于相同输入，批量结果 == 单个结果
    for (int i = 0; i < batch_size; i++) {
        assert(batch_results[i] == single_results[i]);
    }
}
```

#### 性能测试
```cpp
void benchmark_sigmoid_batch() {
    // 测量轮次和时间
    // batch_size = {1, 10, 50, 100}
}
```

### 6. 风险与缓解

| 风险 | 影响 | 缓解措施 |
|------|------|----------|
| 批量 API 不存在 | 高 | 检查现有 API，可能需要自己实现 |
| 批量破坏正确性 | 高 | 详细的单元测试 + 对比测试 |
| 性能提升不如预期 | 中 | 增量实施，先优化关键路径 |
| 内存开销增大 | 低 | 合理的 batch_size 上限 |

### 7. 成功标准

✅ 批量接口实现完成  
✅ 所有单元测试通过  
✅ 正确性测试：批量 == 单个 (逐元素)  
✅ 轮次减少：O(N) → O(1)  
✅ 端到端性能提升验证  

---

## 实施时间线

**Day 1**: 实现批量打开 + 批量 inferComparison  
**Day 2**: 实现 sigmaBatchOffline/Online  
**Day 3**: 测试 + 性能验证  
**Day 4**: 文档 + 总结

---

## 附录：相关 API

### 现有批量 API
```cpp
// from mpc/api.h
void reconstruct(int32_t size, GroupElement* arr, int bw);
void reconstruct(int32_t size, u8* arr);
```

### 需要检查的 API
- `booleanAndOnline`: 是否支持批量？
- `bitxaOnline`: 是否支持批量？
- `evalAllSegments`: 已经是批量（多个分段）
