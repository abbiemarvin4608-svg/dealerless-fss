# Sigmoid/Tanh 离线分段 + DPF 批量生成优化实施计划

## 目标

1. **将 Sigmoid/Tanh 改为离线分段模式**（像 ReLU 一样）
2. **DPF/DCF 批量生成**
3. **OpenMP 并行化**

## 当前架构分析

### ReLU 的工作模式（参考实现）

**Offline**:
```cpp
// src/relu.cpp:316-322
ReluPreprocessing reluOffline(int party_id, int bitwidth, Peer* player) {
    return bitxaOffline(party_id, bitwidth, player);  // 只生成 bitxa 乘法密钥
}
```

**Online**:
```cpp
// 需要外部提供 drelu 比较位
GroupElement reluOnline(int party_id, GroupElement x, u8 drelu,
                        ReluPreprocessing& preprocessing, Peer* player)
```

**DReLU 比较位的生成**：通过分段比较
- Offline: `split()` → `genSegmentKeys()` 生成 DPF 密钥
- Online: `evalAllSegments()` → `inferComparison()` 得到比较位

### Sigmoid 当前模式（需要修改）

**Offline**:
```cpp
// src/activation.cpp:91-116
SigmaActivationKey sigmaOffline(...) {
    key.relu_mul = bitxaOffline(party_id, bits, player);
    key.clip_mul = bitxaOffline(party_id, bits, player);
    key.branch_mul = bitxaOffline(party_id, bits, player);
    return key;
}
```

**Online**:
```cpp
// src/activation.cpp:119-175
GroupElement sigmaOnline(...) {
    const std::vector<u8> input_bits = BitDec(party_id, input, key.bits, player);  // ❌ Online BitDec
    const u8 nonnegative = input_bits.front() ^ public_one;
    
    const std::vector<u8> bound_bits = BitDec(party_id, below_bound_input, key.bits, player);  // ❌ Online BitDec
    const u8 below_bound = bound_bits.front();
    ...
}
```

## 实施方案

### 修改 1: Sigmoid 改为离线分段

#### 1.1 修改 SigmaActivationKey 结构

```cpp
// src/activation.h
struct SigmaActivationKey {
    int bits = 0;
    int frac_bits = 0;
    int q = 10;
    int trunc_bits = 0;
    int nseg = 4;  // 分段数
    
    PublicLUTData delta_table;
    PublicLutKeyPack lut_key;
    ExactTruncationKey trunc_key;
    
    // 改为分段比较密钥（离线生成）
    SegmentKeys sign_cmp_keys;      // 符号判断: x < 0
    SegmentKeys clip_cmp_keys;      // 裁剪判断: |x| < bound
    BooleanAndPreprocessing sign_and_prep;
    BooleanAndPreprocessing clip_and_prep;
    
    // bitxa 乘法密钥（保留）
    BitXAPreprocessing relu_mul;
    BitXAPreprocessing clip_select_mul;
    BitXAPreprocessing branch_select_mul;
};
```

#### 1.2 修改 sigmaOffline

```cpp
// src/activation.cpp
SigmaActivationKey sigmaOffline(int party_id, int bits, int frac_bits,
                                int q, Peer* player) {
    validateParameters(bits, frac_bits, q);
    
    SigmaActivationKey key;
    key.bits = bits;
    key.frac_bits = frac_bits;
    key.q = q;
    key.nseg = 4;  // 默认 4 段
    key.trunc_bits = frac_bits - (q - kBoundLog2);
    key.delta_table = makeSigmaDeltaTable(bits, frac_bits, q);
    key.lut_key = dfss::publicLutOffline(party_id, key.delta_table);
    
    if (key.trunc_bits > 0) {
        key.trunc_key = exactTruncateOffline(party_id, bits, key.trunc_bits, player);
    }
    
    // 离线生成分段比较密钥
    SegmentLayout layout(bits, key.nseg);
    
    // 1. 符号判断: x < 0
    GroupElement threshold_sign(0, bits);
    SplitResult split_sign = split(party_id, threshold_sign, layout, player);
    key.sign_cmp_keys = genSegmentKeys(party_id, split_sign);
    
    // 2. 裁剪判断: magnitude < bound
    const uint64_t bound = (16ULL << frac_bits);
    GroupElement threshold_clip(bound, bits);
    SplitResult split_clip = split(party_id, threshold_clip, layout, player);
    key.clip_cmp_keys = genSegmentKeys(party_id, split_clip);
    
    // 3. 预生成 AND 三元组
    const int num_ands = 2 * (key.nseg - 1);
    key.sign_and_prep = booleanAndOffline(party_id, num_ands, player);
    key.clip_and_prep = booleanAndOffline(party_id, num_ands, player);
    
    // 4. bitxa 乘法密钥
    key.relu_mul = bitxaOffline(party_id, bits, player);
    key.clip_select_mul = bitxaOffline(party_id, bits, player);
    key.branch_select_mul = bitxaOffline(party_id, bits, player);
    
    return key;
}
```

#### 1.3 修改 sigmaOnline

```cpp
// src/activation.cpp
GroupElement sigmaOnline(int party_id, GroupElement input,
                         SigmaActivationKey& key, Peer* player) {
    
    // 1. 符号判断: x < 0 (使用离线生成的分段密钥)
    SegmentLayout layout(key.bits, key.nseg);
    std::vector<std::vector<u8>> input_segments(key.nseg);
    for (int j = 0; j < key.nseg; ++j) {
        const int w = layout.width[j];
        const uint64_t seg_val = (input.value >> layout.shift[j]) & ((1ULL << w) - 1);
        input_segments[j].resize(w);
        for (int i = 0; i < w; ++i) {
            input_segments[j][i] = (seg_val >> i) & 1;
        }
    }
    
    std::vector<SegmentEvalResult> sign_results = 
        evalAllSegments(party_id, key.sign_cmp_keys, input_segments);
    const u8 is_negative = 
        inferComparisonPreprocessed(party_id, sign_results, key.sign_and_prep, player);
    
    const u8 public_one = static_cast<u8>(party_id == SERVER ? 1 : 0);
    const u8 nonnegative = is_negative ^ public_one;
    
    // 2. 计算 |x|
    const GroupElement relu = 
        bitxaOnline(party_id, input, nonnegative, key.relu_mul, player);
    const GroupElement magnitude = relu * 2 - input;
    
    // 3. 裁剪判断: |x| < bound (使用离线生成的分段密钥)
    std::vector<std::vector<u8>> mag_segments(key.nseg);
    for (int j = 0; j < key.nseg; ++j) {
        const int w = layout.width[j];
        const uint64_t seg_val = (magnitude.value >> layout.shift[j]) & ((1ULL << w) - 1);
        mag_segments[j].resize(w);
        for (int i = 0; i < w; ++i) {
            mag_segments[j][i] = (seg_val >> i) & 1;
        }
    }
    
    std::vector<SegmentEvalResult> clip_results = 
        evalAllSegments(party_id, key.clip_cmp_keys, mag_segments);
    const u8 below_bound = 
        inferComparisonPreprocessed(party_id, clip_results, key.clip_and_prep, player);
    
    // 4. 选择、截断、LUT、分支（保持不变）
    const uint64_t bound = (16ULL << key.frac_bits);
    const uint64_t clipped_bound = bound - 1;
    const GroupElement public_clipped_bound(
        (party_id == SERVER ? clipped_bound : 0), key.bits);
    
    const GroupElement select_input = magnitude - public_clipped_bound;
    const GroupElement clipped = public_clipped_bound +
        bitxaOnline(party_id, select_input, below_bound, 
                    key.clip_select_mul, player);
    
    const GroupElement reduced = key.trunc_bits > 0
        ? exactTruncate(party_id, clipped, key.trunc_key, player)
        : clipped;
    
    const GroupElement index(reduced.value, key.q);
    const GroupElement delta = dfss::publicLut(
        party_id, index, key.delta_table, key.lut_key);
    
    const uint64_t scale = (1ULL << key.frac_bits);
    const GroupElement public_scale(
        (party_id == SERVER ? scale : 0), key.bits);
    const GroupElement branch_payload = public_scale - delta * 2;
    const GroupElement branch = bitxaOnline(
        party_id, branch_payload, nonnegative, 
        key.branch_select_mul, player);
    
    return delta + branch;
}
```

### 修改 2: DPF 批量生成

#### 2.1 分析当前瓶颈

当前 `genSegmentKeys` 已经使用了批量生成：
```cpp
// src/gen.cpp:135
result.keys = cggmGen2pcBatch(party_id, split_result.segment_bits, timer);
```

所以 **DPF 批量生成已经实现了**。

但 Sigmoid 中两次调用 `genSegmentKeys` 是串行的：
```cpp
key.sign_cmp_keys = genSegmentKeys(party_id, split_sign);
key.clip_cmp_keys = genSegmentKeys(party_id, split_clip);
```

#### 2.2 优化方案：合并批量生成

创建新的批量接口：

```cpp
// src/gen.h
struct MultiSegmentKeys {
    std::vector<SegmentKeys> key_sets;  // 多组分段密钥
};

// 批量生成多组分段密钥
MultiSegmentKeys genMultiSegmentKeys(
    int party_id,
    const std::vector<SplitResult>& split_results,
    PhaseTimer* timer = nullptr);
```

```cpp
// src/gen.cpp
MultiSegmentKeys genMultiSegmentKeys(
    int party_id,
    const std::vector<SplitResult>& split_results,
    PhaseTimer* timer) {
    
    MultiSegmentKeys result;
    result.key_sets.resize(split_results.size());
    
    // 合并所有分段到一个大批量
    std::vector<std::vector<u8>> all_segments;
    std::vector<int> set_boundaries;  // 记录每组的起始位置
    
    for (const auto& split_result : split_results) {
        set_boundaries.push_back(all_segments.size());
        for (const auto& seg_bits : split_result.segment_bits) {
            all_segments.push_back(seg_bits);
        }
    }
    set_boundaries.push_back(all_segments.size());
    
    // 一次性批量生成所有 DPF
    std::vector<CggmKey> all_keys = 
        cggmGen2pcBatch(party_id, all_segments, timer);
    
    // 分配回各组
    for (size_t i = 0; i < split_results.size(); ++i) {
        int start = set_boundaries[i];
        int end = set_boundaries[i + 1];
        result.key_sets[i].keys.assign(
            all_keys.begin() + start,
            all_keys.begin() + end);
    }
    
    return result;
}
```

在 `sigmaOffline` 中使用：

```cpp
SigmaActivationKey sigmaOffline(...) {
    // ... 前面代码 ...
    
    // 批量生成所有分段密钥
    std::vector<SplitResult> splits = {split_sign, split_clip};
    MultiSegmentKeys multi_keys = genMultiSegmentKeys(party_id, splits);
    
    key.sign_cmp_keys = multi_keys.key_sets[0];
    key.clip_cmp_keys = multi_keys.key_sets[1];
    
    // ... 后续代码 ...
}
```

### 修改 3: OpenMP 并行化

#### 3.1 CMakeLists.txt 添加 OpenMP

```cmake
# 在 CMakeLists.txt 中添加
find_package(OpenMP)
if(OpenMP_CXX_FOUND)
    target_link_libraries(segfss PRIVATE OpenMP::OpenMP_CXX)
    target_compile_definitions(segfss PRIVATE USE_OPENMP)
endif()
```

#### 3.2 在适当位置添加并行

**位置 1: sigmaOffline 中并行生成多个 bitxa 密钥**

```cpp
// src/activation.cpp
SigmaActivationKey sigmaOffline(...) {
    // ... 前面代码 ...
    
#ifdef USE_OPENMP
    #pragma omp parallel sections
    {
        #pragma omp section
        { key.relu_mul = bitxaOffline(party_id, bits, player); }
        
        #pragma omp section
        { key.clip_select_mul = bitxaOffline(party_id, bits, player); }
        
        #pragma omp section
        { key.branch_select_mul = bitxaOffline(party_id, bits, player); }
    }
#else
    key.relu_mul = bitxaOffline(party_id, bits, player);
    key.clip_select_mul = bitxaOffline(party_id, bits, player);
    key.branch_select_mul = bitxaOffline(party_id, bits, player);
#endif
    
    return key;
}
```

**注意**：bitxaOffline 内部使用 player（网络通信），需要确保线程安全。如果有竞争，改为串行。

**位置 2: evalAllSegments 已经使用了并行**

```cpp
// src/gen.cpp:169-179 已经使用 std::async
std::vector<std::future<void>> tasks;
for (size_t j = 0; j < nseg; j++) {
    tasks.emplace_back(std::async(std::launch::async, [&, j]() {
        results[j] = evalSegment(party_id, keys.keys[j], xhat_segments[j]);
    }));
}
```

这已经是并行的，不需要修改。

## 修改文件清单

1. **src/activation.h**
   - 修改 `SigmaActivationKey` 结构

2. **src/activation.cpp**
   - 修改 `sigmaOffline()`
   - 修改 `sigmaOnline()`
   - 添加 `#include "segment.h"`
   - 添加 `#include "gen.h"`

3. **src/gen.h**
   - 添加 `MultiSegmentKeys` 结构
   - 添加 `genMultiSegmentKeys()` 声明

4. **src/gen.cpp**
   - 实现 `genMultiSegmentKeys()`

5. **CMakeLists.txt**
   - 添加 OpenMP 支持

6. **test/correctness/cases_activation.cpp**
   - 更新测试用例验证新实现

## 预期性能提升

### Online 阶段

**Sigmoid 当前（使用 BitDec）**:
- 符号判断: BitDec 32 轮
- 裁剪判断: BitDec 32 轮
- 总计: ~64 轮

**Sigmoid 优化后（使用分段比较）**:
- 符号判断: evalAllSegments (本地) + inferComparison ~3 轮
- 裁剪判断: evalAllSegments (本地) + inferComparison ~3 轮
- 总计: ~6 轮

**Online 轮次减少**: 64 → 6，约 **10× 加速**

### Offline 阶段

**当前**: 两次串行调用 `genSegmentKeys`

**优化后**: 
- 批量生成所有 DPF：通信合并
- 预期加速: 20-30%

## 风险与注意事项

1. **线程安全**: `Peer` 对象的并发访问需要验证
2. **内存开销**: 批量生成会增加内存使用
3. **测试覆盖**: 需要完整的回归测试
4. **向后兼容**: 保留原实现作为 fallback

## 实施步骤

1. **Step 1**: 修改 `SigmaActivationKey` 和 `sigmaOffline` (测试 offline 正确性)
2. **Step 2**: 修改 `sigmaOnline` (测试 online 正确性)
3. **Step 3**: 实现 `genMultiSegmentKeys` 批量生成
4. **Step 4**: 添加 OpenMP 支持
5. **Step 5**: 性能 benchmark 和优化

每步完成后都需要通过 correctness 测试。
