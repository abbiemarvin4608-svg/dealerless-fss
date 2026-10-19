# Sigmoid(-1) 边界问题修复计划

## 问题诊断

### 根本原因
分段比较使用 **无符号比较语义**，但 Sigmoid 需要 **有符号比较**。

**测试失败案例**: `sigmoid(-1)`
- 输入: -1.0 (fixed point) = -4096 = 0xFFFFF000
- 符号判断: `x < 0`，期望返回 1 (true)
- 实际行为: DPF 比较 `0xFFFFF000 < 0x00000000` (无符号) → 0 (false) ❌
- 正确行为: 有符号比较 `-4096 < 0` → 1 (true) ✓

### 为什么会这样？
DPF 树遍历本质上是按位比较，等价于无符号整数比较。对于有符号数：
- 负数: MSB=1，在无符号解释下是大数（如 0xFFFFF000 > 0x00000000）
- 正数: MSB=0，两种解释一致

---

## 解决方案：MSB 翻转

### 数学原理
有符号比较可以转换为无符号比较：

```
signed(x) < signed(r)  ⟺  unsigned(x ⊕ MSB_MASK) < unsigned(r ⊕ MSB_MASK)
```

其中 `MSB_MASK = 1ULL << (bits - 1)`

### 为什么有效？
翻转 MSB 会将有符号数映射到无符号空间，同时保持顺序：

| 原始(有符号) | 原始(32位) | 翻转MSB后 | 无符号值 |
|-------------|-----------|---------|---------|
| -2^31 (最小) | 0x80000000 | 0x00000000 | 0 |
| -4096        | 0xFFFFF000 | 0x7FFFF000 | 2147479552 |
| -1           | 0xFFFFFFFF | 0x7FFFFFFF | 2147483647 |
| 0            | 0x00000000 | 0x80000000 | 2147483648 |
| 1            | 0x00000001 | 0x80000001 | 2147483649 |
| 2^31-1 (最大)| 0x7FFFFFFF | 0xFFFFFFFF | 4294967295 |

**验证案例**: `-4096 < 0`
- 原始: `0xFFFFF000 < 0x00000000` (无符号) → false ❌
- 翻转: `0x7FFFF000 < 0x80000000` (无符号) → true ✓

---

## 实施方案

### 修改 1: 添加有符号比较标志

**文件**: `src/activation.h`

```cpp
struct PiecewiseComparisonKey {
    int bits = 0;
    int nseg = 0;
    uint64_t threshold = 0;
    bool is_signed = false;  // 新增：标记是否为有符号比较
    SegmentKeys segment_keys;
    BooleanAndPreprocessing and_prep;
};
```

### 修改 2: 支持有符号比较的离线阶段

**文件**: `src/activation.h`

```cpp
// 更新函数签名
PiecewiseComparisonKey piecewiseCompareOffline(
    int party_id, int bits, uint64_t threshold, int nseg,
    Peer* player, bool is_signed = false);  // 新增参数
```

**文件**: `src/activation.cpp`

```cpp
PiecewiseComparisonKey piecewiseCompareOffline(
    int party_id, int bits, uint64_t threshold, int nseg,
    Peer* player, bool is_signed) {

    if (nseg <= 0 || nseg > bits) {
        throw std::invalid_argument("piecewiseCompareOffline: invalid nseg");
    }

    PiecewiseComparisonKey key;
    key.bits = bits;
    key.nseg = nseg;
    key.threshold = threshold;
    key.is_signed = is_signed;

    SegmentLayout layout(bits, nseg);

    // 如果是有符号比较，翻转 MSB
    uint64_t threshold_adjusted = threshold;
    if (is_signed) {
        const uint64_t msb_mask = 1ULL << (bits - 1);
        threshold_adjusted = threshold ^ msb_mask;
    }

    GroupElement threshold_elem(threshold_adjusted, bits);
    SplitResult split_result = split(party_id, threshold_elem, layout, player);

    key.segment_keys = genSegmentKeys(party_id, split_result);

    const int num_ands = 2 * (nseg - 1);
    key.and_prep = booleanAndOffline(party_id, num_ands, player);

    return key;
}
```

### 修改 3: 支持有符号比较的在线阶段

**文件**: `src/activation.cpp`

```cpp
u8 piecewiseCompareOnline(
    int party_id, GroupElement x,
    const PiecewiseComparisonKey& key, Peer* player) {

    if (x.bitsize != key.bits) {
        throw std::invalid_argument("piecewiseCompareOnline: input bitsize mismatch");
    }

    // 打开输入值
    GroupElement x_opened = x;
    reconstruct(&x_opened);

    // 如果是有符号比较，翻转 MSB
    if (key.is_signed) {
        const uint64_t msb_mask = 1ULL << (key.bits - 1);
        x_opened.value ^= msb_mask;
    }

    SegmentLayout layout(key.bits, key.nseg);

    // 提取分段
    std::vector<std::vector<u8>> xhat_segments(key.nseg);
    for (int j = 0; j < key.nseg; ++j) {
        const int w = layout.width[j];
        const uint64_t seg_val = (x_opened.value >> layout.shift[j]) & ((1ULL << w) - 1);

        xhat_segments[j].resize(w);
        for (int i = 0; i < w; ++i) {
            xhat_segments[j][i] = (seg_val >> i) & 1;
        }
    }

    // 评估分段
    std::vector<SegmentEvalResult> segment_results =
        evalAllSegments(party_id, key.segment_keys, xhat_segments);

    // 推断结果
    return inferComparisonPreprocessed(party_id, segment_results,
                                       const_cast<BooleanAndPreprocessing&>(key.and_prep),
                                       player);
}
```

### 修改 4: 更新 sigmaOffline 调用

**文件**: `src/activation.cpp`

```cpp
SigmaActivationKey sigmaOffline(int party_id, int bits, int frac_bits,
                                int q, Peer* player) {
    // ... 前面代码不变 ...

    // 符号判断使用有符号比较
    key.sign_cmp = piecewiseCompareOffline(party_id, bits, 0, key.nseg, player,
                                            true);  // is_signed = true

    // 裁剪判断使用无符号比较（magnitude 已经是非负数）
    const uint64_t bound = (16ULL << frac_bits);
    key.clip_cmp = piecewiseCompareOffline(party_id, bits, bound, key.nseg, player,
                                            false);  // is_signed = false

    // ... 后续代码不变 ...
}
```

---

## 验证计划

### 单元测试
创建测试验证 MSB 翻转逻辑：

```cpp
// test/correctness/test_signed_comparison.cpp
void test_signed_comparison() {
    // Test 1: -1 < 0 → true
    // Test 2: 0 < 1 → true
    // Test 3: 1 < 0 → false
    // Test 4: -100 < -50 → true
    // Test 5: edge cases (INT_MIN, INT_MAX)
}
```

### 回归测试
运行完整的 activation 测试套件：
```bash
./build/test/correctness/SEGFSS_CORRECTNESS_TEST r=2 t=9
./build/test/correctness/SEGFSS_CORRECTNESS_TEST r=3 t=9
```

期望结果：**12/12 测试通过** ✅

---

## 替代方案（备选）

如果 MSB 翻转不工作，可以考虑：

### 方案 B: 仅对符号使用 MSB 提取
```cpp
// 不用完整比较，直接提取 MSB
u8 is_negative = (x_opened.value >> (bits - 1)) & 1;
// 需要 1 轮通信来重建 x，但比 BitDec 快
```

### 方案 C: 混合方法
- 符号判断：MSB 提取（1轮）
- 裁剪判断：分段比较（3轮）
- 总计：4轮（仍然比 BitDec 的 64 轮快 16×）

---

## Phase 2: 批量化优化（修复后）

修复 `sigmoid(-1)` 后，实施批量化优化：

### 批量接口设计

```cpp
// Batch offline
std::vector<SigmaActivationKey> sigmaBatchOffline(
    int party_id, int size, int bits, int frac_bits, int q, Peer* player);

// Batch online
std::vector<GroupElement> sigmaBatchOnline(
    int party_id, const std::vector<GroupElement>& inputs,
    std::vector<SigmaActivationKey>& keys, Peer* player);
```

### 优化点
1. **批量打开输入**：一次通信打开所有输入
2. **批量 DPF 评估**：并行评估
3. **批量 AND 操作**：合并 inferComparison 的 AND 门

### 预期收益
- 当前：N 个 Sigmoid × 6 轮 = 6N 轮
- 批量：1 轮（打开）+ 3 轮（评估）+ 2 轮（推断）= 6 轮（常数）
- **加速**: N × 线性到常数轮次

---

## 实施步骤

### Step 1: 修复有符号比较（立即）
1. 添加 `is_signed` 标志
2. 实现 MSB 翻转逻辑
3. 更新 `sigmaOffline` 调用
4. 测试验证

### Step 2: 批量化优化（修复后）
1. 实现批量接口
2. 优化通信轮次
3. 性能基准测试

### Step 3: 端到端验证
1. Google-30 模型测试
2. 与原始 BitDec 对比
3. 文档更新

---

## 风险评估

| 风险 | 概率 | 影响 | 缓解措施 |
|------|------|------|---------|
| MSB 翻转破坏正数比较 | 低 | 高 | 完整回归测试 |
| 性能不如预期 | 低 | 中 | 基准测试验证 |
| 批量化引入新 bug | 中 | 中 | 渐进式实施 |

---

## 成功标准

✅ `sigmoid(-1)` 测试通过  
✅ 所有 12 个 activation 测试通过  
✅ 性能提升 10× 验证  
✅ 批量化实现并测试  
✅ 文档完整更新
