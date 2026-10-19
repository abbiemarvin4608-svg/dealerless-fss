# 详细优化实施方案

## 目录
1. [分段比较替换 BitDec](#1-分段比较替换-bitdec)
2. [批量 Sigmoid/Tanh API](#2-批量-sigmoidtanh-api)
3. [批量 ReLU/线性层接口](#3-批量-relu线性层接口)
4. [DPF/DCF 批量生成](#4-dpfdcf-批量生成)
5. [并行化支持](#5-并行化支持)

---

## 1. 分段比较替换 BitDec

### 1.1 问题分析

**当前实现** (`src/activation.cpp:128-131, 148-151`):
```cpp
// 第一次 BitDec: 判断 input 符号
const std::vector<u8> input_bits = BitDec(party_id, input, key.bits, player);
const u8 nonnegative = input_bits.front() ^ public_one;

// 第二次 BitDec: 判断 |x| < bound
const std::vector<u8> bound_bits = BitDec(party_id, below_bound_input, key.bits, player);
const u8 below_bound = bound_bits.front();
```

**BitDec 工作原理** (`src/dfss/mpc/secure_ops.cpp:931-955`):
- 输入：n 位的加法份额 `⟨x⟩_A`
- 输出：n 位的布尔份额 `⟨b_0⟩_B,...,⟨b_{n-1}⟩_B`
- 算法：串行进位加法器
  ```
  carry_0 = 0
  for i = n-1 down to 0:
    b_i = x_i ⊕ carry_i              // XOR (本地)
    carry_{i+1} = x_i ∧ carry_i      // AND (需要通信，1轮)
  ```
- **轮次**: n 轮 (每位一次依赖的 AND)
- **通信量**: n × (2λ + overhead) ≈ 32 × 40 = 1280 bytes

**问题**:
- 我们只需要 MSB (符号位)，却计算了全部 n 位
- n=32 时需要 32 轮通信
- Google-30: 99 步 × 2 次 BitDec × 32 轮 = 6336 轮

### 1.2 替换方案：分段比较

**核心思路**: 
- 判断符号 = 判断 `x < 0`
- 判断裁剪 = 判断 `|x| < bound`
- 都可以用分段 FSS 比较实现

**分段比较原理** (`src/gen.cpp:186-226`):
1. 将 n 位输入分成 c 段，每段 w=n/c 位
2. 每段生成一个 DPF，评估得到 (L_j, e_j)
3. 用二叉树归约合并: (L_a, e_a) ⊗ (L_b, e_b) = (L_a ⊕ (e_a ∧ L_b), e_a ∧ e_b)
4. 最终得到比较结果 L

**轮次分析**:
- DPF 评估: 并行，0 轮 (offline 预生成)
- 打开掩码输入: 1 轮
- 二叉树归约: ⌈log₂ c⌉ 轮 (每层需要 AND)
- **总轮次**: 1 + ⌈log₂ c⌉

对于 c=4: 1 + 2 = 3 轮 (vs BitDec 的 32 轮，提升 **10.7×**)

### 1.3 具体实现

#### Step 1: 设计新的比较 API

**文件**: `src/activation.h` (新增)

```cpp
// 分段比较密钥 (包含 c 个 DPF)
struct PiecewiseComparisonKey {
    int bits;           // 输入位宽
    int nseg;          // 段数 c
    uint64_t threshold; // 比较阈值 r
    std::vector<dfss::DPFKey> dpf_keys;  // c 个 DPF 密钥
    BooleanAndPreprocessing and_prep;     // 归约用的 AND 三元组
};

// Offline: 生成比较密钥
// 功能: 准备判断 x < threshold 的协议
PiecewiseComparisonKey piecewiseCompareOffline(
    int party_id, 
    int bits,           // 输入位宽 (如 32)
    uint64_t threshold, // 阈值 (如 0 表示判断符号)
    int nseg,          // 段数 (如 4)
    Peer* player);

// Online: 执行比较
// 输入: ⟨x⟩_A (加法份额)
// 输出: ⟨x < threshold⟩_B (布尔份额)
u8 piecewiseCompareOnline(
    int party_id,
    GroupElement x,
    const PiecewiseComparisonKey& key,
    Peer* player);

void freePiecewiseComparisonKey(PiecewiseComparisonKey& key);
```

#### Step 2: 实现 Offline 密钥生成

**文件**: `src/activation.cpp` (新增函数)

```cpp
PiecewiseComparisonKey piecewiseCompareOffline(
    int party_id, int bits, uint64_t threshold, int nseg, Peer* player) {
    
    if (nseg <= 0 || nseg > bits) {
        throw std::invalid_argument("invalid nseg");
    }
    if (bits % nseg != 0) {
        throw std::invalid_argument("bits must be divisible by nseg");
    }
    
    PiecewiseComparisonKey key;
    key.bits = bits;
    key.nseg = nseg;
    key.threshold = threshold;
    
    const int w = bits / nseg;  // 每段位宽
    
    // 1. 为每段生成 DPF
    //    目标函数: f_j(η) = { 1  if r_j < η
    //                        { 0  otherwise
    //    其中 r_j 是 threshold 的第 j 段
    key.dpf_keys.reserve(nseg);
    for (int j = 0; j < nseg; ++j) {
        // 提取 threshold 的第 j 段 (w 位)
        const uint64_t r_j = (threshold >> (j * w)) & ((1ULL << w) - 1);
        
        // 生成 DPF: 在 η > r_j 时输出 1
        // 注意: DPF 域是 2^w，输出是 1-bit
        dfss::DPFKey dpf = dfss::generateDPF(
            party_id, 
            w,          // 输入域位宽
            r_j + 1,    // 分界点 (> r_j 等价于 >= r_j+1)
            1,          // 输出位宽
            player);
        
        key.dpf_keys.push_back(dpf);
    }
    
    // 2. 预生成 AND 三元组
    //    归约需要 2(c-1) 个 AND 门
    //    用二叉树: c=4 需要 2×3=6 个 AND
    const int num_ands = 2 * (nseg - 1);
    key.and_prep = booleanAndOffline(party_id, num_ands, player);
    
    return key;
}
```

**关键点解释**:

1. **分段提取**:
   ```
   threshold = 0x1A2B3C4D (32位)
   分4段 (每段8位):
     r_0 = 0x4D (最低段)
     r_1 = 0x3C
     r_2 = 0x2B
     r_3 = 0x1A (最高段)
   ```

2. **DPF 语义**:
   - 输入: η ∈ [0, 2^w)
   - 输出: 1{η > r_j}
   - 为什么 r_j+1: 因为我们要 `x < threshold`，即每段 `x_j ≤ r_j`

3. **AND 三元组数量**:
   ```
   二叉树归约 c=4:
        层0: (seg0,seg1) → mid0  需要 2 AND
             (seg2,seg3) → mid1  需要 2 AND
        层1: (mid0,mid1) → final 需要 2 AND
        总计: 6 AND
   ```

#### Step 3: 实现 Online 比较

**文件**: `src/activation.cpp` (新增函数)

```cpp
u8 piecewiseCompareOnline(
    int party_id, GroupElement x, 
    const PiecewiseComparisonKey& key, Peer* player) {
    
    if (x.bitsize != key.bits) {
        throw std::invalid_argument("input bitsize mismatch");
    }
    
    const int w = key.bits / key.nseg;
    
    // Step 1: 打开掩码输入 (1轮通信)
    // 用随机数 r 掩盖 x, 打开 x̂ = x + r
    GroupElement mask = generateRandomMask(party_id, key.bits, player);
    GroupElement masked_x = x + mask;
    uint64_t opened_x_hat = openValue(party_id, masked_x, player);  // 1轮
    
    // Step 2: 分段评估 DPF (本地计算，0轮)
    std::vector<SegmentEvalResult> segment_results;
    segment_results.reserve(key.nseg);
    
    for (int j = 0; j < key.nseg; ++j) {
        // 提取 x̂ 的第 j 段
        const uint64_t x_hat_j = (opened_x_hat >> (j * w)) & ((1ULL << w) - 1);
        
        // 评估 DPF
        const u8 dpf_output = dfss::evalDPF(
            party_id, 
            key.dpf_keys[j], 
            x_hat_j,    // 在打开的值上评估
            player);
        
        // 构造 (L_j, e_j)
        SegmentEvalResult seg;
        seg.L_j = dpf_output;  // L_j = DPF(x̂_j)
        seg.e_j = computeEquality(x_hat_j, /* r_j 的相关值 */);  // 细节见下
        segment_results.push_back(seg);
    }
    
    // Step 3: 二叉树归约 (⌈log₂ c⌉ 轮)
    return inferComparisonPreprocessed(
        party_id, segment_results, key.and_prep, player);
}
```

**Step 2 细节补充**: e_j 的计算

```cpp
// e_j 表示 "第 j 段相等"
// 实现: e_j = 1{x̂_j = r_j}
// 但我们只有份额，需要协议计算

// 方法1: 复用 DPF
// e_j = DPF_=(x̂_j) where DPF_= 是相等测试的 DPF
// 生成时在 offline 阶段添加额外的 DPF

// 方法2: 利用 L_j
// 如果 x̂_j > r_j: L_j=1, e_j=0
// 如果 x̂_j = r_j: L_j=0, e_j=1
// 如果 x̂_j < r_j: L_j=0, e_j=0
// 实现: e_j 可通过额外的 DPF 或者在 offline 预计算
```

实际实现中，我们在 `inferComparison` 中已经有处理，直接复用 `gen.cpp:186-226` 的逻辑。

#### Step 4: 修改 Sigmoid 使用新比较

**文件**: `src/activation.cpp` (修改 `sigmaOffline` 和 `sigmaOnline`)

```cpp
// 修改密钥结构
struct SigmaActivationKey {
    int bits;
    int frac_bits;
    int q;
    int trunc_bits;
    PublicLUTData delta_table;
    dfss::PublicLutKeyPack lut_key;
    ExactTruncationKey trunc_key;
    
    // 删除: bitxaOffline 的三个密钥
    // u64 relu_mul, clip_mul, branch_mul;
    
    // 新增: 三个分段比较密钥
    PiecewiseComparisonKey sign_cmp;      // 判断 x < 0
    PiecewiseComparisonKey clip_cmp;      // 判断 |x| < bound
    PiecewiseComparisonKey branch_cmp;    // (可选，复用 sign_cmp)
    
    // 保留 bitxa 用于最后的选择门
    u64 clip_select_mul;
    u64 branch_select_mul;
};

SigmaActivationKey sigmaOffline(
    int party_id, int bits, int frac_bits, int q, Peer* player) {
    
    // ... 参数验证 ...
    
    SigmaActivationKey key;
    key.bits = bits;
    key.frac_bits = frac_bits;
    key.q = q;
    key.trunc_bits = frac_bits - (q - 4);
    key.delta_table = makeSigmaDeltaTable(bits, frac_bits, q);
    key.lut_key = dfss::publicLutOffline(party_id, key.delta_table);
    
    if (key.trunc_bits > 0) {
        key.trunc_key = exactTruncateOffline(party_id, bits, key.trunc_bits, player);
    }
    
    // 生成三个比较密钥
    const int nseg = 4;  // 可配置
    
    // 1. 符号判断: x < 0
    key.sign_cmp = piecewiseCompareOffline(
        party_id, bits, 0, nseg, player);
    
    // 2. 裁剪判断: magnitude < bound
    const uint64_t bound = (16ULL << frac_bits);
    key.clip_cmp = piecewiseCompareOffline(
        party_id, bits, bound, nseg, player);
    
    // 3. bitxa 用于选择 (保留，因为需要在环上做乘法)
    key.clip_select_mul = bitxaOffline(party_id, bits, player);
    key.branch_select_mul = bitxaOffline(party_id, bits, player);
    
    return key;
}

GroupElement sigmaOnline(
    int party_id, GroupElement input,
    SigmaActivationKey& key, Peer* player) {
    
    // 1. 符号判断: x < 0 ? (用分段比较，3轮)
    const u8 is_negative = piecewiseCompareOnline(
        party_id, input, key.sign_cmp, player);
    const u8 public_one = (party_id == SERVER ? 1 : 0);
    const u8 nonnegative = is_negative ^ public_one;
    
    // 2. 计算 |x| (仍需 bitxa 乘法)
    const GroupElement relu = bitxaOnline(
        party_id, input, nonnegative, 
        /* 需要提前生成 */ key.relu_mul, player);
    const GroupElement magnitude = relu * 2 - input;
    
    // 3. 裁剪判断: |x| < bound ? (用分段比较，3轮)
    const u8 below_bound = piecewiseCompareOnline(
        party_id, magnitude, key.clip_cmp, player);
    
    // 4. 选择: below_bound ? magnitude : bound-1
    const GroupElement select_input = magnitude - public_clipped_bound;
    const GroupElement clipped = public_clipped_bound + 
        bitxaOnline(party_id, select_input, below_bound, 
                    key.clip_select_mul, player);
    
    // 5. 截断 + LUT (不变)
    const GroupElement reduced = key.trunc_bits > 0
        ? exactTruncate(party_id, clipped, key.trunc_key, player)
        : clipped;
    const GroupElement index(reduced.value, key.q);
    const GroupElement delta = dfss::publicLut(
        party_id, index, key.delta_table, key.lut_key);
    
    // 6. 分支选择 (不变)
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

### 1.4 轮次对比

| 操作 | BitDec 方案 | 分段比较方案 (c=4) |
|------|------------|-------------------|
| 符号判断 | 32 轮 | 3 轮 (1打开+2归约) |
| 计算 \|x\| | 1 轮 (bitxa) | 1 轮 (bitxa) |
| 裁剪判断 | 32 轮 | 3 轮 |
| 选择门 | 1 轮 | 1 轮 |
| 截断 | s 轮 | s 轮 |
| LUT | 1 轮 | 1 轮 |
| 最终分支 | 1 轮 | 1 轮 |
| **总计** | **68+s 轮** | **10+s 轮** |

提升: **6.8× ↓** (s=12 时: 80→22 轮)

### 1.5 测试方案

**文件**: `test/correctness/test_piecewise_compare.cpp` (新建)

```cpp
#include "activation.h"
#include "test_utils.h"
#include <gtest/gtest.h>

class PiecewiseCompareTest : public ::testing::Test {
protected:
    void SetUp() override {
        player_server = setupPlayer(SERVER);
        player_client = setupPlayer(CLIENT);
    }
    Peer* player_server;
    Peer* player_client;
};

TEST_F(PiecewiseCompareTest, SignTest) {
    const int bits = 32;
    const int nseg = 4;
    const uint64_t threshold = 0;  // 判断 x < 0
    
    // Offline
    auto key_server = piecewiseCompareOffline(
        SERVER, bits, threshold, nseg, player_server);
    auto key_client = piecewiseCompareOffline(
        CLIENT, bits, threshold, nseg, player_client);
    
    // 测试用例
    std::vector<int64_t> test_values = {
        -100, -1, 0, 1, 100,
        INT32_MIN, INT32_MAX
    };
    
    for (int64_t x : test_values) {
        // 秘密共享
        auto [share_s, share_c] = shareValue(x, bits);
        
        // Online (并行执行)
        u8 result_s, result_c;
        std::thread t1([&]() {
            result_s = piecewiseCompareOnline(
                SERVER, share_s, key_server, player_server);
        });
        std::thread t2([&]() {
            result_c = piecewiseCompareOnline(
                CLIENT, share_c, key_client, player_client);
        });
        t1.join(); t2.join();
        
        // 恢复结果
        u8 result = result_s ^ result_c;
        u8 expected = (x < 0) ? 1 : 0;
        
        EXPECT_EQ(result, expected) 
            << "x=" << x << " sign check failed";
    }
    
    freePiecewiseComparisonKey(key_server);
    freePiecewiseComparisonKey(key_client);
}

TEST_F(PiecewiseCompareTest, BoundTest) {
    const int bits = 32;
    const int nseg = 4;
    const uint64_t bound = 1000;
    
    // ... 类似测试 x < bound ...
}

TEST_F(PiecewiseCompareTest, RoundCount) {
    // 验证轮次确实是 1 + ⌈log₂ c⌉
    const int bits = 32;
    
    for (int nseg : {2, 4, 8, 16}) {
        resetRoundCounter(player_server);
        
        auto key = piecewiseCompareOffline(
            SERVER, bits, 0, nseg, player_server);
        GroupElement x(12345, bits);
        piecewiseCompareOnline(
            SERVER, x, key, player_server);
        
        int expected_rounds = 1 + (int)std::ceil(std::log2(nseg));
        int actual_rounds = getRoundCount(player_server);
        
        EXPECT_EQ(actual_rounds, expected_rounds)
            << "nseg=" << nseg;
        
        freePiecewiseComparisonKey(key);
    }
}
```

---

## 2. 批量 Sigmoid/Tanh API

### 2.1 问题分析

**当前实现**: 标量接口
```cpp
GroupElement sigmaOnline(int party_id, GroupElement input, ...);
```

**Google-30 场景**:
- 每个时间步: 100 个 hidden units
- 需要调用: `for (int i = 0; i < 100; ++i) sigma(h[i])`
- 问题: 
  - 100 次函数调用开销
  - 无法批量打开掩码 (100 轮 → 1 轮)
  - DPF 评估串行 (可并行)

### 2.2 批量化原理

**关键观察**:
1. **打开掩码可以批量**: 
   - 原: 每个输入单独 `open(x̂_i)` - 100 轮
   - 新: 一次 `open([x̂_0,...,x̂_99])` - 1 轮

2. **DPF 评估可并行**:
   - 每个输入独立评估 DPF
   - 100 个评估可同时进行

3. **AND 归约可批量**:
   - 原: 每个输入的归约单独做 - 100 × log c 轮
   - 新: 批量 AND - log c 轮 (所有输入共享轮次)

### 2.3 具体实现

#### Step 1: 批量 API 设计

**文件**: `src/activation.h` (新增)

```cpp
// 批量 Sigmoid 密钥 (batch_size 份密钥打包)
struct SigmaBatchKey {
    int batch_size;
    int bits;
    int frac_bits;
    int q;
    int trunc_bits;
    int nseg;  // 分段数
    
    PublicLUTData delta_table;
    dfss::PublicLutKeyPack lut_key;
    ExactTruncationKey trunc_key;  // 批量截断
    
    // batch_size 份分段比较密钥
    std::vector<PiecewiseComparisonKey> sign_cmps;
    std::vector<PiecewiseComparisonKey> clip_cmps;
    
    // 批量 bitxa 密钥
    std::vector<u64> relu_muls;
    std::vector<u64> clip_select_muls;
    std::vector<u64> branch_select_muls;
};

// Offline: 生成批量密钥
SigmaBatchKey sigmaBatchOffline(
    int party_id,
    int batch_size,  // 批量大小 (如 100)
    int bits,
    int frac_bits,
    int q,
    int nseg,        // 分段数 (如 4)
    Peer* player);

// Online: 批量计算 Sigmoid
std::vector<GroupElement> sigmaBatchOnline(
    int party_id,
    const std::vector<GroupElement>& inputs,  // batch_size 个输入
    SigmaBatchKey& key,
    Peer* player);

void freeSigmaBatchKey(SigmaBatchKey& key);
```

#### Step 2: 实现 Offline

**文件**: `src/activation.cpp`

```cpp
SigmaBatchKey sigmaBatchOffline(
    int party_id, int batch_size, int bits, 
    int frac_bits, int q, int nseg, Peer* player) {
    
    if (batch_size <= 0 || batch_size > 1000) {
        throw std::invalid_argument("invalid batch_size");
    }
    
    SigmaBatchKey key;
    key.batch_size = batch_size;
    key.bits = bits;
    key.frac_bits = frac_bits;
    key.q = q;
    key.nseg = nseg;
    key.trunc_bits = frac_bits - (q - 4);
    
    // 1. LUT 表 (所有实例共享)
    key.delta_table = makeSigmaDeltaTable(bits, frac_bits, q);
    key.lut_key = dfss::publicLutOffline(party_id, key.delta_table);
    
    // 2. 截断密钥 (批量)
    if (key.trunc_bits > 0) {
        key.trunc_key = exactTruncateBatchOffline(
            party_id, bits, key.trunc_bits, batch_size, player);
    }
    
    // 3. 为每个实例生成密钥 (可并行)
    key.sign_cmps.reserve(batch_size);
    key.clip_cmps.reserve(batch_size);
    key.relu_muls.reserve(batch_size);
    key.clip_select_muls.reserve(batch_size);
    key.branch_select_muls.reserve(batch_size);
    
    const uint64_t bound = (16ULL << frac_bits);
    
    for (int i = 0; i < batch_size; ++i) {
        // 符号判断
        key.sign_cmps.push_back(
            piecewiseCompareOffline(party_id, bits, 0, nseg, player));
        
        // 裁剪判断
        key.clip_cmps.push_back(
            piecewiseCompareOffline(party_id, bits, bound, nseg, player));
        
        // bitxa 密钥
        key.relu_muls.push_back(bitxaOffline(party_id, bits, player));
        key.clip_select_muls.push_back(bitxaOffline(party_id, bits, player));
        key.branch_select_muls.push_back(bitxaOffline(party_id, bits, player));
    }
    
    return key;
}
```

**优化点**: 
- `piecewiseCompareOffline` 内部的 DPF 生成可批量化 (见优化 #4)
- 循环可多线程并行

#### Step 3: 实现 Online - 批量打开

**文件**: `src/activation.cpp`

```cpp
std::vector<GroupElement> sigmaBatchOnline(
    int party_id,
    const std::vector<GroupElement>& inputs,
    SigmaBatchKey& key,
    Peer* player) {
    
    if (inputs.size() != key.batch_size) {
        throw std::invalid_argument("batch size mismatch");
    }
    
    const int N = key.batch_size;
    std::vector<GroupElement> outputs(N);
    
    // ========== Phase 1: 批量符号判断 ==========
    
    // 1.1 生成掩码 (本地)
    std::vector<GroupElement> sign_masks(N);
    for (int i = 0; i < N; ++i) {
        sign_masks[i] = generateRandomMask(party_id, key.bits, player);
    }
    
    // 1.2 批量打开 x̂ = x + r (1轮)
    std::vector<GroupElement> masked_inputs(N);
    for (int i = 0; i < N; ++i) {
        masked_inputs[i] = inputs[i] + sign_masks[i];
    }
    std::vector<uint64_t> opened_sign = 
        openValuesBatch(party_id, masked_inputs, player);  // ← 1轮
    
    // 1.3 并行评估 DPF (本地)
    std::vector<std::vector<SegmentEvalResult>> sign_segments(N);
    #pragma omp parallel for
    for (int i = 0; i < N; ++i) {
        sign_segments[i] = evalDPFSegments(
            party_id, opened_sign[i], key.sign_cmps[i], player);
    }
    
    // 1.4 批量归约 (log c 轮)
    std::vector<u8> is_negative = 
        inferComparisonBatch(party_id, sign_segments, key.sign_cmps[0].and_prep, player);
    
    std::vector<u8> nonnegative(N);
    const u8 public_one = (party_id == SERVER ? 1 : 0);
    for (int i = 0; i < N; ++i) {
        nonnegative[i] = is_negative[i] ^ public_one;
    }
    
    // ========== Phase 2: 批量计算 |x| ==========
    
    std::vector<GroupElement> magnitudes(N);
    #pragma omp parallel for
    for (int i = 0; i < N; ++i) {
        GroupElement relu = bitxaOnline(
            party_id, inputs[i], nonnegative[i], 
            key.relu_muls[i], player);
        magnitudes[i] = relu * 2 - inputs[i];
    }
    
    // ========== Phase 3: 批量裁剪判断 ==========
    
    // 3.1 批量打开 |x̂| (1轮)
    std::vector<GroupElement> clip_masks(N);
    for (int i = 0; i < N; ++i) {
        clip_masks[i] = generateRandomMask(party_id, key.bits, player);
    }
    std::vector<GroupElement> masked_mags(N);
    for (int i = 0; i < N; ++i) {
        masked_mags[i] = magnitudes[i] + clip_masks[i];
    }
    std::vector<uint64_t> opened_mags = 
        openValuesBatch(party_id, masked_mags, player);  // ← 1轮
    
    // 3.2 并行评估 + 批量归约 (log c 轮)
    std::vector<std::vector<SegmentEvalResult>> clip_segments(N);
    #pragma omp parallel for
    for (int i = 0; i < N; ++i) {
        clip_segments[i] = evalDPFSegments(
            party_id, opened_mags[i], key.clip_cmps[i], player);
    }
    std::vector<u8> below_bound = 
        inferComparisonBatch(party_id, clip_segments, key.clip_cmps[0].and_prep, player);
    
    // ========== Phase 4: 选择 + LUT + 分支 ==========
    
    const uint64_t scale = (1ULL << key.frac_bits);
    const uint64_t bound = (16ULL << key.frac_bits);
    const uint64_t clipped_bound = bound - 1;
    const GroupElement public_scale((party_id == SERVER ? scale : 0), key.bits);
    const GroupElement public_clipped_bound(
        (party_id == SERVER ? clipped_bound : 0), key.bits);
    
    #pragma omp parallel for
    for (int i = 0; i < N; ++i) {
        // 4.1 裁剪选择
        GroupElement select_input = magnitudes[i] - public_clipped_bound;
        GroupElement clipped = public_clipped_bound + 
            bitxaOnline(party_id, select_input, below_bound[i],
                        key.clip_select_muls[i], player);
        
        // 4.2 截断
        GroupElement reduced = key.trunc_bits > 0
            ? exactTruncate(party_id, clipped, key.trunc_key, player)
            : clipped;
        
        // 4.3 LUT
        GroupElement index(reduced.value, key.q);
        GroupElement delta = dfss::publicLut(
            party_id, index, key.delta_table, key.lut_key);
        
        // 4.4 分支选择
        GroupElement branch_payload = public_scale - delta * 2;
        GroupElement branch = bitxaOnline(
            party_id, branch_payload, nonnegative[i],
            key.branch_select_muls[i], player);
        
        outputs[i] = delta + branch;
    }
    
    return outputs;
}
```

**关键函数**: 批量打开

```cpp
// src/mpc/secure_ops.cpp (新增)
std::vector<uint64_t> openValuesBatch(
    int party_id, 
    const std::vector<GroupElement>& shares,
    Peer* player) {
    
    const int N = shares.size();
    if (N == 0) return {};
    
    const int bitwidth = shares[0].bitsize;
    
    // 1. 打包所有份额到一个缓冲区
    std::vector<u8> send_buf(N * sizeof(uint64_t));
    for (int i = 0; i < N; ++i) {
        std::memcpy(&send_buf[i * sizeof(uint64_t)], 
                    &shares[i].value, sizeof(uint64_t));
    }
    
    // 2. 一次通信交换
    std::vector<u8> recv_buf(N * sizeof(uint64_t));
    player->send_and_recv(send_buf.data(), send_buf.size(),
                          recv_buf.data(), recv_buf.size());  // ← 1轮
    
    // 3. 解包并相加
    std::vector<uint64_t> results(N);
    for (int i = 0; i < N; ++i) {
        uint64_t other_share;
        std::memcpy(&other_share, &recv_buf[i * sizeof(uint64_t)], sizeof(uint64_t));
        results[i] = (shares[i].value + other_share) & ((1ULL << bitwidth) - 1);
    }
    
    return results;
}
```

**批量归约**:

```cpp
// src/gen.cpp (新增)
std::vector<u8> inferComparisonBatch(
    int party_id,
    const std::vector<std::vector<SegmentEvalResult>>& all_segments,
    BooleanAndPreprocessing& and_prep,
    Peer* player) {
    
    const int N = all_segments.size();
    const int nseg = all_segments[0].size();
    
    // 所有实例共享轮次进行归约
    std::vector<std::vector<SegmentEvalResult>> levels(N);
    for (int i = 0; i < N; ++i) {
        levels[i] = all_segments[i];
    }
    
    // 二叉树归约 (log c 轮)
    while (levels[0].size() > 1) {
        const int level_size = levels[0].size();
        const int pair_count = level_size / 2;
        
        // 收集所有 AND 输入 (N × pair_count × 2)
        std::vector<u8> all_lhs, all_rhs;
        for (int i = 0; i < N; ++i) {
            for (int p = 0; p < pair_count; ++p) {
                all_lhs.push_back(levels[i][2*p].e_j);
                all_rhs.push_back(levels[i][2*p+1].e_j);
                all_lhs.push_back(levels[i][2*p].e_j);
                all_rhs.push_back(levels[i][2*p+1].L_j);
            }
        }
        
        // 批量 AND (1轮，所有实例共享)
        std::vector<u8> all_products(all_lhs.size());
        and_wrapper(party_id, all_lhs.data(), all_rhs.data(),
                    all_products.data(), all_lhs.size(), player);  // ← 1轮
        
        // 更新下一层
        for (int i = 0; i < N; ++i) {
            std::vector<SegmentEvalResult> next;
            for (int p = 0; p < pair_count; ++p) {
                int idx = i * pair_count * 2 + p * 2;
                SegmentEvalResult combined;
                combined.e_j = all_products[idx];
                combined.L_j = levels[i][2*p].L_j ^ all_products[idx+1];
                next.push_back(combined);
            }
            if ((level_size & 1) != 0) {
                next.push_back(levels[i].back());
            }
            levels[i] = next;
        }
    }
    
    // 提取最终结果
    std::vector<u8> results(N);
    for (int i = 0; i < N; ++i) {
        results[i] = levels[i][0].L_j;
    }
    return results;
}
```

### 2.4 轮次对比 (Google-30: N=100, c=4)

| 阶段 | 标量 (×100) | 批量 |
|------|------------|------|
| 符号判断 | 100×3 = 300 | 1(打开) + 2(归约) = 3 |
| \|x\| 计算 | 100×1 = 100 | 1 (bitxa 串行) |
| 裁剪判断 | 100×3 = 300 | 1 + 2 = 3 |
| 选择+LUT | 100×3 = 300 | 3 (bitxa/LUT 串行) |
| **总计** | **1000 轮** | **10 轮** |

**提升**: **100× ↓**

### 2.5 测试方案

**文件**: `test/correctness/test_sigmoid_batch.cpp`

```cpp
TEST_F(SigmoidTest, BatchCorrectness) {
    const int batch_size = 10;
    const int bits = 32;
    const int frac_bits = 12;
    const int q = 10;
    const int nseg = 4;
    
    // Offline
    auto key_s = sigmaBatchOffline(
        SERVER, batch_size, bits, frac_bits, q, nseg, player_s);
    auto key_c = sigmaBatchOffline(
        CLIENT, batch_size, bits, frac_bits, q, nseg, player_c);
    
    // 测试输入
    std::vector<double> inputs = {
        -5.0, -2.0, -1.0, -0.5, 0.0, 
         0.5,  1.0,  2.0,  5.0, 10.0
    };
    
    // 量化
    const uint64_t scale = (1ULL << frac_bits);
    std::vector<int64_t> inputs_fixed(batch_size);
    for (int i = 0; i < batch_size; ++i) {
        inputs_fixed[i] = (int64_t)(inputs[i] * scale);
    }
    
    // 秘密共享
    auto [shares_s, shares_c] = shareValuesBatch(inputs_fixed, bits);
    
    // Online
    std::vector<GroupElement> results_s, results_c;
    std::thread t1([&]() {
        results_s = sigmaBatchOnline(SERVER, shares_s, key_s, player_s);
    });
    std::thread t2([&]() {
        results_c = sigmaBatchOnline(CLIENT, shares_c, key_c, player_c);
    });
    t1.join(); t2.join();
    
    // 恢复并验证
    for (int i = 0; i < batch_size; ++i) {
        int64_t result = (results_s[i].value + results_c[i].value) & ((1ULL << bits) - 1);
        double result_float = (double)result / scale;
        double expected = 1.0 / (1.0 + std::exp(-inputs[i]));
        
        EXPECT_NEAR(result_float, expected, 0.01)
            << "input=" << inputs[i];
    }
}

TEST_F(SigmoidTest, BatchPerformance) {
    const int batch_size = 100;
    // ... 测量轮次和通信量 ...
    EXPECT_LE(getRoundCount(player_s), 15);  // 应该 <<100
}
```

---

## 3. 批量 ReLU/线性层接口

### 3.1 问题分析

**当前 ReLU** (`src/relu.cpp`):
```cpp
GroupElement reluOnline(int party_id, GroupElement input, DReLUMultKey drelu_input, ...);
```
- 标量接口
- 每层需要遍历调用: `for (int i = 0; i < N; ++i) relu(x[i])`

**当前线性层** (`src/linear.cpp`):
```cpp
// 已经是批量的 (矩阵乘法)
void matmul(GroupElement* output, const GroupElement* input, ...);
```
- 但批次大小固定 4096
- 不支持动态批次

### 3.2 改进方案

#### 改进 3.1: 批量 ReLU

**文件**: `src/relu.h` (新增)

```cpp
// 批量 ReLU
struct ReLUBatchKey {
    int batch_size;
    int bits;
    std::vector<DReLUMultKey> drelu_keys;  // batch_size 个密钥
};

ReLUBatchKey reluBatchOffline(
    int party_id, int batch_size, int bits, Peer* player);

std::vector<GroupElement> reluBatchOnline(
    int party_id,
    const std::vector<GroupElement>& inputs,
    ReLUBatchKey& key,
    Peer* player);
```

**实现**:

```cpp
// src/relu.cpp
std::vector<GroupElement> reluBatchOnline(
    int party_id,
    const std::vector<GroupElement>& inputs,
    ReLUBatchKey& key,
    Peer* player) {
    
    const int N = inputs.size();
    if (N != key.batch_size) {
        throw std::invalid_argument("batch size mismatch");
    }
    
    // ReLU = max(0, x)
    // 实现: drelu = 1{x>=0}, output = drelu * x
    
    // 1. 批量生成 DReLU (分段比较)
    std::vector<u8> drelus(N);
    
    // 方法A: 如果 DReLU 用 DCF
    //   - 批量打开掩码输入 (1轮)
    //   - 并行评估 DCF (本地)
    //   - 批量归约 (log c 轮)
    
    // 方法B: 如果 DReLU 用现有的 inferComparison
    for (int i = 0; i < N; ++i) {
        drelus[i] = computeDReLU(party_id, inputs[i], 
                                  key.drelu_keys[i], player);
    }
    
    // 2. 批量 bitxa (串行，但可优化为批量 AND)
    std::vector<GroupElement> outputs(N);
    for (int i = 0; i < N; ++i) {
        outputs[i] = bitxaOnline(party_id, inputs[i], drelus[i],
                                 key.drelu_keys[i].mult_key, player);
    }
    
    return outputs;
}
```

**优化空间**:
- 如果 DReLU 用分段比较，可批量化 (见 Sigmoid 批量)
- bitxa 的 AND 可批量执行 (100 个 AND 用 1 轮)

#### 改进 3.2: 灵活批次线性层

**当前问题**: `matmul` 批次固定 4096

**改进**: 添加动态批次接口

```cpp
// src/linear.h (新增)
void matmulDynamicBatch(
    GroupElement* output,       // [batch_size, out_dim]
    const GroupElement* input,  // [batch_size, in_dim]
    const int64_t* weight,      // [out_dim, in_dim]
    int batch_size,             // 动态批次
    int in_dim,
    int out_dim,
    int bitwidth,
    Peer* player);
```

**实现**:

```cpp
// src/linear.cpp
void matmulDynamicBatch(
    GroupElement* output, const GroupElement* input,
    const int64_t* weight,
    int batch_size, int in_dim, int out_dim, int bitwidth,
    Peer* player) {
    
    // 如果批次 <= 4096，直接调用现有实现
    if (batch_size <= 4096) {
        return matmul(output, input, weight, batch_size, 
                      in_dim, out_dim, bitwidth, player);
    }
    
    // 否则分块处理
    const int chunk_size = 4096;
    for (int start = 0; start < batch_size; start += chunk_size) {
        int chunk = std::min(chunk_size, batch_size - start);
        matmul(output + start * out_dim,
               input + start * in_dim,
               weight, chunk, in_dim, out_dim, bitwidth, player);
    }
}
```

### 3.3 层级封装

**动机**: 方便调用

**文件**: `src/layer.h` (新建)

```cpp
#pragma once
#include "linear.h"
#include "relu.h"
#include "activation.h"

namespace segfss {

// 全连接层 + ReLU
class FCReLULayer {
public:
    FCReLULayer(int in_dim, int out_dim, int bitwidth);
    
    void offline(int party_id, Peer* player);
    
    std::vector<GroupElement> forward(
        int party_id,
        const std::vector<GroupElement>& input,
        Peer* player);
    
private:
    int in_dim_, out_dim_, bitwidth_;
    int64_t* weight_;
    ReLUBatchKey relu_key_;
};

// RNN 单元 (带 Sigmoid/Tanh)
class RNNCell {
public:
    RNNCell(int input_size, int hidden_size, int bitwidth);
    
    void offline(int party_id, Peer* player);
    
    std::vector<GroupElement> forward(
        int party_id,
        const std::vector<GroupElement>& input,
        const std::vector<GroupElement>& hidden,
        Peer* player);
    
private:
    int input_size_, hidden_size_, bitwidth_;
    int64_t *W_ih_, *W_hh_;
    SigmaBatchKey sigmoid_key_;
    SigmaBatchKey tanh_key_;
};

}  // namespace segfss
```

**实现示例**:

```cpp
// src/layer.cpp
std::vector<GroupElement> RNNCell::forward(
    int party_id,
    const std::vector<GroupElement>& input,
    const std::vector<GroupElement>& hidden,
    Peer* player) {
    
    // 1. 矩阵乘法: i_t = W_ih @ x + W_hh @ h
    std::vector<GroupElement> i_t(hidden_size_);
    matmulDynamicBatch(i_t.data(), input.data(), W_ih_,
                       1, input_size_, hidden_size_, 
                       bitwidth_, player);
    
    std::vector<GroupElement> h_contrib(hidden_size_);
    matmulDynamicBatch(h_contrib.data(), hidden.data(), W_hh_,
                       1, hidden_size_, hidden_size_,
                       bitwidth_, player);
    
    for (int i = 0; i < hidden_size_; ++i) {
        i_t[i] = i_t[i] + h_contrib[i];
    }
    
    // 2. 非线性: h_new = tanh(i_t)
    std::vector<GroupElement> h_new = 
        sigmaBatchOnline(party_id, i_t, tanh_key_, player);
    
    return h_new;
}
```

---

## 4. DPF/DCF 批量生成

### 4.1 问题分析

**当前实现** (`src/dfss/fss/dpf.cpp`):
```cpp
DPFKey generateDPF(int party_id, int bits, uint64_t alpha, int output_bits, ...);
```

**问题**:
- 逐个生成 DPF
- Google-30 sigmoid: 100 个实例 × 2 个比较 × 4 个 DPF = 800 个 DPF
- 每个 DPF 生成独立进行

**优化机会**:
- PRG 扩展可批量
- 密钥材料生成可并行

### 4.2 批量生成 API

**文件**: `src/dfss/fss/dpf.h` (新增)

```cpp
// 批量生成 DPF (相同参数)
struct DPFBatchSpec {
    int bits;
    int output_bits;
    std::vector<uint64_t> alphas;  // 每个 DPF 的 alpha
};

std::vector<DPFKey> generateDPFBatch(
    int party_id,
    const DPFBatchSpec& spec,
    Peer* player);
```

**实现思路**:

```cpp
// src/dfss/fss/dpf.cpp
std::vector<DPFKey> generateDPFBatch(
    int party_id, const DPFBatchSpec& spec, Peer* player) {
    
    const int N = spec.alphas.size();
    const int bits = spec.bits;
    const int output_bits = spec.output_bits;
    
    std::vector<DPFKey> keys(N);
    
    // 并行生成每个 DPF 的种子
    #pragma omp parallel for
    for (int i = 0; i < N; ++i) {
        // 生成随机种子
        keys[i].seed = generateRandomSeed();
    }
    
    // 批量扩展 (如果支持)
    // 或逐个扩展但并行
    #pragma omp parallel for
    for (int i = 0; i < N; ++i) {
        keys[i] = expandDPFKey(party_id, keys[i].seed, 
                               bits, spec.alphas[i], 
                               output_bits, player);
    }
    
    return keys;
}
```

**关键**: 
- 种子生成: 本地，可并行
- PRG 扩展: 本地，可并行
- 密钥交换: 可批量 (一次通信发送所有种子)

### 4.3 集成到 Sigmoid Offline

**修改**: `src/activation.cpp` 中的 `sigmaBatchOffline`

```cpp
SigmaBatchKey sigmaBatchOffline(...) {
    // ... 前面代码 ...
    
    // 批量生成 DPF
    DPFBatchSpec sign_spec, clip_spec;
    sign_spec.bits = bits / nseg;
    sign_spec.output_bits = 1;
    for (int i = 0; i < batch_size * nseg; ++i) {
        sign_spec.alphas.push_back(/* 符号判断的 alpha */);
        clip_spec.alphas.push_back(/* 裁剪判断的 alpha */);
    }
    
    auto sign_dpfs = generateDPFBatch(party_id, sign_spec, player);
    auto clip_dpfs = generateDPFBatch(party_id, clip_spec, player);
    
    // 打包到各个 PiecewiseComparisonKey
    for (int i = 0; i < batch_size; ++i) {
        key.sign_cmps[i].dpf_keys.assign(
            sign_dpfs.begin() + i*nseg,
            sign_dpfs.begin() + (i+1)*nseg);
        key.clip_cmps[i].dpf_keys.assign(
            clip_dpfs.begin() + i*nseg,
            clip_dpfs.begin() + (i+1)*nseg);
    }
    
    return key;
}
```

---

## 5. 并行化支持

### 5.1 OpenMP 多线程

**已有基础**: 代码中使用 `#pragma omp parallel for`

**优化点**:
1. **Offline 密钥生成**:
   - DPF 生成: 各个 DPF 独立
   - Beaver triple: 可批量生成

2. **Online 计算**:
   - DPF 评估: 各个实例独立
   - 本地计算 (ReLU、截断): 可并行

**限制**:
- 通信轮次无法并行 (串行依赖)
- 但可减少函数调用开销

### 5.2 实现示例

**文件**: `src/activation.cpp` (已有的并行)

```cpp
// Offline
#pragma omp parallel for
for (int i = 0; i < batch_size; ++i) {
    key.sign_cmps[i] = piecewiseCompareOffline(...);
}

// Online
#pragma omp parallel for
for (int i = 0; i < N; ++i) {
    sign_segments[i] = evalDPFSegments(...);
}
```

### 5.3 编译选项

**CMakeLists.txt** (确保启用 OpenMP):

```cmake
find_package(OpenMP REQUIRED)
target_link_libraries(segfss PRIVATE OpenMP::OpenMP_CXX)
```

---

## 6. 总结与实施计划

### 6.1 优化收益汇总

| 优化 | 影响模块 | 轮次减少 | 预期加速 |
|------|---------|---------|---------|
| #1 分段比较 | Sigmoid/Tanh | 64→10 | 6.4× |
| #2 批量 Sigmoid | Google-30 | 1000→10 | 100× |
| #3 批量 ReLU | 各 CNN 层 | 小幅 | 2-5× |
| #4 批量 DPF | Offline | - | 并行加速 |
| #5 OpenMP | 全局 | - | 线程数× |

**端到端**: Google-30 WAN
- 现状: 99 步 × 100 实例 × (2 BitDec × 32 轮) ≈ **1000s+**
- 优化后: 99 步 × (批量 10 轮) × 20ms ≈ **20s**
- 提升: **50× ↓**

### 6.2 实施顺序

**Week 1-2**: 基础协议 (P1)
1. 实现 `piecewiseCompareOffline/Online`
2. 测试正确性和轮次
3. 修改 Sigmoid 使用分段比较

**Week 3-4**: 批量化 (P1)
4. 实现 `sigmaBatchOffline/Online`
5. 批量打开、批量归约
6. 端到端测试 Google-30

**Week 5**: 其他优化 (P2)
7. 批量 ReLU
8. 批量 DPF 生成
9. OpenMP 并行

**Week 6**: 集成与验证
10. 性能 benchmark
11. 对比论文结果
12. 修复问题

### 6.3 测试策略

每个优化点都需要:
1. **正确性测试**: 与现有实现逐位对比
2. **性能测试**: 轮次、通信量、延迟
3. **边界测试**: 批量大小、位宽、特殊值

**回归测试**: 确保优化不破坏现有功能

---

## 7. 风险与注意事项

### 7.1 实现风险

1. **分段比较的边界情况**:
   - alpha = 0 时的处理
   - bits 不能整除 nseg
   - DPF 域溢出

2. **批量化的内存开销**:
   - batch_size=100: 100 个密钥
   - 需要估算内存使用

3. **并发安全**:
   - Peer 的线程安全性
   - OpenMP 与网络库的兼容

### 7.2 性能调优

1. **nseg 的选择**:
   - 太小: 轮次多 (log c 大)
   - 太大: DPF 密钥多，通信大
   - 建议: nseg=4 (轮次 3，密钥数可接受)

2. **batch_size 的选择**:
   - Google-30: hidden=100, 固定
   - ResNet: 可配置 (典型 1-128)

3. **编译优化**:
   - `-O3 -march=native`
   - `-fopenmp`
   - Profile-guided optimization

### 7.3 论文一致性

**确保**:
- 复杂度匹配: inferComparison 是 1 + ⌈log₂ c⌉ 轮
- 参数一致: nseg 在论文中说明
- 性能数字: 与论文 Table 对比

---

## 附录: 快速参考

### A.1 新增 API 速查

```cpp
// 分段比较
PiecewiseComparisonKey piecewiseCompareOffline(
    int party_id, int bits, uint64_t threshold, int nseg, Peer* player);
u8 piecewiseCompareOnline(
    int party_id, GroupElement x, const PiecewiseComparisonKey& key, Peer* player);

// 批量 Sigmoid
SigmaBatchKey sigmaBatchOffline(
    int party_id, int batch_size, int bits, int frac_bits, int q, int nseg, Peer* player);
std::vector<GroupElement> sigmaBatchOnline(
    int party_id, const std::vector<GroupElement>& inputs, 
    SigmaBatchKey& key, Peer* player);

// 批量 ReLU
ReLUBatchKey reluBatchOffline(int party_id, int batch_size, int bits, Peer* player);
std::vector<GroupElement> reluBatchOnline(
    int party_id, const std::vector<GroupElement>& inputs,
    ReLUBatchKey& key, Peer* player);

// 批量 DPF
std::vector<DPFKey> generateDPFBatch(
    int party_id, const DPFBatchSpec& spec, Peer* player);

// 批量打开
std::vector<uint64_t> openValuesBatch(
    int party_id, const std::vector<GroupElement>& shares, Peer* player);
```

### A.2 性能指标

| 指标 | 单位 | 目标 |
|------|------|------|
| Sigmoid 轮次 | rounds | ≤15 |
| Google-30 延迟 (WAN) | 秒 | <30 |
| 批量加速比 | × | >50 |
| Offline 时间 | 秒 | <60 |

### A.3 调试技巧

1. **轮次追踪**:
   ```cpp
   int round_before = player->getRoundCount();
   // ... operation ...
   int round_after = player->getRoundCount();
   std::cout << "Rounds: " << (round_after - round_before) << "\n";
   ```

2. **通信量追踪**:
   ```cpp
   size_t bytes_before = player->getTotalBytesSent();
   // ... operation ...
   size_t bytes_after = player->getTotalBytesSent();
   std::cout << "Comm: " << (bytes_after - bytes_before) << " bytes\n";
   ```

3. **正确性验证**:
   ```cpp
   // 对比新旧实现
   u8 result_old = BitDec(...).front();
   u8 result_new = piecewiseCompareOnline(...);
   assert(result_old == result_new);
   ```

---

**文档版本**: v1.0  
**最后更新**: 2024-01-15  
**作者**: TDD Implementation Team
