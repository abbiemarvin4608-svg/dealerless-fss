# TDD 实施路线图与状态追踪

## 当前状态总结

已完成：
- ✅ 优化方案文档 (OPTIMIZATION_PLAN.md)
- ✅ 分段比较 API 设计 (activation.h)
- ⚠️ 分段比较实现 (activation.cpp) - 部分完成
- ⚠️ 测试用例 (cases_piecewise_compare.cpp) - 已创建但未集成

## 实施策略调整

鉴于代码库的复杂性和现有实现的紧密耦合，建议采用**渐进式优化**而非一次性重构：

### Phase 1: 最小可验证原型 (MVP)
**目标**: 证明分段比较可以工作并减少轮次

1. **完成分段比较实现** (1-2天)
   - [x] API 设计
   - [ ] 修复 activation.cpp 中的头文件引用
   - [ ] 实现 piecewiseCompareOffline
   - [ ] 实现 piecewiseCompareOnline
   - [ ] 集成测试到 test runner

2. **选项开关** (0.5天)
   - [ ] 在 SigmaActivationKey 中添加 use_piecewise_compare 标志
   - [ ] 修改 sigmaOffline/Online 支持两种路径
   - [ ] 保持向后兼容

3. **基准测试** (0.5天)
   - [ ] 轮次对比: BitDec vs 分段比较
   - [ ] 通信量对比
   - [ ] 正确性验证

### Phase 2: 批量化 (2-3周)
**目标**: Google-30 端到端加速

1. **批量打开原语** (3天)
   - [ ] openValuesBatch()
   - [ ] 批量 AND (and_wrapper 批量版本)

2. **批量 Sigmoid** (5天)
   - [ ] SigmaBatchKey 结构
   - [ ] sigmaBatchOffline
   - [ ] sigmaBatchOnline
   - [ ] 测试 & benchmark

3. **Google-30 集成** (2天)
   - [ ] 修改驱动代码使用批量 API
   - [ ] 端到端性能测试

### Phase 3: 其他优化 (1-2周)
1. 批量 ReLU (2天)
2. 批量 DPF 生成 (3天)
3. OpenMP 并行化 (2天)

## 当前障碍

### 技术债务
1. **头文件依赖**: activation.cpp 缺少 segment.h 和 gen.h
2. **API 不匹配**: evalAllSegments 需要验证输入格式
3. **测试集成**: cases_piecewise_compare.cpp 需要注册到 test runner

### 解决方案
- 创建独立分支进行实验
- 使用特性开关 (feature flag) 避免破坏现有功能
- 每个优化都有回退路径

## 风险评估

| 风险 | 影响 | 缓解措施 |
|------|------|----------|
| 破坏现有功能 | 高 | 特性开关 + 完整回归测试 |
| 性能未达预期 | 中 | 保留原实现，基准对比 |
| 实现复杂度过高 | 中 | MVP 先行，渐进优化 |
| 与论文不符 | 低 | 文档更新，说明实现差异 |

## 下一步行动

### 立即可做 (不破坏现有代码)
1. 修复 activation.cpp 头文件
2. 完成 piecewiseCompareOffline/Online 实现
3. 添加独立的单元测试

### 需要协调
1. 决定是否在当前分支还是新分支工作
2. 确认测试通过标准
3. 确认性能基准目标

## 代码修复清单

### activation.cpp
```cpp
// 需要添加的头文件
#include "segment.h"
#include "gen.h"

// 需要修复的函数
PiecewiseComparisonKey piecewiseCompareOffline(...) {
    // 使用现有的 split() 和 genSegmentKeys()
}

u8 piecewiseCompareOnline(...) {
    // 使用现有的 evalAllSegments() 和 inferComparisonPreprocessed()
}
```

### cases.h
```cpp
// 需要添加
void check_piecewise_compare(ResultLog& log);
```

### correctness.cpp
```cpp
// 需要添加测试用例注册
{14, "piecewise comparison", check_piecewise_compare},
```

## 时间估算

- **Phase 1 (MVP)**: 2-3 天
  - 修复编译错误: 4 小时
  - 实现 & 测试: 1 天
  - 基准测试: 0.5 天

- **Phase 2 (批量化)**: 2-3 周
  - 核心实现: 1.5 周
  - 测试 & 调试: 0.5 周
  - 集成 & benchmark: 0.5 周

- **Phase 3 (其他优化)**: 1-2 周

**总计**: 约 4-6 周全职工作

## 建议

鉴于时间和复杂度，我建议：

1. **先完成 Phase 1**: 证明分段比较可行
2. **基准测试**: 如果轮次确实减少 6×，继续
3. **论文优先**: 如果时间紧张，在论文中承诺优化但标注"未实现"
4. **后续工作**: 将批量化作为"Future Work"或实验分支

需要我继续完成 Phase 1 的代码修复吗？
