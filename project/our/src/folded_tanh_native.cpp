#include "folded_native.h"
#include <stdexcept>

namespace segfss::folded {
namespace {
SciTruncationConfig outputConfig(SegmentPolicy policy) {
    const reference::Contract contract{32, 27, 5,
        reference::SciCall::truncate_and_reduce, true,
        reference::NumericMode::sci_compatible};
    return makeSciTruncationConfig(contract, DcfBackend::segmented, policy);
}
}

bool tanhNativeAvailable() noexcept { return true; }

std::vector<GroupElement> tanhCoreInput(
    const std::vector<GroupElement>& q12_input) {
    if (q12_input.empty())
        throw std::invalid_argument("folded TanH requires a nonempty input");
    std::vector<GroupElement> result;
    result.reserve(q12_input.size());
    for (const auto& value : q12_input) {
        if (value.bitsize != 32)
            throw std::invalid_argument("folded TanH requires ring32/Q12 input");
        result.push_back(value * uint64_t(2));
    }
    return result;
}

SciTruncationMaterial prepareTanhOutput(
    int role, int count, Peer* player, SegmentPolicy policy,
    const DcfGenerationOptions* options, ActivationMetrics* metrics) {
    ActivationStageScope stage(metrics, true, ActivationStage::output_adapter, player);
    return sciTruncateOfflineBatch(role, count, outputConfig(policy), player, options);
}

std::vector<GroupElement> tanhNativeOutput(
    int role, const std::vector<GroupElement>& sigmoid_of_doubled_input_q12,
    SciTruncationMaterial& output_scale, Peer* player,
    SciTruncationWorkspace* workspace, ActivationMetrics* metrics) {
    if (sigmoid_of_doubled_input_q12.empty() ||
        sigmoid_of_doubled_input_q12.size() != static_cast<std::size_t>(output_scale.count))
        throw std::invalid_argument("folded TanH output adapter shape mismatch");
    const auto config = outputConfig(output_scale.config.segment_policy);
    ActivationStageScope stage(metrics, false, ActivationStage::output_adapter, player);
    std::vector<GroupElement> centered;
    centered.reserve(sigmoid_of_doubled_input_q12.size());
    const GroupElement one(role == SERVER ? 4096 : 0, 32);
    for (const auto& value : sigmoid_of_doubled_input_q12) {
        if (value.bitsize != 32)
            throw std::invalid_argument("folded TanH sigmoid output must use ring32");
        centered.push_back(value * uint64_t(2) - one);
    }
    auto result = sciTruncateOnlineBatch(
        role, centered, config, output_scale, player, workspace);
    for (auto& value : result) value = GroupElement(value.value, 16);
    return result;
}
}  // namespace segfss::folded
