#include "folded_native.h"
#include <stdexcept>

namespace segfss::folded {
SciTruncationMaterial prepareInputExtension(
    int role, int count, Peer* player, SegmentPolicy policy,
    const DcfGenerationOptions* options, ActivationMetrics* metrics) {
    ActivationStageScope stage(metrics, true, ActivationStage::input_adapter, player);
    const reference::Contract contract{16, 32, 0, reference::SciCall::convert,
        true, reference::NumericMode::sci_compatible};
    auto config = makeSciTruncationConfig(contract, DcfBackend::segmented, policy);
    return sciTruncateOfflineBatch(role, count, config, player, options);
}
std::vector<GroupElement> nativeInputToQ12(
    int role, const uint64_t* input, int count, SciTruncationMaterial& extension,
    Peer* player, SciTruncationWorkspace* workspace, ActivationMetrics* metrics,
    std::vector<u8>* input_signs) {
    if (!input || count < 1 || extension.count != count)
        throw std::invalid_argument("folded native input adapter shape mismatch");
    ActivationStageScope stage(metrics, false, ActivationStage::input_adapter, player);
    std::vector<GroupElement> inputs;
    inputs.reserve(count);
    for (int index = 0; index < count; ++index) inputs.emplace_back(input[index], 16);
    auto result = sciTruncateOnlineBatch(role, inputs, extension.config, extension,
        player, workspace, input_signs);
    for (auto& value : result) value = value * uint64_t(32);
    return result;
}
}  // namespace segfss::folded
