#pragma once

#include "activation_metrics.h"
#include "sci_truncation.h"

namespace segfss::folded {
SciTruncationMaterial prepareInputExtension(
    int role, int count, Peer* player, SegmentPolicy segment_policy,
    const DcfGenerationOptions* generation_options, ActivationMetrics* metrics);
std::vector<GroupElement> nativeInputToQ12(
    int role, const uint64_t* input, int count, SciTruncationMaterial& extension,
    Peer* player, SciTruncationWorkspace* workspace, ActivationMetrics* metrics,
    std::vector<u8>* input_signs = nullptr);

bool tanhNativeAvailable() noexcept;
std::vector<GroupElement> tanhCoreInput(const std::vector<GroupElement>& q12_input);
SciTruncationMaterial prepareTanhOutput(
    int role, int count, Peer* player, SegmentPolicy segment_policy,
    const DcfGenerationOptions* generation_options, ActivationMetrics* metrics);
std::vector<GroupElement> tanhNativeOutput(
    int role, const std::vector<GroupElement>& sigmoid_of_doubled_input_q12,
    SciTruncationMaterial& output_scale, Peer* player,
    SciTruncationWorkspace* workspace, ActivationMetrics* metrics);
}  // namespace segfss::folded
