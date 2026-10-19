#pragma once

#include "sci_numeric_oracle.h"
#include "segmented_dcf.h"
#include "ring_packing.h"

namespace segfss {

enum class DcfBackend { full_tree, segmented };

struct SciTruncationConfig {
    reference::Contract contract{1, 1, 0, reference::SciCall::convert};
    DcfBackend backend = DcfBackend::segmented;
    int low_shift_nseg = 0;
    int sign_nseg = 0;
    int wrap_nseg = 0;
    SegmentPolicy segment_policy = SegmentPolicy::explicit_counts;
    // Snapshot of SEGFSS_FUSE_SIGN_WRAP, including contracts that use the
    // original path. A material cannot be consumed under a different flag.
    bool fuse_sign_wrap = false;
};

struct SciTruncationDomains {
    int low_shift_bits = 0;
    int sign_bits = 0;
    int wrap_bits = 0;
    bool need_sign = false;
    bool toward_zero = false;
    bool local_only = false;
    bool fused_sign_wrap = false;
    int correction_count = 0;
};

// max_segments is a policy limit, not a requested nseg. The helper explicitly
// records min(max_segments, domain_bits) for each active comparison domain.
// Hand-written configs instead require exact legal counts, with zero for an
// unused domain. Full-tree uses nseg=1 in each active domain.
SciTruncationConfig makeSciTruncationConfig(
    const reference::Contract& contract, DcfBackend backend, int max_segments);
SciTruncationConfig makeSciTruncationConfig(
    const reference::Contract& contract, DcfBackend backend, SegmentPolicy policy);
SciTruncationDomains sciTruncationDomains(const SciTruncationConfig& config);
bool sameSciTruncationConfig(const SciTruncationConfig& a,
                            const SciTruncationConfig& b);

struct SciTruncationMetrics {
    double mask_preprocessing_ms = 0;
    double comparisons_preprocessing_ms = 0;
    double rounding_preprocessing_ms = 0;
    double masked_input_open_ms = 0;
    double comparisons_online_ms = 0;
    double comparison_scan_ms = 0;
    double rounding_online_ms = 0;
    double masked_bit_conversion_ms = 0;
    std::size_t masked_input_values = 0;
    std::size_t masked_conversion_bits = 0;
    std::size_t online_logical_rounds = 0;
};

struct SciTruncationWorkspace {
    std::vector<std::vector<uint64_t>> queries;
    SegmentedDcfGroupWorkspace comparisons;
};

// Public fields permit the existing trusted local spool to serialize vectors
// and nested DCF materials. No raw pointers or shared key ownership are held.
struct SciTruncationMaterial {
    SciTruncationConfig config;
    int role = 0;
    int count = 0;
    std::vector<uint64_t> input_mask_shares;
    std::vector<uint64_t> mask_high_shares;
    std::vector<u8> mask_sign_bit_shares;
    SegmentedDcfMaterial low_shift;
    SegmentedDcfMaterial sign;
    SegmentedDcfMaterial wrap;
    // Row-major gates: the fused sign/wrap product, then toward-zero
    // rounding, with inactive gates omitted. Both share one online opening.
    BooleanAndPreprocessing rounding;
    std::vector<u8> conversion_mask_bits;
    std::vector<uint64_t> conversion_mask_arithmetic;
    bool consumed = true;
    bool released = false;
    SciTruncationMetrics metrics;

    SciTruncationMaterial() = default;
    SciTruncationMaterial(const SciTruncationMaterial&) = delete;
    SciTruncationMaterial& operator=(const SciTruncationMaterial&) = delete;
    SciTruncationMaterial(SciTruncationMaterial&& other) noexcept;
    SciTruncationMaterial& operator=(SciTruncationMaterial&& other) noexcept;
};

SciTruncationMaterial sciTruncateOfflineBatch(
    int role, int count, const SciTruncationConfig& config, Peer* player,
    const DcfGenerationOptions* generation_options = nullptr);
SciTruncationMaterial sciTruncateOffline(
    int role, const SciTruncationConfig& config, Peer* player);

// expected is checked against the stored numeric mode, rings, operation,
// signedness, shift, backend and domain layouts before any protocol activity.
// input_signs optionally receives the input's secret XOR sign shares when the
// contract already requires a signed comparison. It adds no openings or stock.
std::vector<GroupElement> sciTruncateOnlineBatch(
    int role, const std::vector<GroupElement>& inputs,
    const SciTruncationConfig& expected, SciTruncationMaterial& material,
    Peer* player, SciTruncationWorkspace* workspace = nullptr,
    std::vector<u8>* input_signs = nullptr);
GroupElement sciTruncateOnline(
    int role, GroupElement input, const SciTruncationConfig& expected,
    SciTruncationMaterial& material, Peer* player);

// Concatenates same-config materials into one online batch. Each pointer must
// appear once and inputs contain sum(material.count) rows in pointer order.
// Consumption happens for every donor before the first network operation.
std::vector<GroupElement> sciTruncateOnlineBatch(
    int role, const std::vector<GroupElement>& inputs,
    const SciTruncationConfig& expected,
    const std::vector<SciTruncationMaterial*>& materials, Peer* player,
    SciTruncationMetrics* batch_metrics = nullptr);

void releaseSciTruncationMaterial(SciTruncationMaterial& material);

}  // namespace segfss
