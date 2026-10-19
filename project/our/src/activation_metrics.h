#pragma once

#include <chrono>
#include <cstdint>
#include "mpc/comms.h"
#include "utils/round_trace.h"

namespace segfss {
enum class ActivationStage : unsigned {
    input_adapter, shared_sign_prepare, shared_bitxa_prepare, sign, abs_bitxa,
    clip_compare, clip_bitxa, index_truncation, lut_mask, lut_keygen,
    lut_materialize, lut_tensor_and, lut_leaf_b2a, lut_payload, lut_open,
    lut_product, restore_bitxa, output_adapter, count
};
inline constexpr unsigned kActivationStageCount = unsigned(ActivationStage::count);
struct ActivationStageMetric {
    uint64_t elapsed_us = 0, sent_bytes = 0, received_bytes = 0;
    uint64_t begin_us = 0, end_us = 0;
};
inline uint64_t activationStageTimestampUs(std::chrono::steady_clock::time_point time) {
    return std::chrono::duration_cast<std::chrono::microseconds>(time.time_since_epoch()).count();
}
struct ActivationMetrics {
    ActivationStageMetric offline[kActivationStageCount]{}, online[kActivationStageCount]{};
    uint64_t offline_wall_us = 0, online_wall_us = 0;
    uint64_t resident_bytes = 0, peak_working_bytes = 0;
    uint64_t online_generated = 0, online_loaded_bytes = 0;
};
class ActivationStageScope {
    dfss_round::SemanticScope trace_;
    ActivationStageMetric* target_;
    Peer* peer_;
    std::chrono::steady_clock::time_point start_;
    uint64_t sent_ = 0, received_ = 0;
public:
    ActivationStageScope(ActivationMetrics* metrics, bool offline, ActivationStage stage, Peer* peer)
        : trace_("activation_stage", -1, -1, -1, -1, -1, int(stage), metrics != nullptr),
          target_(metrics ? &(offline ? metrics->offline : metrics->online)[unsigned(stage)] : nullptr), peer_(peer) {
        if (target_) {
            start_ = std::chrono::steady_clock::now();
            sent_ = peer_->bytesSent; received_ = peer_->bytesReceived;
        }
    }
    ~ActivationStageScope() {
        if (target_) {
            const auto end = std::chrono::steady_clock::now();
            const auto begin_us = activationStageTimestampUs(start_);
            target_->elapsed_us += std::chrono::duration_cast<std::chrono::microseconds>(
                end - start_).count();
            if (!target_->begin_us || begin_us < target_->begin_us) target_->begin_us = begin_us;
            const auto end_us = activationStageTimestampUs(end);
            if (end_us > target_->end_us) target_->end_us = end_us;
            target_->sent_bytes += peer_->bytesSent - sent_;
            target_->received_bytes += peer_->bytesReceived - received_;
        }
    }
};
}  // namespace segfss
