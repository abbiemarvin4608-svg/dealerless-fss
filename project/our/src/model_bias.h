#pragma once

#include <cstddef>
#include <cstdint>
#include <vector>

namespace segfss::model {

// Model-side constants are prepared in the same 16-bit ring as the SCI
// tensors.  The input vectors are signed host integers; the stored values are
// two's-complement residues and are never applied to secret shares locally.
struct ExactBiasMaterial {
    std::vector<uint16_t> gate0;
    std::vector<uint16_t> gate1;
    uint64_t model_version = 0;

    bool validFor(uint64_t version) const noexcept {
        return model_version == version && gate0.size() == 100 &&
               gate1.size() == 100;
    }
    std::size_t residentBytes() const noexcept {
        return (gate0.size() + gate1.size()) * sizeof(uint16_t);
    }
};

// SCI's model-constant rule: signed division by 8 truncates toward zero,
// then the result is reduced into Z_(2^16).  This is deliberately separate
// from secret-share truncation.
int16_t divideBiasBy8(int64_t value) noexcept;
uint16_t encodeBias(int64_t value) noexcept;

ExactBiasMaterial prepareExactBias(const std::vector<int64_t>& gate0,
                                   const std::vector<int64_t>& gate1,
                                   uint64_t model_version);

// Broadcast prepared constants over a query batch without changing their
// 16-bit residues.  Ownership is unchanged: these are model-owner constants,
// not an authorization to disclose the model or truncate arbitrary shares.
std::vector<uint16_t> broadcastBias(const std::vector<uint16_t>& bias,
                                    std::size_t batch);

}  // namespace segfss::model
