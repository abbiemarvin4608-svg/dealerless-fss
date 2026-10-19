#include "model_bias.h"

#include <limits>
#include <stdexcept>

namespace segfss::model {

int16_t divideBiasBy8(int64_t value) noexcept {
    // C++ signed division is defined as truncation toward zero.  Keeping the
    // operation in int64_t also handles the int16 boundary inputs safely.
    // The quotient is a residue in Z_(2^16), rather than a host int16_t
    // value.  Converting an out-of-range quotient directly to uint16_t gives
    // the required two's-complement ring reduction without saturation.
    const int64_t quotient = value / 8;
    const uint16_t residue = static_cast<uint16_t>(quotient);
    if (residue <= static_cast<uint16_t>(std::numeric_limits<int16_t>::max())) {
        return static_cast<int16_t>(residue);
    }
    // Convert the upper half explicitly, avoiding an implementation-defined
    // unsigned-to-signed conversion while preserving the ring residue.
    return static_cast<int16_t>(static_cast<int32_t>(residue) - (1 << 16));
}

uint16_t encodeBias(int64_t value) noexcept {
    return static_cast<uint16_t>(divideBiasBy8(value));
}

ExactBiasMaterial prepareExactBias(const std::vector<int64_t>& gate0,
                                   const std::vector<int64_t>& gate1,
                                   uint64_t model_version) {
    if (gate0.size() != 100 || gate1.size() != 100) {
        throw std::invalid_argument("FastGRNN bias vectors must contain 100 values");
    }
    ExactBiasMaterial material;
    material.model_version = model_version;
    material.gate0.reserve(gate0.size());
    material.gate1.reserve(gate1.size());
    for (const auto value : gate0) material.gate0.push_back(encodeBias(value));
    for (const auto value : gate1) material.gate1.push_back(encodeBias(value));
    return material;
}

std::vector<uint16_t> broadcastBias(const std::vector<uint16_t>& bias,
                                    std::size_t batch) {
    if (bias.size() != 100) {
        throw std::invalid_argument("FastGRNN bias vector must contain 100 values");
    }
    std::vector<uint16_t> result;
    result.reserve(bias.size() * batch);
    for (std::size_t i = 0; i < batch; ++i) {
        result.insert(result.end(), bias.begin(), bias.end());
    }
    return result;
}

}  // namespace segfss::model
