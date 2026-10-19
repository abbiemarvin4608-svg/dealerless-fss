// Test cases for batch sigmoid operations
#include "cases.h"
#include "activation.h"
#include <vector>
#include <cmath>

void check_sigmoid_batch(ResultLog& log) {
    constexpr int kBits = 32;
    constexpr int kFracBits = 12;
    constexpr int kQ = 10;
    constexpr int kBatchSize = 3;

    // Test batch sigmoid on multiple inputs
    std::vector<int64_t> test_inputs = {
        static_cast<int64_t>(uint64_t(1) << kFracBits),    // 1.0
        static_cast<int64_t>(uint64_t(2) << kFracBits),    // 2.0
        0                                                   // 0.0
    };

    std::vector<double> expected_outputs = {
        0.73095703125,  // sigmoid(1.0)
        0.88079833984375, // sigmoid(2.0)
        0.5             // sigmoid(0.0)
    };

    // Generate batch keys
    std::vector<segfss::SigmaActivationKey> batch_keys =
        segfss::sigmaBatchOffline(party, kBatchSize, kBits, kFracBits, kQ, peer);

    // Prepare shared inputs
    std::vector<GroupElement> shared_inputs;
    for (int i = 0; i < kBatchSize; i++) {
        shared_inputs.push_back(split_share(
            static_cast<uint64_t>(test_inputs[i]), kBits, 0));
    }

    // Batch sigmoid computation
    std::vector<GroupElement> outputs =
        segfss::sigmaBatchOnline(party, shared_inputs, batch_keys, peer);

    // Reconstruct and verify
    for (int i = 0; i < kBatchSize; i++) {
        reconstruct(&outputs[i]);
        const double value =
            signed_from_twos(outputs[i].value, kBits) /
            static_cast<double>(uint64_t(1) << kFracBits);

        const bool within_tolerance = std::abs(value - expected_outputs[i]) < 0.004;
        log.check_local("batch sigmoid test " + std::to_string(i), within_tolerance, 1);
    }

    // Free keys
    for (auto& key : batch_keys) {
        segfss::freeSigmaActivationKey(key);
    }
}
