// Temporary debug version of activation test
#include "cases.h"
#include "activation.h"

#include <cmath>
#include <cstdint>
#include <stdexcept>
#include <iostream>

void check_activation_debug(ResultLog& log) {
    constexpr int kBits = 32;
    constexpr int kFracBits = 12;
    constexpr int kQ = 10;

    auto secure_sigmoid = [&](int64_t encoded_value, uint64_t server_share) {
        const GroupElement shared_input = split_share(
            static_cast<uint64_t>(encoded_value), kBits, server_share);

        std::cout << "[DEBUG] Input encoded: 0x" << std::hex << static_cast<uint64_t>(encoded_value)
                  << ", server_share: 0x" << server_share << std::dec << std::endl;
        std::cout << "[DEBUG] shared_input.value: 0x" << std::hex << shared_input.value << std::dec << std::endl;

        segfss::SigmaActivationKey key =
            segfss::sigmaOffline(party, kBits, kFracBits, kQ, peer);
        GroupElement output =
            segfss::sigmaOnline(party, shared_input, key, peer);

        std::cout << "[DEBUG] output before reconstruct: 0x" << std::hex << output.value << std::dec << std::endl;

        reconstruct(&output);

        std::cout << "[DEBUG] output after reconstruct: 0x" << std::hex << output.value << std::dec << std::endl;

        const double value =
            signed_from_twos(output.value, kBits) /
            static_cast<double>(uint64_t(1) << kFracBits);

        std::cout << "[DEBUG] Final value: " << value << ", expected ~0.269" << std::endl;

        segfss::freeSigmaActivationKey(key);
        return value;
    };

    std::cout << "\n=== Testing sigmoid(-1) ===" << std::endl;
    const double negative_one = secure_sigmoid(
        -static_cast<int64_t>(uint64_t(1) << kFracBits), 0x2468ACE0ULL);

    std::cout << "Result: " << negative_one << ", tolerance check: "
              << (std::abs(negative_one - 0.26904296875) < 0.004) << std::endl;

    log.check_local("DEBUG sigmoid(-1) tolerance",
                    std::abs(negative_one - 0.26904296875) < 0.004, 1);
}
