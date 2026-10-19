#include "cases.h"
#include "activation.h"

#include <cstdint>
#include <random>
#include <stdexcept>
#include <vector>

namespace {

GroupElement randomized_share(int64_t encoded, int bits,
                              std::mt19937_64& generator) {
    const uint64_t value = twos_from_signed(encoded, bits);
    uint64_t server_share;
    do {
        server_share = reduce_to_bits(generator(), bits);
    } while (server_share == 0 || server_share == value);
    // Both arithmetic shares are nonzero; their sum is the input in the ring.
    return split_share(value, bits, server_share);
}

}  // namespace

void check_activation(ResultLog& log) {
    constexpr int kBits = 32;
    constexpr int kFracBits = segfss::kSigmaDefaultFracBits;
    constexpr int kQ = segfss::kSigmaDefaultQ;
    constexpr int64_t kScale = int64_t(1) << kFracBits;
    constexpr int64_t kBound = 16 * kScale;

    bool rejected_small_ring = false;
    try {
        (void)segfss::makeSigmaDeltaTable(16, 12, kQ);
    } catch (const std::invalid_argument&) {
        rejected_small_ring = true;
    }
    log.check_local("SIGMA rejects a ring without the bound", rejected_small_ring, 1);

    const PublicLUTData table = segfss::makeSigmaDeltaTable(kBits);
    log.check_local("SIGMA q=10 table size", table.values.size(), 1U << kQ);
    log.check_local("SIGMA delta(0)", table.values.front().value, kScale / 2);
    log.check_local("SIGMA delta(bound)", table.values.back().value, 0);
    const GroupElement zero(0, kBits);
    log.check_local("SIGMA sigmoid(0)", segfss::sigmaPlain(zero, kFracBits, kQ).value, kScale / 2);
    log.check_local("SIGMA tanh(0)", segfss::tanhPlain(zero, kFracBits, kQ).value, 0);
    const auto report = segfss::measureSigmaActivationError(kBits, kFracBits, kQ);
    log.check_local("SIGMA sigmoid error bound", report.sigmoid.max_abs < 0.004, 1);
    log.check_local("SIGMA tanh error bound", report.tanh.max_abs < 0.008, 1);

    // The removed raw-opening implementation must not be reachable through
    // its legacy API, including when a caller manually enables the old flag.
    bool rejected_piecewise_offline = false;
    try {
        (void)segfss::piecewiseCompareOffline(party, kBits, 0, 4, peer, true);
    } catch (const std::logic_error&) {
        rejected_piecewise_offline = true;
    }
    log.check_local("unmasked piecewise offline disabled", rejected_piecewise_offline, 1);
    bool rejected_piecewise_online = false;
    try {
        (void)segfss::piecewiseCompareOnline(party, zero, {}, peer);
    } catch (const std::logic_error&) {
        rejected_piecewise_online = true;
    }
    log.check_local("unmasked piecewise online disabled", rejected_piecewise_online, 1);
    segfss::SigmaActivationKey legacy;
    legacy.bits = kBits;
    legacy.use_piecewise_compare = true;
    bool rejected_legacy_mode = false;
    try {
        (void)segfss::sigmaOnline(party, zero, legacy, peer);
    } catch (const std::logic_error&) {
        rejected_legacy_mode = true;
    }
    log.check_local("SIGMA rejects unmasked legacy mode", rejected_legacy_mode, 1);

    std::mt19937_64 generator(0xF553A17ULL);
    std::vector<int64_t> sigmoid_inputs = {
        -2 * kBound, -kBound - 1, -kBound, -kBound + 1,
        -kScale, -1, 0, 1, kScale,
        kBound - 1, kBound, kBound + 1, 2 * kBound,
        -(int64_t(1) << 31), (int64_t(1) << 31) - 1,
    };
    std::uniform_int_distribution<int64_t> random_input(-2 * kBound, 2 * kBound);
    for (int i = 0; i < 4; ++i) {
        sigmoid_inputs.push_back(random_input(generator));
    }
    for (std::size_t i = 0; i < sigmoid_inputs.size(); ++i) {
        const auto input = randomized_share(sigmoid_inputs[i], kBits, generator);
        auto key = segfss::sigmaOffline(party, kBits, peer);
        log.check_local("SIGMA secure BitDec mode[" + std::to_string(i) + "]", key.use_piecewise_compare, 0);
        const auto output = segfss::sigmaOnline(party, input, key, peer);
        const uint64_t expected = segfss::sigmaPlain(
            GroupElement(static_cast<uint64_t>(sigmoid_inputs[i]), kBits), kFracBits, kQ).value;
        log.check_scalar("SIGMA sigmoid exact[" + std::to_string(i) + "]", output, expected);
        segfss::freeSigmaActivationKey(key);
    }

    constexpr int64_t kTanhBound = 8 * kScale;
    std::vector<int64_t> tanh_inputs = {
        -2 * kTanhBound, -kTanhBound - 1, -kTanhBound, -kTanhBound + 1,
        -kScale, 0, kScale,
        kTanhBound - 1, kTanhBound, kTanhBound + 1, 2 * kTanhBound,
    };
    for (int i = 0; i < 4; ++i) {
        tanh_inputs.push_back(random_input(generator));
    }
    for (std::size_t i = 0; i < tanh_inputs.size(); ++i) {
        const auto input = randomized_share(tanh_inputs[i], kBits, generator);
        auto key = segfss::tanhOffline(party, kBits, peer);
        const auto output = segfss::tanhOnline(party, input, key, peer);
        const uint64_t expected = segfss::tanhPlain(
            GroupElement(static_cast<uint64_t>(tanh_inputs[i]), kBits), kFracBits, kQ).value;
        log.check_scalar("SIGMA tanh exact[" + std::to_string(i) + "]", output, expected);
        segfss::freeSigmaActivationKey(key);
    }
}
