#include "cases.h"

#include "exact_truncation.h"

#include <cstdint>
#include <stdexcept>

void check_exact_mul_truncation(ResultLog& log) {
    constexpr int kBits = 8;
    constexpr int kShift = 2;
    constexpr uint64_t kModulus = uint64_t(1) << kBits;

    const GroupElement lhs = split_share(3, kBits, 0x52);
    const GroupElement rhs = split_share(kModulus - 5, kBits, 0x19);
    const uint64_t offline_before = peer->bytesSent + peer->bytesReceived;
    auto key = segfss::exactMulTruncateOffline(
        party, kBits, kShift, peer);
    const uint64_t offline_bytes =
        peer->bytesSent + peer->bytesReceived - offline_before;

    const int32_t rounds_before = numRounds;
    GroupElement output = segfss::exactMulTruncate(
        party, lhs, rhs, key, peer);
    const int32_t online_rounds = numRounds - rounds_before;
    reconstruct(&output);

    const int64_t signed_output = signed_from_twos(output.value,
                                                   kBits - kShift);
    log.check_local("exact MulTr includes offline Beaver and DPF keys",
                    offline_bytes > 0 ? 1 : 0, 1);
    log.check_local("exact MulTr returns floor(-15 / 4) = -4",
                    static_cast<uint64_t>(signed_output == -4), 1);
    log.check_local("SIGMA MulTr uses Beaver and two truncation rounds",
                    static_cast<uint64_t>(online_rounds), 3);

    bool rejected_reuse = false;
    try {
        (void)segfss::exactMulTruncate(party, lhs, rhs, key, peer);
    } catch (const std::logic_error&) {
        rejected_reuse = true;
    }
    log.check_local("exact MulTr rejects key reuse",
                    rejected_reuse ? 1 : 0, 1);
    segfss::freeExactMulTruncationKey(key);
}
