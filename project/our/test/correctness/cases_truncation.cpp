#include "cases.h"

#include "exact_truncation.h"

#include <cstdint>
#include <stdexcept>

void check_exact_truncation(ResultLog& log) {
    constexpr int kBits = 8;
    constexpr int kShift = 2;
    bool all_values_exact = true;
    uint64_t first_offline_bytes = 0;
    uint64_t first_online_bytes = 0;
    int32_t first_online_rounds = 0;
    bool rejected_reuse = false;

    for (uint64_t value = 0; value < (uint64_t(1) << kBits); ++value) {
        const uint64_t server_share = (value * 37 + 0x19) & 0xff;
        const GroupElement input = split_share(value, kBits, server_share);
        const uint64_t offline_before =
            peer->bytesSent + peer->bytesReceived;
        auto key = segfss::exactTruncateOffline(
            party, kBits, kShift, peer);
        if (value == 0) {
            first_offline_bytes =
                peer->bytesSent + peer->bytesReceived - offline_before;
        }

        const uint64_t online_before =
            peer->bytesSent + peer->bytesReceived;
        const int32_t rounds_before = numRounds;
        GroupElement output = segfss::exactTruncate(
            party, input, key, peer);
        if (value == 0) {
            first_online_bytes =
                peer->bytesSent + peer->bytesReceived - online_before;
            first_online_rounds = numRounds - rounds_before;
            try {
                (void)segfss::exactTruncate(party, input, key, peer);
            } catch (const std::logic_error&) {
                rejected_reuse = true;
            }
        }

        reconstruct(&output);
        all_values_exact = all_values_exact &&
                           output.value == (value >> kShift);
        segfss::freeExactTruncationKey(key);
    }

    log.check_local("exact DPF truncation covers the full 8-bit ring",
                    all_values_exact ? 1 : 0, 1);
    log.check_local("exact truncation DPF key generation is offline",
                    first_offline_bytes > 0 ? 1 : 0, 1);
    log.check_local("SIGMA truncation opens its input mask and comparison bit",
                    static_cast<uint64_t>(first_online_rounds), 2);
    log.check_local("SIGMA truncation sends an 8-bit mask and one bit",
                    first_online_bytes, 4);
    log.check_local("exact truncation rejects key reuse",
                    rejected_reuse ? 1 : 0, 1);

    bool wide_values_exact = true;
    for (const int bits : {32, 64}) {
        const int shift = bits == 32 ? 12 : 16;
        const uint64_t values[] = {
            0, 1, (uint64_t(1) << shift) - 1,
            uint64_t(1) << (bits - 1),
            (uint64_t(1) << (bits - 1)) - 1,
            bits == 64 ? UINT64_MAX : UINT32_MAX,
        };
        for (const uint64_t value : values) {
            const GroupElement input = split_share(
                value, bits, value ^ 0x69a4db721ce5f083ULL);
            auto key = segfss::exactTruncateOffline(
                party, bits, shift, peer);
            GroupElement output = segfss::exactTruncate(
                party, input, key, peer);
            reconstruct(&output);
            wide_values_exact = wide_values_exact &&
                output.bitsize == bits - shift &&
                output.value == (value >> shift);
            segfss::freeExactTruncationKey(key);
        }
    }
    log.check_local("SIGMA truncation handles signed ring boundaries",
                    wide_values_exact ? 1 : 0, 1);

    bool rejected_large_shift = false;
    try {
        (void)segfss::exactTruncateOffline(party, 32, 25, peer);
    } catch (const std::invalid_argument&) {
        rejected_large_shift = true;
    }
    log.check_local("SIGMA truncation checks the DPF depth limit",
                    rejected_large_shift ? 1 : 0, 1);
}
