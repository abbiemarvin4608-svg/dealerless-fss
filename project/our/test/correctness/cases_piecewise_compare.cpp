#include "cases.h"
#include "activation.h"
#include "gen.h"

#include <cstdint>
#include <vector>

// Test piecewise comparison (replacement for BitDec in sigmoid)
void check_piecewise_compare(ResultLog& log) {
    constexpr int kBits = 32;
    constexpr int kNseg = 4;

    // Test 1: Sign comparison (x < 0)
    {
        const uint64_t threshold = 0;
        segfss::PiecewiseComparisonKey key =
            segfss::piecewiseCompareOffline(party, kBits, threshold, kNseg, peer);

        // Test negative number
        int64_t negative_val = -100;
        uint64_t encoded = static_cast<uint64_t>(negative_val);
        GroupElement input = split_share(encoded, kBits, 0xABCDEF);

        u8 result = segfss::piecewiseCompareOnline(party, input, key, peer);
        u8 reconstructed = result;
        reconstruct_bool(&reconstructed);

        log.check_local("piecewise compare: -100 < 0", reconstructed, 1);
        segfss::freePiecewiseComparisonKey(key);
    }

    // Test 2: Positive number
    {
        const uint64_t threshold = 0;
        segfss::PiecewiseComparisonKey key =
            segfss::piecewiseCompareOffline(party, kBits, threshold, kNseg, peer);

        int64_t positive_val = 100;
        uint64_t encoded = static_cast<uint64_t>(positive_val);
        GroupElement input = split_share(encoded, kBits, 0x123456);

        u8 result = segfss::piecewiseCompareOnline(party, input, key, peer);
        u8 reconstructed = result;
        reconstruct_bool(&reconstructed);

        log.check_local("piecewise compare: 100 < 0", reconstructed, 0);
        segfss::freePiecewiseComparisonKey(key);
    }

    // Test 3: Bound comparison (|x| < 16*scale)
    {
        constexpr int kFracBits = 12;
        const uint64_t bound = 16ULL << kFracBits;

        segfss::PiecewiseComparisonKey key =
            segfss::piecewiseCompareOffline(party, kBits, bound, kNseg, peer);

        // Test value below bound
        uint64_t val_below = 10ULL << kFracBits;
        GroupElement input_below = split_share(val_below, kBits, 0x111111);

        u8 result_below = segfss::piecewiseCompareOnline(party, input_below, key, peer);
        u8 reconstructed_below = result_below;
        reconstruct_bool(&reconstructed_below);

        log.check_local("piecewise compare: 10 < 16", reconstructed_below, 1);

        // Test value above bound
        uint64_t val_above = 20ULL << kFracBits;
        GroupElement input_above = split_share(val_above, kBits, 0x222222);

        u8 result_above = segfss::piecewiseCompareOnline(party, input_above, key, peer);
        u8 reconstructed_above = result_above;
        reconstruct_bool(&reconstructed_above);

        log.check_local("piecewise compare: 20 < 16", reconstructed_above, 0);

        segfss::freePiecewiseComparisonKey(key);
    }

    // Test 4: Round count verification
    if (party == 0) {
        const uint64_t threshold = 0;
        const int rounds_before = peer->rounds;

        segfss::PiecewiseComparisonKey key =
            segfss::piecewiseCompareOffline(party, kBits, threshold, kNseg, peer);
        GroupElement input = split_share(12345, kBits, 0);
        segfss::piecewiseCompareOnline(party, input, key, peer);

        const int rounds_after = peer->rounds;
        const int expected_rounds = 1 + (int)std::ceil(std::log2(kNseg));
        const int actual_rounds = rounds_after - rounds_before;

        // Allow some tolerance for implementation details
        log.check_local("piecewise compare rounds (nseg=4)",
                        actual_rounds <= expected_rounds + 2, 1);

        segfss::freePiecewiseComparisonKey(key);
    }
}
