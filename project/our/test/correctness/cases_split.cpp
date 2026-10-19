#include "cases.h"

#include "mpc/api.h"

#include <string>
#include <vector>

// Correctness for Pi_Split (Algorithm 1): the ell-bit threshold share is
// bit-decomposed once and sliced into nseg per-segment Boolean shares.
//
// We reconstruct each segment's Boolean bit shares and check they match the
// plaintext bits of the threshold, MS segment first, MS bit first.

using segfss::SegmentLayout;
using segfss::SplitResult;

namespace {

// Reconstruct one Boolean share held by both parties (XOR of the two shares).
u8 reveal_bit(u8 share) {
    reconstruct(&share);
    return share;
}

// check_bit reconstructs by XOR. For a PUBLIC boolean that both parties
// compute identically, feed it one-sided (SERVER contributes b, CLIENT 0)
// so the XOR reconstruction yields b rather than b^b=0.
void check_public_bool(ResultLog& log, const std::string& name, bool value) {
    const u8 share = (party == SERVER) ? static_cast<u8>(value) : u8(0);
    log.check_bit(name, share, static_cast<u8>(value ? 1 : 0));
}

// Verify the static layout (widths sum to ell, shifts are consistent).
void check_layout(ResultLog& log, int ell, int nseg) {
    SegmentLayout layout(ell, nseg);
    int total = 0;
    for (int j = 0; j < nseg; j++) total += layout.width[j];
    check_public_bool(log,
                      "layout widths sum to ell (ell=" + std::to_string(ell) +
                          " nseg=" + std::to_string(nseg) + ")",
                      total == ell);

    // Shifts: MS segment first, each segment sits directly above the next.
    bool shifts_ok = true;
    int expected_shift = ell;
    for (int j = 0; j < nseg; j++) {
        expected_shift -= layout.width[j];
        shifts_ok = shifts_ok && (layout.shift[j] == expected_shift);
    }
    check_public_bool(log,
                      "layout shifts consistent (ell=" + std::to_string(ell) +
                          " nseg=" + std::to_string(nseg) + ")",
                      shifts_ok);
}

}  // namespace

void check_split(ResultLog& log) {
    // Layout sanity across representative (ell, nseg) pairs.
    check_layout(log, 32, 1);
    check_layout(log, 32, 2);
    check_layout(log, 32, 4);
    check_layout(log, 32, 8);
    check_layout(log, 30, 4);  // non-divisible: leading segments get extra bit

    // Functional Pi_Split: decompose a known threshold and check every bit.
    struct SplitCase {
        int ell;
        int nseg;
        uint64_t threshold;
        uint64_t server_share;
    };
    const SplitCase cases[] = {
        {8, 2, 0b10110101, 0x37},
        {8, 4, 0b11001010, 0x91},
        {12, 3, 0xABC, 0x123},
        {16, 4, 0xBEEF, 0x1357},
    };

    for (const auto& tc : cases) {
        SegmentLayout layout(tc.ell, tc.nseg);
        GroupElement r_share =
            split_share(tc.threshold, tc.ell, tc.server_share);

        SplitResult sr = segfss::split(party, r_share, layout, peer);

        // Reconstruct and compare against plaintext bits (MS first).
        int bit_offset = 0;
        bool all_ok = true;
        for (int j = 0; j < tc.nseg; j++) {
            for (int k = 0; k < layout.width[j]; k++) {
                const u8 got = reveal_bit(sr.segment_bits[j][k]);
                const int global_bit = bit_offset + k;  // 0 = MSB of the word
                const u8 want = static_cast<u8>(
                    (tc.threshold >> (tc.ell - 1 - global_bit)) & 1);
                all_ok = all_ok && (got == want);
            }
            bit_offset += layout.width[j];
        }
        check_public_bool(log,
                          "split bits match (ell=" + std::to_string(tc.ell) +
                              " nseg=" + std::to_string(tc.nseg) +
                              " r=" + std::to_string(tc.threshold) + ")",
                          all_ok);
    }
}
