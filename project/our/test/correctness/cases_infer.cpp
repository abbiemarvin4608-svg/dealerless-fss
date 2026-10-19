#include "cases.h"

#include "gen.h"
#include "mpc/api.h"

#include <string>
#include <vector>

// Correctness for Pi_Infer: the cross-segment scan that combines per-segment
// (L_j, e_j) shares into the final comparison share of 1{x < r}.
//
// Comparison decomposition (paper Lemma):
//   1{x < r} = XOR_j ( E_j AND L_j ),   E_j = AND_{i<j} e_i
// where segments run most-significant first, L_j = 1{x^(j) < r^(j)}, and
// e_j = 1{x^(j) == r^(j)}. We build Boolean XOR-shares of the threshold
// digits directly (isolating Pi_Infer from Pi_Split's networking), generate
// per-segment cGGM keys, evaluate at public x, then scan and reconstruct.

using segfss::SegmentLayout;
using segfss::SplitResult;
using segfss::SegmentKeys;
using segfss::SegmentEvalResult;

namespace {

// Build a multi-segment SplitResult carrying Boolean XOR-shares of threshold r.
// The full ell-bit value is XOR-shared (server_share vs r^server_share), then
// sliced per segment MS-first — identical distribution to Pi_Split's output.
SplitResult make_segments(int ell, int nseg, uint64_t r, uint64_t server_share) {
    SegmentLayout layout(ell, nseg);
    SplitResult sr;
    sr.layout = layout;
    sr.segment_bits.resize(nseg);

    const uint64_t local =
        (party == SERVER) ? server_share : (r ^ server_share);
    for (int j = 0; j < nseg; j++) {
        const int width = layout.width[j];
        const int shift = layout.shift[j];
        const uint64_t digit_share =
            (local >> shift) & ((uint64_t(1) << width) - 1);
        sr.segment_bits[j] = msb_bits_from_value(digit_share, width);
    }
    return sr;
}

// Public segment digit of x (MS-first bits) for segment j.
std::vector<u8> segment_digit_bits(uint64_t x, const SegmentLayout& layout,
                                   int j) {
    const int width = layout.width[j];
    const int shift = layout.shift[j];
    const uint64_t digit = (x >> shift) & ((uint64_t(1) << width) - 1);
    return msb_bits_from_value(digit, width);
}

// Sweep every x in [0, 2^ell) for a fixed threshold r and check the
// reconstructed comparison share equals 1{x < r} for all of them.
void check_infer_exhaustive(ResultLog& log, int ell, int nseg, uint64_t r,
                            uint64_t server_share) {
    SegmentLayout layout(ell, nseg);
    SplitResult sr = make_segments(ell, nseg, r, server_share);
    SegmentKeys keys = segfss::genSegmentKeys(party, sr);

    const uint64_t domain = uint64_t(1) << ell;
    bool all_ok = true;
    for (uint64_t x = 0; x < domain; x++) {
        std::vector<std::vector<u8>> xhat_segments(nseg);
        for (int j = 0; j < nseg; j++) {
            xhat_segments[j] = segment_digit_bits(x, layout, j);
        }
        std::vector<SegmentEvalResult> evs =
            segfss::evalAllSegments(party, keys, xhat_segments);

        u8 cmp_share = segfss::inferComparison(party, evs, peer);
        reconstruct(&cmp_share);
        const u8 want = static_cast<u8>(x < r ? 1 : 0);
        all_ok = all_ok && ((cmp_share & 1) == want);
    }

    const std::string tag = "ell=" + std::to_string(ell) +
                            " nseg=" + std::to_string(nseg) +
                            " r=" + std::to_string(r);
    // Feed one-sided so XOR reconstruction yields the SERVER's all_ok.
    const u8 share = (party == SERVER) ? static_cast<u8>(all_ok ? 1 : 0) : u8(0);
    log.check_bit("infer 1{x<r} " + tag, share, u8(1));
}

// End-to-end pipeline over the real networked Pi_Split: decompose a shared
// threshold with split(), generate keys, then check 1{x<r} for a few public x.
// This validates the split->gen handoff (bit ordering + share convention) that
// the direct-construction sweeps above bypass.
void check_pipeline_e2e(ResultLog& log, int ell, int nseg, uint64_t r,
                        uint64_t server_share, const std::vector<uint64_t>& xs) {
    SegmentLayout layout(ell, nseg);
    GroupElement r_share = split_share(r, ell, server_share);
    SplitResult sr = segfss::split(party, r_share, layout, peer);
    SegmentKeys keys = segfss::genSegmentKeys(party, sr);

    bool all_ok = true;
    for (uint64_t x : xs) {
        std::vector<std::vector<u8>> xhat_segments(nseg);
        for (int j = 0; j < nseg; j++) {
            xhat_segments[j] = segment_digit_bits(x, layout, j);
        }
        std::vector<SegmentEvalResult> evs =
            segfss::evalAllSegments(party, keys, xhat_segments);
        u8 cmp_share = segfss::inferComparison(party, evs, peer);
        reconstruct(&cmp_share);
        const u8 want = static_cast<u8>(x < r ? 1 : 0);
        all_ok = all_ok && ((cmp_share & 1) == want);
    }

    const std::string tag = "ell=" + std::to_string(ell) +
                            " nseg=" + std::to_string(nseg) +
                            " r=" + std::to_string(r);
    const u8 share = (party == SERVER) ? static_cast<u8>(all_ok ? 1 : 0) : u8(0);
    log.check_bit("pipeline 1{x<r} " + tag, share, u8(1));
}

}  // namespace

void check_infer(ResultLog& log) {
    // Edge case: nseg=1 (monolithic, no segmentation). E_0=1, so result = L_0.
    check_infer_exhaustive(log, 8, 1, 0x42, 0x13);
    check_infer_exhaustive(log, 10, 1, 0x200, 0x155);

    // Exhaustive over small domains, multiple segment counts and thresholds.
    check_infer_exhaustive(log, 8, 2, 0, 0x37);
    check_infer_exhaustive(log, 8, 2, 1, 0x37);
    check_infer_exhaustive(log, 8, 2, 0b10110101, 0x37);
    check_infer_exhaustive(log, 8, 2, 0xFF, 0x91);
    check_infer_exhaustive(log, 8, 4, 0b11001010, 0x91);
    check_infer_exhaustive(log, 8, 4, 0x80, 0x5A);
    check_infer_exhaustive(log, 10, 5, 0x2AB, 0x123);
    check_infer_exhaustive(log, 12, 3, 0xABC, 0x123);
    check_infer_exhaustive(log, 12, 4, 0x800, 0x1F0);

    // End-to-end integration: real networked split() → gen → eval → infer.
    // Validates the split→gen handoff that direct-construction bypasses.
    check_pipeline_e2e(log, 16, 4, 0xBEEF, 0x1357, {0, 1, 0xBEEE, 0xBEEF, 0xBEF0, 0xFFFF});
    check_pipeline_e2e(log, 20, 5, 0x12345, 0xABCDE, {0, 0x12344, 0x12345, 0x12346, 0xFFFFF});
    check_pipeline_e2e(log, 32, 8, 0xDEADBEEF, 0x12345678,
                       {0, 1, 0xDEADBEEE, 0xDEADBEEF, 0xDEADBEF0, 0xFFFFFFFF});
}
