#include "cases.h"

#include "gen.h"
#include "mpc/api.h"

#include <string>
#include <vector>

// Correctness for Pi_Gen (Algorithm 3): per-segment Boolean-leaf cGGM generation
// (now using our from-scratch cggmGen2pc) plus the Grotto prefix-parity walk.
//
// For a segment digit alpha = r^(j) (secret-shared) and public digit x = xhat^(j),
// evalSegment must return XOR-shares of:
//   L_j = 1{x < alpha}   (accumulated right-sibling tags where x_i = 0)
//   e_j = 1{x == alpha}  (on-path leaf control bit)
// We verify both exhaustively over small segment widths.

using segfss::SegmentKeys;
using segfss::SegmentLayout;
using segfss::SegmentEvalResult;
using segfss::SplitResult;

namespace {

// Reconstruct one Boolean XOR share held across the two parties.
u8 reveal_bit(u8 share) {
    reconstruct(&share);
    return share;
}

// Assert that a locally-computed predicate is TRUE on both parties. Feed the
// SERVER's bit one-sided (CLIENT contributes 0) so XOR reconstruction yields the
// SERVER's value, then compare against the constant 1. (Comparing against the
// value itself would be vacuous.)
void expect_true(ResultLog& log, const std::string& name, bool value) {
    const u8 share = (party == SERVER) ? static_cast<u8>(value ? 1 : 0) : u8(0);
    log.check_bit(name, share, u8(1));
}

// Build a single-segment SplitResult carrying Boolean shares of alpha (MS bit
// first), so we can drive genSegmentKeys / evalSegment directly.
SplitResult make_single_segment(int width, uint64_t alpha,
                                uint64_t server_alpha_share) {
    SplitResult sr;
    sr.layout = SegmentLayout(width, 1);
    // Override to an exact single segment of the requested width.
    sr.layout.ell = width;
    sr.layout.nseg = 1;
    sr.layout.width = {width};
    sr.layout.shift = {0};
    sr.segment_bits.resize(1);
    sr.segment_bits[0] = split_bit_share(alpha, width, server_alpha_share);
    return sr;
}

// Exhaustively verify evalSegment for a given (width, alpha) over all x.
void check_segment_exhaustive(ResultLog& log, int width, uint64_t alpha,
                              uint64_t server_alpha_share) {
    SplitResult sr = make_single_segment(width, alpha, server_alpha_share);
    SegmentKeys keys = segfss::genSegmentKeys(party, sr);

    const uint64_t domain = uint64_t(1) << width;
    bool L_ok = true;
    bool e_ok = true;
    for (uint64_t x = 0; x < domain; x++) {
        const std::vector<u8> xbits = msb_bits_from_value(x, width);
        SegmentEvalResult ev = segfss::evalSegment(party, keys.keys[0], xbits);

        const u8 got_L = reveal_bit(ev.L_j);
        const u8 got_e = reveal_bit(ev.e_j);
        const u8 want_L = static_cast<u8>(x < alpha ? 1 : 0);
        const u8 want_e = static_cast<u8>(x == alpha ? 1 : 0);
        L_ok = L_ok && (got_L == want_L);
        e_ok = e_ok && (got_e == want_e);
    }

    const std::string tag = "w=" + std::to_string(width) +
                            " alpha=" + std::to_string(alpha);
    expect_true(log, "gen L_j=1{x<alpha} " + tag, L_ok);
    expect_true(log, "gen e_j=1{x==alpha} " + tag, e_ok);
}

void check_batched_hash_work(ResultLog& log) {
    const std::vector<int> depths{1, 2, 4};
    const std::vector<uint64_t> alpha{0, 2, 11};
    const std::vector<uint64_t> server_shares{1, 1, 5};
    std::vector<std::vector<u8>> segments;
    for (std::size_t j = 0; j < depths.size(); ++j) {
        segments.push_back(split_bit_share(alpha[j], depths[j],
                                           server_shares[j]));
    }

    segfss::CggmGenerationStats stats;
    const auto keys = segfss::cggmGen2pcBatch(party, segments, nullptr,
                                               &stats);
    log.check_local("batched CGGM internal hashes", stats.internal_hashes, 8);
    // Explicit Peer openings no longer update the legacy global numRounds.
    // This is the inventory of XOR opening batches; OT dependencies and
    // critical-path rounds are measured separately by the production trace.
    log.check_local("batched CGGM XOR opening batches",
                    3 + stats.ef_openings + stats.leaf_cw_openings, 9);
    for (std::size_t j = 0; j < depths.size(); ++j) {
        const uint64_t domain = uint64_t(1) << depths[j];
        for (uint64_t x = 0; x < domain; ++x) {
            const auto bits = msb_bits_from_value(x, depths[j]);
            auto ev = segfss::cggmEval(keys[j], bits);
            reconstruct(&ev.L);
            reconstruct(&ev.e);
            log.check_local("batched CGGM less-than", ev.L, x < alpha[j]);
            log.check_local("batched CGGM equality", ev.e, x == alpha[j]);
        }
    }
}

void check_parallel_batch(ResultLog& log) {
    constexpr int kSegments = 8;
    constexpr int kDepth = 9;
    std::vector<std::vector<u8>> segments;
    for (int j = 0; j < kSegments; ++j) {
        const uint64_t alpha = static_cast<uint64_t>(j * 57 + 9);
        segments.push_back(split_bit_share(alpha, kDepth,
                                           static_cast<uint64_t>(j * 31 + 3)));
    }

    segfss::CggmGenerationStats stats;
    const auto keys = segfss::cggmGen2pcBatch(party, segments, nullptr,
                                               &stats);
    log.check_local("parallel CGGM internal hashes", stats.internal_hashes,
                    kSegments * 255);
    bool all_ok = true;
    for (int j = 0; j < kSegments; ++j) {
        const uint64_t alpha = static_cast<uint64_t>(j * 57 + 9);
        for (uint64_t x : {uint64_t(0), alpha - 1, alpha, alpha + 1,
                           uint64_t(511)}) {
            const auto bits = msb_bits_from_value(x, kDepth);
            auto ev = segfss::cggmEval(keys[j], bits);
            reconstruct(&ev.L);
            reconstruct(&ev.e);
            all_ok = all_ok && ev.L == (x < alpha) && ev.e == (x == alpha);
        }
    }
    log.check_local("parallel CGGM sampled correctness", all_ok, 1);
}

void check_generation_ablation(ResultLog& log, const char* mode, bool merge, bool reuse) {
    setenv("SEGFSS_BIT_BLOCK_MODE", mode, 1);
    setenv("SEGFSS_MERGE_LEAF_CW", merge ? "1" : "0", 1);
    segfss::CggmGenerationWorkspace workspace;
    const std::vector<int> depths{1, 2, 4};
    const std::vector<uint64_t> targets{0, 2, 11};
    std::vector<std::vector<u8>> shares;
    for (std::size_t j = 0; j < depths.size(); ++j)
        shares.push_back(split_bit_share(targets[j], depths[j], 5 & ((1 << depths[j]) - 1)));
    for (int repeat = 0; repeat < 2; ++repeat) {
        segfss::CggmGenerationStats stats;
        auto keys = segfss::cggmGen2pcBatch(party, shares, nullptr, &stats, peer, 1,
                                          reuse ? &workspace : nullptr);
        log.check_local("caller products cover tree and leaf", stats.bit_block_products, 7);
        const bool direct = std::string(mode).find("direct_ot") == 0;
        log.check_local("caller temporary triples", stats.temporary_triples, direct ? 0 : 7);
        log.check_local("caller leaf CW openings", stats.leaf_cw_openings, merge ? 1 : 2);
        if (reuse && repeat) log.check_local("same-shape generation reuses capacity", stats.workspace_capacity_growths, 0);
        bool correct = true;
        for (std::size_t j = 0; j < keys.size(); ++j) {
            for (uint64_t x = 0; x < (uint64_t(1) << depths[j]); ++x) {
                auto value = segfss::cggmEval(keys[j], msb_bits_from_value(x, depths[j]));
                reconstruct(&value.L); reconstruct(&value.e);
                correct = correct && value.L == (x < targets[j]) && value.e == (x == targets[j]);
            }
        }
        log.check_local("generation ablation full-domain comparison", correct, 1);
    }
}

}  // namespace

void check_gen(ResultLog& log) {
    check_batched_hash_work(log);
    check_parallel_batch(log);
    for (const char* mode : {"legacy", "block_ot", "direct_ot_legacy", "direct_ot"})
        check_generation_ablation(log, mode, false, false);
    check_generation_ablation(log, "direct_ot", true, false);
    check_generation_ablation(log, "direct_ot", false, true);
    check_generation_ablation(log, "direct_ot", true, true);
    unsetenv("SEGFSS_BIT_BLOCK_MODE"); unsetenv("SEGFSS_MERGE_LEAF_CW");
    // Width 4: exhaustive over representative alpha (incl. boundaries 0 and max).
    for (uint64_t alpha : {uint64_t(0), uint64_t(1), uint64_t(7), uint64_t(10),
                           uint64_t(15)}) {
        check_segment_exhaustive(log, 4, alpha, 0x5);
    }

    // Width 6: a few interior + boundary points.
    for (uint64_t alpha : {uint64_t(0), uint64_t(31), uint64_t(32),
                           uint64_t(63)}) {
        check_segment_exhaustive(log, 6, alpha, 0x2A);
    }

    // Width 8: boundaries and a couple of interior points.
    for (uint64_t alpha : {uint64_t(0), uint64_t(1), uint64_t(128),
                           uint64_t(200), uint64_t(255)}) {
        check_segment_exhaustive(log, 8, alpha, 0xC3);
    }
}
