#include "cases.h"

#include "linear.h"
#include "ring_packing.h"
#include "mpc/secure_ops.h"

#include <cstdint>
#include <cstdlib>
#include <stdexcept>
#include <vector>

namespace {

constexpr int kBits = 32;
constexpr int kRows = 2;
constexpr int kInner = 3;
constexpr int kCols = 2;

uint64_t ringMask() {
    return (uint64_t(1) << kBits) - 1;
}

std::vector<uint64_t> lhsPlain() {
    return {3, 0xfffffff0ULL, 7, 11, 13, 17};
}

std::vector<uint64_t> rhsPlain() {
    return {19, 23, 29, 31, 37, 41};
}

std::vector<GroupElement> makeShare(const std::vector<uint64_t>& values,
                                    uint64_t seed) {
    std::vector<GroupElement> result(values.size());
    for (std::size_t i = 0; i < values.size(); ++i) {
        const uint64_t server_share =
            (seed + 0x9e3779b97f4a7c15ULL * (i + 1)) & ringMask();
        result[i] = split_share(values[i], kBits, server_share);
    }
    return result;
}

std::vector<GroupElement> makeShareWidth(const std::vector<uint64_t>& values,
                                         uint64_t seed, int bits) {
    const uint64_t mask = bits == 64 ? UINT64_MAX : ((uint64_t(1) << bits) - 1);
    std::vector<GroupElement> result(values.size());
    for (std::size_t i = 0; i < values.size(); ++i) {
        const uint64_t server_share =
            (seed + 0x9e3779b97f4a7c15ULL * (i + 1)) & mask;
        result[i] = split_share(values[i], bits, server_share);
    }
    return result;
}

}  // namespace

void check_matmul(ResultLog& log) {
    const auto lhs = makeShare(lhsPlain(), 0x1234);
    const auto rhs = makeShare(rhsPlain(), 0xabcd);
    auto preprocessing = segfss::matmulOffline(
        party, kRows, kInner, kCols, kBits, peer);
    log.check_local("matrix Beaver small offline COT batches",
                    preprocessing.cot_batches, 1);
    const int32_t rounds_before = numRounds;
    const uint64_t bytes_before = peer->bytesSent + peer->bytesReceived;
    auto output = segfss::matmulOnline(party, lhs, rhs, preprocessing, peer);

    log.check_local("matrix Beaver online rounds", numRounds - rounds_before,
                    1);
    log.check_local("matrix Beaver online bytes per party",
                    peer->bytesSent + peer->bytesReceived - bytes_before,
                    2 * (kRows * kInner + kInner * kCols) * 4);
    log.check_vector("batched secure matmul 2x3 times 3x2", output.data(),
                     {4294967148ULL, 4294967156ULL, 1215, 1353}, kBits);
    log.check_local("matmul preprocessing consumed", preprocessing.consumed,
                    true);
    bool rejected_reuse = false;
    try {
        (void)segfss::matmulOnline(party, lhs, rhs, preprocessing, peer);
    } catch (const std::logic_error&) {
        rejected_reuse = true;
    }
    log.check_local("matrix Beaver mask reuse rejected", rejected_reuse, 1);

    constexpr int rows = 4;
    constexpr int inner = 65;
    constexpr int cols = 16;
    std::vector<uint64_t> large_lhs(rows * inner, 0);
    std::vector<uint64_t> large_rhs(inner * cols, 0);
    for (int i = 0; i < rows; ++i) {
        large_lhs[i * inner + i] = i + 2;
        for (int j = 0; j < cols; ++j) {
            large_rhs[i * cols + j] = j + 3;
        }
    }
    auto large_prep = segfss::matmulOffline(
        party, rows, inner, cols, kBits, peer);
    log.check_local("matrix Beaver tiled offline COT batches",
                    large_prep.cot_batches, 2);
    const int32_t large_rounds_before = numRounds;
    const uint64_t large_bytes_before =
        peer->bytesSent + peer->bytesReceived;
    auto large_output = segfss::matmulOnline(
        party, makeShare(large_lhs, 0x4321),
        makeShare(large_rhs, 0xdcba), large_prep, peer);
    log.check_local("matrix Beaver tiled online rounds",
                    numRounds - large_rounds_before, 1);
    log.check_local("matrix Beaver tiled online bytes per party",
                    peer->bytesSent + peer->bytesReceived - large_bytes_before,
                    2 * (rows * inner + inner * cols) * 4);
    reconstruct(static_cast<int32_t>(large_output.size()),
                large_output.data(), kBits);
    uint64_t mismatches = 0;
    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            mismatches +=
                large_output[i * cols + j].value ==
                        static_cast<uint64_t>((i + 2) * (j + 3))
                    ? 0 : 1;
        }
    }
    log.check_local("matrix Beaver tiled product mismatches", mismatches, 0);

    // Exercise the optional packed opening on a non-byte-aligned ring.  The
    // environment is a public experiment switch and must be set for both
    // parties before entering the protocol.
    setenv("SEGFSS_PACK_RING", "1", 1);
    constexpr int packed_bits = 13;
    auto packed_prep = segfss::matmulOffline(party, 1, 3, 1, packed_bits, peer);
    const uint64_t packed_before = peer->bytesSent + peer->bytesReceived;
    const int32_t packed_rounds_before = numRounds;
    auto packed_output = segfss::matmulOnline(
        party, makeShareWidth({0x123, 0x1abc, 0x77}, 0x55, packed_bits),
        makeShareWidth({3, 5, 7}, 0x77, packed_bits), packed_prep, peer);
    log.check_local("packed Beaver opening rounds", numRounds - packed_rounds_before, 1);
    log.check_local("packed Beaver opening bytes",
                    peer->bytesSent + peer->bytesReceived - packed_before,
                    20 + 2 * segfss::kRingPackingHeaderBytes);
    reconstruct(static_cast<int32_t>(packed_output.size()), packed_output.data(), packed_bits);
    log.check_local("packed Beaver product",
                    packed_output.front().value,
                    static_cast<uint64_t>((0x123 * 3 + 0x1abc * 5 + 0x77 * 7) &
                                          ((uint64_t(1) << packed_bits) - 1)));
    unsetenv("SEGFSS_PACK_RING");
}
