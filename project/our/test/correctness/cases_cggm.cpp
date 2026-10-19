#include "cases.h"

#include "cggm.h"

#include <cryptoTools/Common/Defines.h>

// Correctness for the from-scratch Correlated-GGM (Half-Tree) primitive.
//
// Both parties deterministically generate the SAME (k0, k1) pair from a fixed
// (alpha, delta); SERVER evaluates k0, CLIENT evaluates k1. Each reconstruct()
// XORs the two parties' shares, so comparing to a public constant is
// non-vacuous:
//   * reconstruct(e) == 1{x == alpha}   (leaf control bit = equality)
//   * reconstruct(L) == 1{x <  alpha}   (Grotto prefix-parity comparison)
namespace {

// Fixed global offset with lsb = 1 (0x...11 -> low bit set).
const osuCrypto::block kDelta =
    osuCrypto::toBlock(0x123456789abcdef0ULL, 0xfedcba9876543211ULL);

void check_depth(ResultLog& log, int depth) {
    const uint64_t domain = uint64_t(1) << depth;
    for (uint64_t alpha = 0; alpha < domain; alpha++) {
        const std::vector<u8> alpha_bits =
            msb_bits_from_value(alpha, depth);
        auto keys = segfss::cggmGenDealer(alpha_bits, kDelta);
        const segfss::CggmKey& mykey =
            (party == SERVER) ? keys.first : keys.second;

        for (uint64_t x = 0; x < domain; x++) {
            const std::vector<u8> xhat_bits = msb_bits_from_value(x, depth);
            const segfss::CggmEval ev = segfss::cggmEval(mykey, xhat_bits);

            const u8 expect_eq = (x == alpha) ? 1 : 0;
            const u8 expect_lt = (x < alpha) ? 1 : 0;
            const std::string tag = "d" + std::to_string(depth) + " a" +
                                    std::to_string(alpha) + " x" +
                                    std::to_string(x);
            log.check_bit(tag + " eq", ev.e, expect_eq);
            log.check_bit(tag + " lt", ev.L, expect_lt);
        }
    }
}

}  // namespace

void check_cggm(ResultLog& log) {
    check_depth(log, 2);
    check_depth(log, 3);
    check_depth(log, 4);
}
