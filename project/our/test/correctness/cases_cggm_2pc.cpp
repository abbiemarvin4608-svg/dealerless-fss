#include "cases.h"

#include "cggm.h"

#include <cryptoTools/Common/Defines.h>
#include <cryptoTools/Crypto/PRNG.h>

// Correctness for the DEALERLESS 2PC cGGM key generation (paper Alg. 2 Pi_PREP
// + Alg. 3 Pi_Gen), emulated in a single process by cggmGen2pcSim.
//
// Unlike cases_cggm.cpp (dealer keys, cross-party socket reconstruct), here the
// simulator returns BOTH parties' keys locally, so we combine them locally too:
// XOR-reconstruction of L over the two keys must equal 1{x < alpha}, and of e
// must equal 1{x == alpha}. This is non-vacuous: the two keys carry independent
// root shares (root0 XOR root1 = Delta) and share every public correction word,
// exactly as the real dealerless protocol emits. If Pi_PREP/Pi_Gen do not
// reconstruct to a valid DPF, the XORed evaluations diverge from the constant.
namespace {

void check_depth_2pc(ResultLog& log, osuCrypto::PRNG& prng, int depth) {
    const uint64_t domain = uint64_t(1) << depth;
    for (uint64_t alpha = 0; alpha < domain; alpha++) {
        // Random additive (XOR) split of alpha across the two parties, and
        // random per-party offset shares Delta'_0, Delta'_1.
        const uint64_t a0 = prng.get<uint64_t>() & (domain - 1);
        const uint64_t a1 = alpha ^ a0;
        const std::vector<u8> alpha0_bits = msb_bits_from_value(a0, depth);
        const std::vector<u8> alpha1_bits = msb_bits_from_value(a1, depth);
        const osuCrypto::block d0 = prng.get<osuCrypto::block>();
        const osuCrypto::block d1 = prng.get<osuCrypto::block>();

        auto keys = segfss::cggmGen2pcSim(alpha0_bits, alpha1_bits, d0, d1,
                                          prng);

        for (uint64_t x = 0; x < domain; x++) {
            const std::vector<u8> xhat_bits = msb_bits_from_value(x, depth);
            const segfss::CggmEval e0 = segfss::cggmEval(keys.first, xhat_bits);
            const segfss::CggmEval e1 = segfss::cggmEval(keys.second, xhat_bits);

            const u8 rec_e = e0.e ^ e1.e;
            const u8 rec_l = e0.L ^ e1.L;
            const u8 expect_eq = (x == alpha) ? 1 : 0;
            const u8 expect_lt = (x < alpha) ? 1 : 0;
            const std::string tag = "2pc d" + std::to_string(depth) + " a" +
                                    std::to_string(alpha) + " x" +
                                    std::to_string(x);
            log.check_local(tag + " eq", rec_e, expect_eq);
            log.check_local(tag + " lt", rec_l, expect_lt);
        }
    }
}

}  // namespace

void check_cggm_2pc(ResultLog& log) {
    // Deterministic seed so both test processes agree and runs are reproducible.
    osuCrypto::PRNG prng(osuCrypto::toBlock(0xD1CE5EED0000002AULL,
                                            0x0123456789ABCDEFULL));
    check_depth_2pc(log, prng, 1);
    check_depth_2pc(log, prng, 2);
    check_depth_2pc(log, prng, 3);
    check_depth_2pc(log, prng, 4);
    check_depth_2pc(log, prng, 5);
}
