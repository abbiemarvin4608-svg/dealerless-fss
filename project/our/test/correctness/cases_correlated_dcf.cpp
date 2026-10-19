#include "cases.h"

#include "fss/dpf.h"

void check_correlated_dcf(ResultLog& log) {
    bool all_ok = true;
    std::string first_failure;

    for (int bits = 1; bits <= 5; ++bits) {
        const uint64_t domain = uint64_t(1) << bits;
        const uint64_t mask = domain - 1;
        for (uint64_t alpha = 0; alpha < domain; ++alpha) {
            const uint64_t alpha_share0 = (alpha * 13 + 7) & mask;
            const uint64_t local_alpha =
                party == SERVER ? alpha_share0 : alpha ^ alpha_share0;
            const std::vector<u8> alpha_bits =
                msb_bits_from_value(local_alpha, bits);
            DPFKeyPack key = dfss::keyGenCorrelatedDPFBit(
                party, bits, alpha_bits.data());

            std::vector<u8> outputs(domain);
            for (uint64_t x = 0; x < domain; ++x) {
                outputs[x] = dfss::evalCorrelatedDCFBit(
                    party, GroupElement(x, bits), key);
            }
            reconstruct(static_cast<int32_t>(domain), outputs.data());

            for (uint64_t x = 0; x < domain; ++x) {
                if (outputs[x] != static_cast<u8>(x < alpha)) {
                    all_ok = false;
                    if (first_failure.empty()) {
                        first_failure = "bits=" + std::to_string(bits) +
                            " alpha=" + std::to_string(alpha) +
                            " x=" + std::to_string(x) +
                            " got=" + std::to_string(outputs[x]);
                    }
                    break;
                }
            }
            freeDPFKeyPack(key);
        }
    }

    log.check_local("correlated DPF DCF exhaustive domains 1..5 bits",
                    all_ok ? 1 : 0, 1);
    if (!all_ok && party == SERVER) {
        std::cerr << "First correlated DPF DCF failure: " << first_failure
                  << '\n';
    }
}
