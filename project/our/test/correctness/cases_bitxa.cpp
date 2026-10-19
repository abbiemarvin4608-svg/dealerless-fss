#include "cases.h"

#include "relu.h"

#include "mpc/secure_ops.h"

#include <cstdlib>
#include <cstring>

namespace {

class ScopedEnvironment {
public:
    ScopedEnvironment(const char* name, const char* value) : name_(name) {
        const char* previous = std::getenv(name);
        present_ = previous != nullptr;
        if (previous) previous_ = previous;
        setenv(name, value, 1);
    }
    ~ScopedEnvironment() {
        if (present_) setenv(name_.c_str(), previous_.c_str(), 1);
        else unsetenv(name_.c_str());
    }
private:
    std::string name_, previous_;
    bool present_ = false;
};

void check_bit_block_mode(ResultLog& log, const char* mode, int size) {
    std::vector<u8> bit(size);
    std::vector<block> value(size), output(size);
    std::vector<block> expected(size);
    for (int i = 0; i < size; ++i) {
        const u8 secret_bit = static_cast<u8>((i & 1) ^ ((i >> 1) & 1));
        // Deliberately use non-normalized local shares.  The primitive must
        // reduce choices to their low bit before invoking OT.
        const u8 server_bit = static_cast<u8>(2 * i + ((i >> 1) & 1));
        const block secret_value = osuCrypto::toBlock(
            0x1111000000000000ULL + static_cast<uint64_t>(i),
            0x2222000000000000ULL + static_cast<uint64_t>(7 * i));
        const block server_value = osuCrypto::toBlock(
            0xA0A0000000000000ULL + static_cast<uint64_t>(i),
            0xB0B0000000000000ULL + static_cast<uint64_t>(5 * i));
        bit[i] = party == SERVER ? server_bit
                                 : static_cast<u8>(secret_bit ^ server_bit);
        value[i] = party == SERVER ? server_value : (secret_value ^ server_value);
        expected[i] = secret_bit ? secret_value : ZeroBlock;
    }

    const ScopedEnvironment mode_environment("SEGFSS_BIT_BLOCK_MODE", mode);
    const auto before = peer->bit_block_ledger;
    bitBlockMultiply(party, bit.data(), value.data(), output.data(), size,
                     peer);
    const auto& after = peer->bit_block_ledger;
    const bool native = std::strcmp(mode, "direct_cot") == 0;
    const bool direct = std::strncmp(mode, "direct_ot", 9) == 0 || native;
    const bool limbs = std::strcmp(mode, "legacy") == 0 || std::strcmp(mode, "direct_ot_legacy") == 0;
    log.check_local(std::string(mode) + " immediate products", after.products - before.products, size);
    log.check_local(std::string(mode) + " immediate generated triples",
                    after.temporary_triples_generated - before.temporary_triples_generated, direct ? 0 : size);
    log.check_local(std::string(mode) + " immediate consumed triples",
                    after.temporary_triples_consumed - before.temporary_triples_consumed, direct ? 0 : size);
    log.check_local(std::string(mode) + " immediate e/f openings", after.ef_openings - before.ef_openings,
                    direct ? 0 : 2);
    log.check_local(std::string(mode) + " 64-bit chosen message OTs", after.general_ot_64 - before.general_ot_64,
                    limbs ? 4 * size : 0);
    log.check_local(std::string(mode) + " 128-bit chosen message OTs", after.general_ot_128 - before.general_ot_128,
                    limbs || native ? 0 : 2 * size);
    log.check_local(std::string(mode) + " 128-bit native COTs", after.native_cot_128 - before.native_cot_128,
                    native ? 2 * size : 0);
    reconstruct(size, output.data());
    bool ok = true;
    if (party == SERVER) {
        for (int i = 0; i < size; ++i) {
            ok = ok && std::memcmp(&output[i], &expected[i], sizeof(block)) == 0;
        }
    }
    log.check_local(std::string("bit x block ") + mode, ok ? 1 : 0, 1);
}

void check_ring_cot(ResultLog& log, bool native, int bits, int size) {
    const ScopedEnvironment ring_environment("SEGFSS_NATIVE_RING_COT", native ? "1" : "0");
    const uint64_t mask = bits == 64 ? ~uint64_t(0) : (uint64_t(1) << bits) - 1;
    const std::string label = std::string(native ? "native" : "general") +
        " ring COT width " + std::to_string(bits) + " size " + std::to_string(size);
    std::vector<GroupElement> correlation(size, GroupElement(0, bits));
    std::vector<GroupElement> output(size, GroupElement(0, bits));
    std::vector<u8> choices(size);
    std::vector<uint64_t> expected(size);
    for (int i = 0; i < size; ++i) {
        const uint64_t patterns[] = {0, mask, uint64_t(1) << (bits - 1),
            (uint64_t(1) << (bits - 1)) - 1, uint64_t(0xD5A7139B) * uint64_t(i + 1)};
        correlation[i] = GroupElement(patterns[i % 5], bits);
        // Odd batch sizes put choice 1 at the final element, including the
        // size-one case, so the bounded 16-byte receiver load is exercised.
        choices[i] = static_cast<u8>((i + size) & 1);
        expected[i] = choices[i] ? correlation[i].value : 0;
    }
    for (bool reversed : {false, true}) {
        const bool sender = (party == SERVER) != reversed;
        const auto before = peer->preprocessing_ledger.native_ring_cot_values;
        if (sender) {
            peer->send_cot(correlation.data(), output.data(), size, reversed);
            for (auto& value : output) value = -value;
        } else peer->recv_cot(output.data(), size, choices.data(), reversed);
        log.check_local(label + " endpoint correlations",
            peer->preprocessing_ledger.native_ring_cot_values - before, native ? size : 0);
        log.check_vector(label + (reversed ? " reversed" : " straight"), output.data(), expected, bits);
    }

    // Exercise the callers that rely on arithmetic wrap-around and XOR bits.
    std::vector<u8> bit_shares(size);
    std::vector<GroupElement> values(size, GroupElement(0, bits));
    for (int i = 0; i < size; ++i) {
        const u8 bit0 = static_cast<u8>((i >> 1) & 1);
        const u8 bit1 = static_cast<u8>(i & 1);
        bit_shares[i] = party == SERVER ? bit0 : bit1;
        expected[i] = bit0 ^ bit1;
        const uint64_t server_share = uint64_t(0xBADC0FFEE) * uint64_t(i + 1);
        values[i] = split_share(correlation[i].value, bits, server_share & mask);
        output[i] = GroupElement(0, bits);
    }
    B2A(party, bit_shares.data(), output.data(), size, bits, peer);
    log.check_vector(label + " B2A", output.data(), expected, bits);
    multiplexer(party, bit_shares, values.data(), output.data(), size, peer);
    for (int i = 0; i < size; ++i)
        expected[i] = expected[i] ? correlation[i].value : 0;
    log.check_vector(label + " arithmetic multiplexer", output.data(), expected, bits);
}

}  // namespace

void check_bit_block_modes(ResultLog& log) {
    for (const char* mode : {"legacy", "block_ot", "direct_ot_legacy", "direct_ot"})
        for (int size : {7, 129, 7}) check_bit_block_mode(log, mode, size);
    for (int size : {1, 7, 129, 4097, 7})
        check_bit_block_mode(log, "direct_cot", size);
    for (bool native : {false, true})
        for (int bits : {1, 7, 9, 32, 64})
            for (int size : {1, 7, 129, 4097, 7})
                check_ring_cot(log, native, bits, size);
    // Native extensions advance the PRGs while partially consumed general-OT
    // pools retain their remaining independent pads. Exercise both switches.
    for (bool native : {false, true, false, true, false})
        check_ring_cot(log, native, 32, 7);
    for (const char* mode : {"legacy", "direct_cot", "legacy", "direct_ot", "direct_cot"})
        check_bit_block_mode(log, mode, 7);
}

void check_bitxa(ResultLog& log) {
    constexpr int bitwidth = 4;
    constexpr uint64_t domain = uint64_t(1) << bitwidth;
    constexpr int batch_size = static_cast<int>(2 * domain);

    std::vector<GroupElement> x(batch_size);
    std::vector<u8> y(batch_size);
    std::vector<uint64_t> expected(batch_size);
    for (uint64_t value = 0; value < domain; ++value) {
        for (uint64_t bit = 0; bit <= 1; ++bit) {
            const std::size_t i = static_cast<std::size_t>(2 * value + bit);
            const uint64_t x_share0 = (value * 7 + 3) & (domain - 1);
            const u8 y_share0 = static_cast<u8>((value + bit) & 1U);
            x[i] = split_share(value, bitwidth, x_share0);
            y[i] = party == SERVER ? y_share0
                                   : static_cast<u8>(bit ^ y_share0);
            expected[i] = value * bit;
        }
    }

    std::vector<segfss::BitXAPreprocessing> preprocessing =
        segfss::bitxaOffline(party, batch_size, bitwidth, peer);
    const int32_t rounds_before = numRounds;
    std::vector<GroupElement> output =
        segfss::bitxaOnline(party, x, y, preprocessing, peer);
    log.check_local("BitXA online logical rounds", numRounds - rounds_before,
                    1);
    log.check_vector("BitXA Boolean x arithmetic full 4-bit domain",
                     output.data(), expected, bitwidth);
}
