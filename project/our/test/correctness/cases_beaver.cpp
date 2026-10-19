#include "cases.h"

#include "mpc/secure_ops.h"

#include <cstdint>
#include <string>
#include <vector>

namespace {

uint64_t ringMask(int bits) {
    return bits == 64 ? ~uint64_t(0) : (uint64_t(1) << bits) - 1;
}

uint64_t mix(int index, uint64_t salt) {
    uint64_t z = salt + 0x9e3779b97f4a7c15ULL * static_cast<uint64_t>(index + 1);
    z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9ULL;
    z = (z ^ (z >> 27)) * 0x94d049bb133111ebULL;
    return z ^ (z >> 31);
}

struct Pair {
    uint64_t x;
    uint64_t y;
};

// Deterministic plaintext operands: both processes derive the same pair, so
// each can compute x*y locally.  The first entries hit the carry-heavy corners.
Pair samplePair(int index, int bits) {
    const uint64_t mask = ringMask(bits);
    const uint64_t msb = uint64_t(1) << (bits - 1);
    switch (index) {
    case 0: return {0, mask};
    case 1: return {mask, 0};
    case 2: return {mask, mask};
    case 3: return {1, msb};
    case 4: return {msb, msb};
    default: return {mix(index, 0x5eed) & mask, mix(index, 0xbeef) & mask};
    }
}

std::string label(const char* name, int bits, int size) {
    return std::string(name) + " bits=" + std::to_string(bits) +
           " size=" + std::to_string(size);
}

// SERVER is the COT sender holding x, CLIENT the receiver holding y.  The two
// outputs must be additive shares of x*y mod 2^bits.
void checkCrossTerms(ResultLog& log, int bits, int size) {
    std::vector<GroupElement> input(size, GroupElement(0, bits));
    std::vector<GroupElement> output(size, GroupElement(0, bits));
    std::vector<uint64_t> expected(size);
    for (int i = 0; i < size; ++i) {
        const Pair p = samplePair(i, bits);
        input[i] = GroupElement(party == SERVER ? p.x : p.y, bits);
        expected[i] = (p.x * p.y) & ringMask(bits);
    }
    cross_term_gen(party, input.data(), output.data(), party == SERVER, size,
                   peer);
    reconstruct(size, output.data(), bits);
    uint64_t mismatches = 0;
    for (int i = 0; i < size; ++i) {
        mismatches += output[i].value == expected[i] ? 0 : 1;
    }
    log.check_local(label("cross_term_gen mismatches", bits, size),
                    mismatches, 0);
}

void checkBeaverOffline(ResultLog& log, int bits, int size) {
    std::vector<GroupElement> a(size, GroupElement(0, bits));
    std::vector<GroupElement> b(size, GroupElement(0, bits));
    std::vector<GroupElement> c(size, GroupElement(0, bits));
    beaver_mult_offline(party, a.data(), b.data(), c.data(), peer, size);
    reconstruct(size, a.data(), bits);
    reconstruct(size, b.data(), bits);
    reconstruct(size, c.data(), bits);
    uint64_t mismatches = 0;
    for (int i = 0; i < size; ++i) {
        const uint64_t product = (a[i].value * b[i].value) & ringMask(bits);
        mismatches += c[i].value == product ? 0 : 1;
    }
    log.check_local(label("beaver_mult_offline c!=a*b", bits, size),
                    mismatches, 0);
}

// Makes one party's local counter public; the other party contributes 0.
uint64_t publicCounter(uint64_t local, int owner) {
    GroupElement value(party == owner ? local : 0, 64);
    reconstruct(&value);
    return value.value;
}

// Batched triples use one arithmetic COT per (cross term, bit j).  Bit j only
// needs the low (bits - j) bits of 2^j * x, so the sender's payload is
// sum_j ceil((bits - j) / 8) bytes per cross term instead of two full
// bits-wide OT messages.  The receiver still pays the 128-bit IKNP column.
void checkBeaverCommunication(ResultLog& log) {
    constexpr int kBits = 32;
    constexpr int kSize = 1024;
    constexpr uint64_t kCrossTerms = 2ULL * kSize;
    uint64_t sender_expected = 0;
    for (int j = 0; j < kBits; ++j) {
        sender_expected += kCrossTerms * ((kBits - j + 7) / 8);
    }
    const uint64_t receiver_expected = kCrossTerms * kBits * 16;

    std::vector<GroupElement> a(kSize, GroupElement(0, kBits));
    std::vector<GroupElement> b(kSize, GroupElement(0, kBits));
    std::vector<GroupElement> c(kSize, GroupElement(0, kBits));
    // Warm-up keeps one-time base-OT setup out of the measured window.
    beaver_mult_offline(party, a.data(), b.data(), c.data(), peer, kSize);

    const uint64_t before = peer->bytesSent;
    beaver_mult_offline(party, a.data(), b.data(), c.data(), peer, kSize);
    const uint64_t sent = peer->bytesSent - before;
    log.check_local("beaver offline COT sender bytes (1024 x 32-bit)",
                    publicCounter(sent, SERVER), sender_expected);
    log.check_local("beaver offline IKNP receiver bytes (1024 x 32-bit)",
                    publicCounter(sent, CLIENT), receiver_expected);
}

}  // namespace

void check_beaver(ResultLog& log) {
    // Sizes straddle the 128-OT precompute pool: small batches stay on the
    // pooled OT path, large ones use the length-reduced batched COT.
    for (int bits : {1, 8, 16, 32, 64}) {
        for (int size : {1, 3, 4, 5, 64, 257}) {
            checkCrossTerms(log, bits, size);
        }
    }
    for (int bits : {8, 32, 64}) {
        for (int size : {1, 5, 300}) {
            checkBeaverOffline(log, bits, size);
        }
    }
    checkBeaverCommunication(log);
}
