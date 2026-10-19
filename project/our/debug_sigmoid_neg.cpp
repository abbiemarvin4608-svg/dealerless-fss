#include <iostream>
#include <cstdint>

// Quick debug: check how -1 is encoded in two's complement
int main() {
    constexpr int bits = 32;
    constexpr int frac_bits = 12;

    // Encode -1.0 as fixed point
    int64_t neg_one = -static_cast<int64_t>(uint64_t(1) << frac_bits);
    uint64_t encoded = static_cast<uint64_t>(neg_one);

    std::cout << "Encoded -1.0: 0x" << std::hex << encoded << std::dec << std::endl;
    std::cout << "As signed: " << static_cast<int64_t>(encoded) << std::endl;
    std::cout << "MSB (sign bit): " << ((encoded >> (bits - 1)) & 1) << std::endl;

    // Check if it's < 0 (threshold = 0)
    bool is_negative = (encoded & (1ULL << (bits - 1))) != 0;
    std::cout << "Is negative (MSB check): " << is_negative << std::endl;

    return 0;
}
