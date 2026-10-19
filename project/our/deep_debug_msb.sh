#!/bin/bash
# Deep debug: Add instrumentation to see actual values

cat > /tmp/test_msb_flip.cpp << 'EOF'
#include <iostream>
#include <cstdint>

int main() {
    const int bits = 32;

    // Test case: -4096 (which is -1.0 in fixed point with frac_bits=12)
    int64_t neg_val_signed = -4096;
    uint64_t neg_val = static_cast<uint64_t>(neg_val_signed);
    uint64_t threshold = 0;

    std::cout << "Testing: x < 0 where x = -4096" << std::endl;
    std::cout << "x (signed):    " << neg_val_signed << std::endl;
    std::cout << "x (unsigned):  0x" << std::hex << neg_val << std::dec << std::endl;
    std::cout << "threshold:     0x" << std::hex << threshold << std::dec << std::endl;
    std::cout << std::endl;

    // MSB mask for 32-bit
    const uint64_t msb_mask = 1ULL << (bits - 1);
    std::cout << "MSB mask: 0x" << std::hex << msb_mask << std::dec << std::endl;
    std::cout << std::endl;

    // Apply MSB flip
    uint64_t x_flipped = neg_val ^ msb_mask;
    uint64_t threshold_flipped = threshold ^ msb_mask;

    std::cout << "After MSB flip:" << std::endl;
    std::cout << "x_flipped:         0x" << std::hex << x_flipped << std::dec
              << " = " << x_flipped << std::endl;
    std::cout << "threshold_flipped: 0x" << std::hex << threshold_flipped << std::dec
              << " = " << threshold_flipped << std::endl;
    std::cout << std::endl;

    // Mask to 32 bits
    uint32_t x_32 = static_cast<uint32_t>(x_flipped);
    uint32_t thresh_32 = static_cast<uint32_t>(threshold_flipped);

    std::cout << "Masked to 32-bit:" << std::endl;
    std::cout << "x_32:       0x" << std::hex << x_32 << std::dec << " = " << x_32 << std::endl;
    std::cout << "thresh_32:  0x" << std::hex << thresh_32 << std::dec << " = " << thresh_32 << std::endl;
    std::cout << std::endl;

    // Unsigned comparison
    bool result = x_32 < thresh_32;
    std::cout << "Unsigned comparison: " << x_32 << " < " << thresh_32 << " = " << result << std::endl;
    std::cout << "Expected: true (1)" << std::endl;
    std::cout << "Status: " << (result ? "PASS ✓" : "FAIL ✗") << std::endl;

    return result ? 0 : 1;
}
EOF

g++ -o /tmp/test_msb_flip /tmp/test_msb_flip.cpp && /tmp/test_msb_flip
