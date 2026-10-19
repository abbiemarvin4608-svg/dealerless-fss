#include <iostream>
#include <cstdint>

// Simulate what happens in piecewise comparison for x < 0
int main() {
    // -1.0 in fixed point (12 fractional bits)
    int64_t neg_one = -4096;
    uint64_t neg_one_unsigned = static_cast<uint64_t>(neg_one);
    
    // Threshold for x < 0 is 0
    uint64_t threshold = 0;
    
    std::cout << "Testing: x < 0 where x = -1.0 (fixed point)" << std::endl;
    std::cout << "x (unsigned): 0x" << std::hex << neg_one_unsigned << std::dec << std::endl;
    std::cout << "threshold:    0x" << std::hex << threshold << std::dec << std::endl;
    std::cout << std::endl;
    
    // In 32-bit arithmetic
    uint32_t x_32 = static_cast<uint32_t>(neg_one_unsigned);
    uint32_t threshold_32 = static_cast<uint32_t>(threshold);
    
    std::cout << "x (32-bit):        0x" << std::hex << x_32 << std::dec 
              << " = " << static_cast<int32_t>(x_32) << std::endl;
    std::cout << "threshold (32-bit): 0x" << std::hex << threshold_32 << std::dec 
              << " = " << static_cast<int32_t>(threshold_32) << std::endl;
    std::cout << std::endl;
    
    // Unsigned comparison: 0xFFFFF000 < 0x00000000?
    bool unsigned_cmp = x_32 < threshold_32;
    std::cout << "Unsigned comparison (x < threshold): " << unsigned_cmp << std::endl;
    std::cout << "  -> This is WRONG! 0xFFFFF000 > 0x00000000 in unsigned" << std::endl;
    std::cout << std::endl;
    
    // Signed comparison (what we want)
    bool signed_cmp = static_cast<int32_t>(x_32) < static_cast<int32_t>(threshold_32);
    std::cout << "Signed comparison (x < threshold): " << signed_cmp << std::endl;
    std::cout << "  -> This is CORRECT! -4096 < 0 in signed" << std::endl;
    std::cout << std::endl;
    
    std::cout << "PROBLEM: Segment-based comparison uses UNSIGNED semantics!" << std::endl;
    std::cout << "The DPF tree traversal compares bits, which is unsigned comparison." << std::endl;
    std::cout << "For signed comparison, we need to flip the MSB or adjust the protocol." << std::endl;
    
    return 0;
}
