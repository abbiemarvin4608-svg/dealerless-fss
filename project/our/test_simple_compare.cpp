// Test piecewise comparison directly
#include <iostream>
#include "src/activation.h"
#include "src/segment.h"
#include "build/dealerless-fss/fss-overlay-src/mpc/api.h"

int main() {
    // Initialize MPC
    int party = 2; // SERVER
    int port = 32000;
    
    // For testing: check if x < 0 works correctly for negative numbers
    // Using -4096 (-1.0 in fixed point with frac_bits=12)
    
    std::cout << "Testing piecewise comparison for x < 0" << std::endl;
    std::cout << "Input: -4096 (which is -1.0 in fixed point)" << std::endl;
    
    uint64_t neg_val = static_cast<uint64_t>(-4096);
    std::cout << "As unsigned: 0x" << std::hex << neg_val << std::dec << std::endl;
    std::cout << "Sign bit (bit 31): " << ((neg_val >> 31) & 1) << std::endl;
    
    // The comparison x < 0 should return 1 (true) for negative numbers
    std::cout << "Expected result: is_negative = 1" << std::endl;
    
    return 0;
}
