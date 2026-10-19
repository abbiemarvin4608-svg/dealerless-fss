#!/bin/bash
# Debug sigmoid(-1) failure

set -e

BUILD_DIR="./build/test/correctness"
TEST_BIN="$BUILD_DIR/SEGFSS_CORRECTNESS_TEST"

echo "==================================================================="
echo "Debugging sigmoid(-1) Failure"
echo "==================================================================="
echo ""

# Create a minimal debug test
cat > /tmp/debug_sigmoid_neg.cpp << 'EOF'
#include <iostream>
#include <cstdint>

int main() {
    constexpr int bits = 32;
    constexpr int frac_bits = 12;

    // The test uses this encoding for -1.0
    int64_t neg_one_encoded = -static_cast<int64_t>(uint64_t(1) << frac_bits);
    uint64_t neg_one_unsigned = static_cast<uint64_t>(neg_one_encoded);

    // Server share used in test
    uint64_t server_share = 0x2468ACE0ULL;

    // Client share (reconstructed from test logic)
    uint64_t client_share = neg_one_unsigned - server_share;

    std::cout << "Input: -1.0 (fixed point)" << std::endl;
    std::cout << "Encoded value: 0x" << std::hex << neg_one_unsigned << std::dec << std::endl;
    std::cout << "Server share:  0x" << std::hex << server_share << std::dec << std::endl;
    std::cout << "Client share:  0x" << std::hex << client_share << std::dec << std::endl;
    std::cout << std::endl;

    // Reconstruct
    uint64_t reconstructed = (server_share + client_share) & 0xFFFFFFFFULL;
    std::cout << "Reconstructed: 0x" << std::hex << reconstructed << std::dec << std::endl;
    std::cout << "Expected:      0x" << std::hex << neg_one_unsigned << std::dec << std::endl;
    std::cout << "Match: " << (reconstructed == (neg_one_unsigned & 0xFFFFFFFFULL) ? "YES" : "NO") << std::endl;
    std::cout << std::endl;

    // Check sign bit
    bool sign_bit = (reconstructed >> 31) & 1;
    std::cout << "Sign bit (MSB): " << sign_bit << " (should be 1 for negative)" << std::endl;

    // Expected: is_negative = 1, so nonnegative = is_negative ^ 1 = 0
    std::cout << "Expected is_negative: 1" << std::endl;
    std::cout << "Expected nonnegative: 0" << std::endl;

    return 0;
}
EOF

g++ -o /tmp/debug_sigmoid_neg /tmp/debug_sigmoid_neg.cpp
echo "Running reconstruction test:"
echo "-------------------------------------------------------------------"
/tmp/debug_sigmoid_neg

echo ""
echo "==================================================================="
echo "Now running actual test to capture detailed output"
echo "==================================================================="

# Run with verbose output
$TEST_BIN r=2 t=9 2>&1 | tee /tmp/server_debug.log &
sleep 1
$TEST_BIN r=3 t=9 2>&1 | tee /tmp/client_debug.log

echo ""
echo "Test output:"
cat /tmp/server_debug.log | grep -A2 -B2 "sigmoid(-1)"

echo ""
echo "==================================================================="
