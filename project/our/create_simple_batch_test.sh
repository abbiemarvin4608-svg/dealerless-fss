#!/bin/bash
# Create a simple test to debug batch sigmoid

cat > test_simple_batch.cpp << 'EOF'
#include "activation.h"
#include "mpc/api.h"
#include <iostream>
#include <vector>

extern int party;
extern Peer* peer;

void test_simple_batch() {
    constexpr int kBits = 32;
    constexpr int kFracBits = 12;
    constexpr int kQ = 10;

    // Test single input: 1.0
    int64_t input_val = static_cast<int64_t>(uint64_t(1) << kFracBits);

    std::cout << "[Party " << party << "] Testing single sigmoid(1.0)" << std::endl;

    // Single sigmoid
    auto single_key = segfss::sigmaOffline(party, kBits, kFracBits, kQ, peer);
    GroupElement single_input = split_share(static_cast<uint64_t>(input_val), kBits, 0x12345678ULL);
    GroupElement single_output = segfss::sigmaOnline(party, single_input, single_key, peer);
    reconstruct(&single_output);

    double single_value = signed_from_twos(single_output.value, kBits) /
                         static_cast<double>(uint64_t(1) << kFracBits);

    std::cout << "[Party " << party << "] Single sigmoid(1.0) = " << single_value << std::endl;

    // Batch sigmoid with same input
    std::cout << "[Party " << party << "] Testing batch sigmoid(1.0)" << std::endl;

    std::vector<segfss::SigmaActivationKey> batch_keys =
        segfss::sigmaBatchOffline(party, 1, kBits, kFracBits, kQ, peer);

    std::vector<GroupElement> batch_inputs = {
        split_share(static_cast<uint64_t>(input_val), kBits, 0x12345678ULL)
    };

    std::vector<GroupElement> batch_outputs =
        segfss::sigmaBatchOnline(party, batch_inputs, batch_keys, peer);

    reconstruct(&batch_outputs[0]);

    double batch_value = signed_from_twos(batch_outputs[0].value, kBits) /
                        static_cast<double>(uint64_t(1) << kFracBits);

    std::cout << "[Party " << party << "] Batch sigmoid(1.0) = " << batch_value << std::endl;
    std::cout << "[Party " << party << "] Match: " << (std::abs(single_value - batch_value) < 0.001 ? "YES" : "NO") << std::endl;

    segfss::freeSigmaActivationKey(single_key);
    for (auto& key : batch_keys) {
        segfss::freeSigmaActivationKey(key);
    }
}

int main(int argc, char** argv) {
    // Same setup as correctness tests
    if (argc < 2) {
        std::cerr << "Usage: " << argv[0] << " r=<2|3>" << std::endl;
        return 1;
    }

    party = (argv[1][2] == '2') ? 2 : 3;
    int port = 32000;
    std::string address = "127.0.0.1";

    if (party == 3) {
        peer = new Peer(address, port);
    } else {
        peer = waitForPeer(port);
    }

    test_simple_batch();

    delete peer;
    return 0;
}
EOF

echo "Compiling simple batch test..."
g++ -std=c++17 -I. -Isrc -Ibuild/dealerless-fss/fss-overlay-src \
    test_simple_batch.cpp \
    -Lbuild -lsegfss -Lbuild/dealerless-fss/fss-overlay-src/lib -ldfss \
    -lgmp -lgmpxx -lpthread -lssl -lcrypto \
    -o test_simple_batch 2>&1 | tail -20

if [ -f test_simple_batch ]; then
    echo "Running test..."
    ./test_simple_batch r=2 &
    sleep 1
    ./test_simple_batch r=3
    wait
fi
