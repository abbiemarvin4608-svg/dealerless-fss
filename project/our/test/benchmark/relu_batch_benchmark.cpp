// Measure the ReLU/BitXA batch cost used after a Boolean DReLU result.
// DCF key generation is measured separately by nseg_sweep; this executable
// isolates the arithmetic/Boolean bridge so model projections can show both
// components instead of silently dropping Boolean triples.

#include <chrono>
#include <cstdint>
#include <iostream>
#include <string>
#include <vector>

#include "ArgMapping.h"
#include "commons/group_element.h"
#include "mpc/api.h"
#include "mpc/comms.h"
#include "relu.h"

using namespace sci;
using namespace segfss;

int party_instance = 0;
int party = 0;
int32_t bitlength = 32;
int num_threads = 1;
int port = 32000;
std::string address = "127.0.0.1";
int num_argmax = 1000;
uint8_t choice_bit = 0;
bool verbose = false;
int length = 1;
Peer* client = nullptr;
Peer* server = nullptr;
Dealer* dealer = nullptr;
Peer* peer = nullptr;

namespace {

using Clock = std::chrono::steady_clock;

struct Row {
    double offline_ms = 0.0;
    double online_ms = 0.0;
    uint64_t offline_bytes = 0;
    uint64_t online_bytes = 0;
};

Row run_once(int size, int bits) {
    Row row;
    peer->bytesSent = 0;
    peer->bytesReceived = 0;
    peer->reset_ot_precompute();

    const auto offline_start = Clock::now();
    std::vector<ReluPreprocessing> preprocessing =
        reluOffline(party, size, bits, peer);
    const auto offline_end = Clock::now();
    row.offline_ms = std::chrono::duration<double, std::milli>(
                         offline_end - offline_start)
                         .count();
    row.offline_bytes = peer->bytesSent + peer->bytesReceived;

    std::vector<GroupElement> x(size);
    std::vector<u8> drelu(size);
    for (int i = 0; i < size; ++i) {
        const uint64_t value = static_cast<uint64_t>(i * 17 + 3);
        const uint64_t x_server = static_cast<uint64_t>(0x2468ACE0ULL + i);
        x[i] = GroupElement(
            party == SERVER ? x_server : value - x_server, bits);
        const u8 logical_bit = static_cast<u8>(i & 1U);
        const u8 server_bit = static_cast<u8>((i + 1) & 1U);
        drelu[i] = party == SERVER ? server_bit
                                   : static_cast<u8>(logical_bit ^ server_bit);
    }

    peer->bytesSent = 0;
    peer->bytesReceived = 0;
    const auto online_start = Clock::now();
    std::vector<GroupElement> output =
        reluOnline(party, x, drelu, preprocessing, peer);
    const auto online_end = Clock::now();
    row.online_ms = std::chrono::duration<double, std::milli>(
                        online_end - online_start)
                        .count();
    row.online_bytes = peer->bytesSent + peer->bytesReceived;

    // Output opening is outside the online measurement, matching nseg_sweep.
    reconstruct(size, output.data(), bits);
    return row;
}

}  // namespace

int main(int argc, char** argv) {
    int size = 1;
    int bits = 32;
    int runs = 1;
    ArgMapping amap;
    amap.arg("r", party, "Role: SERVER=2, CLIENT=3");
    amap.arg("p", port, "Port");
    amap.arg("size", size, "Number of batched ReLU values");
    amap.arg("bits", bits, "Ring width");
    amap.arg("runs", runs, "Independent batches");
    amap.parse(argc, argv);
    if (size <= 0 || runs <= 0) {
        std::cerr << "size and runs must be positive\n";
        return 2;
    }

    if (party == CLIENT) {
        server = new Peer(address, port);
        peer = server;
    } else if (party == SERVER) {
        client = waitForPeer(port);
        peer = client;
    } else {
        std::cerr << "Pass r=2 (SERVER) or r=3 (CLIENT).\n";
        return 2;
    }

    Row total;
    for (int run = 0; run < runs; ++run) {
        const Row row = run_once(size, bits);
        total.offline_ms += row.offline_ms;
        total.online_ms += row.online_ms;
        total.offline_bytes += row.offline_bytes;
        total.online_bytes += row.online_bytes;
    }
    total.offline_ms /= runs;
    total.online_ms /= runs;
    total.offline_bytes /= static_cast<uint64_t>(runs);
    total.online_bytes /= static_cast<uint64_t>(runs);

    if (party == SERVER) {
        std::cout << "size,bits,runs,offline_ms,online_ms,"
                     "offline_comm_bytes,online_comm_bytes\n";
        std::cout << size << "," << bits << "," << runs << ","
                  << total.offline_ms << "," << total.online_ms << ","
                  << total.offline_bytes << "," << total.online_bytes
                  << "\n";
    }

    delete peer;
    return 0;
}
