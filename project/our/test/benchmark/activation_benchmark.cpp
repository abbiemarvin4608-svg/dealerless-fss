// Measure one-use SIGMA activation keys at the protocol boundary.
//
// The benchmark deliberately reports key generation (offline) separately from
// evaluation (online).  The final reconstruction is performed after the
// online timer so the row describes a secret-shared activation, matching the
// DCF/DPF benchmark convention.

#include <chrono>
#include <cstdint>
#include <iostream>
#include <stdexcept>
#include <string>

#include "ArgMapping.h"
#include "commons/group_element.h"
#include "mpc/api.h"
#include "mpc/comms.h"
#include "activation.h"

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

Row run_once(const std::string& kind, int bits, int frac_bits, int q) {
    Row row;
    peer->bytesSent = 0;
    peer->bytesReceived = 0;
    peer->reset_ot_precompute();

    const auto offline_start = Clock::now();
    SigmaActivationKey key =
        kind == "tanh"
            ? tanhOffline(party, bits, frac_bits, q, peer)
            : sigmaOffline(party, bits, frac_bits, q, peer);
    if (q == kSigmaDefaultQ &&
        key.delta_table.values.size() != kSigmaDefaultTableSize) {
        throw std::logic_error("standard SIGMA requires a 1024-entry table");
    }
    const auto offline_end = Clock::now();
    row.offline_ms = std::chrono::duration<double, std::milli>(
                         offline_end - offline_start)
                         .count();
    row.offline_bytes = peer->bytesSent + peer->bytesReceived;

    peer->bytesSent = 0;
    peer->bytesReceived = 0;
    const uint64_t scale = uint64_t(1) << frac_bits;
    const uint64_t input_value = scale;  // x=1.0 at the selected precision.
    const uint64_t server_share = 0x13579BDFULL;
    GroupElement input_share(
        party == SERVER ? server_share : input_value - server_share, bits);

    const auto online_start = Clock::now();
    GroupElement output =
        kind == "tanh" ? tanhOnline(party, input_share, key, peer)
                        : sigmaOnline(party, input_share, key, peer);
    const auto online_end = Clock::now();
    row.online_ms = std::chrono::duration<double, std::milli>(
                        online_end - online_start)
                        .count();
    row.online_bytes = peer->bytesSent + peer->bytesReceived;

    // Keep output secret during the measured online phase.  Reconstruct only
    // for the correctness handshake after timing and communication counting.
    reconstruct(&output);
    freeSigmaActivationKey(key);
    return row;
}

void print_row(const std::string& kind, int bits, int frac_bits, int q,
               int runs, const Row& row) {
    if (party != SERVER) return;
    std::cout << "kind,bits,frac_bits,q,runs,offline_ms,online_ms,"
                 "offline_comm_bytes,online_comm_bytes\n";
    std::cout << kind << "," << bits << "," << frac_bits << "," << q
              << "," << runs << "," << row.offline_ms << ","
              << row.online_ms << "," << row.offline_bytes << ","
              << row.online_bytes << "\n";
}

}  // namespace

int main(int argc, char** argv) {
    std::string kind = "sigmoid";
    int bits = 32;
    int frac_bits = kSigmaDefaultFracBits;
    int q = kSigmaDefaultQ;
    int runs = 1;

    ArgMapping amap;
    amap.arg("r", party, "Role: SERVER=2, CLIENT=3");
    amap.arg("p", port, "Port");
    amap.arg("kind", kind, "sigmoid or tanh");
    amap.arg("bits", bits, "Ring width");
    amap.arg("frac", frac_bits, "Fractional bits (SIGMA default: 12)");
    amap.arg("q", q, "SIGMA LUT index width (default: 10, 1024 entries)");
    amap.arg("runs", runs, "Independent one-use keys");
    amap.parse(argc, argv);

    if (kind != "sigmoid" && kind != "tanh") {
        std::cerr << "kind must be sigmoid or tanh\n";
        return 2;
    }
    if (runs <= 0) {
        std::cerr << "runs must be positive\n";
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
        const Row row = run_once(kind, bits, frac_bits, q);
        total.offline_ms += row.offline_ms;
        total.online_ms += row.online_ms;
        total.offline_bytes += row.offline_bytes;
        total.online_bytes += row.online_bytes;
    }
    total.offline_ms /= runs;
    total.online_ms /= runs;
    total.offline_bytes /= static_cast<uint64_t>(runs);
    total.online_bytes /= static_cast<uint64_t>(runs);
    print_row(kind, bits, frac_bits, q, runs, total);

    delete peer;
    return 0;
}
