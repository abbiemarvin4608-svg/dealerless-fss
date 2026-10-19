// E1/E2: Segmentation sweep benchmark.
//
// Measures offline and online cost across nseg configurations to find the
// optimal segmentation count nseg* and demonstrate the feasibility frontier
// (largest ℓ where nseg=1 completes vs nseg>1).
//
// Usage (two terminals):
//   Terminal 1: ./nseg_sweep r=2 p=32000 ell=32 nseg=4
//   Terminal 2: ./nseg_sweep r=3 p=32000 ell=32 nseg=4
//
// Output (SERVER only): CSV row with timing and communication metrics.

#include <chrono>
#include <cstdint>
#include <iostream>
#include <string>

#include "ArgMapping.h"
#include "commons/group_element.h"
#include "mpc/api.h"
#include "mpc/comms.h"

#include "segment.h"
#include "gen.h"

using namespace sci;
using namespace segfss;

// Process-wide runtime globals (required by EzPC/SCI layer).
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

using Clock = std::chrono::high_resolution_clock;
using Duration = std::chrono::duration<double>;

struct BenchmarkResult {
    double offline_time_s;
    double online_time_s;
    uint64_t offline_bytes_sent;
    uint64_t offline_bytes_recv;
    uint64_t online_bytes_sent;
    uint64_t online_bytes_recv;
};

// Benchmark one (ℓ, nseg) configuration.
// Runs: split (BitDec + slice) → genSegmentKeys → scan-triple preprocessing
// (all offline) → evalAllSegments → inferComparison (masked e/f opening only).
// Times offline (split + gen + scan triples) and online (eval + infer) separately.
BenchmarkResult benchmark_one(int ell, int nseg, uint64_t threshold_value, uint64_t input_value) {
    BenchmarkResult result = {};

    // Prepare shares.
    const uint64_t threshold_server_share = 0x12345678ULL;
    GroupElement threshold_share = (party == SERVER)
        ? GroupElement(threshold_server_share, ell)
        : GroupElement(threshold_value - threshold_server_share, ell);

    SegmentLayout layout(ell, nseg);

    // Reset byte counters.
    peer->bytesSent = 0;
    peer->bytesReceived = 0;
    // Treat every measured key generation as an independent offline sample.
    // IKNP keeps a precomputation cursor across calls; without resetting it,
    // the fixed-size OT extension batch is charged only when the cursor wraps,
    // making the reported average communication depend on `runs`.
    peer->reset_ot_precompute();

    // === OFFLINE: split + genSegmentKeys ===
    auto t0_offline = Clock::now();

    SplitResult split_result = split(party, threshold_share, layout, peer);
    SegmentKeys keys = genSegmentKeys(party, split_result);
    // The comparison scan's Boolean Beaver triples depend only on nseg, not
    // on the public evaluation point.  Precompute them here so online infer
    // contains only the masked e/f opening.
    BooleanAndPreprocessing scan_preprocessing = booleanAndOffline(
        party, 2 * (nseg - 1), peer);

    auto t1_offline = Clock::now();
    uint64_t offline_sent = peer->bytesSent;
    uint64_t offline_recv = peer->bytesReceived;

    // === ONLINE: evalAllSegments + inferComparison ===
    // Prepare public input segments (MS-first bit slicing).
    std::vector<std::vector<u8>> input_segments(nseg);
    for (int j = 0; j < nseg; j++) {
        const int w = layout.width[j];
        const int sh = layout.shift[j];
        const uint64_t digit = (input_value >> sh) & ((1ULL << w) - 1);
        input_segments[j].resize(w);
        for (int k = 0; k < w; k++) {
            input_segments[j][k] = static_cast<u8>((digit >> (w - 1 - k)) & 1);
        }
    }

    peer->bytesSent = 0;
    peer->bytesReceived = 0;

    auto t0_online = Clock::now();

    std::vector<SegmentEvalResult> evals = evalAllSegments(party, keys, input_segments);
    u8 cmp_share = inferComparisonPreprocessed(
        party, evals, scan_preprocessing, peer);

    auto t1_online = Clock::now();
    uint64_t online_sent = peer->bytesSent;
    uint64_t online_recv = peer->bytesReceived;

    // Verify correctness (reconstruct comparison bit).
    // The protocol operates in the ell-bit domain, so the expected result must
    // be computed on the values reduced mod 2^ell (not the raw 64-bit inputs).
    reconstruct(&cmp_share);
    const uint64_t domain_mask = (ell >= 64) ? ~0ULL : ((1ULL << ell) - 1);
    const uint64_t threshold_mod = threshold_value & domain_mask;
    const uint64_t input_mod = input_value & domain_mask;
    const u8 expected = (input_mod < threshold_mod) ? 1 : 0;
    if (party == SERVER && (cmp_share & 1) != expected) {
        std::cerr << "CORRECTNESS FAILURE: expected " << int(expected)
                  << ", got " << int(cmp_share & 1) << "\n";
    }

    result.offline_time_s = Duration(t1_offline - t0_offline).count();
    result.online_time_s = Duration(t1_online - t0_online).count();
    result.offline_bytes_sent = offline_sent;
    result.offline_bytes_recv = offline_recv;
    result.online_bytes_sent = online_sent;
    result.online_bytes_recv = online_recv;

    return result;
}

}  // namespace

int main(int argc, char** argv) {
    // Parse arguments.
    int ell = 32;
    int nseg = 4;
    int num_runs = 10;
    uint64_t threshold_val = 0;  // 0 => auto (3/4 of 2^ell)
    uint64_t input_val = 0;      // 0 => auto (1/2 of 2^ell, i.e. below threshold)

    ArgMapping amap;
    amap.arg("r", party, "Role: SERVER=2, CLIENT=3");
    amap.arg("p", port, "Port");
    amap.arg("ell", ell, "Total bitwidth (default 32)");
    amap.arg("nseg", nseg, "Number of segments (default 4)");
    amap.arg("runs", num_runs, "Number of benchmark runs (default 10)");
    amap.arg("threshold", threshold_val, "Threshold value (0=auto: 3/4 of domain)");
    amap.arg("input", input_val, "Input value (0=auto: 1/2 of domain)");
    amap.parse(argc, argv);

    // Auto-pick in-domain values so the comparison is meaningful for any ell.
    const uint64_t domain = (ell >= 64) ? ~0ULL : (1ULL << ell);
    if (threshold_val == 0) threshold_val = (domain >> 1) + (domain >> 2);  // 3/4
    if (input_val == 0) input_val = (domain >> 1);                          // 1/2

    // Connect.
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

    // Warmup run.
    benchmark_one(ell, nseg, threshold_val, input_val);

    // Benchmark runs.
    double total_offline_time = 0.0;
    double total_online_time = 0.0;
    uint64_t total_offline_sent = 0;
    uint64_t total_offline_recv = 0;
    uint64_t total_online_sent = 0;
    uint64_t total_online_recv = 0;

    for (int run = 0; run < num_runs; run++) {
        auto res = benchmark_one(ell, nseg, threshold_val, input_val);
        total_offline_time += res.offline_time_s;
        total_online_time += res.online_time_s;
        total_offline_sent += res.offline_bytes_sent;
        total_offline_recv += res.offline_bytes_recv;
        total_online_sent += res.online_bytes_sent;
        total_online_recv += res.online_bytes_recv;
    }

    // SERVER prints CSV output.
    if (party == SERVER) {
        double avg_offline_time = total_offline_time / num_runs;
        double avg_online_time = total_online_time / num_runs;
        uint64_t avg_offline_sent = total_offline_sent / num_runs;
        uint64_t avg_offline_recv = total_offline_recv / num_runs;
        uint64_t avg_online_sent = total_online_sent / num_runs;
        uint64_t avg_online_recv = total_online_recv / num_runs;
        uint64_t avg_offline_comm = avg_offline_sent + avg_offline_recv;
        uint64_t avg_online_comm = avg_online_sent + avg_online_recv;

        std::cout << "ell,nseg,offline_time_ms,online_time_ms,"
                  << "offline_comm_bytes,online_comm_bytes\n";
        std::cout << ell << "," << nseg << ","
                  << (avg_offline_time * 1000.0) << ","
                  << (avg_online_time * 1000.0) << ","
                  << avg_offline_comm << ","
                  << avg_online_comm << "\n";
    }

    delete peer;
    return 0;
}
