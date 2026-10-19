#include "cggm.h"
#include "mpc/api.h"
#include "../../tools/segmented_dcf_runtime/round_trace.h"

#include <algorithm>
#include <chrono>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <sys/resource.h>

int party_instance = 0, party = 0, bitlength = 32, num_threads = 1, port = 32000;
int num_argmax = 1000, length = 1;
std::string address = "127.0.0.1";
uint8_t choice_bit = 0;
bool verbose = false;
Peer* client = nullptr;
Peer* server = nullptr;
Dealer* dealer = nullptr;
Peer* peer = nullptr;

namespace {
using Clock = std::chrono::steady_clock;
struct Point { Clock::time_point time; uint64_t sent, received; };
Point point() { return {Clock::now(), peer ? peer->bytesSent : 0, peer ? peer->bytesReceived : 0}; }
void report(const char* phase, const Point& before) {
    auto after = point();
    // Peer wrappers meter payload operations after construction. Base-OT
    // initialization runs in the constructor, so read its real NetIO counters.
    if (std::string(phase) == "setup") {
        const auto* pack = peer->getIOPack();
        after.sent = pack->io->counter + pack->io_rev->counter + pack->io_GC->counter;
        after.received = pack->io->recv_counter + pack->io_rev->recv_counter + pack->io_GC->recv_counter;
    }
    std::cout << "METRIC {\"query\":0,\"phase\":\"" << phase
        << "\",\"seconds\":" << std::chrono::duration<double>(after.time - before.time).count()
        << ",\"sent_bytes\":" << after.sent - before.sent
        << ",\"received_bytes\":" << after.received - before.received << "}\n";
    sdcf_round::phase_end(phase, 0);
}
uint64_t vmPeak() {
    std::ifstream status("/proc/self/status");
    std::string line;
    while (std::getline(status, line))
        if (line.rfind("VmPeak:", 0) == 0) return std::stoull(line.substr(7)) * 1024;
    throw std::runtime_error("VmPeak missing from /proc/self/status");
}
uint64_t target(int row, int depth) {
    return (uint64_t(row) * 57 + 9) & ((uint64_t(1) << depth) - 1);
}
}

int main(int argc, char** argv) {
    try {
        if (argc != 10) throw std::invalid_argument("usage: role port count depth tile mode merge reuse aes_offline");
        party = std::stoi(argv[1]); port = std::stoi(argv[2]);
        const int count = std::stoi(argv[3]), depth = std::stoi(argv[4]), tile = std::stoi(argv[5]);
        const std::string mode = argv[6];
        const int merge = std::stoi(argv[7]), reuse = std::stoi(argv[8]);
        const int aes_offline = std::stoi(argv[9]);
        if ((party != SERVER && party != CLIENT) || count < 1 || depth < 1 || depth > 16 || tile < 1 ||
                (merge != 0 && merge != 1) || (reuse != 0 && reuse != 1) ||
                (aes_offline != 0 && aes_offline != 1) ||
                (mode != "legacy" && mode != "block_ot" && mode != "direct_ot_legacy" && mode != "direct_ot"))
            throw std::invalid_argument("invalid public benchmark arguments");
        std::cout.precision(12);
        setenv("SEGFSS_BIT_BLOCK_MODE", mode.c_str(), 1);
        setenv("SEGFSS_MERGE_LEAF_CW", merge ? "1" : "0", 1);
        setenv("SEGFSS_REUSE_GENERATION_WORKSPACE", reuse ? "1" : "0", 1);
        setenv("SEGFSS_AES_OFFLINE", aes_offline ? "1" : "0", 1);
        setenv("SEGFSS_AES_ONLINE", "1", 1);
        dfss_round::callbacks = {sdcf_round_trace_io, sdcf_round_actor_fork,
            sdcf_round_actor_enter, sdcf_round_actor_leave, sdcf_round_actor_join};
        sdcf_round::initialize(party == SERVER ? 1 : 2);
        sdcf_round::phase_begin("setup", 0);
        auto before = point();
        segfss::setCggmThreadLimit(4);
        peer = party == SERVER ? waitForPeer(port) : new Peer(address, port);
        report("setup", before);
        segfss::CggmKeyBatch compact;
        uint64_t products = 0, triples = 0, consumed = 0, ot64 = 0, ot128 = 0;
        uint64_t ef = 0, leaf_openings = 0, internal_hashes = 0, leaf_hashes = 0;
        uint64_t workspace_capacity = 0, workspace_growths = 0, growths_after_first = 0, participants = 0;
        sdcf_round::phase_begin("offline", 0);
        before = point();
        {
            segfss::CggmGenerationWorkspace workspace;
            for (int start = 0; start < count; start += tile) {
                const int size = std::min(tile, count - start);
                std::vector<std::vector<osuCrypto::u8>> shares(size, std::vector<osuCrypto::u8>(depth));
                for (int j = 0; j < size; ++j) {
                    // Fixed XOR-sharing fixture, independent of the later query.
                    const auto alpha = target(start + j, depth);
                    const auto server_share = (uint64_t(start + j) * 31 + 3) & ((uint64_t(1) << depth) - 1);
                    const auto own = party == SERVER ? server_share : alpha ^ server_share;
                    for (int bit = 0; bit < depth; ++bit) shares[j][bit] = (own >> (depth - 1 - bit)) & 1;
                }
                segfss::CggmGenerationStats stats;
                auto keys = segfss::cggmGen2pcBatch(party, shares, nullptr, &stats, peer, 4,
                    reuse ? &workspace : nullptr);
                compact.append(keys);
                products += stats.bit_block_products; triples += stats.temporary_triples;
                consumed += stats.temporary_triples_consumed; ot64 += stats.general_ot_64;
                ot128 += stats.general_ot_128; ef += stats.ef_openings;
                leaf_openings += stats.leaf_cw_openings; internal_hashes += stats.internal_hashes;
                leaf_hashes += stats.leaf_hashes; participants = std::max<uint64_t>(participants, stats.participants);
                workspace_capacity = std::max<uint64_t>(workspace_capacity, stats.workspace_capacity_bytes);
                workspace_growths += stats.workspace_capacity_growths;
                if (start) growths_after_first += stats.workspace_capacity_growths;
            }
        }  // Generation scratch/workspace destroyed before query inputs exist.
        report("offline", before);
        const auto resident = compact.capacityBytes();
        if (compact.size() != std::size_t(count)) throw std::runtime_error("resident key count mismatch");
        std::cout << "CGGM_READY {\"resident_bytes\":" << resident
            << ",\"before_inputs\":true,\"workspace_destroyed\":true,\"material_load_online_ms\":0}\n";

        const uint64_t mask = (uint64_t(1) << depth) - 1;
        std::vector<uint64_t> queries(count);
        for (int row = 0; row < count; ++row) {
            const auto alpha = target(row, depth);
            switch (row % 5) {
                case 0: queries[row] = 0; break;
                case 1: queries[row] = mask; break;
                case 2: queries[row] = (alpha - 1) & mask; break;
                case 3: queries[row] = alpha; break;
                default: queries[row] = (alpha + 1) & mask; break;
            }
        }
        sdcf_round::phase_begin("online", 0);
        before = point();
        segfss::CggmEvaluationStats eval;
        std::vector<segfss::CggmEval> outputs;
        segfss::cggmEvalBatch(compact, queries, {0}, 1, outputs, &eval, true);
        report("online", before);
        rusage usage{}; getrusage(RUSAGE_SELF, &usage);
        std::cout << "CGGM_DETAIL {\"count\":" << count << ",\"depth\":" << depth
            << ",\"tile\":" << tile << ",\"tiles\":" << (count + tile - 1) / tile
            << ",\"mode\":\"" << mode << "\",\"merge_leaf_cw\":" << merge
            << ",\"reuse_workspace\":" << reuse << ",\"aes_offline\":" << aes_offline
            << ",\"products\":" << products
            << ",\"temporary_triples\":" << triples << ",\"consumed_temporary_triples\":" << consumed
            << ",\"general_ot_64\":" << ot64 << ",\"general_ot_128\":" << ot128
            << ",\"ef_openings\":" << ef << ",\"leaf_cw_openings\":" << leaf_openings
            << ",\"keygen_internal_hashes\":" << internal_hashes << ",\"keygen_leaf_hashes\":" << leaf_hashes
            << ",\"keygen_participants\":" << participants << ",\"workspace_peak_capacity_bytes\":" << workspace_capacity
            << ",\"workspace_capacity_growths\":" << workspace_growths
            << ",\"workspace_growths_after_first_tile\":" << growths_after_first
            << ",\"eval_internal_hashes\":" << eval.internal_hashes << ",\"eval_leaf_hashes\":" << eval.leaf_hashes
            << ",\"eval_participants\":" << eval.participants << ",\"resident_bytes\":" << resident
            << ",\"max_rss_bytes\":" << uint64_t(usage.ru_maxrss) * 1024
            << ",\"vm_peak_bytes\":" << vmPeak() << "}\n";

        sdcf_round::phase_begin("output", 0);
        std::vector<uint8_t> own(2 * count), remote(2 * count);
        for (int row = 0; row < count; ++row) { own[2 * row] = outputs[row].L; own[2 * row + 1] = outputs[row].e; }
        if (party == SERVER) { peer->send_u8(own.data(), own.size()); peer->recv_u8(remote.data(), remote.size()); }
        else { peer->recv_u8(remote.data(), remote.size()); peer->send_u8(own.data(), own.size()); }
        for (int row = 0; row < count; ++row)
            if ((own[2 * row] ^ remote[2 * row]) != (queries[row] < target(row, depth)) ||
                    (own[2 * row + 1] ^ remote[2 * row + 1]) != (queries[row] == target(row, depth)))
                throw std::runtime_error("independent comparison oracle mismatch");
        sdcf_round::phase_end("output", 0);
        sdcf_round::phase_begin("teardown", 0);
        compact = {}; delete peer; peer = nullptr;
        sdcf_round::phase_end("teardown", 0); sdcf_round::finish();
        std::cout << "CGGM_BENCHMARK PASS\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "CGGM_BENCHMARK FAIL " << error.what() << '\n'; return 1;
    }
}
