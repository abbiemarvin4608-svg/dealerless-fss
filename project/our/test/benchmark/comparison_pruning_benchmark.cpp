#include "sci_truncation.h"
#include "mpc/api.h"
#include "../../tools/segmented_dcf_runtime/round_trace.h"

#include <chrono>
#include <cstdlib>
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
Point point() { return {Clock::now(), peer->bytesSent, peer->bytesReceived}; }
void report(const char* phase, const Point& before) {
    const auto after = point();
    std::cout << "METRIC {\"query\":0,\"phase\":\"" << phase
        << "\",\"seconds\":" << std::chrono::duration<double>(after.time - before.time).count()
        << ",\"sent_bytes\":" << after.sent - before.sent
        << ",\"received_bytes\":" << after.received - before.received << "}\n";
    sdcf_round::phase_end(phase, 0);
}
uint64_t capacity(const segfss::SciTruncationMaterial& m) {
    return sizeof(m) + (m.input_mask_shares.capacity() + m.mask_high_shares.capacity() +
        m.conversion_mask_arithmetic.capacity()) * sizeof(uint64_t) + m.mask_sign_bit_shares.capacity() +
        m.conversion_mask_bits.capacity() + m.rounding.a.capacity() + m.rounding.b.capacity() +
        m.rounding.c.capacity() + m.low_shift.capacityBytes() + m.sign.capacityBytes() + m.wrap.capacityBytes();
}
}

int main(int argc, char** argv) {
    try {
        if (argc != 6) throw std::invalid_argument("usage: role port count equality leaf");
        party = std::stoi(argv[1]); port = std::stoi(argv[2]);
        const int count = std::stoi(argv[3]), equality = std::stoi(argv[4]), leaf = std::stoi(argv[5]);
        if ((party != SERVER && party != CLIENT) || count < 1 || (equality != 0 && equality != 1) ||
                (leaf != 0 && leaf != 1)) throw std::invalid_argument("invalid public benchmark arguments");
        std::cout.precision(12);
        setenv("SEGFSS_PRUNE_EQUALITY", equality ? "1" : "0", 1);
        setenv("SEGFSS_PRUNE_LEAF_HASHES", leaf ? "1" : "0", 1);
        setenv("SEGFSS_AES_OFFLINE", "1", 1); setenv("SEGFSS_AES_ONLINE", "1", 1);
        setenv("SEGFSS_FUSE_DOMAINS", "1", 1); setenv("SEGFSS_PACK_BITS", "1", 1);
        setenv("SEGFSS_MATERIAL_LAYOUT", "contiguous", 1);
        setenv("SEGFSS_GENERATION_BUDGET_BYTES", "1610612736", 1);
        segfss::setCggmThreadLimit(4);
        dfss_round::callbacks = {sdcf_round_trace_io, sdcf_round_actor_fork,
            sdcf_round_actor_enter, sdcf_round_actor_leave, sdcf_round_actor_join};
        sdcf_round::initialize(party == SERVER ? 1 : 2);
        sdcf_round::phase_begin("setup", 0);
        peer = party == SERVER ? waitForPeer(port) : new Peer(address, port);
        sdcf_round::phase_end("setup", 0);
        const auto config = segfss::makeSciTruncationConfig(
            {32, 32, 14, segfss::reference::SciCall::div_pow2, true},
            segfss::DcfBackend::segmented, segfss::SegmentPolicy::depth_cap_8);
        sdcf_round::phase_begin("offline", 0);
        auto before = point();
        auto material = segfss::sciTruncateOfflineBatch(party, count, config, peer);
        const auto resident = capacity(material);
        report("offline", before);
        std::cout << "PRUNING_READY {\"resident_bytes\":" << resident
            << ",\"before_inputs\":true,\"material_load_online_ms\":0}\n";

        // The private-query shares are constructed only after material generation
        // and resident ownership are complete. No online spool/deserialization.
        std::vector<uint64_t> values(count), own(count), remote(count);
        std::vector<GroupElement> inputs; inputs.reserve(count);
        for (int row = 0; row < count; ++row) {
            const uint64_t value = (uint64_t(row) * 0x9e3779b9 + 0x80000001) & UINT32_MAX;
            const uint64_t share = (uint64_t(row) * 0x7f4a7c15 + 0x8d67341) & UINT32_MAX;
            values[row] = value;
            inputs.emplace_back(party == SERVER ? share : (value - share) & UINT32_MAX, 32);
        }
        sdcf_round::phase_begin("online", 0);
        before = point();
        auto outputs = segfss::sciTruncateOnlineBatch(party, inputs, config, material, peer);
        report("online", before);
        uint64_t gates = 0, triples = 0, consumed = 0, internal = 0, leaves = 0;
        uint64_t keygen_internal = 0, keygen_leaves = 0, skipped = 0, pruned = 0, participants = 0;
        for (const auto* comparison : {&material.low_shift, &material.sign, &material.wrap}) {
            gates += comparison->metrics.scan_and_gates; triples += comparison->scan.size();
            consumed += comparison->scan.cursor; internal += comparison->metrics.eval_internal_hashes;
            leaves += comparison->metrics.eval_leaf_hashes; skipped += comparison->metrics.skipped_leaf_hashes;
            pruned += comparison->metrics.pruned_e_outputs;
            keygen_internal += comparison->metrics.internal_hashes; keygen_leaves += comparison->metrics.leaf_hashes;
            participants = std::max<uint64_t>(participants, comparison->metrics.eval_participants);
        }
        if (gates != triples || triples != consumed) throw std::runtime_error("scan inventory mismatch");
        rusage usage{}; getrusage(RUSAGE_SELF, &usage);
        std::cout << "PRUNING_DETAIL {\"count\":" << count << ",\"prune_equality\":" << equality
            << ",\"prune_leaf_hashes\":" << leaf << ",\"scan_and_gates\":" << gates
            << ",\"generated_scan_triples\":" << triples << ",\"consumed_scan_triples\":" << consumed
            << ",\"eval_internal_hashes\":" << internal << ",\"eval_leaf_hashes\":" << leaves
            << ",\"keygen_internal_hashes\":" << keygen_internal << ",\"keygen_leaf_hashes\":" << keygen_leaves
            << ",\"pruned_e_outputs\":" << pruned << ",\"skipped_leaf_hashes\":" << skipped
            << ",\"online_protocol_rounds\":" << material.metrics.online_logical_rounds
            << ",\"eval_participants\":" << participants << ",\"resident_bytes\":" << resident
            << ",\"max_rss_bytes\":" << uint64_t(usage.ru_maxrss) * 1024 << "}\n";

        sdcf_round::phase_begin("output", 0);
        for (int row = 0; row < count; ++row) own[row] = outputs[row].value;
        if (party == SERVER) { peer->send_u64(own.data(), count); peer->recv_u64(remote.data(), count); }
        else { peer->recv_u64(remote.data(), count); peer->send_u64(own.data(), count); }
        for (int row = 0; row < count; ++row)
            if (((own[row] + remote[row]) & UINT32_MAX) != segfss::reference::evaluate(values[row], config.contract))
                throw std::runtime_error("independent integer oracle mismatch");
        sdcf_round::phase_end("output", 0);
        sdcf_round::phase_begin("teardown", 0);
        segfss::releaseSciTruncationMaterial(material); delete peer; peer = nullptr;
        sdcf_round::phase_end("teardown", 0); sdcf_round::finish();
        std::cout << "PRUNING_BENCHMARK PASS\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "PRUNING_BENCHMARK FAIL " << error.what() << '\n'; return 1;
    }
}
