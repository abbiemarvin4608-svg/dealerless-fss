// Real two-party SDCF-10 scan. Each measured run consumes fresh material.
#include "ArgMapping.h"
#include "mpc/api.h"
#include "mpc/comms.h"
#include "sci_truncation.h"
#include "segmented_dcf.h"

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <sys/resource.h>
#include <vector>

using namespace sci;
using namespace segfss;

int party_instance = 0, party = 0, num_threads = 1, port = 32000;
int32_t bitlength = 32;
std::string address = "127.0.0.1";
int num_argmax = 1000, length = 1;
uint8_t choice_bit = 0;
bool verbose = false;
Peer* client = nullptr;
Peer* server = nullptr;
Dealer* dealer = nullptr;
Peer* peer = nullptr;

namespace {
using Clock = std::chrono::steady_clock;
double elapsed(Clock::time_point a, Clock::time_point b) {
    return std::chrono::duration<double>(b - a).count();
}
uint64_t mask(int bits) { return bits == 64 ? UINT64_MAX : ((uint64_t{1} << bits) - 1); }
struct Counters { uint64_t sent, received, io; };
Counters counters() { return {peer->bytesSent, peer->bytesReceived, peer->rounds}; }

struct Run {
    double offline = 0, online = 0, total = 0;
    double keygen = 0, scan_preprocessing = 0, evaluation = 0, scan = 0;
    double query_prepare = 0, local_evaluation = 0;
    uint64_t eval_participants = 0;
    uint64_t leaf_hashes = 0, generation_participants = 0, keygen_aes_scratch_bytes = 0;
    uint64_t eval_internal_hashes = 0, eval_leaf_hashes = 0, eval_aes_scratch_bytes = 0;
    bool aes_online = false, aes_offline = false;
    double masks = 0, comparison_preprocessing = 0, rounding_preprocessing = 0;
    double opening = 0, comparison_online = 0, rounding_online = 0, b2a_online = 0;
    Counters offline_counters{}, online_counters{};
    uint64_t logical_rounds = 0, scan_gates = 0, scan_layers = 0, internal_hashes = 0;
    uint64_t correction_bits = 0, rounding_gates = 0, masked_input_values = 0;
    uint64_t rss_kib = 0;
    uint64_t expanded_nodes = 0, generation_tiles = 0, peak_working_bytes = 0, resident_bytes = 0;
    uint64_t generation_io_calls = 0, cot_batches = 0, opening_batches = 0;
    uint64_t added_cot_batches = 0, added_opening_batches = 0;
};

Counters difference(Counters a, Counters b) {
    return {a.sent - b.sent, a.received - b.received, a.io - b.io};
}
void add_dcf_metrics(Run& run, const SegmentedDcfMaterial& material) {
    run.keygen += material.metrics.keygen_ms / 1000;
    run.scan_preprocessing += material.metrics.scan_preprocessing_ms / 1000;
    run.evaluation += material.metrics.evaluation_ms / 1000;
    run.query_prepare += material.metrics.query_prepare_ms / 1000;
    run.local_evaluation += material.metrics.local_evaluation_ms / 1000;
    run.eval_participants = std::max<uint64_t>(run.eval_participants, material.metrics.eval_participants);
    run.scan += material.metrics.scan_ms / 1000;
    run.scan_gates += material.metrics.scan_and_gates;
    run.scan_layers += material.metrics.online_scan_layers;
    run.internal_hashes += material.metrics.internal_hashes;
    run.leaf_hashes += material.metrics.leaf_hashes;
    run.generation_participants = std::max<uint64_t>(run.generation_participants, material.metrics.generation_participants);
    run.keygen_aes_scratch_bytes = std::max<uint64_t>(run.keygen_aes_scratch_bytes, material.metrics.keygen_aes_scratch_bytes);
    run.eval_internal_hashes += material.metrics.eval_internal_hashes;
    run.eval_leaf_hashes += material.metrics.eval_leaf_hashes;
    run.eval_aes_scratch_bytes = std::max<uint64_t>(run.eval_aes_scratch_bytes, material.metrics.eval_aes_scratch_bytes);
    run.aes_online = run.aes_online || material.metrics.aes_online;
    run.aes_offline = run.aes_offline || material.metrics.aes_offline;
    run.expanded_nodes += material.metrics.expanded_nodes;
    run.generation_tiles += material.metrics.generation_tiles;
    run.peak_working_bytes = std::max(run.peak_working_bytes, material.metrics.peak_working_bytes);
    run.resident_bytes = std::max(run.resident_bytes, material.metrics.resident_bytes);
    run.generation_io_calls += material.metrics.generation_io_calls;
    run.cot_batches += material.metrics.cot_batches;
    run.opening_batches += material.metrics.opening_batches;
    run.added_cot_batches += material.metrics.added_cot_batches;
    run.added_opening_batches += material.metrics.added_opening_batches;
}
std::vector<uint64_t> thresholds(int count, int bits) {
    std::vector<uint64_t> result(count);
    for (int row = 0; row < count; ++row)
        result[row] = (uint64_t(row) * 0x9e3779b97f4a7c15ULL) & mask(bits);
    return result;
}
std::vector<std::vector<u8>> threshold_shares(const std::vector<uint64_t>& values, int bits) {
    std::vector<std::vector<u8>> result(values.size(), std::vector<u8>(bits));
    const uint64_t value_mask = mask(bits);
    for (std::size_t row = 0; row < values.size(); ++row) {
        const uint64_t share = (values[row] * 0x9e3779b97f4a7c15ULL + 0x6c8e9cf5ULL) & value_mask;
        const uint64_t own = party == SERVER ? share : (values[row] ^ share);
        for (int bit = 0; bit < bits; ++bit)
            result[row][bit] = (own >> (bits - bit - 1)) & 1;
    }
    return result;
}
template <class T> void exchangeOutputShares(const std::vector<T>& own, std::vector<T>& remote);
template <> void exchangeOutputShares(const std::vector<u8>& own, std::vector<u8>& remote) {
    if (party == SERVER) { peer->send_u8(own.data(), own.size()); peer->recv_u8(remote.data(), remote.size()); }
    else { peer->recv_u8(remote.data(), remote.size()); peer->send_u8(own.data(), own.size()); }
}
template <> void exchangeOutputShares(const std::vector<uint64_t>& own, std::vector<uint64_t>& remote) {
    if (party == SERVER) { peer->send_u64(own.data(), own.size()); peer->recv_u64(remote.data(), remote.size()); }
    else { peer->recv_u64(remote.data(), remote.size()); peer->send_u64(own.data(), own.size()); }
}
void verify_dcf(const std::vector<uint64_t>& query, const std::vector<uint64_t>& alpha,
                const std::vector<CggmEval>& output) {
    std::vector<u8> own(2 * output.size()), remote(own.size());
    for (std::size_t i = 0; i < output.size(); ++i) {
        own[2 * i] = output[i].L; own[2 * i + 1] = output[i].e;
    }
    exchangeOutputShares(own, remote);
    // Check synthetic benchmark outputs after the measured window, without
    // retaining reconstructed values in the logs.
    for (std::size_t i = 0; i < output.size(); ++i) {
        if ((own[2 * i] ^ remote[2 * i]) != (query[i] < alpha[i]) ||
            (own[2 * i + 1] ^ remote[2 * i + 1]) != (query[i] == alpha[i])) {
            throw std::runtime_error("DCF output verification failed");
        }
    }
}
void verify_conversion(const reference::Contract& contract,
                       const std::vector<GroupElement>& output,
                       const std::vector<uint64_t>& values) {
    std::vector<uint64_t> own(output.size()), remote(output.size());
    for (std::size_t i = 0; i < output.size(); ++i) own[i] = output[i].value;
    exchangeOutputShares(own, remote);
    for (std::size_t i = 0; i < output.size(); ++i) {
        if (((own[i] + remote[i]) & mask(contract.output_bits)) !=
            reference::evaluate(values[i], contract)) {
            throw std::runtime_error("conversion output verification failed");
        }
    }
}
void layout_json(std::ostream& out, int bits, int nseg) {
    out << '[';
    if (bits) {
        const SegmentLayout layout(bits, nseg);
        for (std::size_t i = 0; i < layout.width.size(); ++i) {
            if (i) out << ',';
            out << layout.width[i];
        }
    }
    out << ']';
}
void run_json(std::ostream& out, const Run& r) {
    out << "{\"offline_seconds\":" << r.offline << ",\"online_seconds\":" << r.online
        << ",\"total_seconds\":" << r.total
        << ",\"offline_sent_bytes\":" << r.offline_counters.sent
        << ",\"offline_received_bytes\":" << r.offline_counters.received
        << ",\"online_sent_bytes\":" << r.online_counters.sent
        << ",\"online_received_bytes\":" << r.online_counters.received
        << ",\"offline_io_calls\":" << r.offline_counters.io
        << ",\"online_io_calls\":" << r.online_counters.io
        << ",\"online_logical_rounds\":" << r.logical_rounds
        << ",\"tree_keygen_seconds\":" << r.keygen
        << ",\"scan_preprocessing_seconds\":" << r.scan_preprocessing
        << ",\"tree_evaluation_seconds\":" << r.evaluation
        << ",\"query_prepare_seconds\":" << r.query_prepare
        << ",\"local_evaluation_seconds\":" << r.local_evaluation
        << ",\"eval_participants\":" << r.eval_participants
        << ",\"scan_seconds\":" << r.scan
        << ",\"mask_b2a_preprocessing_seconds\":" << r.masks
        << ",\"comparisons_preprocessing_seconds\":" << r.comparison_preprocessing
        << ",\"rounding_preprocessing_seconds\":" << r.rounding_preprocessing
        << ",\"masked_input_open_seconds\":" << r.opening
        << ",\"comparisons_online_seconds\":" << r.comparison_online
        << ",\"rounding_online_seconds\":" << r.rounding_online
        << ",\"masked_bit_conversion_seconds\":" << r.b2a_online
        << ",\"scan_gates\":" << r.scan_gates << ",\"scan_layers\":" << r.scan_layers
        << ",\"internal_hashes\":" << r.internal_hashes
        << ",\"leaf_hashes\":" << r.leaf_hashes
        << ",\"generation_participants\":" << r.generation_participants
        << ",\"keygen_aes_scratch_bytes\":" << r.keygen_aes_scratch_bytes
        << ",\"eval_internal_hashes\":" << r.eval_internal_hashes
        << ",\"eval_leaf_hashes\":" << r.eval_leaf_hashes
        << ",\"eval_aes_scratch_bytes\":" << r.eval_aes_scratch_bytes
        << ",\"aes_online\":" << (r.aes_online ? "true" : "false")
        << ",\"aes_offline\":" << (r.aes_offline ? "true" : "false")
        << ",\"extra_hashes\":0"
        << ",\"expanded_nodes\":" << r.expanded_nodes
        << ",\"generation_tiles\":" << r.generation_tiles
        << ",\"peak_generation_working_bytes\":" << r.peak_working_bytes
        << ",\"generation_resident_bytes\":" << r.resident_bytes
        << ",\"generation_io_calls\":" << r.generation_io_calls
        << ",\"cot_batches\":" << r.cot_batches << ",\"opening_batches\":" << r.opening_batches
        << ",\"added_cot_batches\":" << r.added_cot_batches
        << ",\"added_opening_batches\":" << r.added_opening_batches
        << ",\"correction_bits\":" << r.correction_bits
        << ",\"rounding_gates\":" << r.rounding_gates
        << ",\"masked_input_values\":" << r.masked_input_values
        << ",\"peak_rss_kib\":" << r.rss_kib
        << ",\"offline_logical_rounds\":null,\"local_arithmetic_correction_seconds\":null"
        << ",\"output_ring_conversion_seconds\":null}";
}
}  // namespace

int main(int argc, char** argv) {
    int role = 1, bits = 32, comparison_bits = 32, nseg = 1, count = 128, runs = 3, shift = 0;
    int timeout_seconds = 30, signed_arithmetic = 0;
    int cggm_threads = 0;
    std::string kind = "dcf", backend = "segmented", operation, policy_name = "legacy";
    uint64_t tile_keys = 0, generation_budget = 0, generation_reserve = 512ULL << 20;
    ArgMapping args;
    args.arg("role", role, "1=server, 2=client");
    args.arg("port", port, "TCP port");
    args.arg("bits", bits, "input bits");
    args.arg("comparison-bits", comparison_bits, "comparison domain bits");
    args.arg("nseg", nseg, "segment count or conversion policy limit");
    args.arg("segment-policy", policy_name, "legacy, fixed-2, fixed-4 or depth-cap-8");
    args.arg("generation-tile-keys", tile_keys, "public key tile cap; zero unpartitioned");
    args.arg("generation-budget-bytes", generation_budget, "public generation process budget");
    args.arg("generation-reserve-bytes", generation_reserve, "public process/scratch reserve");
    args.arg("count", count, "batch size");
    args.arg("runs", runs, "fresh measured runs");
    args.arg("shift", shift, "conversion shift");
    args.arg("kind", kind, "dcf or conversion");
    args.arg("backend", backend, "full_tree or segmented");
    args.arg("operation", operation, "conversion operation");
    args.arg("signed", signed_arithmetic, "signed conversion");
    args.arg("timeout", timeout_seconds, "external harness budget");
    args.arg("cggm-threads", cggm_threads, "local hashing workers; 0=hardware default");
    args.parse(argc, argv);
    if (cggm_threads < 0) {
        std::cerr << "UNSUPPORTED cggm-threads must be non-negative\n";
        return 3;
    }
    const auto setup_start = Clock::now();
    setCggmThreadLimit(static_cast<std::size_t>(cggm_threads));
    party = role == 1 ? SERVER : CLIENT;
    if (party == CLIENT) { server = new Peer(address, port); peer = server; }
    else { client = waitForPeer(port); peer = client; }
    const double setup_seconds = elapsed(setup_start, Clock::now());
    const auto setup_counters = counters();
    try {
        const auto policy = parseSegmentPolicy(policy_name);
        setenv("SEGFSS_GENERATION_TILE_KEYS", std::to_string(tile_keys).c_str(), 1);
        setenv("SEGFSS_GENERATION_BUDGET_BYTES", std::to_string(generation_budget).c_str(), 1);
        setenv("SEGFSS_GENERATION_RESERVE_BYTES", std::to_string(generation_reserve).c_str(), 1);
        if (kind == "dcf" && policy != SegmentPolicy::explicit_counts && backend == "segmented")
            nseg = segmentCount(comparison_bits, policy);
        if (runs < 1 || count < 1 || (backend != "full_tree" && backend != "segmented") ||
            (backend == "full_tree" && nseg != 1))
            throw std::invalid_argument("invalid sweep configuration");
        reference::Contract contract;
        SciTruncationConfig config;
        SciTruncationDomains domains;
        if (kind == "conversion") {
            contract.input_bits = bits;
            contract.output_bits = operation == "truncate_and_reduce" ? 32 : bits;
            contract.shift = shift;
            contract.call = operation == "truncate_and_reduce" ? reference::SciCall::truncate_and_reduce : reference::SciCall::div_pow2;
            contract.signed_arithmetic = signed_arithmetic;
            contract.numeric_mode = reference::NumericMode::sci_compatible;
            const auto dcf_backend = backend == "full_tree" ? DcfBackend::full_tree : DcfBackend::segmented;
            config = policy == SegmentPolicy::explicit_counts ? makeSciTruncationConfig(contract, dcf_backend, nseg) :
                makeSciTruncationConfig(contract, dcf_backend, policy);
            domains = sciTruncationDomains(config);
        } else if (kind != "dcf") throw std::invalid_argument("unknown primitive kind");
        std::vector<Run> samples;
        for (int iteration = 0; iteration < runs; ++iteration) {
            Run run;
            // Flush unused OT precomputation so every query obtains fresh correlations.
            const auto before = counters();
            if (kind == "dcf") {
                const auto alpha = thresholds(count, comparison_bits);
                const auto shares = threshold_shares(alpha, comparison_bits);
                std::vector<uint64_t> query(count);
                for (int i = 0; i < count; ++i)
                    query[i] = i % 3 == 0 ? (alpha[i] ? alpha[i] - 1 : 0) :
                        (i % 3 == 1 ? ((alpha[i] + 1) & mask(comparison_bits)) : alpha[i]);
                const auto start = Clock::now();
                auto material = segmentedDcfOfflineBatch(party, shares, comparison_bits, nseg, peer);
                const auto prepared = Clock::now();
                const auto online_before = counters();
                const auto output = segmentedDcfOnlineBatch(party, query, material, peer);
                const auto finished = Clock::now();
                run.offline = elapsed(start, prepared);
                run.online = elapsed(prepared, finished);
                run.total = elapsed(start, finished);
                run.offline_counters = difference(online_before, before);
                run.online_counters = difference(counters(), online_before);
                add_dcf_metrics(run, material);
                run.logical_rounds = run.scan_layers;
                verify_dcf(query, alpha, output);
                releaseSegmentedDcfMaterial(material);
            } else {
                std::vector<uint64_t> values(count);
                std::vector<GroupElement> inputs;
                for (int i = 0; i < count; ++i) {
                    values[i] = (uint64_t(i) * 0xd1342543de82ef95ULL + (mask(bits) >> 1)) & mask(bits);
                    const uint64_t share = (uint64_t(i) * 0x9e3779b97f4a7c15ULL + 0x8d67341ULL) & mask(bits);
                    inputs.emplace_back(party == SERVER ? share : (values[i] - share) & mask(bits), bits);
                }
                const auto start = Clock::now();
                auto material = sciTruncateOfflineBatch(party, count, config, peer);
                const auto prepared = Clock::now();
                const auto online_before = counters();
                const auto output = sciTruncateOnlineBatch(party, inputs, config, material, peer);
                const auto finished = Clock::now();
                run.offline = elapsed(start, prepared);
                run.online = elapsed(prepared, finished);
                run.total = elapsed(start, finished);
                run.offline_counters = difference(online_before, before);
                run.online_counters = difference(counters(), online_before);
                for (const auto* dcf : {&material.low_shift, &material.sign, &material.wrap}) add_dcf_metrics(run, *dcf);
                run.masks = material.metrics.mask_preprocessing_ms / 1000;
                run.comparison_preprocessing = material.metrics.comparisons_preprocessing_ms / 1000;
                run.rounding_preprocessing = material.metrics.rounding_preprocessing_ms / 1000;
                run.opening = material.metrics.masked_input_open_ms / 1000;
                run.comparison_online = material.metrics.comparisons_online_ms / 1000;
                run.rounding_online = material.metrics.rounding_online_ms / 1000;
                run.b2a_online = material.metrics.masked_bit_conversion_ms / 1000;
                run.masked_input_values = material.metrics.masked_input_values;
                run.logical_rounds = material.metrics.online_logical_rounds;
                run.correction_bits = material.conversion_mask_bits.size();
                run.rounding_gates = material.rounding.a.size();
                verify_conversion(contract, output, values);
                releaseSciTruncationMaterial(material);
            }
            struct rusage usage{};
            getrusage(RUSAGE_SELF, &usage);
            run.rss_kib = usage.ru_maxrss;
            samples.push_back(run);
        }
        std::ostringstream out;
        out << std::setprecision(12) << "RESULT {\"correctness_passed\":true,\"party\":" << role
            << ",\"cggm_thread_limit\":" << cggm_threads
            << ",\"segment_policy\":\"" << policy_name << "\""
            << ",\"generation_tile_keys\":" << tile_keys
            << ",\"generation_budget_bytes\":" << generation_budget
            << ",\"generation_reserve_bytes\":" << generation_reserve
            << ",\"setup_seconds\":" << setup_seconds
            << ",\"setup_sent_bytes\":" << setup_counters.sent
            << ",\"setup_received_bytes\":" << setup_counters.received
            << ",\"comparison_bits\":" << comparison_bits << ",\"layout\":";
        layout_json(out, comparison_bits, std::min(comparison_bits, nseg));
        out << ",\"active_comparisons\":[";
        if (kind == "conversion") {
            bool first = true;
            for (const auto& entry : {std::pair<int,int>{domains.low_shift_bits, config.low_shift_nseg},
                                     {domains.sign_bits, config.sign_nseg}, {domains.wrap_bits, config.wrap_nseg}}) {
                if (!entry.first) continue;
                if (!first) out << ',';
                first = false;
                out << "{\"bits\":" << entry.first << ",\"nseg\":" << entry.second << ",\"layout\":";
                layout_json(out, entry.first, entry.second); out << '}';
            }
        }
        out << "],\"samples\":[";
        for (std::size_t i = 0; i < samples.size(); ++i) { if (i) out << ','; run_json(out, samples[i]); }
        out << "]}\n";
        std::cout << out.str();
    } catch (const std::bad_alloc&) {
        std::cerr << "OOM out of memory\n"; delete peer; return 137;
    } catch (const std::invalid_argument& error) {
        std::cerr << "UNSUPPORTED " << error.what() << '\n'; delete peer; return 3;
    } catch (const std::exception& error) {
        std::cerr << "SWEEP_ERROR " << error.what() << '\n'; delete peer; return 2;
    }
    delete peer;
    return 0;
}
