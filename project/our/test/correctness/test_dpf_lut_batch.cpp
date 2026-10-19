#include "ArgMapping.h"
#include "dpf_lut.h"
#include "cggm.h"
#include "activation.h"

#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <fstream>
#include <random>
#include <string>
#include <vector>
#include <sys/resource.h>

using namespace sci;
int party_instance = 0, party = 0, num_threads = 1, port = 32116;
int32_t bitlength = 32;
std::string address = "127.0.0.1";
int num_argmax = 1000, length = 1;
uint8_t choice_bit = 0;
bool verbose = false;
Peer *client = nullptr, *server = nullptr, *peer = nullptr;
Dealer* dealer = nullptr;

namespace {
class SavedEnvironment {
public:
    explicit SavedEnvironment(const char* name) : name_(name) {
        const char* value = std::getenv(name);
        present_ = value != nullptr;
        if (value) value_ = value;
    }
    ~SavedEnvironment() {
        if (present_) setenv(name_.c_str(), value_.c_str(), 1);
        else unsetenv(name_.c_str());
    }
private:
    std::string name_, value_;
    bool present_ = false;
};

int checkLocalLutKernels() {
    constexpr std::size_t domain = 1024, target = 1009;
    std::mt19937_64 rng(0x51D0C1EC7ULL);
    segfss::PackedDpfSelectionVector left, right;
    left.reset(domain, 32); right.reset(domain, 32);
    std::vector<uint32_t> left_values(domain), right_values(domain);
    for (std::size_t i = 0; i < domain; ++i) {
        left_values[i] = static_cast<uint32_t>(rng());
        right_values[i] = uint32_t(i == target) - left_values[i];
        left.set(i, left_values[i]); right.set(i, right_values[i]);
    }
    // Dense values force modular overflow; support lengths around SIMD lane
    // boundaries cover tails and unaligned loads at every circular shift.
    std::vector<std::vector<uint32_t>> tables(1, std::vector<uint32_t>(domain));
    for (std::size_t length : {1U, 7U, 8U, 9U, 15U, 16U, 17U, 533U, 1024U}) {
        for (std::size_t begin : {std::size_t(0), domain - length}) {
            std::vector<uint32_t> table(domain);
            for (std::size_t i = begin; i < begin + length; ++i)
                table[i] = static_cast<uint32_t>(rng()) | 1U;
            tables.push_back(std::move(table));
        }
    }
    std::vector<uint32_t> scattered(domain), alternating(domain);
    for (std::size_t i = 0; i < domain; ++i) {
        if (i < 9 || i == 17 || i >= 1021) scattered[i] = static_cast<uint32_t>(rng()) | 1U;
        if (i % 2 == 0) alternating[i] = static_cast<uint32_t>(rng()) | 1U;
    }
    tables.push_back(std::move(scattered));
    tables.push_back(std::move(alternating));
    const auto delta = segfss::prepareDpfLutTable(segfss::makeSigmaDeltaTable(32, 12, 10), 1);
    if (delta->nonzeroCoefficientCount() != 533 || delta->nonzeroRanges().size() != 1 ||
        delta->nonzeroRanges().front() != std::make_pair(std::size_t(0), std::size_t(533)))
        return 1;
    tables.push_back(delta->coefficients());
    std::size_t checked = 0;
    for (std::size_t t = 0; t < tables.size(); ++t) {
        PublicLUTData public_table;
        public_table.Bin = 10; public_table.Bout = 32;
        for (auto value : tables[t]) public_table.values.emplace_back(value, 32);
        const auto prepared = segfss::prepareDpfLutTable(public_table, 1);
        std::size_t nonzero = 0;
        for (auto value : tables[t]) nonzero += value != 0;
        if (prepared->nonzeroCoefficientCount() != nonzero) return 1;
        for (std::size_t shift = 0; shift < domain; ++shift) {
            uint64_t reference_left = 0, reference_right = 0;
            for (std::size_t i = 0; i < domain; ++i) {
                const auto coefficient = tables[t][(i + shift) % domain];
                reference_left += uint64_t(left_values[i]) * coefficient;
                reference_right += uint64_t(right_values[i]) * coefficient;
            }
            if (left.shiftedDotProduct32(tables[t], shift) != uint32_t(reference_left) ||
                right.shiftedDotProduct32(tables[t], shift) != uint32_t(reference_right)) return 1;
            for (bool skip_zero : {false, true}) for (bool simd : {false, true}) {
                const uint32_t actual_left = left.shiftedDotProduct32(*prepared, shift, skip_zero, simd);
                const uint32_t actual_right = right.shiftedDotProduct32(*prepared, shift, skip_zero, simd);
                if (actual_left != uint32_t(reference_left) || actual_right != uint32_t(reference_right) ||
                    uint32_t(actual_left + actual_right) != tables[t][(target + shift) % domain]) {
                    std::cerr << "LUT kernel mismatch: table=" << t << " shift=" << shift
                              << " skip_zero=" << skip_zero << " simd=" << simd << '\n';
                    return 1;
                }
                ++checked;
            }
        }
    }
    // Kernel options are captured before input, including strict flag parsing.
    SavedEnvironment skip_environment("SEGFSS_LUT_SKIP_ZERO"), simd_environment("SEGFSS_LUT_SIMD");
    unsetenv("SEGFSS_LUT_SKIP_ZERO"); unsetenv("SEGFSS_LUT_SIMD");
    const segfss::DpfLutOptions defaults;
    if (defaults.skip_zero_coefficients || defaults.use_simd) return 1;
    setenv("SEGFSS_LUT_SKIP_ZERO", "1", 1); setenv("SEGFSS_LUT_SIMD", "1", 1);
    const segfss::DpfLutOptions enabled;
    if (!enabled.skip_zero_coefficients || !enabled.use_simd) return 1;
    setenv("SEGFSS_LUT_SKIP_ZERO", "invalid", 1);
    bool rejected = false;
    try { (void)segfss::DpfLutOptions(); }
    catch (const std::invalid_argument&) { rejected = true; }
    if (!rejected) return 1;
    std::cout << "LUT_LOCAL_KERNEL_TEST {\"tables\":" << tables.size()
              << ",\"rotations\":1024,\"option_combinations\":4,\"two_share_checks\":" << checked
              << ",\"avx2_available\":" << (segfss::dpfLutSimdAvailable() ? "true" : "false")
              << ",\"delta_nonzero_coefficients\":533,\"pass\":true}\n";
    return 0;
}

int checkLeafCache(Peer* channel, const PublicLUTData& table, int count, int tile) {
    SavedEnvironment cache_environment("SEGFSS_CACHE_LUT_LEAVES");
    SavedEnvironment budget_environment("SEGFSS_GENERATION_BUDGET_BYTES");
    SavedEnvironment reserve_environment("SEGFSS_GENERATION_RESERVE_BYTES");
    unsetenv("SEGFSS_GENERATION_BUDGET_BYTES");
    unsetenv("SEGFSS_GENERATION_RESERVE_BYTES");
    const auto domain = std::size_t(1) << table.Bin;
    std::vector<segfss::PackedDpfSelectionVector> baseline_selectors;
    bool ok = true;
    for (int cached = 0; cached < 2; ++cached) {
        setenv("SEGFSS_CACHE_LUT_LEAVES", cached ? "1" : "0", 1);
        // Replay the same public-fixture local tree randomness to compare
        // individual selection shares. Production queries never reset RNGs.
        channel->randomSource().SetSeed(osuCrypto::toBlock(0x1EAFCAC4ULL, party));
        segfss::DpfLutBatchMetrics metrics;
        auto material = segfss::dpfPublicLutBatchOffline(
            party, table, count, 1700, channel, 7, &metrics, tile);
        ok = ok && metrics.local_tree_expansions == std::size_t(count) * (cached ? 1 : 2) &&
            metrics.local_internal_hashes == metrics.local_tree_expansions * (domain / 2 - 1) &&
            metrics.local_leaf_hashes == metrics.local_tree_expansions * domain &&
            metrics.leaf_cache_peak_bytes == (cached ? std::min(count, tile) * domain * 16 : 0);
        for (std::size_t i = 0; i < material.size(); ++i) {
            if (!cached) baseline_selectors.push_back(material[i].selection_share);
            else for (std::size_t r = 0; r < domain; ++r)
                ok = ok && material[i].selection_share[r].value == baseline_selectors[i][r].value;
        }
        std::vector<GroupElement> inputs;
        for (int i = 0; i < count; ++i)
            inputs.emplace_back(party == SERVER ? 0 : i % domain, table.Bin);
        auto output = segfss::dpfPublicLutBatchOnlineMany(
            party, inputs, material, channel, 1700, {&table}, &metrics).front();
        channel->open_additive(output.data(), output.size(), table.Bout);
        for (int i = 0; i < count; ++i)
            ok = ok && output[i].value == table.values[i % domain].value && material[i].consumed;
        const auto before = channel->bytesSent + channel->bytesReceived;
        bool rejected = false;
        try {
            (void)segfss::dpfPublicLutBatchOnlineMany(party, inputs, material,
                channel, 1700, {&table});
        } catch (const std::invalid_argument&) { rejected = true; }
        ok = ok && rejected && channel->bytesSent + channel->bytesReceived == before;
        std::cout << "LEAF_CACHE_TEST {\"enabled\":" << cached
            << ",\"q\":" << table.Bin << ",\"count\":" << count << ",\"tile\":" << tile
            << ",\"expansions\":" << metrics.local_tree_expansions
            << ",\"cache_peak_bytes\":" << metrics.leaf_cache_peak_bytes
            << ",\"peak_working_bytes\":" << metrics.peak_working_bytes
            << ",\"pass\":" << (ok ? "true" : "false") << "}\n";
        for (auto& entry : material) segfss::releaseDpfPublicLutMaterial(entry);
    }
    // A budget smaller than even one request must fail before preprocessing IO.
    setenv("SEGFSS_GENERATION_BUDGET_BYTES", "1", 1);
    setenv("SEGFSS_GENERATION_RESERVE_BYTES", "0", 1);
    const auto before = channel->bytesSent + channel->bytesReceived;
    bool rejected = false;
    try {
        (void)segfss::dpfPublicLutBatchOffline(party, table, count, 1800, channel, 8, nullptr, tile);
    } catch (const std::invalid_argument&) { rejected = true; }
    ok = ok && rejected && channel->bytesSent + channel->bytesReceived == before;
    if (table.Bin == 10 && count == 33 && tile == 32) {
        // This budget fits the final selectors and mask preprocessing, but
        // needs a smaller tile than the requested 32-key limit.
        constexpr uint64_t budget = 1572864;
        setenv("SEGFSS_GENERATION_BUDGET_BYTES", "1572864", 1);
        segfss::DpfLutBatchMetrics metrics;
        auto material = segfss::dpfPublicLutBatchOffline(
            party, table, count, 1900, channel, 9, &metrics, tile);
        ok = ok && metrics.tiles >= 2 && metrics.leaf_cache_peak_bytes < std::size_t(tile) * domain * 16 &&
            metrics.resident_bytes + metrics.peak_working_bytes <= budget;
        std::vector<GroupElement> masks, selectors;
        for (auto& entry : material) {
            masks.push_back(entry.mask_share);
            for (std::size_t r = 0; r < domain; ++r) selectors.push_back(entry.selection_share[r]);
        }
        channel->open_additive(masks.data(), masks.size(), table.Bin);
        channel->open_additive(selectors.data(), selectors.size(), table.Bout);
        for (int i = 0; i < count; ++i)
            for (std::size_t r = 0; r < domain; ++r)
                ok = ok && selectors[i * domain + r].value == uint64_t(r == masks[i].value);
        for (auto& entry : material) segfss::releaseDpfPublicLutMaterial(entry);
        std::cout << "LEAF_CACHE_BUDGET {\"tiles\":" << metrics.tiles
            << ",\"cache_peak_bytes\":" << metrics.leaf_cache_peak_bytes
            << ",\"pass\":" << (ok ? "true" : "false") << "}\n";
    }
    delete channel;
    return ok ? 0 : 1;
}

struct ProcessMemory {
    uint64_t rss_kib = 0, address_space_kib = 0, peak_rss_kib = 0;
};
ProcessMemory processMemory() {
    ProcessMemory memory;
    std::ifstream status("/proc/self/status");
    std::string field, rest;
    while (status >> field) {
        if (field == "VmRSS:") status >> memory.rss_kib;
        if (field == "VmSize:") status >> memory.address_space_kib;
        std::getline(status, rest);
    }
    rusage usage{};
    if (getrusage(RUSAGE_SELF, &usage) == 0) memory.peak_rss_kib = usage.ru_maxrss;
    return memory;
}
int checkPaperLut(Peer* channel, const PublicLUTData& table, int count, int tile,
                  bool segmented = false) {
    const auto prepared = segfss::prepareDpfLutTable(table, 1);
    segfss::DpfLutOptions options;
    options.tile_size = tile;
    options.participant_limit = 2;
    if (segmented) options.family = segfss::DpfLutKeyFamily::segmented_boolean_tensor_b2a;
    options.segment_widths = segmented ? std::vector<int>{5, 5} : std::vector<int>{10};
    segfss::DpfLutBinding binding;
    binding.context_id = 19;
    binding.operator_id = 41;
    binding.worker_id = 2;
    binding.slot_id = 3;
    binding.element_begin = 101;
    binding.table_version = 1;
    binding.family = options.family;
    bool ok = true;
    std::vector<uint64_t> previous_masks;
    for (uint64_t query : {7U, 8U}) {
        binding.query_id = query;
        segfss::DpfLutBatchMetrics metrics;
        const auto generated = channel->preprocessing_ledger;
        auto material = segfss::dpfPublicLutBatchOffline(
            party, *prepared, count, binding, options, channel, query, &metrics);
        const auto prepare_memory = processMemory();
        ok = ok && metrics.generated == static_cast<uint64_t>(count) &&
            metrics.remaining == static_cast<uint64_t>(count) &&
            metrics.key_trees == static_cast<uint64_t>(count) * (segmented ? 2 : 1) &&
            metrics.tree_max_depth == (segmented ? 5 : 10) &&
            metrics.mask_b2a_values == static_cast<uint64_t>(count) * 10 &&
            metrics.tensor_and_gates == static_cast<uint64_t>(segmented ? count * 1024 : 0) &&
            metrics.leaf_b2a_values == static_cast<uint64_t>(segmented ? count * 1024 : 0) &&
            channel->preprocessing_ledger.boolean_triples - generated.boolean_triples ==
                static_cast<uint64_t>(segmented ? count * 1024 : 0) &&
            channel->preprocessing_ledger.arithmetic_triples == generated.arithmetic_triples &&
            channel->preprocessing_ledger.b2a_values - generated.b2a_values ==
                static_cast<uint64_t>(count) * (segmented ? 1034 : 10);
        // This public-fixture protocol audit is separate from any performance
        // run: reveal masks/selectors only here to verify their common target.
        std::vector<GroupElement> masks, selectors;
        for (const auto& entry : material) {
            masks.push_back(entry.mask_share);
            ok = ok && entry.selection_share.capacityBytes() == 4096 &&
                entry.selection_share.size() == 1024 && entry.key.Bin == 0 &&
                entry.lut_skip_zero == options.skip_zero_coefficients && entry.lut_simd == options.use_simd;
            for (std::size_t u = 0; u < 1024; ++u)
                selectors.push_back(entry.selection_share[u]);
        }
        channel->open_additive(masks.data(), masks.size(), 10);
        channel->open_additive(selectors.data(), selectors.size(), 32);
        std::vector<uint64_t> current_masks;
        bool saw_wrap = false, saw_unwrapped = false;
        for (int i = 0; i < count; ++i) {
            current_masks.push_back(masks[i].value);
            saw_wrap |= static_cast<uint64_t>(i & 1023) < masks[i].value;
            saw_unwrapped |= static_cast<uint64_t>(i & 1023) >= masks[i].value;
            for (std::size_t u = 0; u < 1024; ++u)
                ok = ok && selectors[static_cast<std::size_t>(i) * 1024 + u].value ==
                    static_cast<uint64_t>(u == masks[i].value);
        }
        if (count >= 1024) ok = ok && saw_wrap && saw_unwrapped;
        if (!previous_masks.empty()) ok = ok && previous_masks != current_masks;
        previous_masks = std::move(current_masks);
        std::mt19937_64 shares(0x45AE57ULL + query);
        std::vector<GroupElement> inputs;
        for (int i = 0; i < count; ++i) {
            const auto left = shares() & 1023;
            inputs.emplace_back(party == SERVER ? left : (i & 1023) - left, 10);
        }
        const auto offline = channel->preprocessing_ledger;
        const auto products = channel->bit_block_ledger.products;
        const auto rounds = channel->rounds;
        auto outputs = segfss::dpfPublicLutBatchOnline(
            party, inputs, material, binding, channel, *prepared, &metrics);
        ok = ok && metrics.online_open_batches == 1 && metrics.remaining == 0 &&
            metrics.online_generated == 0 && metrics.online_loaded_bytes == 0 &&
            channel->bit_block_ledger.products == products &&
            channel->preprocessing_ledger.b2a_values == offline.b2a_values &&
            channel->preprocessing_ledger.boolean_triples == offline.boolean_triples &&
            channel->preprocessing_ledger.arithmetic_triples == offline.arithmetic_triples &&
            channel->rounds - rounds == 2 &&
            metrics.online_sent_bytes == static_cast<uint64_t>(count) * 2 &&
            metrics.online_received_bytes == static_cast<uint64_t>(count) * 2;
        channel->open_additive(outputs.data(), outputs.size(), 32);
        for (int i = 0; i < count; ++i)
            ok = ok && outputs[i].value == table.values[i & 1023].value;
        const auto bytes = channel->bytesSent + channel->bytesReceived;
        bool rejected = false;
        try {
            (void)segfss::dpfPublicLutBatchOnline(
                party, inputs, material, binding, channel, *prepared);
        } catch (const std::invalid_argument&) { rejected = true; }
        ok = ok && rejected && bytes == channel->bytesSent + channel->bytesReceived;
        std::cout << "PAPER_LUT {\"query\":" << query << ",\"count\":" << count
                  << ",\"family\":\"" << (segmented ? "segmented_boolean_tensor_b2a" : "full_q_arithmetic")
                  << "\",\"mask_us\":" << metrics.mask_stage.elapsed_us
                  << ",\"keygen_us\":" << metrics.tree_keygen_stage.elapsed_us
                  << ",\"payload_us\":" << metrics.payload_stage.elapsed_us
                  << ",\"materialize_us\":" << metrics.materialize_stage.elapsed_us
                  << ",\"eval_all_us\":" << metrics.eval_all_us
                  << ",\"offline_us\":" << metrics.offline_us
                  << ",\"local_tree_expansions\":" << metrics.local_tree_expansions
                  << ",\"local_internal_hashes\":" << metrics.local_internal_hashes
                  << ",\"local_leaf_hashes\":" << metrics.local_leaf_hashes
                  << ",\"leaf_cache_peak_bytes\":" << metrics.leaf_cache_peak_bytes
                  << ",\"open_us\":" << metrics.open_stage.elapsed_us
                  << ",\"product_us\":" << metrics.product_stage.elapsed_us
                  << ",\"offline_sent\":" << metrics.offline_sent_bytes
                  << ",\"offline_received\":" << metrics.offline_received_bytes
                  << ",\"online_sent\":" << metrics.online_sent_bytes
                  << ",\"online_received\":" << metrics.online_received_bytes
                  << ",\"resident_bytes\":" << metrics.resident_bytes
                  << ",\"peak_working_bytes\":" << metrics.peak_working_bytes
                  << ",\"prepare_rss_kib\":" << prepare_memory.rss_kib
                  << ",\"prepare_as_kib\":" << prepare_memory.address_space_kib
                  << ",\"prepare_peak_rss_kib\":" << prepare_memory.peak_rss_kib
                  << ",\"lut_segments\":" << (segmented ? 2 : 1)
                  << ",\"key_trees\":" << metrics.key_trees
                  << ",\"tree_max_depth\":" << metrics.tree_max_depth
                  << ",\"mask_b2a_values\":" << metrics.mask_b2a_values
                  << ",\"tensor_and_gates\":" << metrics.tensor_and_gates
                  << ",\"leaf_b2a_values\":" << metrics.leaf_b2a_values
                  << ",\"stages\":{";
        bool first_stage = true;
        auto stage = [&](const char* name, const segfss::DpfLutStageMetric& value) {
            if (!first_stage) std::cout << ',';
            first_stage = false;
            std::cout << '"' << name << "\":{\"us\":" << value.elapsed_us
                      << ",\"sent\":" << value.sent_bytes
                      << ",\"received\":" << value.received_bytes
                      << ",\"channel_rounds\":" << value.channel_rounds << '}';
        };
        stage("mask", metrics.mask_stage);
        stage("tree_keygen", metrics.tree_keygen_stage);
        stage("payload", metrics.payload_stage);
        stage("materialize", metrics.materialize_stage);
        stage("tensor_and", metrics.tensor_and_stage);
        stage("leaf_b2a", metrics.leaf_b2a_stage);
        stage("open", metrics.open_stage);
        stage("product", metrics.product_stage);
        std::cout << "},\"one_hot\":true,\"pass\":" << (ok ? "true" : "false") << "}\n";
        for (auto& entry : material) segfss::releaseDpfPublicLutMaterial(entry);
    }
    auto changed_table = table;
    changed_table.values.back() = GroupElement(changed_table.values.back().value + 1, 32);
    const auto changed_prepared = segfss::prepareDpfLutTable(changed_table, 1);
    for (int failure = 0; failure < 14; ++failure) {
        binding.query_id = 20 + failure;
        auto material = segfss::dpfPublicLutBatchOffline(
            party, *prepared, 3, binding, options, channel, 20 + failure);
        auto wrong = binding;
        if (failure == 0) ++wrong.query_id;
        if (failure == 1) ++wrong.operator_id;
        if (failure == 2) ++wrong.worker_id;
        if (failure == 3) ++wrong.slot_id;
        if (failure == 4) ++wrong.element_begin;
        if (failure == 5) ++wrong.table_version;
        if (failure == 6) ++wrong.context_id;
        if (failure == 7) material.back().family = segmented
            ? segfss::DpfLutKeyFamily::full_q_arithmetic
            : segfss::DpfLutKeyFamily::segmented_boolean_tensor_b2a;
        if (failure == 8) material.pop_back();
        segfss::DpfPublicLutMaterial moved;
        if (failure == 11) moved = std::move(material.back());
        if (failure == 12) ++material.back().lut_segments;
        if (failure == 13) ++material.back().segment_bits[0];
        const auto bytes = channel->bytesSent + channel->bytesReceived;
        bool rejected = false;
        try {
            (void)segfss::dpfPublicLutBatchOnline(failure == 9 ? 5 - party : party,
                std::vector<GroupElement>(3, GroupElement(0, 10)), material,
                wrong, channel, failure == 10 ? *changed_prepared : *prepared);
        } catch (const std::invalid_argument&) { rejected = true; }
        ok = ok && rejected && bytes == channel->bytesSent + channel->bytesReceived;
        for (auto& entry : material) {
            ok = ok && entry.consumed;
            segfss::releaseDpfPublicLutMaterial(entry);
        }
        segfss::releaseDpfPublicLutMaterial(moved);
    }
    for (int failure = 0; failure < 3; ++failure) {
        auto malformed = options;
        auto wrong_binding = binding;
        if (failure == 0) malformed.segment_widths = {4, 6};
        if (failure == 1) malformed.family = static_cast<segfss::DpfLutKeyFamily>(42);
        if (failure == 2) wrong_binding.family = segmented
            ? segfss::DpfLutKeyFamily::full_q_arithmetic
            : segfss::DpfLutKeyFamily::segmented_boolean_tensor_b2a;
        const auto bytes = channel->bytesSent + channel->bytesReceived;
        bool rejected = false;
        try {
            (void)segfss::dpfPublicLutBatchOffline(party, *prepared, 3, wrong_binding,
                malformed, channel, 1);
        } catch (const std::invalid_argument&) { rejected = true; }
        ok = ok && rejected && bytes == channel->bytesSent + channel->bytesReceived;
    }
    std::cout << "PAPER_LUT_CONTRACT queries=2 malformed_inventories=14 malformed_options=3 "
              << "reject_before_IO=1 consume_all=1 status=" << (ok ? "passed" : "failed") << '\n';
    delete channel;
    return ok ? 0 : 1;
}
}  // namespace

int main(int argc, char** argv) {
    int q = 3, bits = 9, other = 7, count = 11, tile = 256, mode = 0;
    ArgMapping args;
    args.arg("r", party, "SERVER=2 CLIENT=3");
    args.arg("p", port, "Port");
    args.arg("q", q, "Index width");
    args.arg("bits", bits, "Selection/output ring");
    args.arg("other", other, "Second output ring");
    args.arg("count", count, "Independent requests");
    args.arg("tile", tile, "Offline generation tile");
    args.arg("mode", mode, "0 many outputs, 1 designated table, 2 full-q paper, 3 segmented paper, 4 leaf-cache equivalence, 5 local kernels");
    args.parse(argc, argv);
    if (mode == 5) return checkLocalLutKernels();
    if (party != SERVER && party != CLIENT) return 2;
    if (q < 1 || q > (mode >= 2 ? 10 : 8) || count < 1 || bits < 1 || bits > 62 ||
        other < 1 || other > bits || tile < 1) return 2;
    Peer* channel = party == SERVER ? waitForPeer(port) : new Peer(address, port);
    // Exercise the production slot boundary: preprocessing and lookup must
    // use the explicit channel even when the legacy global peer is unset.
    peer = nullptr;
    segfss::setCggmThreadLimit(2);
    const uint64_t index_mask = (uint64_t(1) << q) - 1;
    const uint64_t wide_mask = (uint64_t(1) << bits) - 1;
    const uint64_t narrow_mask = (uint64_t(1) << other) - 1;
    PublicLUTData first, second;
    first.Bin = second.Bin = q;
    first.Bout = bits;
    second.Bout = other;
    for (uint64_t x = 0; x <= index_mask; ++x) {
        // Large, irregular coefficients exercise reduction and field carry.
        first.values.emplace_back((wide_mask - 13 * x) & wide_mask, bits);
        second.values.emplace_back((3 + 29 * x * x) & narrow_mask, other);
    }
    if (mode == 4) return checkLeafCache(channel, first, count, tile);
    if (mode == 2 || mode == 3) {
        if (q != 10 || bits != 32) return 2;
        return checkPaperLut(channel, first, count, tile, mode == 3);
    }
    segfss::DpfLutBatchMetrics metrics;
    auto materials = segfss::dpfPublicLutBatchOffline(
        party, first, static_cast<std::size_t>(count), 1700, channel, 7,
        &metrics, static_cast<std::size_t>(tile));
    bool ok = metrics.generated == static_cast<uint64_t>(count) &&
        metrics.generation_epoch == 7 && metrics.eval_points ==
        static_cast<uint64_t>(count) * (index_mask + 1) &&
        metrics.online_generated == 0 && metrics.online_loaded_bytes == 0;
    for (const auto& material : materials) {
        ok = ok && !material.consumed && !material.released &&
            material.selection_share.size() == index_mask + 1 &&
            material.selection_share.capacityBytes() ==
                (index_mask + 1) * static_cast<uint64_t>((bits + 7) / 8);
    }
    // Private shares first appear after the complete resident batch exists.
    std::mt19937_64 rng(0x45AE57ULL);
    std::vector<GroupElement> inputs;
    for (int i = 0; i < count; ++i) {
        const uint64_t share = rng() & index_mask;
        const uint64_t x = static_cast<uint64_t>(i) & index_mask;
        inputs.emplace_back(party == SERVER ? share : x - share, q);
    }
    std::vector<std::vector<GroupElement>> outputs;
    PublicLUTData alternative = first;
    for (uint64_t x = 0; x <= index_mask; ++x)
        alternative.values[x] = GroupElement((19 + 21*x*x) & wide_mask, bits);
    if (mode == 0) {
        outputs = segfss::dpfPublicLutBatchOnlineMany(
            party, inputs, materials, channel, 1700, {&first, &second}, &metrics);
    } else {
        std::vector<const PublicLUTData*> designated;
        for (int i = 0; i < count; ++i)
            designated.push_back((i & 1) ? &alternative : &first);
        outputs.push_back(segfss::dpfPublicLutBatchOnlinePerInput(
            party, inputs, materials, channel, 1700, designated, &metrics));
    }
    ok = ok && outputs.size() == (mode == 0 ? 2U : 1U) &&
        metrics.consumed == static_cast<uint64_t>(count) && metrics.remaining == 0 &&
        metrics.online_open_batches == 1 && metrics.online_generated == 0 &&
        metrics.online_loaded_bytes == 0;
    for (std::size_t output = 0; output < outputs.size(); ++output) {
        const auto& table = output == 0 ? first : second;
        peer = channel;
        reconstruct(count, outputs[output].data(), table.Bout);
        peer = nullptr;
        for (int i = 0; i < count; ++i) {
            const auto& expected_table = mode == 1 && (i & 1) ? alternative : table;
            ok = ok && outputs[output][i].bitsize == table.Bout &&
                outputs[output][i].value == expected_table.values[i & index_mask].value;
        }
    }
    const uint64_t before = channel->bytesSent + channel->bytesReceived;
    bool rejected = false;
    try {
        (void)segfss::dpfPublicLutBatchOnlineMany(
            party, inputs, materials, channel, 1700, {&first, &second});
    } catch (const std::invalid_argument&) { rejected = true; }
    ok = ok && rejected && before == channel->bytesSent + channel->bytesReceived;
    // Malformed metadata must poison the entire resident inventory without
    // any exchange. Independent fresh batches exercise epoch, request identity,
    // missing material, input width, and designated output-table validation.
    if (count <= 300) {
        for (int failure = 0; failure < 5; ++failure) {
            auto bad = segfss::dpfPublicLutBatchOffline(party, first, 3,
                4000 + failure * 3, channel, 20 + failure, nullptr, 3);
            std::vector<GroupElement> bad_inputs(3, GroupElement(0, q));
            auto bad_table = first;
            if (failure == 0) ++bad.back().generation_epoch;
            if (failure == 1) ++bad.back().request_id;
            if (failure == 2) bad.pop_back();
            if (failure == 3) bad_inputs.back().bitsize = q + 1;
            if (failure == 4) bad_table.values.back().bitsize = bits - 1;
            const auto bytes = channel->bytesSent + channel->bytesReceived;
            bool failed = false;
            try {
                (void)segfss::dpfPublicLutBatchOnlineMany(party, bad_inputs,
                    bad, channel, 4000 + failure * 3, {&bad_table});
            } catch (const std::invalid_argument&) { failed = true; }
            ok = ok && failed && bytes == channel->bytesSent + channel->bytesReceived;
            for (auto& item : bad) {
                ok = ok && item.consumed;
                segfss::releaseDpfPublicLutMaterial(item);
            }
        }
    }
    std::cout << "{\"party\":" << party << ",\"q\":" << q
              << ",\"bits\":" << bits << ",\"other\":" << other
              << ",\"requests\":" << count << ",\"mode\":" << mode << ",\"tiles\":" << metrics.tiles
              << ",\"resident_bytes\":" << metrics.resident_bytes
              << ",\"peak_working_bytes\":" << metrics.peak_working_bytes
              << ",\"offline_sent\":" << metrics.offline_sent_bytes
              << ",\"offline_received\":" << metrics.offline_received_bytes
              << ",\"online_sent\":" << metrics.online_sent_bytes
              << ",\"online_received\":" << metrics.online_received_bytes
              << ",\"offline_us\":" << metrics.offline_us
              << ",\"keygen_us\":" << metrics.keygen_us
              << ",\"eval_all_us\":" << metrics.eval_all_us
              << ",\"online_open_us\":" << metrics.online_open_us
              << ",\"inner_product_us\":" << metrics.inner_product_us
              << ",\"pass\":" << (ok ? "true" : "false") << "}\n";
    for (auto& material : materials) segfss::releaseDpfPublicLutMaterial(material);
    delete channel;
    return ok ? 0 : 1;
}
