#include "segmented_dcf.h"
#include "fss/dpf.h"
#include "mpc/api.h"

#include <algorithm>
#include <atomic>
#include <cstdint>
#include <cstdlib>
#include <future>
#include <iostream>
#include <stdexcept>
#include <string>
#include <thread>
#include <type_traits>
#include <utility>

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
void require(bool condition, const std::string& message) {
    if (!condition) throw std::runtime_error(message);
}

template <class Function>
void invalid(Function&& function, const std::string& name) {
    bool caught = false;
    try { function(); } catch (const std::invalid_argument&) { caught = true; }
    require(caught, name + " must reject invalid input");
}

std::vector<u8> bitsOf(uint64_t value, int bits) {
    std::vector<u8> output(bits);
    for (int bit = 0; bit < bits; ++bit) output[bit] = (value >> (bits - bit - 1)) & 1;
    return output;
}

uint64_t maskOf(int bits) { return bits == 64 ? UINT64_MAX : (uint64_t(1) << bits) - 1; }

std::vector<u8> thresholdShare(uint64_t threshold, int bits) {
    const uint64_t share = (threshold * 0x9e3779b97f4a7c15ULL + 0x6c8e9cf5ULL) & maskOf(bits);
    return bitsOf(party == SERVER ? share : threshold ^ share, bits);
}

void validation() {
    static_assert(!std::is_copy_constructible<segfss::SegmentedDcfMaterial>::value,
                  "comparison material must have unique ownership");
    static_assert(!std::is_copy_assignable<segfss::SegmentedDcfMaterial>::value,
                  "comparison material cannot be copied by assignment");
    for (const auto shape : {std::pair<int, int>{0, 1}, {65, 1}, {1, 0}, {1, 2}, {8, -1}}) {
        invalid([&] { segfss::SegmentLayout layout(shape.first, shape.second); }, "layout");
    }
    invalid([] { segfss::segmentedDcfOffline(2, {0}, 1, 0, nullptr); }, "offline null peer");
    const segfss::SegmentLayout wide(64, 1);
    require(segfss::segmentDigit(GroupElement(UINT64_MAX, 64), wide, 0).value == UINT64_MAX,
            "64-bit segment must preserve the complete unsigned word");
    segfss::DcfGenerationOptions tiled;
    tiled.tile_keys = 3;
    tiled.budget_bytes = 64ULL << 20;
    tiled.reserve_bytes = 1ULL << 20;
    const auto plan = segfss::planDcfGeneration(7, segfss::SegmentLayout(17, 3), tiled);
    require(plan.tiles.size() == 7 && plan.tiles.back().key_count == 3,
            "generation tiles preserve row-major odd segments");
    const auto tail = segfss::planDcfGeneration(7, segfss::SegmentLayout(9, 2), tiled);
    require(tail.tiles.size() == 5 && tail.tiles.back().key_count == 2,
            "generation plan retains the partial final tile");
    for (const auto& tile : tail.tiles)
        require(tile.working_bytes + tail.resident_bytes + tiled.reserve_bytes <= tiled.budget_bytes,
                "tile reserve includes resident material and public process reserve");
    tiled.budget_bytes = 1;
    invalid([&] { segfss::planDcfGeneration(128, segfss::SegmentLayout(32, 2), tiled); },
            "insufficient generation budget");

    // These gate counts are worked examples of the balanced merge tree:
    // L = L_left XOR (e_left AND L_right); only requested e outputs add an AND.
    for (const auto shape : {std::pair<int, std::size_t>{1, 0}, {2, 1}, {3, 3},
                             {4, 4}, {5, 7}, {6, 8}, {7, 10}, {8, 11}}) {
        const auto less = segfss::planDcfScan(shape.first, false, true);
        const auto both = segfss::planDcfScan(shape.first, true, true);
        const auto original = segfss::planDcfScan(shape.first, false, false);
        require(less.and_gates_per_row == shape.second &&
                both.and_gates_per_row == std::size_t(2 * (shape.first - 1)) &&
                original.and_gates_per_row == both.and_gates_per_row,
                "public demand plan must retain all required intermediate equalities");
        require(less.need_e.front().back() == 0 &&
                std::count(less.need_e.front().begin(), less.need_e.front().end(), u8(0)) == 1,
                "only the final leaf equality is unused in an L-only tree");
    }
}

void sameShareLeafPruning() {
    std::vector<segfss::CggmKey> keys;
    std::vector<std::vector<u8>> queries;
    std::vector<u8> need_e;
    // Same keys, every last direction, L-only and (L,e), mixed depths.
    for (int depth : {1, 3, 8}) for (int direction : {0, 1}) for (bool equal : {false, true}) {
        auto pair = segfss::cggmGenDealer(bitsOf(1, depth));
        keys.push_back(std::move(pair.first));
        keys.push_back(std::move(pair.second));
        queries.push_back(bitsOf(direction, depth));
        queries.push_back(bitsOf(direction, depth));
        need_e.push_back(equal); need_e.push_back(equal);
    }
    segfss::CggmKeyBatch compact; compact.append(keys);
    std::vector<uint64_t> words;
    for (const auto& bits : queries) words.push_back(bits.back());
    for (bool aes : {false, true}) {
        setenv("SEGFSS_AES_ONLINE", aes ? "1" : "0", 1);
        segfss::CggmEvaluationStats original_stats, pruned_stats, compact_stats;
        const auto original = segfss::cggmEvalBatch(keys, queries, need_e, &original_stats, false);
        const auto pruned = segfss::cggmEvalBatch(keys, queries, need_e, &pruned_stats, true);
        std::vector<segfss::CggmEval> compact_output;
        segfss::cggmEvalBatch(compact, words, {0}, 1, compact_output, need_e, &compact_stats, true);
        require(original_stats.leaf_hashes == 48 && pruned_stats.leaf_hashes == 24 &&
                compact_stats.leaf_hashes == 24,
                "actual leaf hashes: L/e directions 0/1 use 2/1; L-only uses 1/0");
        for (std::size_t index = 0; index < keys.size(); ++index) {
            const auto scalar = segfss::cggmEval(keys[index], queries[index], need_e[index], true);
            require(pruned[index].L == original[index].L &&
                    pruned[index].e == original[index].e &&
                    compact_output[index].L == original[index].L &&
                    compact_output[index].e == original[index].e &&
                    scalar.L == original[index].L && scalar.e == original[index].e,
                    "leaf pruning preserves the same material's required local shares");
        }
    }
    unsetenv("SEGFSS_AES_ONLINE");
    std::cout << "same_share_leaf_pruning mixed_depths=1,3,8 aes=0,1 PASS\n";
}

// Exhaustive reconstructed-domain check. Key generation is the deterministic
// 2PC simulator; only the segment scan here uses plaintext reconstructed bits.
void localExhaustive() {
    osuCrypto::PRNG random(osuCrypto::toBlock(0x020002ULL, 0x53444346ULL));
    uint64_t comparisons = 0;
    for (int bits = 1; bits <= 8; ++bits) {
        const uint64_t domain = uint64_t(1) << bits;
        for (uint64_t alpha = 0; alpha < domain; ++alpha) {
            auto full = segfss::cggmGenDealer(bitsOf(alpha, bits));
            for (int nseg = 1; nseg <= bits; ++nseg) {
                segfss::SegmentLayout layout(bits, nseg);
                std::vector<std::pair<segfss::CggmKey, segfss::CggmKey>> segments;
                for (int segment = 0; segment < nseg; ++segment) {
                    const int width = layout.width[segment];
                    const uint64_t target = (alpha >> layout.shift[segment]) & maskOf(width);
                    const uint64_t share = random.get<uint64_t>() & maskOf(width);
                    segments.push_back(segfss::cggmGen2pcSim(bitsOf(share, width),
                        bitsOf(share ^ target, width), random.get<osuCrypto::block>(),
                        random.get<osuCrypto::block>(), random));
                }
                for (uint64_t query = 0; query < domain; ++query) {
                    u8 less = 0, equal = 1;
                    for (int segment = 0; segment < nseg; ++segment) {
                        const auto query_bits = bitsOf(query >> layout.shift[segment], layout.width[segment]);
                        const auto a = segfss::cggmEval(segments[segment].first, query_bits);
                        const auto b = segfss::cggmEval(segments[segment].second, query_bits);
                        less ^= equal & (a.L ^ b.L);
                        equal &= a.e ^ b.e;
                    }
                    const auto query_bits = bitsOf(query, bits);
                    const auto a = segfss::cggmEval(full.first, query_bits);
                    const auto b = segfss::cggmEval(full.second, query_bits);
                    require(less == (query < alpha) && equal == (query == alpha) &&
                            less == (a.L ^ b.L) && equal == (a.e ^ b.e),
                            "local exhaustive mismatch: bits=" + std::to_string(bits) +
                            " nseg=" + std::to_string(nseg) + " alpha=" + std::to_string(alpha) +
                            " query=" + std::to_string(query));
                    ++comparisons;
                }
            }
        }
        std::cout << "local_2pc_simulation bits=" << bits << " nseg=1.." << bits << " PASS\n";
    }
    std::cout << "local_2pc_simulation comparisons=" << comparisons
              << " assertions=" << comparisons * 4 << " PASS\n";
}

void localBatchExecution() {
    std::vector<segfss::CggmKey> first_keys, second_keys;
    std::vector<std::vector<u8>> queries;
    std::vector<segfss::CggmEval> expected;
    // Mixed depths cover uneven segments, singleton leaves, and 64-bit
    // domains; the total walk depth exceeds the parallel execution threshold.
    for (int row = 0; row < 128; ++row) {
        for (int depth : {1, 4, 7, 16, 64}) {
            const uint64_t mask = maskOf(depth);
            const uint64_t alpha = (row * 0x9e3779b97f4a7c15ULL) & mask;
            const uint64_t candidates[] = {0, mask, alpha, (alpha - 1) & mask,
                                           (alpha + 1) & mask};
            const uint64_t query = candidates[row % 5];
            auto keys = segfss::cggmGenDealer(bitsOf(alpha, depth),
                osuCrypto::toBlock(0x53444346ULL, static_cast<uint64_t>(2 * row + 1)));
            first_keys.push_back(std::move(keys.first));
            second_keys.push_back(std::move(keys.second));
            queries.push_back(bitsOf(query, depth));
            expected.push_back({static_cast<u8>(query < alpha),
                                static_cast<u8>(query == alpha)});
        }
    }
    segfss::setCggmThreadLimit(1);
    const auto serial_first = segfss::cggmEvalBatch(first_keys, queries);
    const auto serial_second = segfss::cggmEvalBatch(second_keys, queries);
    segfss::CggmEvaluationStats l_only_stats;
    const auto l_only = segfss::cggmEvalBatch(first_keys, queries, &l_only_stats, false);
    require(l_only.size() == serial_first.size() && l_only_stats.leaf_hashes <= serial_first.size() * 2,
            "L-only cGGM evaluation must prune unused leaf hashes");
    for (std::size_t index = 0; index < l_only.size(); ++index)
        require(l_only[index].L == serial_first[index].L && l_only[index].e == 0,
                "L-only cGGM output");
    segfss::CggmKeyBatch packed_first, packed_second;
    packed_first.append(first_keys);
    packed_second.append(second_keys);
    std::vector<uint64_t> public_queries;
    std::vector<int> shifts{0};
    for (const auto& bits : queries) {
        uint64_t query = 0;
        for (auto bit : bits) query = (query << 1) | bit;
        public_queries.push_back(query);
    }
    std::vector<segfss::CggmEval> packed_output;
    segfss::cggmEvalBatch(packed_first, public_queries, shifts, 1, packed_output);
    for (std::size_t index = 0; index < packed_output.size(); ++index)
        require(packed_output[index].L == serial_first[index].L &&
                packed_output[index].e == serial_first[index].e, "compact query evaluation");
    for (const std::size_t count : {1u, 7u, 8u, 9u, 257u, 640u}) {
        std::vector<segfss::CggmKey> subset(first_keys.begin(), first_keys.begin() + count);
        std::vector<std::vector<u8>> bits(queries.begin(), queries.begin() + count);
        std::vector<uint64_t> words(public_queries.begin(), public_queries.begin() + count);
        segfss::CggmKeyBatch compact;
        compact.append(subset);
        for (const std::size_t threads : {1u, 4u}) {
            segfss::setCggmThreadLimit(threads);
            setenv("SEGFSS_AES_ONLINE", "1", 1);
            segfss::CggmEvaluationStats compact_stats, legacy_stats;
            const auto legacy = segfss::cggmEvalBatch(subset, bits, &legacy_stats);
            segfss::cggmEvalBatch(compact, words, shifts, 1, packed_output, &compact_stats);
            std::size_t internal_hashes = 0;
            for (std::size_t index = 0; index < count; ++index) {
                internal_hashes += subset[index].depth - 1;
                require(legacy[index].L == serial_first[index].L && legacy[index].e == serial_first[index].e &&
                        packed_output[index].L == serial_first[index].L && packed_output[index].e == serial_first[index].e,
                        "batched AES preserves individual shares, mixed-depth mapping and tail order");
            }
            require(compact_stats.batched_aes && legacy_stats.batched_aes &&
                    compact_stats.internal_hashes == internal_hashes && compact_stats.leaf_hashes == 2 * count &&
                    legacy_stats.internal_hashes == internal_hashes && legacy_stats.leaf_hashes == 2 * count &&
                    compact_stats.scratch_peak_bytes > 0 && compact_stats.participants <= threads,
                    "batched AES reports exact hashes and bounded participants");
            unsetenv("SEGFSS_AES_ONLINE");
        }
    }
    segfss::setCggmThreadLimit(1);
    auto moved_batch = std::move(packed_second);
    segfss::cggmEvalBatch(moved_batch, public_queries, shifts, 1, packed_output);
    for (std::size_t index = 0; index < packed_output.size(); ++index)
        require(packed_output[index].L == serial_second[index].L &&
                packed_output[index].e == serial_second[index].e, "moved compact evaluation");
    segfss::setCggmThreadLimit(4);
    const auto configured_limit = segfss::cggmThreadLimit();
    for (bool aes : {false, true}) {
        setenv("SEGFSS_AES_ONLINE", aes ? "1" : "0", 1);
        std::vector<std::future<void>> capped_callers;
        for (int caller = 0; caller < 4; ++caller) {
            capped_callers.emplace_back(std::async(std::launch::async, [&] {
                segfss::CggmThreadScope branch(2);
                {
                    segfss::CggmThreadScope worker(1);
                    segfss::CggmEvaluationStats expanded_stats, compact_stats;
                    const auto expanded = segfss::cggmEvalBatch(first_keys, queries, &expanded_stats);
                    std::vector<segfss::CggmEval> compact;
                    segfss::cggmEvalBatch(moved_batch, public_queries, shifts, 1, compact, &compact_stats);
                    require(expanded_stats.participants == 1 && compact_stats.participants == 1 &&
                            worker.observedParticipants() == 1 && worker.effectiveLimit() == 1,
                            "four activation callers keep expanded and compact AES walks on their owner");
                    for (std::size_t row = 0; row < expanded.size(); ++row)
                        require((expanded[row].L ^ compact[row].L) == expected[row].L &&
                                (expanded[row].e ^ compact[row].e) == expected[row].e,
                                "scoped hashing budget preserves comparison and equality");
                }
                require(branch.effectiveLimit() == std::min(std::size_t(2), configured_limit) &&
                        branch.observedParticipants() == 1,
                        "nested worker budget restores and reports into its calling branch");
            }));
        }
        for (auto& caller : capped_callers) caller.get();
        require(segfss::cggmThreadLimit() == configured_limit,
                "worker-local budgets do not mutate the process executor");
        segfss::CggmEvaluationStats after;
        segfss::cggmEvalBatch(first_keys, queries, &after);
        require(after.participants == std::min(configured_limit, std::size_t(3)),
                "executor capacity is restored after independent scoped workers finish");
    }
    unsetenv("SEGFSS_AES_ONLINE");
    const auto parallel_first = segfss::cggmEvalBatch(first_keys, queries);
    const auto parallel_second = segfss::cggmEvalBatch(second_keys, queries);
    require(serial_first.size() == expected.size() && parallel_first.size() == expected.size(),
            "cGGM batch preserves input count");
    for (std::size_t row = 0; row < expected.size(); ++row) {
        const auto scalar_first = segfss::cggmEval(first_keys[row], queries[row]);
        const auto scalar_second = segfss::cggmEval(second_keys[row], queries[row]);
        require(serial_first[row].L == parallel_first[row].L &&
                serial_first[row].e == parallel_first[row].e &&
                serial_second[row].L == parallel_second[row].L &&
                serial_second[row].e == parallel_second[row].e,
                "cGGM parallel batch must preserve individual shares and ordering");
        require(scalar_first.L == parallel_first[row].L &&
                scalar_first.e == parallel_first[row].e &&
                scalar_second.L == parallel_second[row].L &&
                scalar_second.e == parallel_second[row].e,
                "cGGM parallel batch must equal independent scalar walks");
        require((parallel_first[row].L ^ parallel_second[row].L) == expected[row].L &&
                (parallel_first[row].e ^ parallel_second[row].e) == expected[row].e,
                "cGGM parallel batch must reconstruct unsigned comparison and equality");
    }
    require(segfss::cggmEvalBatch({}, {}).empty(), "empty local cGGM batch");
    invalid([&] { segfss::cggmEvalBatch(first_keys, {}); }, "batch count mismatch");
    auto malformed = first_keys;
    malformed[0].depth = 0;
    invalid([&] { segfss::cggmEvalBatch(malformed, queries); }, "zero-depth key");
    malformed = first_keys;
    malformed.back().internal_cw.clear();
    invalid([&] { segfss::cggmEvalBatch(malformed, queries); }, "correction word count");
    auto short_queries = queries;
    short_queries.back().pop_back();
    invalid([&] { segfss::cggmEvalBatch(first_keys, short_queries); }, "input depth mismatch");

    // Independent calls may share the configured local executor. Reconfigure
    // between batches, repeat the same limit, and recover after rejected input;
    // none of these operations may change output shares or their ordering.
    for (std::size_t threads : {4, 4, 2, 1, 3, 4}) {
        segfss::setCggmThreadLimit(threads);
        for (int repeat = 0; repeat < 12; ++repeat) {
            const auto output = segfss::cggmEvalBatch(first_keys, queries);
            for (std::size_t row = 0; row < output.size(); ++row) {
                require(output[row].L == serial_first[row].L &&
                        output[row].e == serial_first[row].e,
                        "repeated cGGM batch preserves shares after worker reconfiguration");
            }
        }
    }
    std::atomic<bool> begin{false};
    std::vector<std::future<void>> callers;
    for (int caller = 0; caller < 3; ++caller) {
        callers.emplace_back(std::async(std::launch::async, [&] {
            while (!begin.load(std::memory_order_acquire)) std::this_thread::yield();
            for (int repeat = 0; repeat < 24; ++repeat) {
                const auto output = segfss::cggmEvalBatch(second_keys, queries);
                for (std::size_t row = 0; row < output.size(); ++row) {
                    require(output[row].L == serial_second[row].L &&
                            output[row].e == serial_second[row].e,
                            "concurrent local cGGM batches preserve shares and ordering");
                }
            }
        }));
    }
    begin.store(true, std::memory_order_release);
    for (int resize = 0; resize < 24; ++resize) {
        segfss::setCggmThreadLimit(resize % 3 == 0 ? 1 : 4);
    }
    for (auto& caller : callers) caller.get();
    segfss::setCggmThreadLimit(1);
    std::cout << "local_cggm_batch count=" << expected.size()
              << " threads=1,4 mixed_depths=1,4,7,16,64 PASS\n";
}

std::vector<segfss::CggmEval> openResults(const std::vector<segfss::CggmEval>& shares) {
    std::vector<u8> own(2 * shares.size()), other(own.size());
    for (std::size_t row = 0; row < shares.size(); ++row) {
        own[2 * row] = shares[row].L;
        own[2 * row + 1] = shares[row].e;
    }
    if (party == SERVER) {
        peer->send_u8(own.data(), own.size());
        peer->recv_u8(other.data(), other.size());
    } else {
        peer->recv_u8(other.data(), other.size());
        peer->send_u8(own.data(), own.size());
    }
    std::vector<segfss::CggmEval> output(shares.size());
    for (std::size_t row = 0; row < output.size(); ++row) {
        output[row] = {static_cast<u8>(own[2 * row] ^ other[2 * row]),
                       static_cast<u8>(own[2 * row + 1] ^ other[2 * row + 1])};
    }
    return output;
}

void verifyBatch(int bits, int nseg, const std::vector<uint64_t>& thresholds,
                 const std::vector<uint64_t>& queries,
                 const segfss::DcfGenerationOptions& options = segfss::dcfGenerationOptions()) {
    std::vector<std::vector<u8>> shares;
    for (uint64_t threshold : thresholds) shares.push_back(thresholdShare(threshold, bits));
    auto material = segfss::segmentedDcfOfflineBatch(party, shares, bits, nseg, peer, options);
    const auto result = openResults(segfss::segmentedDcfOnlineBatch(party, queries, material, peer));
    for (std::size_t row = 0; row < queries.size(); ++row) {
        require(result[row].L == (queries[row] < thresholds[row]) &&
                result[row].e == (queries[row] == thresholds[row]),
                "socket batch mismatch bits=" + std::to_string(bits) + " nseg=" +
                std::to_string(nseg) + " row=" + std::to_string(row));
    }
    std::size_t expected_layers = 0;
    for (int active = nseg; active > 1; active = (active + 1) / 2) ++expected_layers;
    require(material.metrics.online_scan_layers == expected_layers &&
            material.scan.cursor == 2 * queries.size() * (nseg - 1), "scan layers and gate accounting");
    uint64_t internal = 0, leaf = 0;
    for (int width : material.layout.width) {
        internal += (uint64_t(1) << (width - 1)) - 1;
        leaf += uint64_t(1) << width;
    }
    require(material.metrics.internal_hashes == thresholds.size() * internal &&
            material.metrics.leaf_hashes == thresholds.size() * leaf && material.metrics.extra_hashes == 0,
            "tiling must preserve the original total hash work");
    invalid([&] { segfss::segmentedDcfOnlineBatch(party, queries, material, peer); }, "double consumption");
    if (party == SERVER) {
        std::cout << "socket_2pc bits=" << bits << " nseg=" << nseg << " count=" << queries.size()
                  << " depths=";
        for (int width : material.layout.width) std::cout << width << ',';
        std::cout << " scan_layers=" << material.metrics.online_scan_layers
                  << " scan_and_gates=" << material.metrics.scan_and_gates << " PASS\n";
    }
    segfss::releaseSegmentedDcfMaterial(material);
    segfss::releaseSegmentedDcfMaterial(material);
    invalid([&] { segfss::segmentedDcfOnlineBatch(party, queries, material, peer); }, "released material");
}

void tiledPolicyTests() {
    setenv("SEGFSS_MATERIAL_LAYOUT", "contiguous", 1);
    for (int bits : {1, 8, 9, 16, 17, 24, 25, 31, 32}) {
        const uint64_t maximum = maskOf(bits);
        const std::vector<uint64_t> thresholds{0, maximum, maximum / 2, 1, maximum - 1, 2 & maximum, maximum / 2 + 1};
        const std::vector<uint64_t> queries{0, maximum - 1, maximum / 2, 0, maximum, 2 & maximum, maximum / 2};
        for (auto policy : {segfss::SegmentPolicy::fixed_2, segfss::SegmentPolicy::fixed_4,
                            segfss::SegmentPolicy::depth_cap_8}) {
            for (std::size_t cap : {0, 1, 3, 8}) {
                segfss::DcfGenerationOptions options;
                options.tile_keys = cap;
                options.budget_bytes = 64ULL << 20;
                options.reserve_bytes = 1ULL << 20;
                verifyBatch(bits, segfss::segmentCount(bits, policy), thresholds, queries, options);
            }
        }
    }
    unsetenv("SEGFSS_MATERIAL_LAYOUT");
}

void participantLimitTests() {
    std::vector<uint64_t> thresholds(128),queries(128);
    for(std::size_t row=0;row<thresholds.size();++row) {
        thresholds[row]=(row*117113)&65535;
        queries[row]=row%2?thresholds[row]:(thresholds[row]-1)&65535;
    }
    for(bool aes:{false,true}) for(std::size_t limit:{1,2}) {
        setenv("SEGFSS_AES_OFFLINE",aes?"1":"0",1);
        segfss::DcfGenerationOptions options;
        options.participant_limit=limit;
        std::vector<std::vector<u8>> shares;
        for(auto threshold:thresholds) shares.push_back(thresholdShare(threshold,16));
        auto material=segfss::segmentedDcfOfflineBatch(party,shares,16,2,peer,options);
        require(material.metrics.generation_participants<=limit,
                "public per-request generation participant limit was exceeded");
        auto result=segfss::segmentedDcfOnlineBatch(party,queries,material,peer);
        auto opened=openResults(result);
        for(std::size_t row=0;row<thresholds.size();++row)
            require(opened[row].L==(queries[row]<thresholds[row]),"bounded generation integer oracle");
        segfss::releaseSegmentedDcfMaterial(material);
    }
    if(party==SERVER) std::cout<<"generation_participant_limits pool=4 requests=1,2 aes=0,1 values=512 PASS\n";
}

void demandedSocketTests() {
    for (bool compact : {false, true}) for (bool prune_equality : {false, true})
        for (bool prune_leaf : {false, true}) for (bool need_e : {false, true}) {
        setenv("SEGFSS_MATERIAL_LAYOUT", compact ? "contiguous" : "legacy", 1);
        setenv("SEGFSS_AES_ONLINE", "1", 1);
        for (int nseg : {1, 2, 3, 4, 5, 7, 8}) {
            const int bits = 16, count = 9;
            std::vector<uint64_t> thresholds(count), queries(count);
            std::vector<std::vector<u8>> shares;
            for (int row = 0; row < count; ++row) {
                thresholds[row] = (row * 117113U) & 65535U;
                queries[row] = row % 3 == 0 ? thresholds[row] :
                    row % 3 == 1 ? (thresholds[row] - 1) & 65535U : (thresholds[row] + 1) & 65535U;
                shares.push_back(thresholdShare(thresholds[row], bits));
            }
            segfss::DcfGenerationOptions options;
            options.tile_keys = 11;
            options.prune_equality = prune_equality;
            options.prune_leaf_hashes = prune_leaf;
            auto material = segfss::segmentedDcfOfflineBatch(party, shares, bits, nseg, peer, options, need_e);
            const auto gates = count * segfss::planDcfScan(nseg, need_e, prune_equality).and_gates_per_row;
            require(material.scan.size() == gates && material.metrics.scan_and_gates == gates,
                    "generated triple inventory follows the public output plan");
            if (!need_e) {
                const auto sent = peer->bytesSent, received = peer->bytesReceived;
                invalid([&] { segfss::segmentedDcfOnlineBatchGroup(
                    party, {{&queries, &material, true}}, peer); }, "request exceeding offline output demand");
                require(!material.consumed && sent == peer->bytesSent && received == peer->bytesReceived,
                        "invalid output upgrade must reject before consumption or I/O");
            }
            const auto opened = openResults(segfss::segmentedDcfOnlineBatch(party, queries, material, peer));
            require(material.scan.cursor == gates, "demanded scan consumes exactly its generated inventory");
            for (int row = 0; row < count; ++row)
                require(opened[row].L == (queries[row] < thresholds[row]) &&
                        opened[row].e == (need_e && queries[row] == thresholds[row]),
                        "demanded scan reconstructs independent L/(L,e) integer comparison oracle");
        }
    }
    unsetenv("SEGFSS_MATERIAL_LAYOUT"); unsetenv("SEGFSS_AES_ONLINE");
    if (party == SERVER) std::cout << "demanded_socket_scan layouts=2 equality=0,1 leaf=0,1 outputs=L,Le nseg=1,2,3,4,5,7,8 PASS\n";
}

void socketTests() {
    invalid([] { segfss::segmentedDcfOffline(party, {0}, 1, 0, peer); }, "zero nseg");
    invalid([] { segfss::segmentedDcfOffline(party, {0}, 1, 2, peer); }, "excess nseg");
    invalid([] { segfss::segmentedDcfOffline(party, {0}, 0, 1, peer); }, "zero width");
    invalid([] { segfss::segmentedDcfOffline(party, {0}, 65, 1, peer); }, "excess width");
    invalid([] { segfss::segmentedDcfOffline(party, {2}, 1, 1, peer); }, "non-Boolean threshold");
    invalid([] { segfss::segmentedDcfOffline(party, {0}, 2, 1, peer); }, "threshold width");
    invalid([] { segfss::segmentedDcfOfflineBatch(party, {}, 1, 1, peer); }, "empty batch");
    invalid([] { segfss::segmentedDcfOffline(4, {0}, 1, 1, peer); }, "invalid role");

    const char* offline_setting = std::getenv("SEGFSS_AES_OFFLINE");
    const std::string saved_offline = offline_setting ? offline_setting : "0";
    for (const bool batch : {false, true}) {
        setenv("SEGFSS_AES_OFFLINE", batch ? "1" : "0", 1);
        std::vector<std::vector<u8>> shares;
        std::vector<uint64_t> targets;
        for (int row = 0; row < 9; ++row) {
            const int depth = 1 + row % 7;
            targets.push_back((row * 19) & maskOf(depth));
            shares.push_back(thresholdShare(targets.back(), depth));
        }
        segfss::CggmGenerationStats stats;
        const auto keys = segfss::cggmGen2pcBatch(party, shares, nullptr, &stats);
        std::size_t internal = 0, leaves = 0;
        for (std::size_t row = 0; row < keys.size(); ++row) {
            internal += (std::size_t(1) << (keys[row].depth - 1)) - 1;
            leaves += std::size_t(1) << keys[row].depth;
            for (uint64_t query = 0; query <= maskOf(keys[row].depth); ++query) {
                const auto opened = openResults({segfss::cggmEval(keys[row], bitsOf(query, keys[row].depth))});
                require(opened.front().L == (query < targets[row]) && opened.front().e == (query == targets[row]),
                        "scalar and batched offline AES reconstruct the independent comparison oracle");
            }
        }
        require(stats.batched_aes == batch && stats.internal_hashes == internal && stats.leaf_hashes == leaves &&
                stats.participants >= 1, "offline AES reports complete hash accounting");
    }
    setenv("SEGFSS_AES_OFFLINE", saved_offline.c_str(), 1);

    uint64_t comparisons = 0;
    for (int bits = 1; bits <= 5; ++bits) {
        const uint64_t domain = uint64_t(1) << bits;
        std::vector<uint64_t> thresholds, queries;
        for (uint64_t alpha = 0; alpha < domain; ++alpha) {
            for (uint64_t query = 0; query < domain; ++query) {
                thresholds.push_back(alpha);
                queries.push_back(query);
            }
        }
        for (int nseg = 1; nseg <= bits; ++nseg) {
            verifyBatch(bits, nseg, thresholds, queries);
            comparisons += queries.size();
        }
        // Independent legacy full-tree DPF-based DCF implementation, not a
        // second invocation of the segmented wrapper with nseg=1.
        for (uint64_t alpha = 0; alpha < domain; ++alpha) {
            const auto share = thresholdShare(alpha, bits);
            auto key = dfss::keyGenCorrelatedDPFBit(party, bits, share.data());
            std::vector<segfss::CggmEval> legacy(domain);
            for (uint64_t query = 0; query < domain; ++query)
                legacy[query].L = dfss::evalCorrelatedDCFBit(party, GroupElement(query, bits), key);
            const auto opened = openResults(legacy);
            for (uint64_t query = 0; query < domain; ++query)
                require(opened[query].L == (query < alpha), "legacy full-tree DCF mismatch");
            freeDPFKeyPack(key);
        }
    }
    if (party == SERVER) std::cout << "socket_2pc exhaustive_comparisons=" << comparisons
                                   << " legacy_fulltree_comparisons=1364 PASS\n";

    for (int bits : {8, 16, 20, 24, 32, 64}) {
        const uint64_t maximum = maskOf(bits);
        std::vector<uint64_t> thresholds, queries;
        for (int row = 0; row < 128; ++row) {
            uint64_t alpha = (row * 0x9e3779b97f4a7c15ULL) & maximum;
            const uint64_t candidates[] = {0, maximum, alpha, (alpha - 1) & maximum,
                                          (alpha + 1) & maximum, uint64_t(1) << (bits - 1)};
            thresholds.push_back(alpha);
            queries.push_back(candidates[row % 6]);
        }
        for (int nseg : {((bits + 7) / 8), ((bits + 7) / 8) + 1, bits}) {
            if (nseg <= bits) verifyBatch(bits, nseg, thresholds, queries);
        }
    }

    std::vector<uint64_t> thresholds(12800), queries(12800);
    for (std::size_t row = 0; row < thresholds.size(); ++row) {
        thresholds[row] = (row * 0x9e3779b9ULL) & UINT32_MAX;
        queries[row] = (row % 2 == 0) ? thresholds[row] : (row * 117113ULL) & UINT32_MAX;
    }
    verifyBatch(32, 4, thresholds, queries);

    auto original = segfss::segmentedDcfOffline(party, thresholdShare(19, 6), 6, 3, peer);
    auto moved = std::move(original);
    invalid([&] { segfss::segmentedDcfOnline(party, 18, original, peer); }, "moved-from material");
    invalid([&] { segfss::segmentedDcfOnline(party, 64, moved, peer); }, "out-of-domain query");
    invalid([&] { segfss::segmentedDcfOnlineBatch(party, {18, 19}, moved, peer); }, "query count mismatch");
    const int saved_width = moved.layout.width[0];
    ++moved.layout.width[0];
    invalid([&] { segfss::segmentedDcfOnline(party, 18, moved, peer); }, "corrupt layout");
    moved.layout.width[0] = saved_width;
    auto single = openResults({segfss::segmentedDcfOnline(party, 18, moved, peer)});
    require(single.front().L == 1 && single.front().e == 0, "single API output");
    verifyBatch(6, 3, {19}, {18});
    if (party == SERVER) std::cout << "socket_2pc ownership_single_batch_boundaries PASS\n";
}
}  // namespace

int main(int argc, char** argv) {
    try {
        validation();
        segfss::setCggmThreadLimit(1);
        if (argc == 1) {
            sameShareLeafPruning();
            localBatchExecution();
            localExhaustive();
        } else {
            require(argc == 3 || argc == 4,
                    "usage: SEGMENTED_DCF_TEST [role port [cggm_threads]]");
            if (argc == 4) {
                const auto mode=std::string(argv[3]);
                const int threads = mode == "tiling" || mode == "demand" ? 1 : mode == "limits" ? 4 : std::stoi(mode);
                require(threads > 0, "cggm_threads must be positive");
                segfss::setCggmThreadLimit(static_cast<std::size_t>(threads));
            }
            party = std::stoi(argv[1]);
            port = std::stoi(argv[2]);
            require(party == SERVER || party == CLIENT, "role must be 2 or 3");
            peer = party == SERVER ? waitForPeer(port) : new Peer(address, port);
            if (argc == 4 && std::string(argv[3]) == "tiling") tiledPolicyTests();
            else if (argc == 4 && std::string(argv[3]) == "limits") participantLimitTests();
            else if (argc == 4 && std::string(argv[3]) == "demand") demandedSocketTests();
            else socketTests();
            delete peer;
            peer = nullptr;
        }
        std::cout << "segmented DCF: PASS\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "segmented DCF: FAIL " << error.what() << '\n';
        return 1;
    }
}
