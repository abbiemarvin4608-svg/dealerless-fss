#include "segmented_dcf.h"
#include "mpc/api.h"
#include "runtime_options.h"

#include <algorithm>
#include <chrono>
#include <cstdlib>
#include <limits>
#include <iostream>
#include <stdexcept>
#include <utility>
#include <unordered_set>

namespace segfss {
namespace {
using Clock = std::chrono::steady_clock;

uint64_t addBound(uint64_t a, uint64_t b) {
    return b > UINT64_MAX - a ? UINT64_MAX : a + b;
}
uint64_t multiplyBound(uint64_t a, uint64_t b) {
    return b && a > UINT64_MAX / b ? UINT64_MAX : a * b;
}
uint64_t publicInteger(const char* name, uint64_t fallback) {
    const char* text = std::getenv(name);
    if (!text) return fallback;
    const std::string value(text);
    if (value.empty() || value.find_first_not_of("0123456789") != std::string::npos)
        throw std::invalid_argument(std::string(name) + " must be a nonnegative integer");
    return std::stoull(value);
}

double milliseconds(Clock::time_point start) {
    return std::chrono::duration<double, std::milli>(Clock::now() - start).count();
}

void validateShape(int role, int bits, int nseg, std::size_t count, Peer* player) {
    if ((role != SERVER && role != CLIENT) || bits < 1 || bits > 64 ||
        nseg < 1 || nseg > bits || count == 0 ||
        count > static_cast<std::size_t>(std::numeric_limits<int>::max() / (4 * bits)) ||
        player == nullptr || player->role() != role) {
        throw std::invalid_argument("segmented DCF: invalid role/bits/nseg/count/peer");
    }
}

void validateMaterial(int role, const std::vector<uint64_t>& queries,
                      const SegmentedDcfMaterial& material, Peer* player) {
    validateShape(role, material.bits, material.nseg, queries.size(), player);
    const SegmentLayout expected(material.bits, material.nseg);
    const std::size_t gates = queries.size() *
        planDcfScan(material.nseg, material.need_e, material.prune_equality).and_gates_per_row;
    if (material.consumed || material.released || material.role != role ||
        material.count != static_cast<int>(queries.size()) ||
        material.layout.ell != expected.ell || material.layout.nseg != expected.nseg ||
        material.layout.width != expected.width || material.layout.shift != expected.shift ||
        material.keyCount() != queries.size() * material.nseg ||
        material.scan.a.size() != gates || material.scan.b.size() != gates ||
        material.scan.c.size() != gates || material.scan.cursor != 0) {
        throw std::invalid_argument("segmented DCF: consumed, released or malformed material");
    }
    for (std::size_t row = 0; row < queries.size(); ++row) {
        if (material.bits < 64 && (queries[row] >> material.bits) != 0) {
            throw std::invalid_argument("segmented DCF: query outside comparison domain");
        }
        for (int segment = 0; segment < material.nseg; ++segment) {
            const auto key = material.keyView(row * material.nseg + segment);
            if (key.depth != expected.width[segment]) {
                throw std::invalid_argument("segmented DCF: malformed segment key");
            }
        }
    }
}
}  // namespace

bool contiguousDcfMaterials() {
    const char* value = std::getenv("SEGFSS_MATERIAL_LAYOUT");
    if (!value || std::string(value) == "legacy") return false;
    if (std::string(value) == "contiguous") return true;
    throw std::invalid_argument("SEGFSS_MATERIAL_LAYOUT must be legacy or contiguous");
}

DcfGenerationOptions dcfGenerationOptions() {
    DcfGenerationOptions options;
    options.tile_keys = publicInteger("SEGFSS_GENERATION_TILE_KEYS", 0);
    options.budget_bytes = publicInteger("SEGFSS_GENERATION_BUDGET_BYTES", 0);
    options.reserve_bytes = publicInteger("SEGFSS_GENERATION_RESERVE_BYTES", options.reserve_bytes);
    options.participant_limit = publicInteger("SEGFSS_CGGM_THREADS", cggmThreadLimit());
    options.prune_equality = runtimeFlag("SEGFSS_PRUNE_EQUALITY");
    options.prune_leaf_hashes = runtimeFlag("SEGFSS_PRUNE_LEAF_HASHES");
    if (!options.participant_limit || options.budget_bytes > (1ULL << 40) ||
        options.reserve_bytes > (1ULL << 40))
        throw std::invalid_argument("invalid public generation resource configuration");
    return options;
}

DcfScanPlan planDcfScan(int nseg, bool need_e, bool prune_equality) {
    if (nseg < 1 || nseg > 64) throw std::invalid_argument("invalid DCF scan segment count");
    DcfScanPlan plan;
    for (int active = nseg;; active = (active + 1) / 2) {
        plan.need_e.emplace_back(active, 1);
        if (active == 1) break;
    }
    if (prune_equality) {
        plan.need_e.back()[0] = need_e;
        for (std::size_t upper = plan.need_e.size() - 1; upper > 0; --upper) {
            auto& lower = plan.need_e[upper - 1];
            const auto& demand = plan.need_e[upper];
            for (std::size_t parent = 0; parent < demand.size(); ++parent) {
                const std::size_t left = 2 * parent;
                if (left + 1 < lower.size()) {
                    lower[left] = 1; // needed by this parent's L, even if its e is unused
                    lower[left + 1] = demand[parent];
                } else lower[left] = demand[parent]; // carried prefix has no new gate
            }
        }
    }
    for (std::size_t level = 0; level + 1 < plan.need_e.size(); ++level) {
        const std::size_t pairs = plan.need_e[level].size() / 2;
        std::size_t gates = pairs; // one e_left AND L_right per merge
        for (std::size_t pair = 0; pair < pairs; ++pair)
            gates += plan.need_e[level + 1][pair];
        plan.and_gates_per_layer.push_back(gates);
        plan.and_gates_per_row += gates;
    }
    return plan;
}

DcfGenerationPlan planDcfGeneration(std::size_t count, const SegmentLayout& layout,
                                    const DcfGenerationOptions& options, bool need_e) {
    const SegmentLayout expected(layout.ell, layout.nseg);
    if (!count || layout.width != expected.width || layout.shift != expected.shift ||
        !options.participant_limit || count > std::numeric_limits<std::size_t>::max() / layout.nseg)
        throw std::invalid_argument("invalid public generation shape");
    DcfGenerationPlan plan;
    plan.participant_limit = options.participant_limit;
    const auto keys = count * layout.nseg;
    // Retained keys, scan triples, threshold vectors and allocator slack remain
    // resident while the next tile expands. Reserve the complete final output.
    const uint64_t final_keys = addBound(multiplyBound(keys, sizeof(CggmKey) + sizeof(CggmKeyRecord)),
        multiplyBound(multiplyBound(count, layout.ell - layout.nseg), 2 * sizeof(osuCrypto::block)));
    const uint64_t thresholds = multiplyBound(count, 2 * layout.ell + sizeof(std::vector<u8>));
    plan.resident_bytes = addBound(options.resident_bytes, addBound(final_keys,
        addBound(thresholds, multiplyBound(count,
            3 * planDcfScan(layout.nseg, need_e, options.prune_equality).and_gates_per_row))));
    uint64_t available = UINT64_MAX;
    if (options.budget_bytes) {
        if (plan.resident_bytes >= options.budget_bytes ||
            options.reserve_bytes >= options.budget_bytes - plan.resident_bytes)
            throw std::invalid_argument("generation budget cannot reserve resident material");
        available = options.budget_bytes - plan.resident_bytes - options.reserve_bytes;
    }
    const auto limit = options.tile_keys ? std::min(options.tile_keys, keys) : keys;
    std::size_t begin = 0;
    while (begin < keys) {
        DcfGenerationTile tile;
        tile.begin = begin;
        while (begin + tile.key_count < keys && tile.key_count < limit) {
            const int depth = layout.width[(begin + tile.key_count) % layout.nseg];
            const uint64_t leaves = uint64_t(1) << (depth - 1);
            const uint64_t internal = leaves - 1;
            // At the final resize, old/new nodes plus retained hashes coexist.
            // The additive term bounds OT/transcript buffers and metadata/CWs.
            const auto working = addBound(multiplyBound(leaves, 2 * sizeof(osuCrypto::block)),
                4096 + uint64_t(depth) * 512);
            if (options.budget_bytes && (working > available || tile.working_bytes > available - working)) {
                if (!tile.key_count) throw std::invalid_argument("generation budget cannot fit one expanded tree");
                break;
            }
            tile.working_bytes = addBound(tile.working_bytes, working);
            tile.expanded_nodes = addBound(tile.expanded_nodes, addBound(leaves, internal));
            tile.internal_hashes = addBound(tile.internal_hashes, internal);
            tile.leaf_hashes = addBound(tile.leaf_hashes, multiplyBound(leaves, 2));
            tile.max_depth = std::max(tile.max_depth, depth);
            ++tile.key_count;
        }
        plan.peak_working_bytes = std::max(plan.peak_working_bytes, tile.working_bytes);
        begin += tile.key_count;
        plan.tiles.push_back(tile);
    }
    return plan;
}

std::size_t SegmentedDcfMaterial::keyCount() const {
    if (!keys.empty() && compact_keys.size())
        throw std::invalid_argument("segmented DCF: ambiguous key ownership");
    return keys.empty() ? compact_keys.size() : keys.size();
}

CggmKeyView SegmentedDcfMaterial::keyView(std::size_t index) const {
    if (keys.empty()) return compact_keys.view(index);
    const auto& key = keys.at(index);
    if (key.depth < 1 || key.depth > 64 || key.internal_cw.size() != static_cast<std::size_t>(key.depth - 1))
        throw std::invalid_argument("segmented DCF: malformed legacy key");
    return {key.depth, key.root, key.hcw, key.internal_cw.data(), {key.lcw[0], key.lcw[1]}};
}

void SegmentedDcfMaterial::compactKeys() {
    if (keys.empty()) return;
    if (compact_keys.size()) throw std::logic_error("segmented DCF: duplicate compact keys");
    compact_keys.append(keys);
    std::vector<CggmKey>().swap(keys);
}

std::size_t SegmentedDcfMaterial::capacityBytes() const {
    std::size_t bytes = keys.capacity() * sizeof(CggmKey) + compact_keys.capacityBytes();
    for (const auto& key : keys) bytes += key.internal_cw.capacity() * sizeof(osuCrypto::block);
    return bytes + scan.a.capacity() + scan.b.capacity() + scan.c.capacity();
}

SegmentedDcfMaterial::SegmentedDcfMaterial(SegmentedDcfMaterial&& other) noexcept {
    *this = std::move(other);
}

SegmentedDcfMaterial& SegmentedDcfMaterial::operator=(SegmentedDcfMaterial&& other) noexcept {
    if (this != &other) {
        role = other.role;
        bits = other.bits;
        nseg = other.nseg;
        count = other.count;
        segment_policy = other.segment_policy;
        layout = std::move(other.layout);
        keys = std::move(other.keys);
        compact_keys = std::move(other.compact_keys);
        scan = std::move(other.scan);
        consumed = other.consumed;
        released = other.released;
        need_e = other.need_e;
        prune_equality = other.prune_equality;
        prune_leaf_hashes = other.prune_leaf_hashes;
        metrics = other.metrics;
        other.count = 0;
        other.consumed = true;
        other.released = true;
    }
    return *this;
}

SegmentedDcfMaterial segmentedDcfOfflineBatch(
    int role, const std::vector<std::vector<u8>>& thresholds,
    int bits, int nseg, Peer* player) {
    return segmentedDcfOfflineBatch(role, thresholds, bits, nseg, player, dcfGenerationOptions());
}

SegmentedDcfMaterial segmentedDcfOfflineBatch(
    int role, const std::vector<std::vector<u8>>& thresholds,
    int bits, int nseg, Peer* player, const DcfGenerationOptions& options) {
    return segmentedDcfOfflineBatch(role, thresholds, bits, nseg, player, options, true);
}

SegmentedDcfMaterial segmentedDcfOfflineBatch(
    int role, const std::vector<std::vector<u8>>& thresholds,
    int bits, int nseg, Peer* player, const DcfGenerationOptions& options,
    bool need_e) {
    validateShape(role, bits, nseg, thresholds.size(), player);
    for (const auto& row : thresholds) {
        if (row.size() != static_cast<std::size_t>(bits) ||
            std::any_of(row.begin(), row.end(), [](u8 bit) { return bit > 1; })) {
            throw std::invalid_argument("segmented DCF: malformed threshold bit shares");
        }
    }
    SegmentedDcfMaterial material;
    material.role = role;
    material.bits = bits;
    material.nseg = nseg;
    material.count = static_cast<int>(thresholds.size());
    material.need_e = need_e;
    material.prune_equality = options.prune_equality;
    material.prune_leaf_hashes = options.prune_leaf_hashes;
    material.layout = SegmentLayout(bits, nseg);
    const auto plan = planDcfGeneration(thresholds.size(), material.layout, options, need_e);
    const bool compact = contiguousDcfMaterials();
    const auto key_count = thresholds.size() * nseg;
    if (compact) {
        material.compact_keys.records.reserve(key_count);
        material.compact_keys.correction_words.reserve(thresholds.size() * (bits - nseg));
    } else material.keys.reserve(key_count);
    auto start = Clock::now();
    const auto before_sent = player->bytesSent, before_received = player->bytesReceived, before_io = player->rounds;
    const bool reuse_workspace = runtimeFlag("SEGFSS_REUSE_GENERATION_WORKSPACE");
    CggmGenerationWorkspace generation_workspace;
    for (const auto& tile : plan.tiles) {
        std::vector<std::vector<u8>> segments;
        segments.reserve(tile.key_count);
        for (std::size_t index = tile.begin; index < tile.begin + tile.key_count; ++index) {
            const int segment = index % nseg;
            const auto& row = thresholds[index / nseg];
            const int offset = bits - material.layout.shift[segment] - material.layout.width[segment];
            segments.emplace_back(row.begin() + offset, row.begin() + offset + material.layout.width[segment]);
        }
        CggmGenerationStats stats;
        auto keys = cggmGen2pcBatch(role, segments, nullptr, &stats, player, plan.participant_limit,
                                   reuse_workspace ? &generation_workspace : nullptr);
        if (compact) material.compact_keys.append(keys);
        else for (auto& key : keys) material.keys.push_back(std::move(key));
        material.metrics.internal_hashes += stats.internal_hashes;
        material.metrics.leaf_hashes += stats.leaf_hashes;
        material.metrics.generation_participants = std::max(material.metrics.generation_participants, stats.participants);
        material.metrics.keygen_aes_scratch_bytes = std::max(material.metrics.keygen_aes_scratch_bytes, stats.scratch_peak_bytes);
        material.metrics.aes_offline = material.metrics.aes_offline || stats.batched_aes;
        material.metrics.expanded_nodes += tile.expanded_nodes;
        material.metrics.cot_batches += tile.max_depth > 1 ? 4 : 2;
        material.metrics.opening_batches += tile.max_depth - 1 + stats.ef_openings + stats.leaf_cw_openings;
        material.metrics.bit_block_products += stats.bit_block_products;
        material.metrics.temporary_product_triples += stats.temporary_triples;
        material.metrics.temporary_product_triples_consumed += stats.temporary_triples_consumed;
        material.metrics.general_ot_64 += stats.general_ot_64;
        material.metrics.general_ot_128 += stats.general_ot_128;
        material.metrics.native_cot_128 += stats.native_cot_128;
        material.metrics.product_ef_openings += stats.ef_openings;
        material.metrics.generation_workspace_capacity_bytes = std::max<uint64_t>(
            material.metrics.generation_workspace_capacity_bytes, stats.workspace_capacity_bytes);
        material.metrics.generation_workspace_growths += stats.workspace_capacity_growths;
    }
    material.metrics.keygen_ms = milliseconds(start);
    material.metrics.generation_tiles = plan.tiles.size();
    material.metrics.peak_working_bytes = plan.peak_working_bytes;
    material.metrics.resident_bytes = plan.resident_bytes;
    material.metrics.generation_io_calls = player->rounds - before_io;
    material.metrics.generation_sent_bytes = player->bytesSent - before_sent;
    material.metrics.generation_received_bytes = player->bytesReceived - before_received;
    const int max_depth = material.layout.maxWidth();
    material.metrics.added_cot_batches = material.metrics.cot_batches - (max_depth > 1 ? 4 : 2);
    const auto stages = max_depth > 1 ? 2 : 1;
    const char* product_mode = std::getenv("SEGFSS_BIT_BLOCK_MODE");
    const bool direct = product_mode && (std::string(product_mode).find("direct_ot") == 0 ||
                                        std::string(product_mode) == "direct_cot");
    const auto leaf_openings = runtimeFlag("SEGFSS_MERGE_LEAF_CW") ? 1 : 2;
    material.metrics.added_opening_batches = material.metrics.opening_batches -
        (max_depth - 1 + (direct ? 0 : 2 * stages) + leaf_openings);
    if (options.tile_keys || options.budget_bytes || reuse_workspace ||
        runtimeFlag("SEGFSS_MERGE_LEAF_CW") || (product_mode && std::string(product_mode) != "legacy")) {
        std::cout << "DCF_GENERATION {\"bits\":" << bits << ",\"count\":" << thresholds.size()
            << ",\"nseg\":" << nseg << ",\"tiles\":" << plan.tiles.size()
            << ",\"tile_keys\":" << plan.tiles.front().key_count
            << ",\"tail_keys\":" << plan.tiles.back().key_count
            << ",\"peak_working_bytes\":" << plan.peak_working_bytes
            << ",\"resident_bytes\":" << plan.resident_bytes
            << ",\"reserve_bytes\":" << options.reserve_bytes
            << ",\"budget_bytes\":" << options.budget_bytes
            << ",\"participant_limit\":" << plan.participant_limit
            << ",\"internal_hashes\":" << material.metrics.internal_hashes
            << ",\"leaf_hashes\":" << material.metrics.leaf_hashes
            << ",\"extra_hashes\":0,\"cot_batches\":" << material.metrics.cot_batches
            << ",\"opening_batches\":" << material.metrics.opening_batches
            << ",\"added_cot_batches\":" << material.metrics.added_cot_batches
            << ",\"added_opening_batches\":" << material.metrics.added_opening_batches
            << ",\"io_calls\":" << material.metrics.generation_io_calls
            << ",\"sent_bytes\":" << material.metrics.generation_sent_bytes
            << ",\"received_bytes\":" << material.metrics.generation_received_bytes
            << ",\"bit_block_products\":" << material.metrics.bit_block_products
            << ",\"temporary_product_triples\":" << material.metrics.temporary_product_triples
            << ",\"temporary_product_triples_consumed\":" << material.metrics.temporary_product_triples_consumed
            << ",\"general_ot_64\":" << material.metrics.general_ot_64
            << ",\"general_ot_128\":" << material.metrics.general_ot_128
            << ",\"native_cot_128\":" << material.metrics.native_cot_128
            << ",\"product_ef_openings\":" << material.metrics.product_ef_openings
            << ",\"reuse_workspace\":" << (reuse_workspace ? "true" : "false")
            << ",\"workspace_capacity_bytes\":" << material.metrics.generation_workspace_capacity_bytes
            << ",\"workspace_capacity_growths\":" << material.metrics.generation_workspace_growths
            << ",\"tile_plan\":[";
        bool first = true;
        for (const auto& tile : plan.tiles) {
            if (!first) std::cout << ',';
            first = false;
            std::cout << "{\"begin\":" << tile.begin << ",\"keys\":" << tile.key_count
                << ",\"working_bytes\":" << tile.working_bytes
                << ",\"expanded_nodes\":" << tile.expanded_nodes
                << ",\"internal_hashes\":" << tile.internal_hashes
                << ",\"leaf_hashes\":" << tile.leaf_hashes << '}';
        }
        std::cout << "]}" << std::endl;
    }
    material.metrics.scan_and_gates = thresholds.size() *
        planDcfScan(nseg, need_e, material.prune_equality).and_gates_per_row;
    start = Clock::now();
    material.scan = booleanAndOffline(role, static_cast<int>(material.metrics.scan_and_gates), player);
    material.metrics.scan_preprocessing_ms = milliseconds(start);
    material.consumed = false;
    return material;
}

SegmentedDcfMaterial segmentedDcfOffline(int role, const std::vector<u8>& threshold,
                                       int bits, int nseg, Peer* player) {
    return segmentedDcfOfflineBatch(role, {threshold}, bits, nseg, player);
}

std::vector<CggmEval> segmentedDcfOnlineBatch(
    int role, const std::vector<uint64_t>& queries,
    SegmentedDcfMaterial& material, Peer* player) {
    return segmentedDcfOnlineBatchGroup(role, {{&queries, &material, material.need_e}}, player).front();
}

namespace {
void evaluateLocal(const std::vector<uint64_t>& queries, SegmentedDcfMaterial& material,
                   bool requested_need_e, SegmentedDcfWorkspace& scratch) {
    const int count = material.count;
    auto start = Clock::now();
    scratch.query_bits.resize(material.keys.size());
    for (int row = 0; !material.keys.empty() && row < count; ++row) {
        for (int segment = 0; segment < material.nseg; ++segment) {
            auto& bits = scratch.query_bits[row * material.nseg + segment];
            bits.clear();
            bits.reserve(material.layout.width[segment]);
            for (int bit = material.layout.width[segment] - 1; bit >= 0; --bit) {
                bits.push_back((queries[row] >>
                                (material.layout.shift[segment] + bit)) & 1);
            }
        }
    }
    // All cGGM walks are local.  Evaluate them as one batch before the
    // communication-dependent Boolean scan below.
    material.metrics.query_prepare_ms = milliseconds(start);
    start = Clock::now();
    CggmEvaluationStats eval_stats;
    scratch.plan = planDcfScan(material.nseg, material.need_e, material.prune_equality);
    scratch.layer = 0;
    scratch.leaf_need_e.resize(material.keyCount());
    for (std::size_t key = 0; key < scratch.leaf_need_e.size(); ++key) {
        // An unpruned scan still needs all leaf e's. For a singleton, no
        // scan can use an equality output the caller did not request.
        scratch.leaf_need_e[key] = material.nseg == 1 ? requested_need_e :
                                  scratch.plan.need_e.front()[key % material.nseg];
    }
    if (material.keys.empty())
        cggmEvalBatch(material.compact_keys, queries, material.layout.shift, material.nseg,
                      scratch.level, scratch.leaf_need_e, &eval_stats, material.prune_leaf_hashes);
    else scratch.level = cggmEvalBatch(material.keys, scratch.query_bits, scratch.leaf_need_e,
                                     &eval_stats, material.prune_leaf_hashes);
    material.metrics.local_evaluation_ms = milliseconds(start);
    material.metrics.evaluation_ms = material.metrics.query_prepare_ms + material.metrics.local_evaluation_ms;
    material.metrics.eval_participants = eval_stats.participants;
    material.metrics.eval_internal_hashes = eval_stats.internal_hashes;
    material.metrics.eval_leaf_hashes = eval_stats.leaf_hashes;
    material.metrics.eval_aes_scratch_bytes = eval_stats.scratch_peak_bytes;
    material.metrics.aes_online = eval_stats.batched_aes;
    material.metrics.skipped_leaf_hashes = 2 * material.keyCount() - eval_stats.leaf_hashes;
    material.metrics.pruned_e_outputs = material.count * std::size_t(std::count(
        scratch.plan.need_e.front().begin(), scratch.plan.need_e.front().end(), u8(0)));
    if (material.nseg == 1 && !requested_need_e) material.metrics.pruned_e_outputs = material.count;
    for (std::size_t level = 1; level < scratch.plan.need_e.size(); ++level)
        for (std::size_t pair = 0; pair < scratch.plan.need_e[level - 1].size() / 2; ++pair)
            if (!scratch.plan.need_e[level][pair]) material.metrics.pruned_e_outputs += material.count;
    scratch.active = material.nseg;
}
}

std::vector<std::vector<CggmEval>> segmentedDcfOnlineBatchGroup(
    int role, const std::vector<SegmentedDcfBatchView>& groups, Peer* player,
    SegmentedDcfGroupWorkspace* workspace) {
    std::unordered_set<SegmentedDcfMaterial*> seen;
    for (const auto& group : groups) {
        if (!group.queries || !group.material || !seen.insert(group.material).second)
            throw std::invalid_argument("segmented DCF: invalid or repeated group");
        validateMaterial(role, *group.queries, *group.material, player);
        if (group.need_e && !group.material->need_e)
            throw std::invalid_argument("segmented DCF: online output exceeds offline demand");
    }
    SegmentedDcfGroupWorkspace local;
    auto& scratch = workspace ? *workspace : local;
    scratch.domains.resize(groups.size());
    scratch.scan_layers = 0;
    for (const auto& group : groups) group.material->consumed = true;
    for (std::size_t index = 0; index < groups.size(); ++index)
        evaluateLocal(*groups[index].queries, *groups[index].material,
                      groups[index].need_e, scratch.domains[index]);
    const auto scan_start = Clock::now();
    while (true) {
        scratch.and_groups.clear();
        for (std::size_t domain = 0; domain < groups.size(); ++domain) {
            auto& state = scratch.domains[domain];
            auto& material = *groups[domain].material;
            if (state.active <= 1) continue;
            const int pairs = state.active / 2;
            state.a.resize(material.count * state.plan.and_gates_per_layer[state.layer]);
            state.b.resize(state.a.size()); state.c.resize(state.a.size());
            std::size_t index = 0;
            for (int row = 0; row < material.count; ++row) for (int pair = 0; pair < pairs; ++pair) {
                const auto left = state.level[row * state.active + 2 * pair];
                const auto right = state.level[row * state.active + 2 * pair + 1];
                if (state.plan.need_e[state.layer + 1][pair]) {
                    state.a[index] = left.e; state.b[index++] = right.e;
                }
                state.a[index] = left.e; state.b[index++] = right.L;
            }
            scratch.and_groups.push_back({state.a.data(), state.b.data(), state.c.data(),
                                          int(state.c.size()), &material.scan});
        }
        if (scratch.and_groups.empty()) break;
        booleanAndOnlineBatch(role, scratch.and_groups, player, &scratch.and_workspace);
        ++scratch.scan_layers;
        for (std::size_t domain = 0; domain < groups.size(); ++domain) {
            auto& state = scratch.domains[domain];
            auto& material = *groups[domain].material;
            if (state.active <= 1) continue;
            const int pairs = state.active / 2, next_count = (state.active + 1) / 2;
            state.next.resize(material.count * next_count);
            std::size_t index = 0;
            for (int row = 0; row < material.count; ++row) {
                for (int pair = 0; pair < pairs; ++pair) {
                    const u8 equal = state.plan.need_e[state.layer + 1][pair] ? state.c[index++] : 0;
                    state.next[row * next_count + pair] = {
                        static_cast<u8>(state.level[row * state.active + 2 * pair].L ^ state.c[index++]), equal};
                }
                if (state.active & 1) state.next[row * next_count + next_count - 1] = state.level[row * state.active + state.active - 1];
            }
            state.level.swap(state.next);
            state.active = next_count;
            ++state.layer;
            ++material.metrics.online_scan_layers;
        }
    }
    scratch.scan_ms = milliseconds(scan_start);
    std::vector<std::vector<CggmEval>> outputs;
    for (std::size_t domain = 0; domain < groups.size(); ++domain) {
        // Group members share this elapsed interval; callers use group scan_ms once.
        groups[domain].material->metrics.scan_ms = scratch.scan_ms;
        if (!groups[domain].need_e)
            for (auto& output : scratch.domains[domain].level) output.e = 0;
        if (groups[domain].material->scan.cursor != groups[domain].material->scan.size())
            throw std::logic_error("segmented DCF: scan material was not consumed exactly");
        outputs.push_back(scratch.domains[domain].level);
    }
    return outputs;
}

CggmEval segmentedDcfOnline(int role, uint64_t query,
                           SegmentedDcfMaterial& material, Peer* player) {
    return segmentedDcfOnlineBatch(role, {query}, material, player).front();
}

void releaseSegmentedDcfMaterial(SegmentedDcfMaterial& material) {
    material.consumed = true;
    material.released = true;
    std::vector<CggmKey>().swap(material.keys);
    material.compact_keys = {};
    material.scan = {};
}

}  // namespace segfss
