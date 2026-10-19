#pragma once

#include "gen.h"

#include <cstdint>
#include <vector>

namespace segfss {

struct DcfGenerationOptions {
    std::size_t tile_keys = 0; // zero retains one unpartitioned batch
    uint64_t budget_bytes = 0; // zero disables budget-driven partitioning
    uint64_t reserve_bytes = 512ULL << 20;
    uint64_t resident_bytes = 0;
    std::size_t participant_limit = 1;
    bool prune_equality = false;
    bool prune_leaf_hashes = false;
};
// Level zero is the leaf row; each later level pairs adjacent high/low
// prefixes and carries an odd final prefix. Demands are public and identical
// for every comparison row. Equality is propagated backwards from the root.
struct DcfScanPlan {
    std::vector<std::vector<u8>> need_e;
    std::vector<std::size_t> and_gates_per_layer;
    std::size_t and_gates_per_row = 0;
};
DcfScanPlan planDcfScan(int nseg, bool need_e, bool prune_equality);
struct DcfGenerationTile {
    std::size_t begin = 0, key_count = 0;
    uint64_t working_bytes = 0, expanded_nodes = 0, internal_hashes = 0, leaf_hashes = 0;
    int max_depth = 0;
};
struct DcfGenerationPlan {
    std::vector<DcfGenerationTile> tiles;
    uint64_t resident_bytes = 0, peak_working_bytes = 0;
    std::size_t participant_limit = 1;
};
DcfGenerationOptions dcfGenerationOptions();
DcfGenerationPlan planDcfGeneration(std::size_t count, const SegmentLayout& layout,
                                    const DcfGenerationOptions& options, bool need_e = true);

struct SegmentedDcfMetrics {
    double keygen_ms = 0;
    double scan_preprocessing_ms = 0;
    double evaluation_ms = 0;
    double query_prepare_ms = 0;
    double local_evaluation_ms = 0;
    double scan_ms = 0;
    std::size_t eval_participants = 0;
    std::size_t internal_hashes = 0;
    std::size_t leaf_hashes = 0;
    std::size_t generation_participants = 0;
    std::size_t keygen_aes_scratch_bytes = 0;
    std::size_t eval_internal_hashes = 0;
    std::size_t eval_leaf_hashes = 0;
    std::size_t eval_aes_scratch_bytes = 0;
    bool aes_online = false, aes_offline = false;
    std::size_t scan_and_gates = 0;
    std::size_t online_scan_layers = 0;
    uint64_t extra_hashes = 0, expanded_nodes = 0;
    uint64_t generation_tiles = 0, peak_working_bytes = 0, resident_bytes = 0;
    uint64_t generation_io_calls = 0, generation_sent_bytes = 0, generation_received_bytes = 0;
    uint64_t cot_batches = 0, opening_batches = 0, added_cot_batches = 0, added_opening_batches = 0;
    std::size_t pruned_e_outputs = 0;
    std::size_t skipped_leaf_hashes = 0;
    uint64_t bit_block_products = 0, temporary_product_triples = 0, temporary_product_triples_consumed = 0;
    uint64_t general_ot_64 = 0, general_ot_128 = 0, product_ef_openings = 0;
    uint64_t native_cot_128 = 0;
    uint64_t generation_workspace_capacity_bytes = 0, generation_workspace_growths = 0;
};

// One row owns nseg consecutive keys. Public data fields support the existing
// offline spool format; online validates their shape before consuming them.
struct SegmentedDcfMaterial {
    int role = 0;
    int bits = 0;
    int nseg = 0;
    int count = 0;
    SegmentPolicy segment_policy = SegmentPolicy::explicit_counts;
    SegmentLayout layout;
    std::vector<CggmKey> keys;
    CggmKeyBatch compact_keys;
    BooleanAndPreprocessing scan;
    bool consumed = true;
    bool released = false;
    bool need_e = true;
    bool prune_equality = false;
    bool prune_leaf_hashes = false;
    SegmentedDcfMetrics metrics;

    SegmentedDcfMaterial() = default;
    SegmentedDcfMaterial(const SegmentedDcfMaterial&) = delete;
    SegmentedDcfMaterial& operator=(const SegmentedDcfMaterial&) = delete;
    SegmentedDcfMaterial(SegmentedDcfMaterial&& other) noexcept;
    SegmentedDcfMaterial& operator=(SegmentedDcfMaterial&& other) noexcept;
    std::size_t keyCount() const;
    CggmKeyView keyView(std::size_t index) const;
    void compactKeys();
    std::size_t capacityBytes() const;
};

bool contiguousDcfMaterials();

struct SegmentedDcfBatchView {
    const std::vector<uint64_t>* queries;
    SegmentedDcfMaterial* material;
    bool need_e = true;
};
struct SegmentedDcfWorkspace {
    std::vector<std::vector<u8>> query_bits;
    std::vector<CggmEval> level, next;
    std::vector<u8> a, b, c;
    std::vector<u8> leaf_need_e;
    DcfScanPlan plan;
    int active = 0;
    std::size_t layer = 0;
};
struct SegmentedDcfGroupWorkspace {
    std::vector<SegmentedDcfWorkspace> domains;
    std::vector<BooleanAndBatchView> and_groups;
    BooleanAndWorkspace and_workspace;
    double scan_ms = 0;
    std::size_t scan_layers = 0;
};

// Complete independent comparisons; same scan layers share one masked opening.
std::vector<std::vector<CggmEval>> segmentedDcfOnlineBatchGroup(
    int role, const std::vector<SegmentedDcfBatchView>& groups, Peer* player,
    SegmentedDcfGroupWorkspace* workspace = nullptr);

// Every threshold row contains exactly bits XOR shares, most significant bit
// first. Thresholds are already Boolean shared: no private-input BitDec runs.
// nseg=1 uses a full cGGM tree and requires no online scan communication.
SegmentedDcfMaterial segmentedDcfOfflineBatch(
    int role, const std::vector<std::vector<u8>>& threshold_bit_shares_msb,
    int bits, int nseg, Peer* player);
SegmentedDcfMaterial segmentedDcfOfflineBatch(
    int role, const std::vector<std::vector<u8>>& threshold_bit_shares_msb,
    int bits, int nseg, Peer* player, const DcfGenerationOptions& options);
SegmentedDcfMaterial segmentedDcfOfflineBatch(
    int role, const std::vector<std::vector<u8>>& threshold_bit_shares_msb,
    int bits, int nseg, Peer* player, const DcfGenerationOptions& options,
    bool need_e);
SegmentedDcfMaterial segmentedDcfOffline(
    int role, const std::vector<u8>& threshold_bit_shares_msb,
    int bits, int nseg, Peer* player);

// The queries are public unsigned bits-wide values. Returned L/e are XOR
// shares of query < threshold / query == threshold, respectively. Each
// material is consumed once, even if a later network operation fails.
std::vector<CggmEval> segmentedDcfOnlineBatch(
    int role, const std::vector<uint64_t>& public_queries,
    SegmentedDcfMaterial& material, Peer* player);
CggmEval segmentedDcfOnline(int role, uint64_t public_query,
                           SegmentedDcfMaterial& material, Peer* player);

// Idempotent release; a released or moved-from material cannot be evaluated.
void releaseSegmentedDcfMaterial(SegmentedDcfMaterial& material);

}  // namespace segfss
