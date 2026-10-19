#pragma once

#include <cstddef>
#include <cstdint>
#include <vector>
#include <memory>
#include <array>
#include <utility>

#include "commons/group_element.h"
#include "commons/public_data.h"
#include "fss/dpf.h"
#include "mpc/api.h"

namespace segfss {

enum class DpfLutKeyFamily {
    full_q_arithmetic = 0,
    segmented_boolean_tensor_b2a = 1
};

struct DpfLutOptions {
    DpfLutOptions();
    DpfLutKeyFamily family = DpfLutKeyFamily::full_q_arithmetic;
    std::size_t tile_size = 1024;
    std::size_t participant_limit = 1;
    // Empty selects the family contract: {10} or {5,5}.
    std::vector<int> segment_widths;
    // Opt-in local kernels, frozen with the prepared material. The constructor
    // reads SEGFSS_LUT_SKIP_ZERO and SEGFSS_LUT_SIMD (both default to false).
    bool skip_zero_coefficients = false;
    bool use_simd = false;
};

// The SIMD option falls back to the portable scalar kernel when unavailable.
bool dpfLutSimdAvailable() noexcept;

struct DpfLutBinding {
    uint64_t context_id = 0, query_id = 0, operator_id = 0;
    uint64_t worker_id = 0, slot_id = 0, element_begin = 0, table_version = 1;
    DpfLutKeyFamily family = DpfLutKeyFamily::full_q_arithmetic;
    bool operator==(const DpfLutBinding& other) const noexcept;
    bool operator!=(const DpfLutBinding& other) const noexcept { return !(*this == other); }
};

// An immutable public table and continuous coefficients prepared once,
// before private input. The fingerprint binds contents as well as version.
class PreparedDpfLutTable {
public:
    PreparedDpfLutTable(const PublicLUTData& table, uint64_t version);
    const PublicLUTData& table() const noexcept { return table_; }
    const std::vector<uint32_t>& coefficients() const noexcept { return coefficients_; }
    const std::vector<std::pair<std::size_t, std::size_t>>& nonzeroRanges() const noexcept {
        return nonzero_ranges_;
    }
    std::size_t nonzeroCoefficientCount() const noexcept { return nonzero_count_; }
    uint64_t version() const noexcept { return version_; }
    uint64_t fingerprint() const noexcept { return fingerprint_; }
private:
    PublicLUTData table_;
    std::vector<uint32_t> coefficients_;
    // Half-open intervals depend only on the immutable public coefficients.
    std::vector<std::pair<std::size_t, std::size_t>> nonzero_ranges_;
    std::size_t nonzero_count_ = 0;
    uint64_t version_ = 0, fingerprint_ = 0;
};

std::shared_ptr<const PreparedDpfLutTable> prepareDpfLutTable(
    const PublicLUTData& table, uint64_t version);

// Continuous little-endian ring elements, without per-element host metadata.
// A 9--16 bit arithmetic selection share occupies two bytes per domain point.
class PackedDpfSelectionVector {
public:
    PackedDpfSelectionVector() = default;
    PackedDpfSelectionVector(const PackedDpfSelectionVector&) = default;
    PackedDpfSelectionVector& operator=(const PackedDpfSelectionVector&) = default;
    PackedDpfSelectionVector(PackedDpfSelectionVector&& other) noexcept;
    PackedDpfSelectionVector& operator=(PackedDpfSelectionVector&& other) noexcept;
    void assign(const std::vector<GroupElement>& values, int output_bits);
    void reset(std::size_t count, int output_bits);
    void set(std::size_t index, uint64_t value);
    GroupElement operator[](std::size_t index) const;
    uint32_t shiftedDotProduct32(const std::vector<uint32_t>& coefficients,
                                std::size_t shift) const;
    uint32_t shiftedDotProduct32(const PreparedDpfLutTable& table,
                                std::size_t shift, bool skip_zero, bool use_simd) const;
    std::size_t size() const noexcept { return count_; }
    std::size_t capacityBytes() const noexcept { return bytes_.capacity(); }
    int bitsize() const noexcept { return output_bits_; }
    void clear() noexcept { bytes_.clear(); count_ = 0; }
    void shrink_to_fit() { bytes_.shrink_to_fit(); }
private:
    std::vector<uint8_t> bytes_;
    std::size_t count_ = 0;
    int output_bits_ = 0;
};

struct DpfLutStageMetric {
    uint64_t elapsed_us = 0, sent_bytes = 0, received_bytes = 0, channel_rounds = 0;
    uint64_t begin_us = 0, end_us = 0;
};

struct DpfLutBatchMetrics {
    uint64_t generation_epoch = 0;
    uint64_t generated = 0, consumed = 0, remaining = 0;
    uint64_t tiles = 0, eval_points = 0, online_open_batches = 0;
    uint64_t key_trees = 0, mask_b2a_values = 0, tensor_and_gates = 0, leaf_b2a_values = 0;
    // Local materialization work, excluding interactive key generation.
    uint64_t local_tree_expansions = 0, local_internal_hashes = 0, local_leaf_hashes = 0;
    std::size_t leaf_cache_peak_bytes = 0;
    int tree_max_depth = 0;
    std::size_t resident_bytes = 0, peak_working_bytes = 0;
    uint64_t offline_sent_bytes = 0, offline_received_bytes = 0;
    uint64_t online_sent_bytes = 0, online_received_bytes = 0;
    uint64_t offline_us = 0, mask_conversion_us = 0, keygen_us = 0;
    uint64_t eval_all_us = 0, online_open_us = 0, inner_product_us = 0;
    // Preparation never occurs in an online invocation. These counters are
    // explicit lifecycle evidence, not a subtraction from online timings.
    uint64_t online_generated = 0, online_loaded_bytes = 0;
    // These actual stages are disjoint. Historical keygen_us remains inclusive
    // of payload adaptation; use tree_keygen_stage + payload_stage to split it.
    DpfLutStageMetric mask_stage, tree_keygen_stage, payload_stage, materialize_stage;
    DpfLutStageMetric tensor_and_stage, leaf_b2a_stage, open_stage, product_stage;
};

// A public-function LUT materialized with a dealerless arithmetic correlated
// DPF.  The target is a fresh random mask r generated before the query.  Each
// party retains an arithmetic one-hot share at r and an arithmetic share of r.
// Neither the target nor the one-hot vector is
// revealed during preparation.
struct DpfPublicLutMaterial {
    DpfPublicLutMaterial() = default;
    DpfPublicLutMaterial(const DpfPublicLutMaterial&) = delete;
    DpfPublicLutMaterial& operator=(const DpfPublicLutMaterial&) = delete;
    DpfPublicLutMaterial(DpfPublicLutMaterial&& other) noexcept;
    DpfPublicLutMaterial& operator=(DpfPublicLutMaterial&& other) noexcept;
    int index_bits = 0;
    int output_bits = 0;
    int key_group_size = 0;
    uint64_t request_id = 0;
    uint64_t generation_epoch = 0;
    Peer* owner_peer = nullptr;
    DpfLutBinding binding;
    DpfLutKeyFamily family = DpfLutKeyFamily::full_q_arithmetic;
    int lut_segments = 1;
    std::array<int, 2> segment_bits{{10, 0}};
    uint64_t table_fingerprint = 0;
    bool paper_lookup = false;
    bool lut_skip_zero = false, lut_simd = false;
    GroupElement mask_share;
    // Arithmetic shares of the one-hot function at the random mask.
    // The public table is supplied to online evaluation and is not duplicated
    // for every request.
    PackedDpfSelectionVector selection_share;
    DPFKeyPack key;
    bool consumed = false;
    bool released = false;

    std::size_t residentBytes() const noexcept;
    bool hasValidPaperLayout() const noexcept;
};

// Generate one material item.  All communication and full-domain DPF
// evaluation happen before the caller loads a private query.
DpfPublicLutMaterial dpfPublicLutOffline(
    int party_id, const PublicLUTData& table, uint64_t request_id,
    Peer* player, uint64_t generation_epoch = 1);

// Same-depth trees share every interactive key-generation stage. Bounded
// tiles limit scratch memory; all final vectors remain resident on return.
std::vector<DpfPublicLutMaterial> dpfPublicLutBatchOffline(
    int party_id, const PublicLUTData& table, std::size_t count,
    uint64_t request_id_base, Peer* player, uint64_t generation_epoch = 1,
    DpfLutBatchMetrics* metrics = nullptr, std::size_t tile_size = 1024);

// Paper q10/ring32 lifecycle. Both key families produce identical resident
// selectors and use the same subtract-mask opening and local inner product.
std::vector<DpfPublicLutMaterial> dpfPublicLutBatchOffline(
    int party_id, const PreparedDpfLutTable& table, std::size_t count,
    const DpfLutBinding& binding, const DpfLutOptions& options, Peer* player,
    uint64_t generation_epoch, DpfLutBatchMetrics* metrics = nullptr);
std::vector<GroupElement> dpfPublicLutBatchOnline(
    int party_id, const std::vector<GroupElement>& inputs,
    std::vector<DpfPublicLutMaterial>& materials, const DpfLutBinding& expected,
    Peer* player, const PreparedDpfLutTable& table,
    DpfLutBatchMetrics* metrics = nullptr);

// Multiple public outputs share one masked-index opening and one one-hot
// vector. Each table retains its own arithmetic output ring. Materials must
// have been generated in the largest of those rings.
std::vector<std::vector<GroupElement>> dpfPublicLutBatchOnlineMany(
    int party_id, const std::vector<GroupElement>& inputs,
    std::vector<DpfPublicLutMaterial>& materials, Peer* player,
    uint64_t request_id_base, const std::vector<const PublicLUTData*>& tables,
    DpfLutBatchMetrics* metrics = nullptr);

// Each input selects its public LUT (for example, different exponential
// digits). All indices are still opened together, and each local convolution
// evaluates only its designated table. Shape and identities are checked for
// the complete batch before any opening.
std::vector<GroupElement> dpfPublicLutBatchOnlinePerInput(
    int party_id, const std::vector<GroupElement>& inputs,
    std::vector<DpfPublicLutMaterial>& materials, Peer* player,
    uint64_t request_id_base,
    const std::vector<const PublicLUTData*>& table_for_input,
    DpfLutBatchMetrics* metrics = nullptr);

// Open all masked indices in one exchange and compute the resident local
// selector's public-table inner product. Every material is consumed before I/O; malformed
// batches, wrong identities, repeated consumption, and inventory shortages
// fail without an online fallback.
std::vector<GroupElement> dpfPublicLutBatchOnline(
    int party_id, const std::vector<GroupElement>& inputs,
    std::vector<DpfPublicLutMaterial>& materials, Peer* player,
    uint64_t request_id_base, const PublicLUTData& table);

GroupElement dpfPublicLutOnline(
    int party_id, GroupElement input, DpfPublicLutMaterial& material,
    Peer* player, const PublicLUTData& table);

void releaseDpfPublicLutMaterial(DpfPublicLutMaterial& material) noexcept;

}  // namespace segfss
