#pragma once

#include <cstdint>

#include "commons/group_element.h"
#include "commons/public_data.h"
#include "buildingblock/lut.h"
#include "exact_truncation.h"
#include "relu.h"
#include "gen.h"
#include "cggm.h"
#include "dpf_lut.h"
#include "activation_metrics.h"
#include "private_msb.h"

namespace segfss {

// Standalone SIGMA defaults. Explicit caller precision remains supported so
// a network precision scan keeps one Q(ell,s) format throughout its graph.
inline constexpr int kSigmaDefaultFracBits = 12;
inline constexpr int kSigmaDefaultQ = 10;
inline constexpr int kSigmaDefaultTableSize = 1 << kSigmaDefaultQ;
inline constexpr int kSigmaBoundLog2 = 4;
enum class ActivationSignBackend { bitdec, segmented_fss };

// Legacy unmasked piecewise-comparison interface retained for source
// compatibility. Offline/online calls reject use because that implementation
// opened the raw input. Secure activations use BitDec below.
struct PiecewiseComparisonKey {
    int bits = 0;
    int nseg = 0;
    uint64_t threshold = 0;
    bool is_signed = false;  // If true, use signed comparison via MSB flip
    SegmentKeys segment_keys;
    BooleanAndPreprocessing and_prep;
};

// Disabled: throws std::logic_error before any protocol communication.
PiecewiseComparisonKey piecewiseCompareOffline(
    int party_id, int bits, uint64_t threshold, int nseg, Peer* player,
    bool is_signed = false);

// Disabled: throws std::logic_error before any input is opened.
u8 piecewiseCompareOnline(
    int party_id, GroupElement x,
    const PiecewiseComparisonKey& key, Peer* player);

void freePiecewiseComparisonKey(PiecewiseComparisonKey& key);

// SIGMA's folded delta table.  The default q=10 table has 1024 entries over
// [0,16), with six fractional index bits when frac_bits=12.
PublicLUTData makeSigmaDeltaTable(
    int output_bits, int frac_bits = kSigmaDefaultFracBits,
    int q = kSigmaDefaultQ);

// One-use preprocessing for the dealerless secure folded lookup. BitDec
// returns secret Boolean shares for the sign and bound tests; BitXA selects
// the magnitude, half-open clipped value and output branch.
struct SigmaActivationKey {
    Peer* owner_peer = nullptr;
    int bits = 0;
    int frac_bits = 0;
    int clip_bits = 0;
    bool native_signed16_input = false;
    bool reuse_input_sign = false;
    int q = kSigmaDefaultQ;
    int trunc_bits = 0;
    int nseg = 2;
    ActivationSignBackend sign_backend = ActivationSignBackend::bitdec;
    PublicLUTData delta_table;
    std::shared_ptr<const PublicLUTData> public_delta_table;
    std::shared_ptr<const PreparedDpfLutTable> prepared_delta_table;
    PublicLutKeyPack lut_key;
    // cGGM equality evaluation and B2A materialize the masked output table
    // offline. Online opens its masked index and consumes the table once.
    struct SegmentedLutMaterial {
        int index_bits = 0;
        int output_bits = 0;
        int nseg = 2;
        GroupElement mask_share;
        SegmentKeys keys;
        BooleanAndPreprocessing equality_and;
        std::vector<GroupElement> output_share_table;
        double online_ms = 0.0;
        bool consumed = false;
    } segmented_lut;
    // Optional dealerless correlated-DPF public-LUT material.  The backend is
    // selected during offline preparation and consumes this material exactly
    // once during the masked-index opening.
    DpfPublicLutMaterial dpf_lut;
    bool use_dpf_lut = false;
    // The LUT index is a bounded integer reduction.  Keep its exact TR key
    // so the half-open [0, 2^q) table domain cannot wrap on a random carry.
    ExactTruncationKey trunc_key;

    // Kept for source compatibility. sigmaOffline leaves this false;
    // sigmaOnline/batch reject true rather than opening raw activations.
    bool use_piecewise_compare = false;

    // Legacy fields; no piecewise keys are generated for activations.
    PiecewiseComparisonKey sign_cmp;   // For sign test
    PiecewiseComparisonKey clip_cmp;   // For bound test

    // One-use keys for the BitDec/BitXA path.
    BitXAPreprocessing relu_mul;
    BitXAPreprocessing clip_mul;    // Legacy unused field.
    BitXAPreprocessing branch_mul;  // Legacy unused field.

    // Half-open clipping and sigmoid branch selection.
    BitXAPreprocessing clip_select_mul;
    BitXAPreprocessing branch_select_mul;
    BooleanAndPreprocessing sign_and;
    BooleanAndPreprocessing clip_and;
};

struct ActivationStatistics {
    uint64_t generated = 0, consumed = 0, failed_batches = 0;
    uint64_t offline_sent = 0, offline_received = 0;
    uint64_t online_sent = 0, online_received = 0;
};

// The native signed16 extension can supply the original input sign. Scaling
// by 32 (or 64 for TanH) preserves it in ring32. Bind these secret XOR shares
// to the same one-use prepared query as the activation inventory.
struct ActivationInputSignShares {
    Peer* owner_peer = nullptr;
    DpfLutBinding binding;
    uint64_t generation_epoch = 0;
    std::vector<u8> shares;
    bool consumed = false;
    ActivationInputSignShares() = default;
    ActivationInputSignShares(const ActivationInputSignShares&) = delete;
    ActivationInputSignShares& operator=(const ActivationInputSignShares&) = delete;
    ActivationInputSignShares(ActivationInputSignShares&&) = default;
    ActivationInputSignShares& operator=(ActivationInputSignShares&&) = default;
};

// One caller owns a communication/OT channel and an independent one-use
// activation inventory. The channel's PRNG and counters stay with that owner.
struct ActivationContext {
    const int party_id;
    Peer* const player;
    const uint64_t identity;
    uint64_t generation_epoch = 0;
    std::size_t material_cursor = 0;
    bool prepared = false;
    std::vector<SigmaActivationKey> materials;
    BooleanAndWorkspace boolean_workspace;
    ActivationStatistics statistics;
    bool paper_lut = false;
    DpfLutBinding lut_binding;
    DpfLutOptions lut_options;
    DpfLutBatchMetrics lut_metrics;
    std::shared_ptr<const PreparedDpfLutTable> public_table;
    ActivationSignBackend sign_backend = ActivationSignBackend::bitdec;
    int sign_segments = 0;
    // Only the exact native signed16/Q7 adapter may narrow the clip predicate
    // or reuse signs. Arithmetic, BitXA and the LUT remain in ring32.
    bool native_signed16_input = false;
    bool reuse_input_sign = false;
    int clip_bits = 0;  // zero follows the current arithmetic ring width
    // zero follows sign_segments for each preparation, including changes
    // between fresh queries; explicit native widths remain independent.
    int clip_segments = 0;
    struct SignMaterial {
        MsbMaterial material;
        DpfLutBinding binding;
        uint64_t generation_epoch = 0;
        bool clip = false;
    } sign_material, clip_material;

    ActivationContext(int role, Peer* channel, uint64_t context_identity);
    ~ActivationContext();
    ActivationContext(const ActivationContext&) = delete;
    ActivationContext& operator=(const ActivationContext&) = delete;
    osuCrypto::PRNG& randomSource() noexcept { return player->randomSource(); }
};

void validateActivationSignInventory(const ActivationContext& context, int count, int bits);

void prepareSigmaActivation(ActivationContext& context, int batch_size,
                            int bits, int frac_bits = kSigmaDefaultFracBits,
                            int q = kSigmaDefaultQ);
void prepareSigmaActivation(ActivationContext& context, int batch_size,
    int bits, int frac_bits, int q, const DpfLutOptions& options,
    const DpfLutBinding& binding, DpfLutBatchMetrics* metrics = nullptr,
    ActivationMetrics* stages = nullptr);
std::vector<GroupElement> sigmaBatchOnline(
    ActivationContext& context, const std::vector<GroupElement>& inputs);
std::vector<GroupElement> sigmaBatchOnline(
    ActivationContext& context, const std::vector<GroupElement>& inputs,
    ActivationMetrics* stages);
std::vector<GroupElement> sigmaBatchOnline(
    ActivationContext& context, const std::vector<GroupElement>& inputs,
    ActivationMetrics* stages, ActivationInputSignShares* input_signs);
std::vector<GroupElement> tanhBatchOnline(
    ActivationContext& context, const std::vector<GroupElement>& inputs);

SigmaActivationKey sigmaOffline(int party_id, int bits, int frac_bits,
                                int q, Peer* player);
inline SigmaActivationKey sigmaOffline(int party_id, int bits, Peer* player) {
    return sigmaOffline(party_id, bits, kSigmaDefaultFracBits,
                        kSigmaDefaultQ, player);
}
GroupElement sigmaOnline(int party_id, GroupElement input,
                         SigmaActivationKey& key, Peer* player);

SigmaActivationKey tanhOffline(int party_id, int bits, int frac_bits,
                               int q, Peer* player);
inline SigmaActivationKey tanhOffline(int party_id, int bits, Peer* player) {
    return tanhOffline(party_id, bits, kSigmaDefaultFracBits,
                       kSigmaDefaultQ, player);
}
// tanhOnline forms 2*x in the input ring.  Callers therefore provide a
// bounded fixed-point preactivation with enough headroom for this doubling;
// with f=12 and q=10, the folded table saturates at |x| >= 8.
GroupElement tanhOnline(int party_id, GroupElement input,
                        SigmaActivationKey& key, Peer* player);

void freeSigmaActivationKey(SigmaActivationKey& key);

// ========== Batch Sigmoid/Tanh Operations ==========

// Batch offline: Generate keys for batch_size sigmoid operations
std::vector<SigmaActivationKey> sigmaBatchOffline(
    int party_id, int batch_size, int bits, int frac_bits, int q, Peer* player);
inline std::vector<SigmaActivationKey> sigmaBatchOffline(
    int party_id, int batch_size, int bits, Peer* player) {
    return sigmaBatchOffline(party_id, batch_size, bits,
                             kSigmaDefaultFracBits, kSigmaDefaultQ, player);
}

// All input-dependent openings and Boolean carry gates are batched across
// the activation vector. Each entry consumes independent offline material.
std::vector<GroupElement> sigmaBatchOnline(
    int party_id,
    const std::vector<GroupElement>& inputs,
    std::vector<SigmaActivationKey>& keys,
    Peer* player);

inline std::vector<SigmaActivationKey> tanhBatchOffline(
    int party_id, int batch_size, int bits, int frac_bits, int q, Peer* player) {
    return sigmaBatchOffline(party_id, batch_size, bits, frac_bits, q, player);
}
std::vector<GroupElement> tanhBatchOnline(
    int party_id,
    const std::vector<GroupElement>& inputs,
    std::vector<SigmaActivationKey>& keys,
    Peer* player);

// Disabled legacy batch comparison; throws before any input is opened.
std::vector<u8> batchPiecewiseCompareOnline(
    int party_id,
    const std::vector<GroupElement>& inputs,
    const std::vector<PiecewiseComparisonKey>& keys,
    Peer* player);

// Cleartext fixed-point mirrors used to validate the public SIGMA table and
// to generate precision reports.  The secure protocol consumes the same
// PublicLUTData through the segmented dealerless cGGM LUT after its
// DReLU/abs/index steps.
GroupElement sigmaPlain(GroupElement input,
                        int frac_bits = kSigmaDefaultFracBits,
                        int q = kSigmaDefaultQ);
GroupElement tanhPlain(GroupElement input,
                       int frac_bits = kSigmaDefaultFracBits,
                       int q = kSigmaDefaultQ);

struct SigmaActivationError {
    double max_abs = 0.0;
    double max_ulp = 0.0;
    double at_input = 0.0;
};

struct SigmaActivationErrorReport {
    SigmaActivationError sigmoid;
    SigmaActivationError tanh;
};

// Exhaustively compare the fixed-point table with real sigmoid/tanh over the
// representable interval [-16,16) at the requested fractional precision.
SigmaActivationErrorReport measureSigmaActivationError(
    int input_bits, int frac_bits = kSigmaDefaultFracBits,
    int q = kSigmaDefaultQ);

}  // namespace segfss
