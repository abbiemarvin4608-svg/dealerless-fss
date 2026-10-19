#pragma once

#include <cstddef>
#include <vector>
#include "commons/types.h"
#include "segment.h"
#include "cggm.h"  // CggmKey
#include "instrumentation.h"

// Π_Gen: Per-segment Boolean-leaf DPF generation (Algorithm 3 offline)
// and Grotto prefix-parity evaluation (Algorithm 3 online).

namespace segfss {

struct SegmentKeys {
    std::vector<CggmKey> keys;  // One cGGM key per segment (rewired from DPFKeyPack)
};

// Boolean Beaver material for the cross-segment scan.  The triples are
// independent of the public evaluation point, so a caller can generate them
// with the rest of the offline material and consume them during infer.
struct BooleanAndPreprocessing {
    std::vector<u8> a;
    std::vector<u8> b;
    std::vector<u8> c;
    std::size_t cursor = 0;

    std::size_t size() const { return a.size(); }
};

// Generate `gate_count` Boolean Beaver triples using the same KKOT-backed
// TripleGenerator as the secure AND primitive.  Communication is charged to
// the caller's current phase; this function does not open any model input.
BooleanAndPreprocessing booleanAndOffline(int party_id, int gate_count,
                                          Peer* player);

struct BooleanOpenWorkspace {
    std::vector<u8> packed, remote_packed;
};

struct BooleanAndBatchView {
    const u8* a;
    const u8* b;
    u8* output;
    int count;
    BooleanAndPreprocessing* material;
};
struct BooleanAndWorkspace {
    std::vector<u8> opened;
    BooleanOpenWorkspace wire;
};
void booleanAndOnlineBatch(int party_id, const std::vector<BooleanAndBatchView>& groups,
                          Peer* player, BooleanAndWorkspace* workspace = nullptr);

// XOR-open masked Boolean shares, with optional LSB-first bit packing.
void openBooleanBits(int party_id, std::vector<u8>& values, Peer* player,
                     BooleanOpenWorkspace* workspace = nullptr);

// Consume preprocessed triples for a batched Boolean AND.  The only network
// traffic here is the masked e/f opening, so it can be timed as online work.
void booleanAndOnline(int party_id, const u8* dataA, const u8* dataB,
                      u8* output, int size,
                      BooleanAndPreprocessing& preprocessing, Peer* player);

// Offline: Generate correlated Boolean-leaf DPF keys for each segment.
// Takes the Boolean-shared segment bits from Π_Split.
// Returns one DPF key per segment.
// Optional timer enables E3 phase breakdown (delegates to cggmGen2pc).
SegmentKeys genSegmentKeys(int party_id, const SplitResult& split_result,
                           PhaseTimer* timer = nullptr);

// Online evaluation result for one segment.
struct SegmentEvalResult {
    u8 L_j;  // Boolean share of comparison bit (Grotto prefix parity)
    u8 e_j;  // Boolean share of equality bit (leaf control bit)
};

// Online: Evaluate one segment's DPF key at public digit xhat_seg_j.
// Implements Grotto prefix-parity walk: accumulate right-sibling control bits
// when x_i=0 to compute the comparison share L_j, and return the leaf control
// bit e_j for equality.
SegmentEvalResult evalSegment(int party_id, const CggmKey& key,
                               const std::vector<u8>& xhat_seg_j);

// Online: Evaluate all segments at their respective public digits.
// xhat_segments[j] is the j-th segment's digit (MSB-first bit vector).
std::vector<SegmentEvalResult> evalAllSegments(
    int party_id, const SegmentKeys& keys,
    const std::vector<std::vector<u8>>& xhat_segments);

// Π_Infer: Cross-segment scan to compute the final comparison share.
// Takes per-segment evaluation results (L_j, e_j) and computes the Boolean
// share of 1{xhat < r} via the paper's comparison decomposition:
//   1{xhat < r} = XOR_j ( E_j AND L_j )
// where E_j = AND_{i<j} e_i (prefix product of equality bits).
//
// Implemented as a balanced associative scan over (L_j,e_j): exactly
// 2*(nseg-1) AND gates, ceil(log2(nseg)) dependent AND rounds, and local XORs.
u8 inferComparison(int party_id,
                   const std::vector<SegmentEvalResult>& segment_results,
                   Peer* player);

// Variant used when the scan triples were generated in the offline phase.
u8 inferComparisonPreprocessed(
    int party_id, const std::vector<SegmentEvalResult>& segment_results,
    BooleanAndPreprocessing& preprocessing, Peer* player);

}  // namespace segfss
