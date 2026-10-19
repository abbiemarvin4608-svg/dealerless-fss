#pragma once

// ---------------------------------------------------------------------------
// Segment decomposition for dealerless 32-bit FSS comparison.
//
// Implements the paper's parallel multi-segment decomposition:
//   - Pi_Split (Algorithm 1): decompose the ell-bit threshold share into
//     nseg per-segment Boolean shares via a single BitDec, then slice.
//   - Segment layout: an ell-bit value splits into nseg segments, most
//     significant first, with per-segment widths wseg_j ~= ell / nseg.
//
// The comparison  1{xhat < r}  decomposes (Lemma "Comparison decomposition")
// into per-segment comparison bits L_j and prefix-equality bits E_j:
//     1{xhat < r} = XOR_j ( E_j AND L_j ),
// where L_j = 1{xhat^(j) < r^(j)} and E_j = AND_{i<j} 1{xhat^(i) = r^(i)}.
// ---------------------------------------------------------------------------

#include <cstdint>
#include <string>
#include <vector>

#include "commons/group_element.h"
#include "mpc/comms.h"

namespace segfss {

enum class SegmentPolicy { explicit_counts, fixed_2, fixed_4, depth_cap_8 };
SegmentPolicy parseSegmentPolicy(const std::string& name);
const char* segmentPolicyName(SegmentPolicy policy);
int segmentCount(int bits, SegmentPolicy policy);

// Describes how an ell-bit domain is cut into nseg segments (MS segment first).
// Segment widths are as equal as possible; when ell is not divisible by nseg
// the leading segments each take one extra bit.
struct SegmentLayout {
    int ell = 32;           // total input bit-length
    int nseg = 1;           // number of segments
    std::vector<int> width; // per-segment bit widths, index 0 = most significant
    std::vector<int> shift; // low-bit offset of each segment within the ell-bit word

    SegmentLayout() = default;
    SegmentLayout(int ell, int nseg);

    int size() const { return nseg; }
    int maxWidth() const;   // max_j wseg_j : critical-path depth after splitting
};

// Extract segment j (public value) from a public ell-bit word, MS segment = 0.
// Returns a wseg_j-bit GroupElement holding the digit xhat^(j).
GroupElement segmentDigit(const GroupElement& x, const SegmentLayout& layout,
                          int j);

// ---------------------------------------------------------------------------
// Pi_Split (Algorithm 1).
//
// Input:  party_id, the ell-bit threshold arithmetic share <r>, and a layout.
// Output: for each segment j, the Boolean shares of the bits of r^(j),
//         MS bit first, as a vector<u8> of length wseg_j.
//
// Cost (per the paper): one BitDec call dominates at O(ell) rounds and
// O(ell^2 * lambda) communication; the per-segment slicing is local.
// ---------------------------------------------------------------------------
struct SplitResult {
    SegmentLayout layout;
    // segment_bits[j] holds wseg_j Boolean shares (MS bit first) of r^(j).
    std::vector<std::vector<u8>> segment_bits;
};

SplitResult split(int party_id, GroupElement threshold_share,
                  const SegmentLayout& layout, Peer* player);

}  // namespace segfss
