#include "segment.h"

#include "mpc/secure_ops.h"

#include <algorithm>
#include <cassert>
#include <stdexcept>

namespace segfss {

SegmentPolicy parseSegmentPolicy(const std::string& name) {
    if (name == "legacy") return SegmentPolicy::explicit_counts;
    if (name == "fixed-2") return SegmentPolicy::fixed_2;
    if (name == "fixed-4") return SegmentPolicy::fixed_4;
    if (name == "depth-cap-8") return SegmentPolicy::depth_cap_8;
    throw std::invalid_argument("unknown segment policy");
}

const char* segmentPolicyName(SegmentPolicy policy) {
    switch (policy) {
    case SegmentPolicy::explicit_counts: return "legacy";
    case SegmentPolicy::fixed_2: return "fixed-2";
    case SegmentPolicy::fixed_4: return "fixed-4";
    case SegmentPolicy::depth_cap_8: return "depth-cap-8";
    }
    throw std::invalid_argument("unknown segment policy");
}

int segmentCount(int bits, SegmentPolicy policy) {
    if (bits < 0 || bits > 64) throw std::invalid_argument("segment policy: domain outside [0,64]");
    switch (policy) {
    case SegmentPolicy::fixed_2: return std::min(2, bits);
    case SegmentPolicy::fixed_4: return std::min(4, bits);
    case SegmentPolicy::depth_cap_8: return (bits + 7) / 8;
    default: throw std::invalid_argument("segment policy requires an explicit fixed or depth cap policy");
    }
}

// ---------------------------------------------------------------------------
// SegmentLayout
// ---------------------------------------------------------------------------
SegmentLayout::SegmentLayout(int ell_, int nseg_)
    : ell(ell_), nseg(nseg_) {
    if (ell < 1 || ell > 64 || nseg < 1 || nseg > ell) {
        throw std::invalid_argument("SegmentLayout: require 1 <= nseg <= ell <= 64");
    }
    width.resize(nseg);
    shift.resize(nseg);

    // Distribute ell bits as evenly as possible: the first (ell % nseg)
    // segments get ceil(ell/nseg) bits, the rest get floor(ell/nseg).
    const int base_width = ell / nseg;
    const int extra_count = ell % nseg;

    int bit_offset = 0;
    for (int j = 0; j < nseg; j++) {
        width[j] = base_width + (j < extra_count ? 1 : 0);
        shift[j] = ell - bit_offset - width[j];  // MS segment first
        bit_offset += width[j];
    }
}

int SegmentLayout::maxWidth() const {
    return *std::max_element(width.begin(), width.end());
}

// ---------------------------------------------------------------------------
// Extract public segment digit.
//
// The j-th segment (MS=0) spans bits [ell - sum_{i<j} w_i - 1 : shift[j]],
// i.e., the wseg_j bits starting at bit position shift[j] (0=LSB).
// ---------------------------------------------------------------------------
GroupElement segmentDigit(const GroupElement& x, const SegmentLayout& layout,
                          int j) {
    assert(j >= 0 && j < layout.nseg);
    const uint64_t mask = layout.width[j] == 64 ? UINT64_MAX :
                          (uint64_t(1) << layout.width[j]) - 1;
    const uint64_t digit = (x.value >> layout.shift[j]) & mask;
    return GroupElement(digit, layout.width[j]);
}

// ---------------------------------------------------------------------------
// Pi_Split (Algorithm 1).
//
// From the paper (Section 4.2, lines ~610–625):
//   1. Run BitDec once on the ell-bit threshold share <r> to obtain Boolean
//      shares <r_1>, ..., <r_ell> of the bits of r (MSB first).
//   2. For each segment j, slice out the bits corresponding to r^(j) locally.
//   3. Each segment carries its own root seed into generation (Sec 4.3), so
//      the nseg shallow GGM trees of depth wseg_j are built concurrently.
//
// BitDec is the distributed bit-decomposition protocol from [Distributed-FSS]
// with cost O(ell) rounds and O(ell^2 * lambda) communication.
//
// The reference library's BitDec is `std::vector<u8> BitDec(party_id, input,
// bits, player)` and returns MSB-first Boolean shares.
// ---------------------------------------------------------------------------
SplitResult split(int party_id, GroupElement threshold_share,
                  const SegmentLayout& layout, Peer* player) {
    // Step 1: BitDec on the full ell-bit threshold.
    std::vector<u8> r_bits = BitDec(party_id, threshold_share, layout.ell, player);
    assert(r_bits.size() == static_cast<size_t>(layout.ell));

    // Step 2: Slice the bits into per-segment shares.
    // r_bits[0] is the MSB. Segment j's bits start at bit_offset below.
    SplitResult result;
    result.layout = layout;
    result.segment_bits.resize(layout.nseg);

    int bit_offset = 0;
    for (int j = 0; j < layout.nseg; j++) {
        const int w = layout.width[j];
        result.segment_bits[j].resize(w);
        for (int k = 0; k < w; k++) {
            result.segment_bits[j][k] = r_bits[bit_offset + k];
        }
        bit_offset += w;
    }

    return result;
}

}  // namespace segfss
