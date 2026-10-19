#pragma once

// E4 ablation instrumentation: cost-counted variants of cGGM expansion and
// segment evaluation, for isolating the Half-Tree and Grotto contributions.
//
//   (i)  Half-Tree cGGM vs standard GGM: countExpansion() reports the number
//        of hash (PRG) calls each backend needs to build a depth-`d` tree.
//   (ii) Grotto prefix-parity vs full-domain: cggmEvalFullDomain() reconstructs
//        the same {L,e} as cggmEval but by expanding all 2^depth leaves; the
//        optional hash counter exposes the O(N) vs O(log N) gap.

#include <cstdint>
#include <vector>

#include <cryptoTools/Common/Defines.h>

#include "cggm.h"

namespace segfss {

// Hash-call counts to build a depth-`d` GGM tree under each backend.
struct ExpansionCost {
    int depth = 0;
    uint64_t half_tree_hashes = 0;    // 1 hash / internal node (right = H^parent)
    uint64_t standard_ggm_hashes = 0; // 2 hashes / internal node (independent L,R)
};

// Count expansion hashes for a full depth-`d` binary tree under both backends.
ExpansionCost countExpansion(int depth);

// Full-domain evaluator: expand every leaf, reconstruct L=1{xhat<alpha} and
// e=1{xhat==alpha} by summing leaf parities. If hash_count != nullptr, adds the
// number of hash calls performed. Semantically identical (per party share) to
// cggmEval, but O(2^depth) instead of O(depth).
CggmEval cggmEvalFullDomain(const CggmKey& key,
                            const std::vector<osuCrypto::u8>& xhat_bits,
                            uint64_t* hash_count);

// Prefix-parity evaluator with a hash counter (same result as cggmEval).
CggmEval cggmEvalPrefixCounted(const CggmKey& key,
                               const std::vector<osuCrypto::u8>& xhat_bits,
                               uint64_t* hash_count);

}  // namespace segfss
