#include "ablation.h"

#include <cstring>
#include <stdexcept>
#include <vector>

#include "cggm.h"
#include "fss/internal/ggm.h"  // dfss::internal::ccrHash

namespace segfss {

// Block LSB helpers (mirror the file-local ones in cggm.cpp; deliberately
// self-contained to avoid the reference library's static-init-order fiasco).
namespace {
inline osuCrypto::u8 blockLsb(const osuCrypto::block& b) {
    return static_cast<osuCrypto::u8>(_mm_cvtsi128_si64x(b) & 1);
}
inline osuCrypto::block setBlockLsb(const osuCrypto::block& input,
                                    osuCrypto::u8 bit) {
    const osuCrypto::block one = _mm_set_epi64x(0, 1);
    if (blockLsb(input) != (bit & 1)) {
        return _mm_xor_si128(input, one);
    }
    return input;
}
}  // namespace

ExpansionCost countExpansion(int depth) {
    if (depth <= 0) {
        throw std::invalid_argument("countExpansion: depth must be > 0");
    }

    ExpansionCost cost;
    cost.depth = depth;

    // Half-Tree cGGM: 1 hash per internal node (right child = left ^ parent).
    // Standard GGM: 2 hashes per internal node (independent PRG for L and R).
    // Count nodes at internal levels 1..depth-1.
    uint64_t internal_nodes = 0;
    for (int level = 1; level < depth; level++) {
        internal_nodes += (1ULL << (level - 1));
    }

    cost.half_tree_hashes = internal_nodes;  // 1 hash / node
    cost.standard_ggm_hashes = internal_nodes * 2;  // 2 hashes / node

    return cost;
}

CggmEval cggmEvalPrefixCounted(const CggmKey& key,
                               const std::vector<osuCrypto::u8>& xhat_bits,
                               uint64_t* hash_count) {
    const int depth = key.depth;
    if (static_cast<int>(xhat_bits.size()) != depth) {
        throw std::invalid_argument("cggmEvalPrefixCounted: size mismatch");
    }

    uint64_t hashes = 0;
    osuCrypto::block levelNode = key.root;
    osuCrypto::u8 t = blockLsb(levelNode);
    osuCrypto::u8 L = 0;

    // Internal levels: 1 hash per level.
    for (int i = 0; i < depth - 1; i++) {
        const osuCrypto::u8 dir = xhat_bits[i] & 1;
        const osuCrypto::block H = dfss::internal::ccrHash(levelNode);
        hashes++;
        const osuCrypto::block corr = t ? key.internal_cw[i] : osuCrypto::ZeroBlock;
        const osuCrypto::block left = H ^ corr;
        const osuCrypto::block right = H ^ levelNode ^ corr;
        if (dir == 0) {
            L ^= blockLsb(right);
            levelNode = left;
        } else {
            levelNode = right;
        }
        t = blockLsb(levelNode);
    }

    // Leaf level: 2 hashes (q0, q1).
    const osuCrypto::u8 finalDir = xhat_bits[depth - 1] & 1;
    const osuCrypto::block q0 = dfss::internal::ccrHash(levelNode);
    const osuCrypto::block q1 = dfss::internal::ccrHash(levelNode ^ osuCrypto::OneBlock);
    hashes += 2;

    const osuCrypto::block leafCwL = setBlockLsb(key.hcw, key.lcw[0]);
    const osuCrypto::block leafCwR = setBlockLsb(key.hcw, key.lcw[1]);
    const osuCrypto::block leftLeaf = q0 ^ (t ? leafCwL : osuCrypto::ZeroBlock);
    const osuCrypto::block rightLeaf = q1 ^ (t ? leafCwR : osuCrypto::ZeroBlock);

    osuCrypto::u8 e;
    if (finalDir == 0) {
        L ^= blockLsb(rightLeaf);
        e = blockLsb(leftLeaf);
    } else {
        e = blockLsb(rightLeaf);
    }

    if (hash_count) *hash_count += hashes;
    return {L, e};
}

CggmEval cggmEvalFullDomain(const CggmKey& key,
                            const std::vector<osuCrypto::u8>& xhat_bits,
                            uint64_t* hash_count) {
    const int depth = key.depth;
    if (static_cast<int>(xhat_bits.size()) != depth) {
        throw std::invalid_argument("cggmEvalFullDomain: size mismatch");
    }

    // Reconstruct xhat as an index in [0, 2^depth).
    uint64_t xhat = 0;
    for (int i = 0; i < depth; i++) {
        xhat = (xhat << 1) | (xhat_bits[i] & 1);
    }

    uint64_t hashes = 0;

    // Expand the full internal tree to get all 2^(depth-1) penultimate nodes.
    std::vector<osuCrypto::block> level_nodes(1, key.root);
    for (int lev = 0; lev < depth - 1; lev++) {
        std::vector<osuCrypto::block> next;
        next.reserve(level_nodes.size() * 2);
        for (const auto& nd : level_nodes) {
            const osuCrypto::u8 t = blockLsb(nd);
            const osuCrypto::block H = dfss::internal::ccrHash(nd);
            hashes++;
            const osuCrypto::block corr = t ? key.internal_cw[lev] : osuCrypto::ZeroBlock;
            const osuCrypto::block left = H ^ corr;
            const osuCrypto::block right = H ^ nd ^ corr;
            next.push_back(left);
            next.push_back(right);
        }
        level_nodes = std::move(next);
    }

    // Expand all 2^depth leaves and accumulate parities.
    const uint64_t N = 1ULL << depth;
    osuCrypto::u8 L = 0;
    osuCrypto::u8 e = 0;

    for (uint64_t leaf_idx = 0; leaf_idx < N; leaf_idx++) {
        const uint64_t parent_idx = leaf_idx >> 1;
        const osuCrypto::u8 sigma = static_cast<osuCrypto::u8>(leaf_idx & 1);
        const osuCrypto::block parent = level_nodes[parent_idx];
        const osuCrypto::u8 t = blockLsb(parent);

        const osuCrypto::block q = dfss::internal::ccrHash(
            sigma ? (parent ^ osuCrypto::OneBlock) : parent);
        hashes++;

        const osuCrypto::block leafCw = setBlockLsb(key.hcw, key.lcw[sigma]);
        const osuCrypto::block leaf = q ^ (t ? leafCw : osuCrypto::ZeroBlock);
        const osuCrypto::u8 parity = blockLsb(leaf);

        // L = XOR of parities for all indices > xhat.
        if (leaf_idx > xhat) {
            L ^= parity;
        }
        // e = parity at xhat.
        if (leaf_idx == xhat) {
            e = parity;
        }
    }

    if (hash_count) *hash_count += hashes;
    return {L, e};
}

}  // namespace segfss
