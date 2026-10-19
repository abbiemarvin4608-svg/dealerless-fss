#pragma once

#include <vector>

#include "commons/group_element.h"
#include "commons/keypack.h"
#include "mpc/comms.h"
#include "cggm.h"

namespace segfss {

struct ExactTruncationKey {
    int input_bits = 0;
    int output_bits = 0;
    int shift = 0;
    GroupElement input_mask_share;
    GroupElement mask_high_share;
    GroupElement comparison_mask_arith_share;
    u8 comparison_mask_bool_share = 0;
    DPFKeyPack comparison_key;
    CggmKey context_comparison_key;
    bool use_context_comparison = false;
    bool consumed = false;
};

struct ExactMulTruncationKey {
    int input_bits = 0;
    int shift = 0;
    GroupElement triple_a;
    GroupElement triple_b;
    GroupElement triple_c;
    ExactTruncationKey truncation;
    bool consumed = false;
};

ExactTruncationKey exactTruncateOffline(int party_id, int input_bits,
                                        int shift, Peer* player);
std::vector<ExactTruncationKey> exactTruncateBatchOffline(
    int party_id, int count, int input_bits, int shift, Peer* player);
GroupElement exactTruncate(int party_id, GroupElement input,
                           ExactTruncationKey& key, Peer* player);
// Open all masked inputs, then all masked carry bits, in two batch exchanges.
std::vector<GroupElement> exactTruncateBatch(
    int party_id, const std::vector<GroupElement>& inputs,
    const std::vector<ExactTruncationKey*>& keys, Peer* player);

ExactMulTruncationKey exactMulTruncateOffline(int party_id, int input_bits,
                                              int shift, Peer* player);
GroupElement exactMulTruncate(int party_id, GroupElement lhs,
                              GroupElement rhs, ExactMulTruncationKey& key,
                              Peer* player);

void freeExactTruncationKey(ExactTruncationKey& key);
void freeExactMulTruncationKey(ExactMulTruncationKey& key);

}  // namespace segfss
