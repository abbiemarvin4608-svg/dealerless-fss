#pragma once

#include <vector>

#include "commons/group_element.h"
#include "commons/types.h"
#include "mpc/comms.h"

namespace segfss {

// One-time preprocessing for the BitXA protocol in Algorithm 1 of the
// supplied construction.  delta_y is held both as a Boolean XOR share and as
// an arithmetic share; delta_z is an arithmetic share of delta_x * delta_y.
// A mask must never be reused after an online call.
struct BitXAPreprocessing {
    int bitwidth = 0;
    GroupElement delta_x;
    u8 delta_y_bool = 0;
    GroupElement delta_y_arith;
    GroupElement delta_z;
    bool consumed = false;
};

// Offline phase.  The two multiplications used to convert the random Boolean
// mask and to form delta_z occur entirely here.
std::vector<BitXAPreprocessing> bitxaOffline(
    int party_id, int size, int bitwidth, Peer* player);
BitXAPreprocessing bitxaOffline(int party_id, int bitwidth, Peer* player);

// Online phase.  x is additively shared in Z_{2^bitwidth}; y is Boolean
// XOR-shared.  All (x + delta_x, y XOR delta_y) values are opened in one
// combined message round, after which z = x*y is computed locally.
std::vector<GroupElement> bitxaOnline(
    int party_id, const std::vector<GroupElement>& x,
    const std::vector<u8>& y, std::vector<BitXAPreprocessing>& preprocessing,
    Peer* player);
GroupElement bitxaOnline(int party_id, GroupElement x, u8 y,
                         BitXAPreprocessing& preprocessing, Peer* player);

// ReLU specialization: drelu must be a Boolean share of 1{x >= 0}.  These
// aliases make the offline/online split explicit at the neural-network layer
// while using BitXA directly, without an online B2A followed by Beaver Mul.
using ReluPreprocessing = BitXAPreprocessing;

std::vector<ReluPreprocessing> reluOffline(
    int party_id, int size, int bitwidth, Peer* player);
ReluPreprocessing reluOffline(int party_id, int bitwidth, Peer* player);

std::vector<GroupElement> reluOnline(
    int party_id, const std::vector<GroupElement>& x,
    const std::vector<u8>& drelu,
    std::vector<ReluPreprocessing>& preprocessing, Peer* player);
GroupElement reluOnline(int party_id, GroupElement x, u8 drelu,
                        ReluPreprocessing& preprocessing, Peer* player);

}  // namespace segfss
