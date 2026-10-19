#pragma once

#include <cstddef>
#include <vector>

#include "commons/group_element.h"
#include "mpc/comms.h"

namespace segfss {

// Matrix Beaver triple over Z_(2^bitwidth), stored as additive shares.
// Matrices use row-major layout: U[rows,inner], V[inner,cols], W[rows,cols].
struct MatmulPreprocessing {
    int rows = 0;
    int inner = 0;
    int cols = 0;
    int bitwidth = 0;
    std::vector<GroupElement> u;
    std::vector<GroupElement> v;
    std::vector<GroupElement> w;
    std::size_t cot_batches = 0;
    bool consumed = false;
};

MatmulPreprocessing matmulOffline(int party_id, int rows, int inner, int cols,
                                  int bitwidth, Peer* player);

// Opens the two masked input matrices together, then computes the output
// shares locally. Each preprocessing object may be consumed once.
std::vector<GroupElement> matmulOnline(
    int party_id, const std::vector<GroupElement>& lhs,
    const std::vector<GroupElement>& rhs, MatmulPreprocessing& preprocessing,
    Peer* player);

}  // namespace segfss
