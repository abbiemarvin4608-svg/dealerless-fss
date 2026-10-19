#include "linear.h"

#include <algorithm>
#include <cstdint>
#include <limits>
#include <stdexcept>

#include "mpc/api.h"
#include "mpc/secure_ops.h"
#include "runtime_options.h"
#include "ring_packing.h"

namespace segfss {
namespace {

std::size_t checkedProduct(std::size_t lhs, std::size_t rhs) {
    if (lhs > std::numeric_limits<std::size_t>::max() / rhs) {
        throw std::invalid_argument("matmul dimensions overflow");
    }
    return lhs * rhs;
}

void validateContext(int party_id, Peer* player) {
    if (party_id != SERVER && party_id != CLIENT) {
        throw std::invalid_argument("matmul requires SERVER or CLIENT");
    }
    if (player == nullptr || player != peer) {
        throw std::invalid_argument("matmul requires the active peer");
    }
}

// The generic reconstruct path stores one, two, four, or eight bytes per
// ring element.  Beaver E/F values have a public common width, so they can be
// serialized bit densely without changing the opened residues.  Keep the
// legacy path available for matched baseline runs and make this an explicit
// opt-in runtime mode.
void openPackedRing(int party_id, std::vector<GroupElement>& values,
                    int bitwidth, Peer* player) {
    std::vector<uint64_t> local;
    local.reserve(values.size());
    for (const auto& value : values) {
        if (value.bitsize != bitwidth)
            throw std::invalid_argument("packed matmul opening width mismatch");
        local.push_back(value.value);
    }
    const auto remote = exchangeRingValues(party_id == SERVER, local, bitwidth,
        [&](const u8* data, int size) { player->send_u8(data, size); },
        [&](u8* data, int size) { player->recv_u8(data, size); });
    const uint64_t mask = reference::mask(bitwidth);
    for (std::size_t i = 0; i < values.size(); ++i)
        values[i].value = (values[i].value + remote[i]) & mask;
    ++numRounds;
}

}  // namespace

MatmulPreprocessing matmulOffline(int party_id, int rows, int inner, int cols,
                                  int bitwidth, Peer* player) {
    validateContext(party_id, player);
    if (rows <= 0 || inner <= 0 || cols <= 0 || bitwidth < 1 ||
        bitwidth > 64) {
        throw std::invalid_argument("matmul dimensions or bitwidth invalid");
    }
    const auto u_size = checkedProduct(rows, inner);
    const auto v_size = checkedProduct(inner, cols);
    const auto w_size = checkedProduct(rows, cols);
    const auto products = checkedProduct(w_size, inner);
    if (products > static_cast<std::size_t>(std::numeric_limits<int>::max() /
                                            (2 * bitwidth))) {
        throw std::invalid_argument("matmul OT batch size exceeds int range");
    }

    MatmulPreprocessing prep;
    prep.rows = rows;
    prep.inner = inner;
    prep.cols = cols;
    prep.bitwidth = bitwidth;
    prep.u.resize(u_size, GroupElement(0, bitwidth));
    prep.v.resize(v_size, GroupElement(0, bitwidth));
    prep.w.resize(w_size, GroupElement(0, bitwidth));

    auto rng = secure_prng();
    for (auto& value : prep.u) {
        value = random_ge_from_prng(rng, bitwidth);
    }
    for (auto& value : prep.v) {
        value = random_ge_from_prng(rng, bitwidth);
    }

    // The two cross products U_0 V_1 and U_1 V_0 use one batched COT.
    // Bound temporary storage for large layers while batching each tile.
    constexpr std::size_t kProductsPerBatch = 4096;
    std::vector<GroupElement> ot_inputs;
    std::vector<GroupElement> cross_shares;
    const std::size_t max_batch = std::min(kProductsPerBatch, products);
    ot_inputs.reserve(2 * max_batch);
    cross_shares.reserve(2 * max_batch);
    for (std::size_t base = 0; base < products; base += kProductsPerBatch) {
        const std::size_t count = std::min(kProductsPerBatch, products - base);
        ++prep.cot_batches;
        ot_inputs.assign(2 * count, GroupElement(0, bitwidth));
        cross_shares.assign(2 * count, GroupElement(0, bitwidth));
        for (std::size_t offset = 0; offset < count; ++offset) {
            const std::size_t flat = base + offset;
            const std::size_t output_index = flat / inner;
            const std::size_t k = flat % inner;
            const std::size_t i = output_index / cols;
            const std::size_t j = output_index % cols;
            const GroupElement& u = prep.u[i * inner + k];
            const GroupElement& v = prep.v[k * cols + j];
            prep.w[output_index] = prep.w[output_index] + u * v;
            ot_inputs[offset] = party_id == SERVER ? u : v;
            ot_inputs[count + offset] = party_id == SERVER ? v : u;
        }
        cross_term_gen(party_id, ot_inputs.data(), cross_shares.data(),
                       party_id == SERVER, static_cast<int>(2 * count),
                       player);
        for (std::size_t offset = 0; offset < count; ++offset) {
            const std::size_t output_index = (base + offset) / inner;
            prep.w[output_index] = prep.w[output_index] +
                cross_shares[offset] + cross_shares[count + offset];
        }
    }
    return prep;
}

std::vector<GroupElement> matmulOnline(
    int party_id, const std::vector<GroupElement>& lhs,
    const std::vector<GroupElement>& rhs, MatmulPreprocessing& preprocessing,
    Peer* player) {
    validateContext(party_id, player);
    if (preprocessing.consumed) {
        throw std::logic_error("matmul preprocessing already consumed");
    }
    const int rows = preprocessing.rows;
    const int inner = preprocessing.inner;
    const int cols = preprocessing.cols;
    const int bitwidth = preprocessing.bitwidth;
    if (rows <= 0 || inner <= 0 || cols <= 0 || bitwidth < 1 ||
        bitwidth > 64 || lhs.size() != preprocessing.u.size() ||
        rhs.size() != preprocessing.v.size() ||
        preprocessing.w.size() != static_cast<std::size_t>(rows) * cols) {
        throw std::invalid_argument("matmul input shape mismatch");
    }
    for (const auto& value : lhs) {
        if (value.bitsize != bitwidth) {
            throw std::invalid_argument("matmul lhs bitwidth mismatch");
        }
    }
    for (const auto& value : rhs) {
        if (value.bitsize != bitwidth) {
            throw std::invalid_argument("matmul rhs bitwidth mismatch");
        }
    }

    const std::size_t u_size = lhs.size();
    std::vector<GroupElement> opened(u_size + rhs.size(),
                                      GroupElement(0, bitwidth));
    for (std::size_t i = 0; i < u_size; ++i) {
        opened[i] = lhs[i] - preprocessing.u[i];
    }
    for (std::size_t i = 0; i < rhs.size(); ++i) {
        opened[u_size + i] = rhs[i] - preprocessing.v[i];
    }
    preprocessing.consumed = true;
    if (runtimeFlag("SEGFSS_PACK_RING")) {
        openPackedRing(party_id, opened, bitwidth, player);
    } else {
        reconstruct(static_cast<int32_t>(opened.size()), opened.data(), bitwidth);
    }

    std::vector<GroupElement> result = preprocessing.w;
    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            GroupElement& sum = result[i * cols + j];
            for (int k = 0; k < inner; ++k) {
                const GroupElement& e = opened[i * inner + k];
                const GroupElement& f = opened[u_size + k * cols + j];
                sum = sum + e * preprocessing.v[k * cols + j] +
                    preprocessing.u[i * inner + k] * f;
                if (party_id == CLIENT) {
                    sum = sum + e * f;
                }
            }
        }
    }
    return result;
}

}  // namespace segfss
