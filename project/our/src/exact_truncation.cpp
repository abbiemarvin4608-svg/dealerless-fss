#include "exact_truncation.h"

#include <stdexcept>
#include <limits>
#include <chrono>
#include <unordered_set>
#include <vector>

#include "fss/dpf.h"
#include "mpc/api.h"
#include "mpc/secure_ops.h"
#include "gen.h"

namespace segfss {
namespace {

void validateParameters(int party_id, int input_bits, int shift,
                        Peer* player) {
    if (party_id != SERVER && party_id != CLIENT) {
        throw std::invalid_argument("exact truncation requires SERVER or CLIENT");
    }
    if (player == nullptr) {
        throw std::invalid_argument(
            "exact truncation requires a peer");
    }
    if (input_bits < 2 || input_bits > 64 || shift <= 0 ||
        shift >= input_bits || shift > 24) {
        throw std::invalid_argument(
            "exact truncation requires 2 <= bits <= 64 and 0 < shift < bits with shift <= 24");
    }
}

u8 evaluateComparison(int party_id, const GroupElement& masked,
                      const ExactTruncationKey& key) {
    if (!key.use_context_comparison)
        return dfss::evalCorrelatedDCFBit(party_id, masked, key.comparison_key);
    std::vector<u8> bits(key.shift);
    for (int i = 0; i < key.shift; ++i)
        bits[i] = (masked.value >> (key.shift - 1 - i)) & 1U;
    return cggmEval(key.context_comparison_key, bits).L;
}

void openCarryBits(u8* values, int count, Peer* player) {
    const auto started = std::chrono::steady_clock::now();
    player->open_xor(values, count);
    const auto elapsed = std::chrono::duration_cast<std::chrono::microseconds>(
        std::chrono::steady_clock::now() - started).count();
    player->opening_microseconds += elapsed;
    if (player == peer) {
        ++numRounds;
        reconstructMicroseconds += elapsed;
    }
}

}  // namespace

std::vector<ExactTruncationKey> exactTruncateBatchOffline(
    int party_id, int count, int input_bits, int shift, Peer* player) {
    validateParameters(party_id, input_bits, shift, player);
    if (count <= 0 || count > std::numeric_limits<int>::max() / (shift + 1))
        throw std::invalid_argument("exact truncation: invalid batch size");
    auto& rng = player->randomSource();
    std::vector<u8> low_bits(static_cast<std::size_t>(count) * (shift + 1));
    for (auto& bit : low_bits) bit = rng.get<u8>() & 1U;
    std::vector<GroupElement> arithmetic(low_bits.size(), GroupElement(0, input_bits));
    B2A(party_id, low_bits.data(), arithmetic.data(), arithmetic.size(), input_bits, player);
    std::vector<std::vector<u8>> targets(count, std::vector<u8>(shift));
    std::vector<ExactTruncationKey> keys(count);
    for (int i = 0; i < count; ++i) {
        auto& key = keys[i];
        const auto offset = static_cast<std::size_t>(i) * (shift + 1);
        key.input_bits = input_bits;
        key.output_bits = input_bits - shift;
        key.shift = shift;
        key.mask_high_share = random_ge_from_prng(rng, key.output_bits);
        key.input_mask_share = GroupElement(key.mask_high_share.value << shift, input_bits);
        for (int bit = 0; bit < shift; ++bit) {
            key.input_mask_share = key.input_mask_share + arithmetic[offset + bit] *
                                  (uint64_t(1) << bit);
            targets[i][bit] = low_bits[offset + shift - 1 - bit];
        }
        key.comparison_mask_bool_share = low_bits[offset + shift];
        key.comparison_mask_arith_share = GroupElement(arithmetic[offset + shift].value,
                                                      key.output_bits);
        key.use_context_comparison = true;
    }
    auto comparisons = cggmGen2pcBatch(party_id, targets, nullptr, nullptr,
                                      player, cggmThreadLimit());
    for (int i = 0; i < count; ++i) keys[i].context_comparison_key = std::move(comparisons[i]);
    return keys;
}

ExactTruncationKey exactTruncateOffline(int party_id, int input_bits,
                                        int shift, Peer* player) {
    validateParameters(party_id, input_bits, shift, player);

    ExactTruncationKey output;
    output.input_bits = input_bits;
    output.output_bits = input_bits - shift;
    output.shift = shift;

    auto& rng = player->randomSource();
    std::vector<u8> low_bits(shift + 1);
    std::vector<GroupElement> arithmetic_bits(
        shift + 1, GroupElement(0, input_bits));
    for (u8& bit : low_bits) {
        bit = static_cast<u8>(rng.get<u8>() & 1U);
    }
    B2A(party_id, low_bits.data(), arithmetic_bits.data(),
        shift + 1, input_bits, player);

    output.mask_high_share = random_ge_from_prng(rng, output.output_bits);
    output.input_mask_share = GroupElement(
        output.mask_high_share.value << shift, input_bits);
    for (int i = 0; i < shift; ++i) {
        output.input_mask_share = output.input_mask_share +
            arithmetic_bits[i] * (uint64_t(1) << i);
    }

    output.comparison_mask_bool_share = low_bits[shift];
    output.comparison_mask_arith_share = GroupElement(
        arithmetic_bits[shift].value, output.output_bits);

    std::vector<u8> low_bits_msb(shift);
    for (int i = 0; i < shift; ++i) {
        low_bits_msb[i] = low_bits[shift - 1 - i];
    }
    if (player == peer) {
        output.comparison_key = dfss::keyGenCorrelatedDPFBit(
            party_id, shift, low_bits_msb.data());
    } else {
        output.context_comparison_key = cggmGen2pcBatch(
            party_id, {low_bits_msb}, nullptr, nullptr, player, 1).front();
        output.use_context_comparison = true;
    }
    return output;
}

GroupElement exactTruncate(int party_id, GroupElement input,
                           ExactTruncationKey& key, Peer* player) {
    if ((party_id != SERVER && party_id != CLIENT) || player == nullptr) {
        throw std::invalid_argument(
            "exactTruncate requires a valid peer");
    }
    if (key.consumed) {
        throw std::logic_error("exact truncation preprocessing was reused");
    }
    if (input.bitsize != key.input_bits || key.shift <= 0 ||
        key.shift >= key.input_bits ||
        key.output_bits != key.input_bits - key.shift) {
        throw std::invalid_argument("exact truncation input/key mismatch");
    }

    key.consumed = true;
    GroupElement masked = input + key.input_mask_share;
    player->open_additive(&masked, 1, masked.bitsize);

    const uint64_t low_mask = (uint64_t(1) << key.shift) - 1;
    u8 masked_comparison = evaluateComparison(
        party_id, GroupElement(masked.value & low_mask, key.shift),
        key);
    masked_comparison ^= key.comparison_mask_bool_share;
    openCarryBits(&masked_comparison, 1, player);

    const uint64_t opened_bit = masked_comparison & 1U;
    const GroupElement comparison_mask(
        key.comparison_mask_arith_share.value, key.output_bits);
    const GroupElement carry =
        GroupElement((party_id - SERVER) * opened_bit, key.output_bits) +
        comparison_mask - comparison_mask * (2 * opened_bit);
    const GroupElement masked_high(masked.value >> key.shift,
                                   key.output_bits);
    return masked_high * static_cast<uint64_t>(party_id - SERVER) -
           key.mask_high_share - carry;
}

std::vector<GroupElement> exactTruncateBatch(
    int party_id, const std::vector<GroupElement>& inputs,
    const std::vector<ExactTruncationKey*>& keys, Peer* player) {
    if ((party_id != SERVER && party_id != CLIENT) || player == nullptr ||
        inputs.size() != keys.size()) {
        throw std::invalid_argument("exactTruncateBatch invalid context or shape");
    }
    if (inputs.empty()) return {};
    const int input_bits = inputs.front().bitsize;
    const int shift = keys.front() ? keys.front()->shift : 0;
    validateParameters(party_id, input_bits, shift, player);
    std::unordered_set<ExactTruncationKey*> seen;
    for (size_t i = 0; i < keys.size(); ++i) {
        const auto* key = keys[i];
        if (key == nullptr || inputs[i].bitsize != input_bits ||
            key->input_bits != input_bits || key->shift != shift ||
            key->output_bits != input_bits - shift) {
            throw std::invalid_argument("exactTruncateBatch input/key mismatch");
        }
        if (key->consumed) {
            throw std::logic_error("exact truncation preprocessing was reused");
        }
        if (!seen.insert(keys[i]).second) {
            throw std::invalid_argument("exactTruncateBatch duplicate key");
        }
    }

    std::vector<GroupElement> masked;
    masked.reserve(inputs.size());
    for (size_t i = 0; i < inputs.size(); ++i) {
        keys[i]->consumed = true;
        masked.push_back(inputs[i] + keys[i]->input_mask_share);
    }
    player->open_additive(masked.data(), static_cast<int>(masked.size()), input_bits);
    const uint64_t low_mask = (uint64_t(1) << shift) - 1;
    std::vector<u8> comparison(masked.size());
    for (size_t i = 0; i < masked.size(); ++i) {
        comparison[i] = evaluateComparison(
            party_id, GroupElement(masked[i].value & low_mask, shift),
            *keys[i]) ^ keys[i]->comparison_mask_bool_share;
    }
    openCarryBits(comparison.data(), static_cast<int>(comparison.size()), player);

    const int output_bits = input_bits - shift;
    std::vector<GroupElement> outputs;
    outputs.reserve(inputs.size());
    for (size_t i = 0; i < inputs.size(); ++i) {
        const uint64_t opened_bit = comparison[i] & 1U;
        const GroupElement comparison_mask(
            keys[i]->comparison_mask_arith_share.value, output_bits);
        const GroupElement carry =
            GroupElement((party_id - SERVER) * opened_bit, output_bits) +
            comparison_mask - comparison_mask * (2 * opened_bit);
        outputs.push_back(
            GroupElement(masked[i].value >> shift, output_bits) *
                static_cast<uint64_t>(party_id - SERVER) -
            keys[i]->mask_high_share - carry);
    }
    return outputs;
}

ExactMulTruncationKey exactMulTruncateOffline(int party_id, int input_bits,
                                              int shift, Peer* player) {
    validateParameters(party_id, input_bits, shift, player);

    ExactMulTruncationKey key;
    key.input_bits = input_bits;
    key.shift = shift;
    key.triple_a = GroupElement(0, input_bits);
    key.triple_b = GroupElement(0, input_bits);
    key.triple_c = GroupElement(0, input_bits);
    beaver_mult_offline(party_id, &key.triple_a, &key.triple_b,
                        &key.triple_c, player, 1);
    key.truncation = exactTruncateOffline(
        party_id, input_bits, shift, player);
    return key;
}

GroupElement exactMulTruncate(int party_id, GroupElement lhs,
                              GroupElement rhs, ExactMulTruncationKey& key,
                              Peer* player) {
    if ((party_id != SERVER && party_id != CLIENT) || player == nullptr) {
        throw std::invalid_argument(
            "exactMulTruncate requires a valid peer");
    }
    if (key.consumed) {
        throw std::logic_error("exact MulTr preprocessing was reused");
    }
    if (lhs.bitsize != key.input_bits || rhs.bitsize != key.input_bits ||
        key.shift <= 0 || key.shift >= key.input_bits) {
        throw std::invalid_argument("exact MulTr input/key mismatch");
    }

    GroupElement masked_inputs[2] = {
        lhs - key.triple_a,
        rhs - key.triple_b,
    };
    player->open_additive(masked_inputs, 2, key.input_bits);
    GroupElement product =
        key.triple_c + key.triple_b * masked_inputs[0] +
        key.triple_a * masked_inputs[1] +
        masked_inputs[1] * masked_inputs[0] *
            static_cast<uint64_t>(party_id - SERVER);
    GroupElement output = exactTruncate(
        party_id, product, key.truncation, player);
    key.consumed = true;
    return output;
}

void freeExactTruncationKey(ExactTruncationKey& key) {
    freeDPFKeyPack(key.comparison_key);
    key = ExactTruncationKey();
}

void freeExactMulTruncationKey(ExactMulTruncationKey& key) {
    freeExactTruncationKey(key.truncation);
    key = ExactMulTruncationKey();
}

}  // namespace segfss
