#include "activation.h"

#include <chrono>
#include <stdexcept>
#include <utility>

#include "mpc/api.h"

namespace segfss {
namespace {

std::vector<u8> signBitsOnline(
    int party_id, const std::vector<GroupElement>& inputs,
    std::vector<SigmaActivationKey>& keys,
    BooleanAndPreprocessing SigmaActivationKey::* field, Peer* player,
    BooleanAndWorkspace* workspace, int bits) {
    const size_t count = inputs.size();
    std::vector<u8> carry(count, 0), lhs(2 * count), rhs(2 * count),
                    product(2 * count);
    BooleanAndPreprocessing round_material;
    round_material.a.resize(2 * count);
    round_material.b.resize(2 * count);
    round_material.c.resize(2 * count);
    // Add the two arithmetic shares in Boolean form. The top-bit sum is
    // the comparison result; only lower-bit carries need secure AND gates.
    for (int bit = 0; bit < bits - 1; ++bit) {
        for (size_t i = 0; i < count; ++i) {
            const u8 local = static_cast<u8>((inputs[i].value >> bit) & 1U);
            lhs[2 * i] = party_id == SERVER ? local : 0;
            rhs[2 * i] = party_id == CLIENT ? local : 0;
            lhs[2 * i + 1] = local;
            rhs[2 * i + 1] = carry[i];
            const auto& material = keys[i].*field;
            for (size_t gate = 0; gate < 2; ++gate) {
                round_material.a[2 * i + gate] = material.a[material.cursor + gate];
                round_material.b[2 * i + gate] = material.b[material.cursor + gate];
                round_material.c[2 * i + gate] = material.c[material.cursor + gate];
            }
        }
        round_material.cursor = 0;
        booleanAndOnlineBatch(party_id,
            {{lhs.data(), rhs.data(), product.data(),
              static_cast<int>(2 * count), &round_material}}, player, workspace);
        for (size_t i = 0; i < count; ++i) {
            carry[i] = product[2 * i] ^ product[2 * i + 1];
            (keys[i].*field).cursor += 2;
        }
    }
    for (size_t i = 0; i < count; ++i) {
        carry[i] ^= static_cast<u8>((inputs[i].value >> (bits - 1)) & 1U);
    }
    return carry;
}

std::vector<GroupElement> mixedProductsOnline(
    int party_id, const std::vector<GroupElement>& inputs,
    const std::vector<u8>& bits, std::vector<SigmaActivationKey>& keys,
    BitXAPreprocessing SigmaActivationKey::* field, Peer* player) {
    std::vector<BitXAPreprocessing> materials;
    materials.reserve(keys.size());
    for (auto& key : keys) materials.push_back(key.*field);
    auto result = bitxaOnline(party_id, inputs, bits, materials, player);
    for (size_t i = 0; i < keys.size(); ++i) keys[i].*field = std::move(materials[i]);
    return result;
}

std::vector<GroupElement> lookupOnline(
    const std::vector<GroupElement>& indices,
    std::vector<SigmaActivationKey>& keys, Peer* player) {
    const auto started = std::chrono::steady_clock::now();
    const int q = keys.front().q;
    std::vector<GroupElement> masked;
    masked.reserve(indices.size());
    for (size_t i = 0; i < indices.size(); ++i) {
        auto& material = keys[i].segmented_lut;
        material.consumed = true;
        masked.push_back(indices[i] + material.mask_share);
    }
    player->open_additive(masked.data(), static_cast<int>(masked.size()), q);
    std::vector<GroupElement> outputs;
    outputs.reserve(indices.size());
    const uint64_t mask = (uint64_t(1) << q) - 1;
    for (size_t i = 0; i < indices.size(); ++i) {
        outputs.push_back(keys[i].segmented_lut.output_share_table[masked[i].value & mask]);
    }
    const double elapsed = std::chrono::duration<double, std::milli>(
        std::chrono::steady_clock::now() - started).count();
    for (auto& key : keys) key.segmented_lut.online_ms = elapsed / keys.size();
    return outputs;
}

std::vector<GroupElement> dpfLookupOnline(
    int party_id, const std::vector<GroupElement>& indices,
    std::vector<SigmaActivationKey>& keys, Peer* player,
    DpfLutBatchMetrics* metrics, const DpfLutBinding* binding) {
    std::vector<DpfPublicLutMaterial> materials;
    materials.reserve(keys.size());
    for (auto& key : keys) materials.push_back(std::move(key.dpf_lut));
    try {
        auto result = keys.front().prepared_delta_table
            ? dpfPublicLutBatchOnline(party_id, indices, materials,
                binding ? *binding : materials.front().binding, player,
                *keys.front().prepared_delta_table, metrics)
            : dpfPublicLutBatchOnlineMany(party_id, indices, materials, player,
                materials.front().request_id, {keys.front().public_delta_table.get()},
                metrics).front();
        for (size_t i = 0; i < keys.size(); ++i)
            keys[i].dpf_lut = std::move(materials[i]);
        return result;
    } catch (...) {
        for (size_t i = 0; i < keys.size(); ++i)
            keys[i].dpf_lut = std::move(materials[i]);
        throw;
    }
}

void validateBatch(int party_id, const std::vector<GroupElement>& inputs,
                   const std::vector<SigmaActivationKey>& keys, Peer* player,
                   const DpfLutBinding* binding = nullptr, const ActivationContext* context = nullptr,
                   const ActivationInputSignShares* input_signs = nullptr) {
    if (inputs.empty() || inputs.size() != keys.size() ||
        (party_id != SERVER && party_id != CLIENT) ||
        player == nullptr) {
        throw std::invalid_argument("sigmaBatchOnline: invalid batch or peer");
    }
    const auto& reference = keys.front();
    const bool segmented_sign = reference.sign_backend == ActivationSignBackend::segmented_fss;
    if (reference.sign_backend != ActivationSignBackend::bitdec && !segmented_sign)
        throw std::invalid_argument("activation sign backend is invalid");
    if (context && (context->sign_backend != reference.sign_backend ||
        context->native_signed16_input != reference.native_signed16_input ||
        context->reuse_input_sign != reference.reuse_input_sign ||
        (context->clip_bits ? context->clip_bits : reference.bits) != reference.clip_bits))
        throw std::invalid_argument("activation comparison contract changed after preparation");
    if (reference.reuse_input_sign) {
        if (!context || !context->paper_lut || !input_signs || input_signs->consumed ||
            input_signs->owner_peer != player || input_signs->binding != context->lut_binding ||
            input_signs->generation_epoch != context->generation_epoch ||
            input_signs->shares.size() != inputs.size())
            throw std::invalid_argument("activation input sign owner or query mismatch");
        for (const auto bit : input_signs->shares)
            if (bit > 1) throw std::invalid_argument("activation input sign is not a Boolean share");
    } else if (input_signs) {
        throw std::invalid_argument("activation did not request reused input signs");
    }
    if (segmented_sign) {
        if (!context) throw std::invalid_argument("private segmented sign requires an owning activation context");
        validateActivationSignInventory(*context, inputs.size(), reference.bits);
    }
    const auto* expected = binding ? binding :
        (reference.prepared_delta_table ? &reference.dpf_lut.binding : nullptr);
    if (expected && inputs.size() - 1 > UINT64_MAX - expected->element_begin)
        throw std::invalid_argument("sigmaBatchOnline: invalid paper element range");
    for (size_t i = 0; i < keys.size(); ++i) {
        const auto& key = keys[i];
        const auto& material = key.segmented_lut;
        const bool valid_shape =
            key.owner_peer == player &&
            key.q > 0 && key.q <= 12 && key.bits >= 2 && key.bits <= 62 &&
            key.public_delta_table &&
            key.public_delta_table->Bin == key.q &&
            key.public_delta_table->Bout == key.bits &&
            key.public_delta_table->values.size() == (size_t(1) << key.q) &&
            bool(key.prepared_delta_table) == bool(reference.prepared_delta_table) &&
            (!key.prepared_delta_table || key.prepared_delta_table->fingerprint() ==
                                           reference.prepared_delta_table->fingerprint()) &&
            inputs[i].bitsize == key.bits &&
            key.bits == reference.bits && key.q == reference.q &&
            key.clip_bits == reference.clip_bits && key.clip_bits >= 2 &&
            (key.clip_bits == key.bits || (key.clip_bits == 22 && key.bits == 32 &&
                key.frac_bits == 12 && key.q == 10 && key.native_signed16_input)) &&
            key.native_signed16_input == reference.native_signed16_input &&
            key.reuse_input_sign == reference.reuse_input_sign &&
            (!key.reuse_input_sign || key.native_signed16_input) &&
            key.sign_backend == reference.sign_backend &&
            key.use_dpf_lut == reference.use_dpf_lut &&
            key.frac_bits == reference.frac_bits &&
            key.trunc_bits == reference.trunc_bits &&
            material.index_bits == key.q && material.output_bits == key.bits &&
            material.mask_share.bitsize == key.q &&
            ((!key.use_dpf_lut &&
              material.output_share_table.size() == (size_t(1) << key.q)) ||
             (key.use_dpf_lut && key.dpf_lut.index_bits == key.q &&
              key.dpf_lut.output_bits == key.bits &&
              key.dpf_lut.selection_share.size() == (size_t(1) << key.q)));
        if (!valid_shape) {
            throw std::invalid_argument("sigmaBatchOnline: heterogeneous or invalid input/key");
        }
        if (expected) {
            const auto& lut = key.dpf_lut;
            const auto& table = key.prepared_delta_table;
            if (!table || !key.use_dpf_lut || player->role() != party_id ||
                expected->context_id == 0 || expected->table_version != table->version() ||
                !lut.paper_lookup || lut.owner_peer != player || lut.binding != *expected ||
                lut.family != expected->family ||
                !lut.hasValidPaperLayout() ||
                (lut.family != DpfLutKeyFamily::full_q_arithmetic &&
                 lut.family != DpfLutKeyFamily::segmented_boolean_tensor_b2a) ||
                lut.request_id != expected->element_begin + i ||
                lut.table_fingerprint != table->fingerprint() ||
                lut.generation_epoch == 0 ||
                lut.generation_epoch != reference.dpf_lut.generation_epoch ||
                lut.mask_share.bitsize != key.q || lut.key_group_size != key.q ||
                lut.selection_share.bitsize() != key.bits)
                throw std::invalid_argument("sigmaBatchOnline: paper LUT inventory binding mismatch");
        } else if (key.dpf_lut.paper_lookup) {
            throw std::invalid_argument("sigmaBatchOnline: paper LUT requires its prepared table");
        }
        const bool valid_bitxa =
            key.relu_mul.bitwidth == key.bits &&
            key.clip_select_mul.bitwidth == key.bits &&
            key.branch_select_mul.bitwidth == key.bits &&
            key.relu_mul.delta_x.bitsize == key.bits &&
            key.relu_mul.delta_y_arith.bitsize == key.bits &&
            key.relu_mul.delta_z.bitsize == key.bits &&
            key.clip_select_mul.delta_x.bitsize == key.bits &&
            key.clip_select_mul.delta_y_arith.bitsize == key.bits &&
            key.clip_select_mul.delta_z.bitsize == key.bits &&
            key.branch_select_mul.delta_x.bitsize == key.bits &&
            key.branch_select_mul.delta_y_arith.bitsize == key.bits &&
            key.branch_select_mul.delta_z.bitsize == key.bits;
        if (!valid_bitxa) {
            throw std::invalid_argument("sigmaBatchOnline: invalid BitXA material shape at index " +
                                        std::to_string(i));
        }
        const auto valid_sign_stock = [&](const BooleanAndPreprocessing& stock, size_t expected) {
            return stock.cursor == 0 && stock.a.size() == expected &&
                   stock.b.size() == expected && stock.c.size() == expected;
        };
        if (key.use_piecewise_compare ||
            (!key.use_dpf_lut && material.consumed) ||
            (key.use_dpf_lut && (key.dpf_lut.consumed || key.dpf_lut.released)) ||
            key.relu_mul.consumed || key.clip_select_mul.consumed ||
            key.branch_select_mul.consumed ||
            !valid_sign_stock(key.sign_and, segmented_sign || key.reuse_input_sign ?
                0 : static_cast<size_t>(2) * (key.bits - 1)) ||
            !valid_sign_stock(key.clip_and, segmented_sign ?
                0 : static_cast<size_t>(2) * (key.clip_bits - 1)) ||
            (key.trunc_bits > 0 && key.trunc_key.consumed)) {
            throw std::logic_error("sigmaBatchOnline: invalid or consumed one-use material");
        }
    }
}

}  // namespace

void validateActivationSignInventory(const ActivationContext& context, int count, int bits) {
    const int clip_bits = context.clip_bits ? context.clip_bits : bits;
    const int clip_segments = context.clip_segments ? context.clip_segments : context.sign_segments;
    if (context.sign_backend == ActivationSignBackend::bitdec) {
        if (context.sign_segments != 0 || context.clip_segments != 0 ||
            context.sign_material.material.count || context.clip_material.material.count)
            throw std::invalid_argument("BitDec activation has unrequested private MSB inventory");
        return;
    }
    if (context.sign_backend != ActivationSignBackend::segmented_fss || !context.paper_lut ||
        context.lut_binding.context_id != context.identity || !context.generation_epoch ||
        context.materials.size() != size_t(count))
        throw std::invalid_argument("private activation sign owner or binding mismatch");
    for (const auto* stock : {&context.sign_material, &context.clip_material}) {
        if (stock == &context.sign_material && context.reuse_input_sign) {
            if (stock->material.count || stock->material.width || stock->material.nseg ||
                stock->material.owner_peer || !stock->material.keys.empty() ||
                !stock->material.mask_shares.empty() || !stock->material.mask_sign_shares.empty() ||
                stock->material.scan.size())
                throw std::invalid_argument("reused input sign has unrequested private MSB inventory");
            continue;
        }
        if (stock->binding != context.lut_binding || stock->generation_epoch != context.generation_epoch ||
            stock->clip != (stock == &context.clip_material))
            throw std::invalid_argument("private activation sign query or purpose mismatch");
        validatePrivateMsbMaterial(context.party_id, stock->material, count,
            stock->clip ? clip_bits : bits,
            stock->clip ? clip_segments : context.sign_segments, context.player);
    }
    for (const auto& key : context.materials)
        if (key.sign_backend != context.sign_backend || key.dpf_lut.binding != context.lut_binding ||
            key.dpf_lut.generation_epoch != context.generation_epoch ||
            key.clip_bits != clip_bits || key.reuse_input_sign != context.reuse_input_sign ||
            key.native_signed16_input != context.native_signed16_input)
            throw std::invalid_argument("private activation sign/LUT generation mismatch");
}

std::vector<u8> batchPiecewiseCompareOnline(
    int, const std::vector<GroupElement>&,
    const std::vector<PiecewiseComparisonKey>&, Peer*) {
    throw std::logic_error(
        "unmasked batch piecewise comparison is disabled; use the BitDec activation path");
}

static std::vector<GroupElement> sigmaBatchOnlineImpl(
    int party_id, const std::vector<GroupElement>& inputs,
    std::vector<SigmaActivationKey>& keys, Peer* player,
    BooleanAndWorkspace* workspace, DpfLutBatchMetrics* metrics = nullptr,
    const DpfLutBinding* binding = nullptr, ActivationMetrics* stages = nullptr,
    ActivationContext* context = nullptr, ActivationInputSignShares* input_signs = nullptr) {
    // Validation failures happen before the first network operation, but the
    // caller has still crossed the online boundary.  Fail closed for the
    // complete batch so a malformed entry cannot be repaired and retried with
    // the remaining resident material.  The same guard also covers protocol
    // exceptions after a masked opening has already been sent.
    auto invalidateBatch = [&] {
        if (input_signs) input_signs->consumed = true;
        for (auto& key : keys) {
            key.segmented_lut.consumed = true;
            key.dpf_lut.consumed = true;
        }
    };
    try {
        validateBatch(party_id, inputs, keys, player, binding, context, input_signs);
        if (input_signs) input_signs->consumed = true;
        // Once any masked message may leave the process, this entire activation
        // material must remain unusable even if a later protocol stage throws.
        for (auto& key : keys) {
            if (!key.use_dpf_lut) key.segmented_lut.consumed = true;
        }
        const int bits = keys.front().bits;
        const int q = keys.front().q;
        const int frac_bits = keys.front().frac_bits;
        const u8 public_one = static_cast<u8>(party_id == SERVER);
        std::vector<u8> nonnegative;
        {
            ActivationStageScope stage(stages, false, ActivationStage::sign, player);
            if (keys.front().reuse_input_sign) {
                nonnegative = input_signs->shares;
            } else if (keys.front().sign_backend == ActivationSignBackend::segmented_fss) {
                std::vector<uint64_t> shares;
                for (const auto& input : inputs) shares.push_back(input.value);
                nonnegative.resize(inputs.size());
                privateMsbOnline(party_id, shares.data(), nonnegative.data(), context->sign_material.material, player);
            } else nonnegative = signBitsOnline(
                party_id, inputs, keys, &SigmaActivationKey::sign_and, player, workspace, bits);
        }
        for (auto& value : nonnegative) value ^= public_one;
        std::vector<GroupElement> relu;
        {
            ActivationStageScope stage(stages, false, ActivationStage::abs_bitxa, player);
            relu = mixedProductsOnline(
                party_id, inputs, nonnegative, keys, &SigmaActivationKey::relu_mul, player);
        }
        const uint64_t bound = uint64_t(1) << (frac_bits + kSigmaBoundLog2);
        const GroupElement public_bound(party_id == SERVER ? bound : 0, bits);
        const GroupElement public_clipped_bound(party_id == SERVER ? bound - 1 : 0, bits);
        std::vector<GroupElement> magnitudes(inputs.size()), bound_inputs(inputs.size()),
                                  selection_inputs(inputs.size());
        for (size_t i = 0; i < inputs.size(); ++i) {
            magnitudes[i] = relu[i] * uint64_t(2) - inputs[i];
            // For the exact signed16/Q7 native adapter, |u|-2^16 lies in
            // [-2^16, 2^21-2^16] even for TanH's doubled minimum input.
            // Reduction to signed22 preserves this predicate; keep the
            // magnitude and selection arithmetic in the original ring32.
            bound_inputs[i] = GroupElement((magnitudes[i] - public_bound).value,
                keys.front().clip_bits);
            selection_inputs[i] = magnitudes[i] - public_clipped_bound;
        }
        std::vector<u8> below_bound;
        {
            ActivationStageScope stage(stages, false, ActivationStage::clip_compare, player);
            if (keys.front().sign_backend == ActivationSignBackend::segmented_fss) {
                std::vector<uint64_t> shares;
                for (const auto& input : bound_inputs) shares.push_back(input.value);
                below_bound.resize(inputs.size());
                privateMsbOnline(party_id, shares.data(), below_bound.data(), context->clip_material.material, player);
            } else below_bound = signBitsOnline(
                party_id, bound_inputs, keys, &SigmaActivationKey::clip_and, player, workspace,
                keys.front().clip_bits);
        }
        std::vector<GroupElement> clipped;
        {
            ActivationStageScope stage(stages, false, ActivationStage::clip_bitxa, player);
            clipped = mixedProductsOnline(party_id, selection_inputs, below_bound,
                keys, &SigmaActivationKey::clip_select_mul, player);
        }
        for (auto& value : clipped) value = value + public_clipped_bound;
        std::vector<GroupElement> reduced = clipped;
        if (keys.front().trunc_bits > 0) {
            ActivationStageScope stage(stages, false, ActivationStage::index_truncation, player);
            std::vector<ExactTruncationKey*> truncation_keys;
            truncation_keys.reserve(keys.size());
            for (auto& key : keys) truncation_keys.push_back(&key.trunc_key);
            reduced = exactTruncateBatch(party_id, clipped, truncation_keys, player);
        }
        std::vector<GroupElement> indices;
        indices.reserve(inputs.size());
        for (const auto& value : reduced) indices.emplace_back(value.value, q);
        const auto delta = keys.front().use_dpf_lut
            ? dpfLookupOnline(party_id, indices, keys, player, metrics, binding)
            : lookupOnline(indices, keys, player);
        const GroupElement public_scale(party_id == SERVER ? uint64_t(1) << frac_bits : 0, bits);
        std::vector<GroupElement> payloads;
        payloads.reserve(inputs.size());
        for (const auto& value : delta) payloads.push_back(public_scale - value * uint64_t(2));
        std::vector<GroupElement> outputs;
        {
            ActivationStageScope stage(stages, false, ActivationStage::restore_bitxa, player);
            outputs = mixedProductsOnline(party_id, payloads, nonnegative,
                keys, &SigmaActivationKey::branch_select_mul, player);
        }
        for (size_t i = 0; i < outputs.size(); ++i) outputs[i] = outputs[i] + delta[i];
        return outputs;
    } catch (...) {
        invalidateBatch();
        throw;
    }
}

std::vector<GroupElement> sigmaBatchOnline(
    int party_id, const std::vector<GroupElement>& inputs,
    std::vector<SigmaActivationKey>& keys, Peer* player) {
    return sigmaBatchOnlineImpl(party_id, inputs, keys, player, nullptr);
}

static std::vector<GroupElement> tanhBatchOnlineImpl(
    int party_id, const std::vector<GroupElement>& inputs,
    std::vector<SigmaActivationKey>& keys, Peer* player,
    BooleanAndWorkspace* workspace, DpfLutBatchMetrics* metrics = nullptr,
    const DpfLutBinding* binding = nullptr, ActivationContext* context = nullptr) {
    auto invalidateBatch = [&] {
        for (auto& key : keys) {
            key.segmented_lut.consumed = true;
            key.dpf_lut.consumed = true;
        }
    };
    try {
        // Validate before constructing the doubled shares, but keep the same
        // batch fail-closed contract as SIGMA when a suffix is malformed or
        // already consumed.
        validateBatch(party_id, inputs, keys, player, binding, context);
        std::vector<GroupElement> doubled;
        doubled.reserve(inputs.size());
        for (const auto& input : inputs) doubled.push_back(input * uint64_t(2));
        auto outputs = sigmaBatchOnlineImpl(party_id, doubled, keys, player, workspace,
                                            metrics, binding, nullptr, context);
        const GroupElement public_scale(
            party_id == SERVER ? uint64_t(1) << keys.front().frac_bits : 0,
            keys.front().bits);
        for (auto& output : outputs) output = output * uint64_t(2) - public_scale;
        return outputs;
    } catch (...) {
        invalidateBatch();
        throw;
    }
}

std::vector<GroupElement> tanhBatchOnline(
    int party_id, const std::vector<GroupElement>& inputs,
    std::vector<SigmaActivationKey>& keys, Peer* player) {
    return tanhBatchOnlineImpl(party_id, inputs, keys, player, nullptr);
}

static std::vector<GroupElement> consumeActivation(
    ActivationContext& context, const std::vector<GroupElement>& inputs, bool tanh,
    ActivationMetrics* stages = nullptr, ActivationInputSignShares* input_signs = nullptr) {
    if (!context.prepared || context.material_cursor != 0)
        throw std::logic_error("activation context has no unconsumed batch");
    const auto sent = context.player->bytesSent;
    const auto received = context.player->bytesReceived;
    // Crossing online irreversibly consumes this inventory, including shape
    // failures. A fresh prepare is the only way to issue another request.
    context.prepared = false;
    context.material_cursor = context.materials.size();
    context.statistics.consumed += context.materials.size();
    try {
        if (tanh && context.reuse_input_sign)
            throw std::invalid_argument("native reused signs require the explicit SIGMA core input");
        auto result = tanh
            ? tanhBatchOnlineImpl(context.party_id, inputs, context.materials,
                                  context.player, &context.boolean_workspace,
                                  context.paper_lut ? &context.lut_metrics : nullptr,
                                  context.paper_lut ? &context.lut_binding : nullptr, &context)
            : sigmaBatchOnlineImpl(context.party_id, inputs, context.materials,
                                   context.player, &context.boolean_workspace,
                                   context.paper_lut ? &context.lut_metrics : nullptr,
                                   context.paper_lut ? &context.lut_binding : nullptr, stages, &context, input_signs);
        context.statistics.online_sent += context.player->bytesSent - sent;
        context.statistics.online_received += context.player->bytesReceived - received;
        return result;
    } catch (...) {
        if (input_signs) input_signs->consumed = true;
        context.sign_material.material.consumed = true;
        context.clip_material.material.consumed = true;
        context.statistics.online_sent += context.player->bytesSent - sent;
        context.statistics.online_received += context.player->bytesReceived - received;
        ++context.statistics.failed_batches;
        throw;
    }
}

std::vector<GroupElement> sigmaBatchOnline(
    ActivationContext& context, const std::vector<GroupElement>& inputs) {
    return consumeActivation(context, inputs, false);
}

std::vector<GroupElement> sigmaBatchOnline(
    ActivationContext& context, const std::vector<GroupElement>& inputs,
    ActivationMetrics* stages) {
    return consumeActivation(context, inputs, false, stages);
}

std::vector<GroupElement> sigmaBatchOnline(
    ActivationContext& context, const std::vector<GroupElement>& inputs,
    ActivationMetrics* stages, ActivationInputSignShares* input_signs) {
    return consumeActivation(context, inputs, false, stages, input_signs);
}

std::vector<GroupElement> tanhBatchOnline(
    ActivationContext& context, const std::vector<GroupElement>& inputs) {
    return consumeActivation(context, inputs, true);
}

}  // namespace segfss
