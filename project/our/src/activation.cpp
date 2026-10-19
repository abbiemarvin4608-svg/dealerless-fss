#include "activation.h"

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <stdexcept>
#include <utility>
#include <vector>

#include <cryptoTools/Crypto/PRNG.h>

#include "mpc/api.h"
#include "mpc/secure_ops.h"

namespace segfss {
namespace {

constexpr int kBoundLog2 = kSigmaBoundLog2;  // B = 2^4 = 16.

void validateParameters(int bits, int frac_bits, int q) {
    if (bits < 2 || bits > 62) {
        throw std::invalid_argument("SIGMA activation requires 2 <= bits <= 62");
    }
    if (frac_bits < 0 || frac_bits >= bits || frac_bits + kBoundLog2 >= 62) {
        throw std::invalid_argument("invalid SIGMA fractional precision");
    }
    if (frac_bits + kBoundLog2 >= bits) {
        throw std::invalid_argument(
            "SIGMA bound requires bits > frac_bits + log2(bound)");
    }
    if (q <= kBoundLog2 || q >= 31) {
        throw std::invalid_argument("SIGMA q must satisfy 4 < q < 31");
    }
}

uint64_t maskForBits(int bits) {
    return (uint64_t(1) << bits) - 1;
}

int64_t signedValue(GroupElement input) {
    const uint64_t raw = input.value & maskForBits(input.bitsize);
    const uint64_t sign = uint64_t(1) << (input.bitsize - 1);
    if ((raw & sign) == 0) {
        return static_cast<int64_t>(raw);
    }
    return static_cast<int64_t>(raw - (uint64_t(1) << input.bitsize));
}

uint64_t encodeValue(int64_t value, int bits) {
    return static_cast<uint64_t>(value) & maskForBits(bits);
}

std::vector<u8> publicBits(uint64_t value, int bits) {
    std::vector<u8> result(static_cast<size_t>(bits));
    for (int i = 0; i < bits; ++i) {
        result[static_cast<size_t>(i)] =
            static_cast<u8>((value >> (bits - 1 - i)) & 1U);
    }
    return result;
}

std::vector<SigmaActivationKey::SegmentedLutMaterial> segmentedLutOfflineBatch(
    int party_id, int count, const PublicLUTData& table, int nseg,
    Peer* player) {
    const int index_bits = table.Bin;
    const int output_bits = table.Bout;
    // Materializing every shifted output takes O(2^(2q)) local operations.
    // Bound this explicit implementation rather than accidentally allocating
    // or computing a table with billions of entries for generic q callers.
    if (index_bits <= 0 || index_bits > 12 || output_bits <= 0 ||
        output_bits > 62 || nseg < 1 || nseg > index_bits || count <= 0 ||
        player == nullptr) {
        throw std::invalid_argument("invalid materialized segmented LUT parameters (q <= 12)");
    }
    const size_t domain = size_t(1) << index_bits;
    auto& rng = player->randomSource();
    std::vector<u8> mask_bits(static_cast<size_t>(count) * index_bits);
    rng.get(mask_bits.data(), mask_bits.size());
    for (u8& bit : mask_bits) bit &= 1U;
    std::vector<GroupElement> arithmetic_bits(mask_bits.size(),
                                               GroupElement(0, index_bits));
    B2A(party_id, mask_bits.data(), arithmetic_bits.data(),
        static_cast<int>(mask_bits.size()), index_bits, player);

    const SegmentLayout layout(index_bits, nseg);
    std::vector<SigmaActivationKey::SegmentedLutMaterial> materials(count);
    std::vector<std::vector<u8>> targets;
    targets.reserve(static_cast<size_t>(count) * nseg);
    for (int i = 0; i < count; ++i) {
        auto& material = materials[i];
        material.index_bits = index_bits;
        material.output_bits = output_bits;
        material.nseg = nseg;
        material.mask_share = GroupElement(0, index_bits);
        for (int bit = 0; bit < index_bits; ++bit) {
            material.mask_share = material.mask_share +
                arithmetic_bits[static_cast<size_t>(i) * index_bits + bit] *
                (uint64_t(1) << bit);
        }
        for (int segment = 0; segment < nseg; ++segment) {
            const int width = layout.width[segment];
            std::vector<u8> target;
            target.reserve(width);
            for (int k = 0; k < width; ++k) {
                const int global = layout.shift[segment] + width - 1 - k;
                target.push_back(mask_bits[static_cast<size_t>(i) * index_bits + global]);
            }
            targets.push_back(std::move(target));
        }
    }
    auto generated_keys = cggmGen2pcBatch(party_id, targets, nullptr, nullptr, player);
    for (int i = 0; i < count; ++i) {
        for (int segment = 0; segment < nseg; ++segment) {
            materials[i].keys.keys.push_back(std::move(
                generated_keys[static_cast<size_t>(i) * nseg + segment]));
        }
    }

    // Every equality point is public and independent of the later input.
    // Cache each segment's local digit evaluation, then AND whole vectors
    // together with fresh triples while still in the offline phase.
    const size_t entries = static_cast<size_t>(count) * domain;
    std::vector<u8> equality(entries), rhs(entries), product(entries);
    auto and_prep = booleanAndOffline(
        party_id, static_cast<int>(entries * (nseg - 1)), player);
    for (int segment = 0; segment < nseg; ++segment) {
        const int width = layout.width[segment];
        const size_t digit_domain = size_t(1) << width;
        for (int i = 0; i < count; ++i) {
            std::vector<u8> digit_equality(digit_domain);
            for (size_t digit = 0; digit < digit_domain; ++digit) {
                digit_equality[digit] = evalSegment(
                    party_id, materials[i].keys.keys[segment],
                    publicBits(digit, width)).e_j;
            }
            for (size_t j = 0; j < domain; ++j) {
                const u8 value = digit_equality[
                    (j >> layout.shift[segment]) & (digit_domain - 1)];
                (segment == 0 ? equality : rhs)[static_cast<size_t>(i) * domain + j] = value;
            }
        }
        if (segment > 0) {
            booleanAndOnline(party_id, equality.data(), rhs.data(),
                product.data(), static_cast<int>(entries), and_prep, player);
            equality.swap(product);
        }
    }
    std::vector<GroupElement> arithmetic(entries, GroupElement(0, output_bits));
    B2A(party_id, equality.data(), arithmetic.data(),
        static_cast<int>(entries), output_bits, player);

    // L_b[t] = sum_j a_b[j] T[(t-j) mod N]. Unsigned overflow is exact
    // modulo 2^64; masking gives the desired 2^output_bits ring. Duplicating
    // the reversed table makes each dot product a contiguous integer loop.
    std::vector<uint64_t> reversed_table(2 * domain);
    for (size_t j = 0; j < 2 * domain; ++j) {
        reversed_table[j] = table.values[(domain - (j & (domain - 1))) &
                                         (domain - 1)].value;
    }
    std::vector<uint64_t> arithmetic_values(domain);
    for (int i = 0; i < count; ++i) {
        auto& output = materials[i].output_share_table;
        output.resize(domain, GroupElement(0, output_bits));
        for (size_t j = 0; j < domain; ++j) {
            arithmetic_values[j] = arithmetic[static_cast<size_t>(i) * domain + j].value;
        }
        for (size_t t = 0; t < domain; ++t) {
            const uint64_t* rotated = reversed_table.data() +
                ((domain - t) & (domain - 1));
            uint64_t sum = 0;
            for (size_t j = 0; j < domain; ++j) {
                sum += arithmetic_values[j] * rotated[j];
            }
            output[t] = GroupElement(sum & maskForBits(output_bits), output_bits);
        }
    }
    return materials;
}

BooleanAndPreprocessing takeAndMaterial(
    BooleanAndPreprocessing& batch, size_t offset, size_t count) {
    BooleanAndPreprocessing result;
    result.a.assign(batch.a.begin() + offset, batch.a.begin() + offset + count);
    result.b.assign(batch.b.begin() + offset, batch.b.begin() + offset + count);
    result.c.assign(batch.c.begin() + offset, batch.c.begin() + offset + count);
    return result;
}

uint64_t tableIndex(int64_t magnitude, int frac_bits, int q) {
    const int64_t domain_units = int64_t(1) << (frac_bits + kBoundLog2);
    const int64_t clipped = std::min<int64_t>(
        std::max<int64_t>(magnitude, 0), domain_units - 1);
    const int table_frac_bits = q - kBoundLog2;
    int64_t index = clipped;
    if (frac_bits >= table_frac_bits) {
        index >>= (frac_bits - table_frac_bits);
    } else {
        index <<= (table_frac_bits - frac_bits);
    }
    const int64_t max_index = (int64_t(1) << q) - 1;
    return static_cast<uint64_t>(std::min<int64_t>(index, max_index));
}

uint64_t deltaEncoded(uint64_t index, int output_bits, int frac_bits, int q) {
    const int table_frac_bits = q - kBoundLog2;
    const long double argument =
        static_cast<long double>(index) /
        static_cast<long double>(uint64_t(1) << table_frac_bits);
    const long double scale =
        static_cast<long double>(uint64_t(1) << frac_bits);
    return static_cast<uint64_t>(std::floor(
        scale / (1.0L + std::exp(argument)))) & maskForBits(output_bits);
}

uint64_t sigmaEncoded(int64_t signed_input, int bits, int frac_bits, int q) {
    const int64_t magnitude = signed_input < 0 ? -signed_input : signed_input;
    const uint64_t delta = deltaEncoded(
        tableIndex(magnitude, frac_bits, q), bits, frac_bits, q);
    const uint64_t scale = uint64_t(1) << frac_bits;
    const uint64_t encoded = signed_input > 0 ? scale - delta : delta;
    return encoded & maskForBits(bits);
}

double decodeValue(uint64_t encoded, int bits, int frac_bits) {
    GroupElement value(encoded, bits);
    return static_cast<double>(signedValue(value)) /
           static_cast<double>(uint64_t(1) << frac_bits);
}

}  // namespace

SigmaActivationKey sigmaOffline(int party_id, int bits, int frac_bits,
                                int q, Peer* player) {
    auto keys = sigmaBatchOffline(party_id, 1, bits, frac_bits, q, player);
    return std::move(keys.front());
}

static std::vector<SigmaActivationKey> sigmaBatchOfflineImpl(
    int party_id, int batch_size, int bits, int frac_bits, int q, Peer* player,
    const DpfLutOptions* options, const DpfLutBinding* binding,
    std::shared_ptr<const PreparedDpfLutTable> prepared_table,
    DpfLutBatchMetrics* lut_metrics, uint64_t generation_epoch = 1,
    ActivationMetrics* stages = nullptr, bool segmented_sign = false,
    bool reuse_input_sign = false, int clip_bits = 0,
    bool native_signed16_input = false) {
    validateParameters(bits, frac_bits, q);
    if (!clip_bits) clip_bits = bits;
    if ((party_id != SERVER && party_id != CLIENT) ||
        player == nullptr || batch_size <= 0 || q > 12) {
        throw std::invalid_argument(
            "sigmaBatchOffline requires a valid peer, positive batch and q <= 12");
    }
    const int table_frac_bits = q - kBoundLog2;
    if (frac_bits < table_frac_bits) {
        throw std::invalid_argument(
            "sigmaOffline requires enough fractional bits for LUT indexing");
    }
    const char* nseg_env = std::getenv("SEGFSS_FSS_LUT_NSEG");
    const int lut_nseg = nseg_env ? std::atoi(nseg_env) : 2;
    const int nseg = options
        ? (options->family == DpfLutKeyFamily::full_q_arithmetic ? 1 : 2)
        : std::max(1, std::min(q, lut_nseg));
    const auto cached_table = prepared_table
        ? std::shared_ptr<const PublicLUTData>(prepared_table, &prepared_table->table())
        : std::make_shared<const PublicLUTData>(makeSigmaDeltaTable(bits, frac_bits, q));
    const auto& table = *cached_table;
    std::vector<SigmaActivationKey> keys;
    keys.reserve(batch_size);
    const bool use_dpf_lut = options || [] {
        const char* value = std::getenv("SEGFSS_DPF_LUT");
        return value != nullptr && (value[0] == '1' || value[0] == 'y' ||
                                    value[0] == 'Y' || value[0] == 't' ||
                                    value[0] == 'T');
    }();
    // Keep equality-vector conversion memory bounded even for Batch=128
    // network schedules with tens of thousands of independent activations.
    constexpr int kOfflineChunk = 64;
    // DPF request identities are contiguous within one online batch.  Keep
    // the DPF path together so a single masked-index opening can validate all
    // materials; cGGM retains its bounded-memory chunks.
    const int offline_chunk = use_dpf_lut ? batch_size : kOfflineChunk;
    for (int offset = 0; offset < batch_size; offset += offline_chunk) {
        const int count = std::min(offline_chunk, batch_size - offset);
        std::vector<SigmaActivationKey::SegmentedLutMaterial> materials;
        std::vector<DpfPublicLutMaterial> dpf_materials;
        if (!use_dpf_lut) {
            materials = segmentedLutOfflineBatch(
                party_id, count, table, nseg, player);
        } else {
            dpf_materials = options
                ? dpfPublicLutBatchOffline(party_id, *prepared_table, count,
                    *binding, *options, player, generation_epoch, lut_metrics)
                : dpfPublicLutBatchOffline(party_id, table, static_cast<std::size_t>(count),
                    static_cast<uint64_t>(offset), player);
        }
        const size_t sign_gates = reuse_input_sign ? 0 : static_cast<size_t>(2) * (bits - 1);
        const size_t clip_gates = static_cast<size_t>(2) * (clip_bits - 1);
        const size_t gates_per_row = sign_gates + clip_gates;
        BooleanAndPreprocessing and_material;
        if (!segmented_sign) {
            ActivationStageScope stage(stages, true, ActivationStage::shared_sign_prepare, player);
            and_material = booleanAndOffline(
                party_id, static_cast<int>(count * gates_per_row), player);
        }
        std::vector<BitXAPreprocessing> mixed_products;
        {
            ActivationStageScope stage(stages, true, ActivationStage::shared_bitxa_prepare, player);
            mixed_products = bitxaOffline(party_id, 3 * count, bits, player);
        }
        std::vector<ExactTruncationKey> truncation;
        const int trunc_bits = frac_bits - table_frac_bits;
        if (trunc_bits > 0) {
            ActivationStageScope stage(stages, true, ActivationStage::index_truncation, player);
            truncation = exactTruncateBatchOffline(party_id, count, bits,
                                                  trunc_bits, player);
        }
        for (int i = 0; i < count; ++i) {
            SigmaActivationKey key;
            key.owner_peer = player;
            key.bits = bits;
            key.frac_bits = frac_bits;
            key.clip_bits = clip_bits;
            key.native_signed16_input = native_signed16_input;
            key.reuse_input_sign = reuse_input_sign;
            key.q = q;
            key.trunc_bits = frac_bits - table_frac_bits;
            key.nseg = nseg;
            key.sign_backend = segmented_sign ? ActivationSignBackend::segmented_fss : ActivationSignBackend::bitdec;
            key.delta_table.Bin = q;
            key.delta_table.Bout = bits;
            key.public_delta_table = cached_table;
            key.prepared_delta_table = prepared_table;
            key.use_dpf_lut = use_dpf_lut;
            if (!use_dpf_lut) {
                key.segmented_lut = std::move(materials[i]);
            } else {
                key.dpf_lut = std::move(dpf_materials[i]);
                // Keep metadata mirrored for diagnostics and memory reports.
                key.segmented_lut.index_bits = q;
                key.segmented_lut.output_bits = bits;
                key.segmented_lut.mask_share = key.dpf_lut.mask_share;
            }
            if (key.trunc_bits > 0) {
                key.trunc_key = std::move(truncation[i]);
            }
            if (!segmented_sign) {
                key.sign_and = takeAndMaterial(and_material, static_cast<size_t>(i) * gates_per_row, sign_gates);
                key.clip_and = takeAndMaterial(and_material, static_cast<size_t>(i) * gates_per_row + sign_gates, clip_gates);
            }
            key.relu_mul = std::move(mixed_products[3 * i]);
            key.clip_select_mul = std::move(mixed_products[3 * i + 1]);
            key.branch_select_mul = std::move(mixed_products[3 * i + 2]);
            keys.push_back(std::move(key));
        }
    }
    return keys;
}

std::vector<SigmaActivationKey> sigmaBatchOffline(
    int party_id, int batch_size, int bits, int frac_bits, int q, Peer* player) {
    return sigmaBatchOfflineImpl(party_id, batch_size, bits, frac_bits, q,
                                 player, nullptr, nullptr, nullptr, nullptr);
}

ActivationContext::ActivationContext(int role, Peer* channel,
                                     uint64_t context_identity)
    : party_id(role), player(channel), identity(context_identity) {
    if (!channel || (role != SERVER && role != CLIENT) ||
        channel->role() != role || context_identity == 0)
        throw std::invalid_argument("activation context: invalid role, channel or identity");
}

ActivationContext::~ActivationContext() {
    for (auto& material : materials) freeSigmaActivationKey(material);
}

void prepareSigmaActivation(ActivationContext& context, int batch_size,
                            int bits, int frac_bits, int q) {
    if (context.prepared)
        throw std::logic_error("activation context has an unconsumed batch");
    for (auto& material : context.materials) freeSigmaActivationKey(material);
    context.materials.clear();
    const auto sent = context.player->bytesSent;
    const auto received = context.player->bytesReceived;
    context.materials = sigmaBatchOffline(context.party_id, batch_size, bits,
                                          frac_bits, q, context.player);
    context.paper_lut = false;
    context.sign_backend = ActivationSignBackend::bitdec;
    context.sign_segments = 0;
    context.native_signed16_input = false;
    context.reuse_input_sign = false;
    context.clip_bits = 0;
    context.clip_segments = 0;
    context.sign_material = ActivationContext::SignMaterial{};
    context.clip_material = ActivationContext::SignMaterial{};
    ++context.generation_epoch;
    context.material_cursor = 0;
    context.prepared = true;
    context.statistics.generated += context.materials.size();
    context.statistics.offline_sent += context.player->bytesSent - sent;
    context.statistics.offline_received += context.player->bytesReceived - received;
}

void prepareSigmaActivation(ActivationContext& context, int batch_size,
    int bits, int frac_bits, int q, const DpfLutOptions& options,
    const DpfLutBinding& binding, DpfLutBatchMetrics* metrics, ActivationMetrics* stages) {
    if (context.prepared)
        throw std::logic_error("activation context has an unconsumed batch");
    if (bits != 32 || frac_bits != 12 || q != 10 ||
        binding.context_id != context.identity || context.generation_epoch == UINT64_MAX)
        throw std::invalid_argument("paper activation numeric format or context mismatch");
    const bool segmented_sign = context.sign_backend == ActivationSignBackend::segmented_fss;
    const int clip_bits = context.clip_bits ? context.clip_bits : bits;
    const int clip_segments = context.clip_segments ? context.clip_segments : context.sign_segments;
    if ((!segmented_sign && (context.sign_backend != ActivationSignBackend::bitdec || context.sign_segments != 0)) ||
        (segmented_sign && (context.sign_segments < 1 || context.sign_segments >= bits)))
        throw std::invalid_argument("paper activation sign backend or segment count mismatch");
    if ((clip_bits != bits && clip_bits != 22) ||
        ((context.reuse_input_sign || clip_bits != bits) && !context.native_signed16_input) ||
        (!segmented_sign && context.clip_segments != 0) ||
        (segmented_sign && (clip_segments < 1 || clip_segments >= clip_bits)))
        throw std::invalid_argument("paper activation native sign or clip contract mismatch");
    for (auto& material : context.materials) freeSigmaActivationKey(material);
    context.materials.clear();
    if (!context.public_table || context.public_table->version() != binding.table_version)
        context.public_table = prepareDpfLutTable(
            makeSigmaDeltaTable(bits, frac_bits, q), binding.table_version);
    const auto sent = context.player->bytesSent;
    const auto received = context.player->bytesReceived;
    context.sign_material = ActivationContext::SignMaterial{};
    context.clip_material = ActivationContext::SignMaterial{};
    if (segmented_sign) {
        if (!context.reuse_input_sign) {
            ActivationStageScope stage(stages, true, ActivationStage::sign, context.player);
            context.sign_material.material = privateMsbOffline(context.party_id, batch_size, bits, context.sign_segments, context.player);
        }
        {
            ActivationStageScope stage(stages, true, ActivationStage::clip_compare, context.player);
            context.clip_material.material = privateMsbOffline(context.party_id, batch_size,
                clip_bits, clip_segments, context.player);
        }
    }
    context.materials = sigmaBatchOfflineImpl(context.party_id, batch_size, bits,
        frac_bits, q, context.player, &options, &binding, context.public_table,
        &context.lut_metrics, context.generation_epoch + 1, stages, segmented_sign,
        context.reuse_input_sign, clip_bits, context.native_signed16_input);
    context.paper_lut = true;
    context.lut_binding = binding;
    context.lut_options = options;
    ++context.generation_epoch;
    if (segmented_sign) {
        for (auto* stock : {&context.sign_material, &context.clip_material}) {
            if (!stock->material.count) continue;
            stock->binding = binding;
            stock->generation_epoch = context.generation_epoch;
        }
        context.clip_material.clip = true;
    }
    context.material_cursor = 0;
    context.prepared = true;
    context.statistics.generated += context.materials.size();
    context.statistics.offline_sent += context.player->bytesSent - sent;
    context.statistics.offline_received += context.player->bytesReceived - received;
    if (metrics) *metrics = context.lut_metrics;
}

GroupElement sigmaOnline(int party_id, GroupElement input,
                         SigmaActivationKey& key, Peer* player) {
    std::vector<SigmaActivationKey> keys;
    keys.push_back(std::move(key));
    try {
        auto output = sigmaBatchOnline(party_id, {input}, keys, player);
        key = std::move(keys.front());
        return output.front();
    } catch (...) {
        key = std::move(keys.front());
        throw;
    }
}

SigmaActivationKey tanhOffline(int party_id, int bits, int frac_bits,
                               int q, Peer* player) {
    return sigmaOffline(party_id, bits, frac_bits, q, player);
}

GroupElement tanhOnline(int party_id, GroupElement input,
                        SigmaActivationKey& key, Peer* player) {
    const GroupElement doubled = input * uint64_t(2);
    const GroupElement sigmoid = sigmaOnline(party_id, doubled, key, player);
    const uint64_t scale = uint64_t(1) << key.frac_bits;
    const GroupElement public_scale(
        static_cast<uint64_t>(party_id == SERVER ? scale : 0), key.bits);
    return sigmoid * uint64_t(2) - public_scale;
}

void freeSigmaActivationKey(SigmaActivationKey& key) {
    freePublicLutKeyPack(key.lut_key);
    freeExactTruncationKey(key.trunc_key);

    freePiecewiseComparisonKey(key.sign_cmp);
    freePiecewiseComparisonKey(key.clip_cmp);

    key.segmented_lut.keys.keys.clear();
    releaseDpfPublicLutMaterial(key.dpf_lut);
    key.segmented_lut = SigmaActivationKey::SegmentedLutMaterial();
    key = SigmaActivationKey();
}

PublicLUTData makeSigmaDeltaTable(int output_bits, int frac_bits, int q) {
    validateParameters(output_bits, frac_bits, q);
    PublicLUTData table;
    table.Bin = q;
    table.Bout = output_bits;
    table.values.resize(size_t(1) << q);
    for (uint64_t i = 0; i < table.values.size(); ++i) {
        table.values[i] = GroupElement(
            deltaEncoded(i, output_bits, frac_bits, q), output_bits);
    }
    return table;
}

GroupElement sigmaPlain(GroupElement input, int frac_bits, int q) {
    validateParameters(input.bitsize, frac_bits, q);
    return GroupElement(sigmaEncoded(signedValue(input), input.bitsize,
                                      frac_bits, q),
                        input.bitsize);
}

GroupElement tanhPlain(GroupElement input, int frac_bits, int q) {
    validateParameters(input.bitsize, frac_bits, q);
    const int64_t doubled = signedValue(GroupElement(
        encodeValue(signedValue(input) * 2, input.bitsize), input.bitsize));
    const uint64_t sigma = sigmaEncoded(doubled, input.bitsize, frac_bits, q);
    const int64_t output = static_cast<int64_t>(sigma) * 2 -
                           (int64_t(1) << frac_bits);
    return GroupElement(encodeValue(output, input.bitsize), input.bitsize);
}

SigmaActivationErrorReport measureSigmaActivationError(
    int input_bits, int frac_bits, int q) {
    validateParameters(input_bits, frac_bits, q);
    const int64_t domain_units = int64_t(1) << (frac_bits + kBoundLog2);
    const long double scale =
        static_cast<long double>(uint64_t(1) << frac_bits);
    SigmaActivationErrorReport report;
    for (int64_t input_units = -domain_units;
         input_units < domain_units; ++input_units) {
        const GroupElement input(encodeValue(input_units, input_bits),
                                 input_bits);
        const long double real_input =
            static_cast<long double>(input_units) / scale;

        const double sigmoid_approx = decodeValue(
            sigmaPlain(input, frac_bits, q).value, input_bits, frac_bits);
        const double tanh_approx = decodeValue(
            tanhPlain(input, frac_bits, q).value, input_bits, frac_bits);
        const double sigmoid_ref = static_cast<double>(
            1.0L / (1.0L + std::exp(-real_input)));
        const double tanh_ref = static_cast<double>(std::tanh(real_input));

        const double sigmoid_error = std::abs(sigmoid_approx - sigmoid_ref);
        if (sigmoid_error > report.sigmoid.max_abs) {
            report.sigmoid.max_abs = sigmoid_error;
            report.sigmoid.max_ulp = sigmoid_error * static_cast<double>(scale);
            report.sigmoid.at_input = static_cast<double>(real_input);
        }
        const double tanh_error = std::abs(tanh_approx - tanh_ref);
        if (tanh_error > report.tanh.max_abs) {
            report.tanh.max_abs = tanh_error;
            report.tanh.max_ulp = tanh_error * static_cast<double>(scale);
            report.tanh.at_input = static_cast<double>(real_input);
        }
    }
    return report;
}

}  // namespace segfss

// ========== Piecewise Comparison Implementation ==========

namespace segfss {

PiecewiseComparisonKey piecewiseCompareOffline(
    int, int, uint64_t, int, Peer*, bool) {
    throw std::logic_error(
        "unmasked piecewise comparison is disabled; use the BitDec activation path");
}

u8 piecewiseCompareOnline(
    int, GroupElement, const PiecewiseComparisonKey&, Peer*) {
    throw std::logic_error(
        "unmasked piecewise comparison is disabled; use the BitDec activation path");
}

void freePiecewiseComparisonKey(PiecewiseComparisonKey& key) {
    key.segment_keys.keys.clear();
    key.and_prep = BooleanAndPreprocessing();
    key = PiecewiseComparisonKey();
}

}  // namespace segfss
