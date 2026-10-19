#include "sci_truncation.h"
#include "runtime_options.h"

#include "mpc/api.h"
#include "mpc/secure_ops.h"

#include <algorithm>
#include <chrono>
#include <iterator>
#include <limits>
#include <stdexcept>
#include <unordered_set>
#include <utility>

namespace segfss {


namespace {
using Clock = std::chrono::steady_clock;

double elapsedMs(Clock::time_point start) {
    return std::chrono::duration<double, std::milli>(Clock::now() - start).count();
}

SciTruncationDomains domainsFor(const reference::Contract& contract, bool fuse_sign_wrap = false) {
    reference::validate(contract);
    SciTruncationDomains domains;
    const int shift = std::max(0, contract.shift);
    const int left = std::max(0, -contract.shift);
    const bool extend = contract.output_bits > contract.input_bits - shift + left;
    domains.low_shift_bits = shift;
    domains.wrap_bits = extend ? contract.input_bits : 0;
    domains.toward_zero = shift > 0 && contract.signed_arithmetic &&
        contract.call == reference::SciCall::div_pow2 &&
        contract.numeric_mode == reference::NumericMode::sci_compatible;
    domains.need_sign = contract.signed_arithmetic && (extend || domains.toward_zero);
    domains.sign_bits = domains.need_sign ? contract.input_bits - 1 : 0;
    domains.local_only = shift == 0 && !extend;
    domains.fused_sign_wrap = fuse_sign_wrap && extend && contract.signed_arithmetic;
    if (domains.fused_sign_wrap) domains.wrap_bits = 0;
    domains.correction_count = (shift > 0) + extend +
        (extend && contract.signed_arithmetic && !domains.fused_sign_wrap) + domains.toward_zero;
    return domains;
}

void validateRuntimeConfig(const SciTruncationConfig& config) {
    if (config.fuse_sign_wrap != runtimeFlag("SEGFSS_FUSE_SIGN_WRAP"))
        throw std::invalid_argument("SCI truncation: sign/wrap fusion flag differs from material config");
}

void validateContext(int role, int count, Peer* player) {
    if ((role != SERVER && role != CLIENT) || !player || player->role() != role || count < 1 ||
        count > std::numeric_limits<int>::max() / 256) {
        throw std::invalid_argument("SCI truncation: invalid role/count/peer");
    }
}

void validateComparison(const SegmentedDcfMaterial& material, int role,
                        int count, int bits, int nseg, bool need_e) {
    if (bits == 0) {
        if (material.count != 0 || material.keyCount() || !material.scan.a.empty())
            throw std::invalid_argument("SCI truncation: unexpected inactive comparison material");
        return;
    }
    const SegmentLayout layout(bits, nseg);
    const std::size_t gates = static_cast<std::size_t>(count) *
        planDcfScan(nseg, need_e, material.prune_equality).and_gates_per_row;
    if (material.role != role || material.count != count || material.bits != bits ||
        material.nseg != nseg || material.need_e != need_e || material.consumed || material.released ||
        material.layout.ell != bits || material.layout.nseg != nseg ||
        material.layout.width != layout.width || material.layout.shift != layout.shift ||
        material.keyCount() != static_cast<std::size_t>(count) * nseg ||
        material.scan.a.size() != gates || material.scan.b.size() != gates ||
        material.scan.c.size() != gates || material.scan.cursor != 0) {
        throw std::invalid_argument("SCI truncation: invalid comparison material");
    }
    for (std::size_t index = 0; index < material.keyCount(); ++index) {
        const auto key = material.keyView(index);
        if (key.depth != layout.width[index % nseg])
            throw std::invalid_argument("SCI truncation: invalid comparison key");
    }
}

SciTruncationDomains validateMaterial(int role, const SciTruncationConfig& expected,
                                     const SciTruncationMaterial& material, Peer* player) {
    validateContext(role, material.count, player);
    validateRuntimeConfig(expected);
    const auto domains = sciTruncationDomains(expected);
    if (!sameSciTruncationConfig(expected, material.config) || material.role != role ||
        material.consumed || material.released) {
        throw std::invalid_argument("SCI truncation: contract mismatch or unavailable material");
    }
    const std::size_t masks = domains.local_only ? 0 : material.count;
    const std::size_t sign_masks = domains.need_sign ? material.count : 0;
    const std::size_t conversions = static_cast<std::size_t>(material.count) * domains.correction_count;
    const std::size_t rounding = material.count *
        static_cast<std::size_t>(domains.toward_zero + domains.fused_sign_wrap);
    if (material.input_mask_shares.size() != masks || material.mask_high_shares.size() != masks ||
        material.mask_sign_bit_shares.size() != sign_masks ||
        material.conversion_mask_bits.size() != conversions ||
        material.conversion_mask_arithmetic.size() != conversions ||
        material.rounding.a.size() != rounding || material.rounding.b.size() != rounding ||
        material.rounding.c.size() != rounding || material.rounding.cursor != 0) {
        throw std::invalid_argument("SCI truncation: invalid mask or rounding material");
    }
    validateComparison(material.low_shift, role, material.count, domains.low_shift_bits, expected.low_shift_nseg,
                       domains.toward_zero);
    validateComparison(material.sign, role, material.count, domains.sign_bits, expected.sign_nseg, false);
    validateComparison(material.wrap, role, material.count, domains.wrap_bits, expected.wrap_nseg, false);
    for (const auto* comparison : {&material.low_shift, &material.sign, &material.wrap})
        if (comparison->count && comparison->segment_policy != expected.segment_policy)
            throw std::invalid_argument("SCI truncation: comparison policy differs from material contract");
    return domains;
}

std::vector<uint64_t> openMaskedInput(int role, const std::vector<GroupElement>& inputs,
                                    const SciTruncationMaterial& material, Peer* player) {
    const uint64_t mask = reference::mask(material.config.contract.input_bits);
    std::vector<uint64_t> opened(inputs.size()), remote(inputs.size());
    for (std::size_t row = 0; row < inputs.size(); ++row)
        opened[row] = (inputs[row].value + material.input_mask_shares[row]) & mask;
    if (runtimeFlag("SEGFSS_PACK_RING")) {
        remote = exchangeRingValues(role == SERVER, opened, material.config.contract.input_bits,
            [&](const u8* data, int size) { player->send_u8(data, size); },
            [&](u8* data, int size) { player->recv_u8(data, size); });
    } else if (role == SERVER) {
        player->send_u64(opened.data(), opened.size());
        player->recv_u64(remote.data(), remote.size());
    } else {
        player->recv_u64(remote.data(), remote.size());
        player->send_u64(opened.data(), opened.size());
    }
    for (std::size_t row = 0; row < opened.size(); ++row) opened[row] = (opened[row] + remote[row]) & mask;
    return opened;
}

std::vector<CggmEval> compareDomain(int role, const std::vector<uint64_t>& opened,
                                  SegmentedDcfMaterial& material, Peer* player,
                                  SciTruncationWorkspace* workspace) {
    if (material.count == 0) return std::vector<CggmEval>(opened.size());
    std::vector<uint64_t> local_queries;
    if (workspace) workspace->queries.resize(1);
    auto& queries = workspace ? workspace->queries.front() : local_queries;
    queries.resize(opened.size());
    const uint64_t mask = reference::mask(material.bits);
    for (std::size_t row = 0; row < opened.size(); ++row) queries[row] = opened[row] & mask;
    return segmentedDcfOnlineBatchGroup(role, {{&queries, &material, material.need_e}}, player,
        workspace ? &workspace->comparisons : nullptr).front();
}

std::vector<uint64_t> convertMaskedBits(int role, const std::vector<u8>& values,
                                      const SciTruncationMaterial& material, Peer* player) {
    std::vector<u8> opened(values.size());
    for (std::size_t bit = 0; bit < values.size(); ++bit)
        opened[bit] = values[bit] ^ material.conversion_mask_bits[bit];
    openBooleanBits(role, opened, player);
    const uint64_t mask = reference::mask(material.config.contract.output_bits);
    std::vector<uint64_t> arithmetic(values.size());
    for (std::size_t bit = 0; bit < values.size(); ++bit) {
        const uint64_t value = opened[bit];
        const uint64_t random = material.conversion_mask_arithmetic[bit];
        arithmetic[bit] = ((role == SERVER ? value : 0) + random - 2 * value * random) & mask;
    }
    return arithmetic;
}

template <class T>
void append(std::vector<T>& destination, const std::vector<T>& source) {
    destination.insert(destination.end(), source.begin(), source.end());
}

void appendComparison(SegmentedDcfMaterial& destination, SegmentedDcfMaterial& source) {
    if (source.count == 0) return;
    if (destination.count == 0) {
        destination = std::move(source);
        return;
    }
    if (destination.need_e != source.need_e || destination.prune_equality != source.prune_equality ||
        destination.prune_leaf_hashes != source.prune_leaf_hashes)
        throw std::invalid_argument("SCI truncation: differing comparison demand plans");
    if (destination.compact_keys.size() || source.compact_keys.size()) {
        destination.compactKeys(); source.compactKeys();
        destination.compact_keys.append(source.compact_keys);
    } else {
        destination.keys.insert(destination.keys.end(),
            std::make_move_iterator(source.keys.begin()), std::make_move_iterator(source.keys.end()));
    }
    append(destination.scan.a, source.scan.a);
    append(destination.scan.b, source.scan.b);
    append(destination.scan.c, source.scan.c);
    destination.count += source.count;
    destination.metrics.keygen_ms += source.metrics.keygen_ms;
    destination.metrics.scan_preprocessing_ms += source.metrics.scan_preprocessing_ms;
    destination.metrics.internal_hashes += source.metrics.internal_hashes;
    destination.metrics.leaf_hashes += source.metrics.leaf_hashes;
    destination.metrics.generation_participants = std::max(destination.metrics.generation_participants,
                                                          source.metrics.generation_participants);
    destination.metrics.keygen_aes_scratch_bytes = std::max(destination.metrics.keygen_aes_scratch_bytes,
                                                          source.metrics.keygen_aes_scratch_bytes);
    destination.metrics.aes_offline = destination.metrics.aes_offline || source.metrics.aes_offline;
    destination.metrics.scan_and_gates += source.metrics.scan_and_gates;
    releaseSegmentedDcfMaterial(source);
}
}  // namespace

SciTruncationDomains sciTruncationDomains(const SciTruncationConfig& config) {
    const auto domains = domainsFor(config.contract, config.fuse_sign_wrap);
    if (config.backend != DcfBackend::full_tree && config.backend != DcfBackend::segmented)
        throw std::invalid_argument("SCI truncation: unknown comparison backend");
    segmentPolicyName(config.segment_policy);
    for (const auto shape : {std::pair<int, int>{domains.low_shift_bits, config.low_shift_nseg},
                            {domains.sign_bits, config.sign_nseg}, {domains.wrap_bits, config.wrap_nseg}}) {
        if ((shape.first == 0 && shape.second != 0) ||
            (shape.first > 0 && (shape.second < 1 || shape.second > shape.first)) ||
            (shape.first > 0 && config.backend == DcfBackend::full_tree && shape.second != 1))
            throw std::invalid_argument("SCI truncation: invalid explicit comparison segmentation");
        if (config.backend == DcfBackend::segmented && config.segment_policy != SegmentPolicy::explicit_counts &&
            shape.second != segmentCount(shape.first, config.segment_policy))
            throw std::invalid_argument("SCI truncation: policy differs from comparison segment count");
    }
    return domains;
}

SciTruncationConfig makeSciTruncationConfig(
    const reference::Contract& contract, DcfBackend backend, int max_segments) {
    if (max_segments < 1 || max_segments > 64)
        throw std::invalid_argument("SCI truncation: max_segments must be in [1,64]");
    const bool fuse_sign_wrap = runtimeFlag("SEGFSS_FUSE_SIGN_WRAP");
    const auto domains = domainsFor(contract, fuse_sign_wrap);
    const int limit = backend == DcfBackend::full_tree ? 1 : max_segments;
    SciTruncationConfig config{contract, backend, std::min(limit, domains.low_shift_bits),
        std::min(limit, domains.sign_bits), std::min(limit, domains.wrap_bits)};
    config.fuse_sign_wrap = fuse_sign_wrap;
    sciTruncationDomains(config);
    return config;
}

SciTruncationConfig makeSciTruncationConfig(
    const reference::Contract& contract, DcfBackend backend, SegmentPolicy policy) {
    if (policy == SegmentPolicy::explicit_counts)
        throw std::invalid_argument("SCI truncation: legacy policy requires max_segments");
    auto config = makeSciTruncationConfig(contract, backend, 1);
    const auto domains = domainsFor(contract, config.fuse_sign_wrap);
    config.segment_policy = policy;
    if (backend == DcfBackend::segmented) {
        config.low_shift_nseg = segmentCount(domains.low_shift_bits, policy);
        config.sign_nseg = segmentCount(domains.sign_bits, policy);
        config.wrap_nseg = segmentCount(domains.wrap_bits, policy);
    }
    sciTruncationDomains(config);
    return config;
}

bool sameSciTruncationConfig(const SciTruncationConfig& a, const SciTruncationConfig& b) {
    return a.contract.input_bits == b.contract.input_bits &&
        a.contract.output_bits == b.contract.output_bits && a.contract.shift == b.contract.shift &&
        a.contract.call == b.contract.call && a.contract.signed_arithmetic == b.contract.signed_arithmetic &&
        a.contract.numeric_mode == b.contract.numeric_mode && a.backend == b.backend &&
        a.low_shift_nseg == b.low_shift_nseg && a.sign_nseg == b.sign_nseg && a.wrap_nseg == b.wrap_nseg &&
        a.segment_policy == b.segment_policy && a.fuse_sign_wrap == b.fuse_sign_wrap;
}

SciTruncationMaterial::SciTruncationMaterial(SciTruncationMaterial&& other) noexcept {
    *this = std::move(other);
}

SciTruncationMaterial& SciTruncationMaterial::operator=(SciTruncationMaterial&& other) noexcept {
    if (this != &other) {
        config = other.config;
        role = other.role;
        count = other.count;
        input_mask_shares = std::move(other.input_mask_shares);
        mask_high_shares = std::move(other.mask_high_shares);
        mask_sign_bit_shares = std::move(other.mask_sign_bit_shares);
        low_shift = std::move(other.low_shift);
        sign = std::move(other.sign);
        wrap = std::move(other.wrap);
        rounding = std::move(other.rounding);
        conversion_mask_bits = std::move(other.conversion_mask_bits);
        conversion_mask_arithmetic = std::move(other.conversion_mask_arithmetic);
        consumed = other.consumed;
        released = other.released;
        metrics = other.metrics;
        other.count = 0;
        other.consumed = true;
        other.released = true;
    }
    return *this;
}

SciTruncationMaterial sciTruncateOfflineBatch(
    int role, int count, const SciTruncationConfig& config, Peer* player,
    const DcfGenerationOptions* generation_options) {
    validateContext(role, count, player);
    validateRuntimeConfig(config);
    const auto domains = sciTruncationDomains(config);
    SciTruncationMaterial material;
    material.config = config;
    material.role = role;
    material.count = count;
    if (domains.local_only) {
        material.consumed = false;
        return material;
    }
    const auto& contract = config.contract;
    const int bits = contract.input_bits;
    const int shift = std::max(0, contract.shift);
    const int stride = bits + domains.correction_count;
    const int arithmetic_bits = std::max(bits, contract.output_bits);
    const uint64_t input_mask = reference::mask(bits), output_mask = reference::mask(contract.output_bits);
    auto start = Clock::now();
    osuCrypto::PRNG random(osuCrypto::sysRandomSeed());
    std::vector<u8> sampled(static_cast<std::size_t>(count) * stride);
    random.get(sampled.data(), sampled.size());
    for (auto& bit : sampled) bit &= 1;
    std::vector<GroupElement> arithmetic(sampled.size(), GroupElement(0, arithmetic_bits));
    // One offline B2A batch provides additive shares of every mask bit in a
    // ring wide enough to derive both r and floor(r/2^shift) without rounding.
    B2A(role, sampled.data(), arithmetic.data(), arithmetic.size(), arithmetic_bits, player);
    material.input_mask_shares.resize(count);
    material.mask_high_shares.resize(count);
    if (domains.need_sign) material.mask_sign_bit_shares.resize(count);
    material.conversion_mask_bits.resize(static_cast<std::size_t>(count) * domains.correction_count);
    material.conversion_mask_arithmetic.resize(material.conversion_mask_bits.size());
    for (int row = 0; row < count; ++row) {
        uint64_t input_share = 0, high_share = 0;
        for (int bit = 0; bit < bits; ++bit) {
            input_share += arithmetic[row * stride + bit].value << bit;
            if (bit >= shift) high_share += arithmetic[row * stride + bit].value << (bit - shift);
        }
        material.input_mask_shares[row] = input_share & input_mask;
        material.mask_high_shares[row] = high_share & output_mask;
        if (domains.need_sign) material.mask_sign_bit_shares[row] = sampled[row * stride + bits - 1];
        for (int correction = 0; correction < domains.correction_count; ++correction) {
            const int index = row * domains.correction_count + correction;
            material.conversion_mask_bits[index] = sampled[row * stride + bits + correction];
            material.conversion_mask_arithmetic[index] = arithmetic[row * stride + bits + correction].value & output_mask;
        }
    }
    material.metrics.mask_preprocessing_ms = elapsedMs(start);
    start = Clock::now();
    auto generate = [&](int domain_bits, int nseg, bool need_e, SegmentedDcfMaterial& destination) {
        if (domain_bits == 0) return;
        std::vector<std::vector<u8>> thresholds(count, std::vector<u8>(domain_bits));
        for (int row = 0; row < count; ++row)
            for (int bit = 0; bit < domain_bits; ++bit)
                thresholds[row][bit] = sampled[row * stride + domain_bits - bit - 1];
        auto options = generation_options ? *generation_options : dcfGenerationOptions();
        options.resident_bytes += sampled.capacity() + arithmetic.capacity() * sizeof(GroupElement) +
            (material.input_mask_shares.capacity() + material.mask_high_shares.capacity() +
             material.conversion_mask_arithmetic.capacity()) * sizeof(uint64_t) +
            material.mask_sign_bit_shares.capacity() + material.conversion_mask_bits.capacity() +
            material.low_shift.capacityBytes() + material.sign.capacityBytes() + material.wrap.capacityBytes();
        destination = segmentedDcfOfflineBatch(role, thresholds, domain_bits, nseg, player, options, need_e);
        destination.segment_policy = config.segment_policy;
    };
    generate(domains.low_shift_bits, config.low_shift_nseg, domains.toward_zero, material.low_shift);
    generate(domains.sign_bits, config.sign_nseg, false, material.sign);
    generate(domains.wrap_bits, config.wrap_nseg, false, material.wrap);
    material.metrics.comparisons_preprocessing_ms = elapsedMs(start);
    start = Clock::now();
    if (domains.toward_zero || domains.fused_sign_wrap)
        material.rounding = booleanAndOffline(role, count *
            static_cast<int>(domains.toward_zero + domains.fused_sign_wrap), player);
    material.metrics.rounding_preprocessing_ms = elapsedMs(start);
    material.consumed = false;
    return material;
}

SciTruncationMaterial sciTruncateOffline(int role, const SciTruncationConfig& config, Peer* player) {
    return sciTruncateOfflineBatch(role, 1, config, player);
}

std::vector<GroupElement> sciTruncateOnlineBatch(
    int role, const std::vector<GroupElement>& inputs, const SciTruncationConfig& expected,
    SciTruncationMaterial& material, Peer* player, SciTruncationWorkspace* workspace,
    std::vector<u8>* input_signs) {
    const auto domains = validateMaterial(role, expected, material, player);
    if (input_signs && !domains.need_sign)
        throw std::invalid_argument("SCI truncation: requested sign is absent from this contract");
    if (inputs.size() != static_cast<std::size_t>(material.count))
        throw std::invalid_argument("SCI truncation: input count mismatch");
    for (const auto& input : inputs)
        if (input.bitsize != expected.contract.input_bits)
            throw std::invalid_argument("SCI truncation: input ring mismatch");
    material.consumed = true;
    const auto& contract = expected.contract;
    const int shift = std::max(0, contract.shift), left = std::max(0, -contract.shift);
    const uint64_t output_mask = reference::mask(contract.output_bits);
    std::vector<GroupElement> outputs(inputs.size(), GroupElement(0, contract.output_bits));
    if (domains.local_only) {
        for (std::size_t row = 0; row < inputs.size(); ++row)
            outputs[row].value = (inputs[row].value << left) & output_mask;
        return outputs;
    }
    auto start = Clock::now();
    const auto opened = openMaskedInput(role, inputs, material, player);
    material.metrics.masked_input_open_ms = elapsedMs(start);
    material.metrics.masked_input_values = inputs.size();
    material.metrics.online_logical_rounds = 1;
    start = Clock::now();
    std::vector<CggmEval> low, sign_compare, wrap;
    if (runtimeFlag("SEGFSS_FUSE_DOMAINS")) {
        SciTruncationWorkspace local;
        auto& scratch = workspace ? *workspace : local;
        scratch.queries.resize(3);
        std::vector<SegmentedDcfBatchView> groups;
        std::vector<std::vector<CggmEval>*> targets;
        std::size_t index = 0;
        for (const auto& entry : {std::make_pair(&material.low_shift, &low),
                                 std::make_pair(&material.sign, &sign_compare), std::make_pair(&material.wrap, &wrap)}) {
            if (!entry.first->count) { entry.second->resize(opened.size()); continue; }
            auto& queries = scratch.queries[index++];
            queries.resize(opened.size());
            const auto mask = reference::mask(entry.first->bits);
            for (std::size_t row = 0; row < opened.size(); ++row) queries[row] = opened[row] & mask;
            groups.push_back({&queries, entry.first, entry.first->need_e}); targets.push_back(entry.second);
        }
        auto results = segmentedDcfOnlineBatchGroup(role, groups, player, &scratch.comparisons);
        for (std::size_t domain = 0; domain < results.size(); ++domain) *targets[domain] = std::move(results[domain]);
        material.metrics.online_logical_rounds += scratch.comparisons.scan_layers;
        material.metrics.comparison_scan_ms = scratch.comparisons.scan_ms;
    } else {
        low = compareDomain(role, opened, material.low_shift, player, workspace);
        sign_compare = compareDomain(role, opened, material.sign, player, workspace);
        wrap = compareDomain(role, opened, material.wrap, player, workspace);
        material.metrics.online_logical_rounds += material.low_shift.metrics.online_scan_layers +
            material.sign.metrics.online_scan_layers + material.wrap.metrics.online_scan_layers;
        material.metrics.comparison_scan_ms = material.low_shift.metrics.scan_ms +
            material.sign.metrics.scan_ms + material.wrap.metrics.scan_ms;
    }
    material.metrics.comparisons_online_ms = elapsedMs(start);
    std::vector<u8> signs(inputs.size()), nonzero(inputs.size()), rounding(inputs.size());
    std::vector<u8> fused_product(domains.fused_sign_wrap ? inputs.size() : 0);
    for (std::size_t row = 0; row < inputs.size(); ++row) {
        if (domains.need_sign)
            signs[row] = sign_compare[row].L ^ material.mask_sign_bit_shares[row] ^
                (role == SERVER ? static_cast<u8>(opened[row] >> (contract.input_bits - 1)) : 0);
        nonzero[row] = low[row].e ^ static_cast<u8>(role == SERVER);
    }
    start = Clock::now();
    if (domains.fused_sign_wrap) {
        const int gates_per_row = domains.toward_zero + domains.fused_sign_wrap;
        std::vector<u8> lhs(inputs.size() * gates_per_row), rhs(lhs.size()), products(lhs.size());
        for (std::size_t row = 0; row < inputs.size(); ++row) {
            std::size_t index = row * gates_per_row;
            if (domains.fused_sign_wrap) {
                // b is public. Flip just SERVER's share so the two XOR
                // reconstructions are a XOR b and l XOR b respectively.
                const u8 b = role == SERVER ? opened[row] >> (contract.input_bits - 1) : 0;
                lhs[index] = material.mask_sign_bit_shares[row] ^ b;
                rhs[index++] = sign_compare[row].L ^ b;
            }
            if (domains.toward_zero) {
                lhs[index] = signs[row]; rhs[index] = nonzero[row];
            }
        }
        booleanAndOnline(role, lhs.data(), rhs.data(), products.data(), products.size(), material.rounding, player);
        for (std::size_t row = 0; row < inputs.size(); ++row) {
            std::size_t index = row * gates_per_row;
            if (domains.fused_sign_wrap) fused_product[row] = products[index++];
            if (domains.toward_zero) rounding[row] = products[index];
        }
        ++material.metrics.online_logical_rounds;
    } else if (domains.toward_zero) {
        booleanAndOnline(role, signs.data(), nonzero.data(), rounding.data(), inputs.size(), material.rounding, player);
        ++material.metrics.online_logical_rounds;
    }
    material.metrics.rounding_online_ms = elapsedMs(start);
    std::vector<u8> corrections(inputs.size() * domains.correction_count);
    for (std::size_t row = 0; row < inputs.size(); ++row) {
        std::size_t index = row * domains.correction_count;
        if (shift > 0) corrections[index++] = low[row].L;
        if (domains.fused_sign_wrap) corrections[index++] = fused_product[row];
        else if (domains.wrap_bits > 0) {
            corrections[index++] = wrap[row].L;
            if (contract.signed_arithmetic) corrections[index++] = signs[row];
        }
        if (domains.toward_zero) corrections[index] = rounding[row];
    }
    start = Clock::now();
    const auto arithmetic = convertMaskedBits(role, corrections, material, player);
    material.metrics.masked_bit_conversion_ms = elapsedMs(start);
    material.metrics.masked_conversion_bits = corrections.size();
    ++material.metrics.online_logical_rounds;
    // floor_unsigned(x/2^s) = z_high-r_high-borrow+2^(k-s)*wrap;
    // signed floor subtracts 2^(k-s)*sign, and signed div adds sign*nonzero.
    const int high_bits = contract.input_bits - shift;
    const uint64_t scale = high_bits < 64 ? uint64_t(1) << high_bits : 0;
    for (std::size_t row = 0; row < inputs.size(); ++row) {
        uint64_t output = (role == SERVER ? opened[row] >> shift : 0) - material.mask_high_shares[row];
        std::size_t index = row * domains.correction_count;
        if (shift > 0) output -= arithmetic[index++];
        if (domains.fused_sign_wrap) {
            // Let a=r_MSB, l=[z_low < r_low], b=z_MSB. Then
            // wrap-sign = (1-2*b) * ((a XOR b) AND (l XOR b)).
            const uint64_t correction = scale * arithmetic[index++];
            if (opened[row] >> (contract.input_bits - 1)) output -= correction;
            else output += correction;
        } else if (domains.wrap_bits > 0) {
            output += scale * arithmetic[index++];
            if (contract.signed_arithmetic) output -= scale * arithmetic[index++];
        }
        if (domains.toward_zero) output += arithmetic[index];
        outputs[row].value = (output << left) & output_mask;
    }
    if (input_signs) *input_signs = std::move(signs);
    return outputs;
}

GroupElement sciTruncateOnline(int role, GroupElement input, const SciTruncationConfig& expected,
                              SciTruncationMaterial& material, Peer* player) {
    return sciTruncateOnlineBatch(role, {input}, expected, material, player).front();
}

std::vector<GroupElement> sciTruncateOnlineBatch(
    int role, const std::vector<GroupElement>& inputs, const SciTruncationConfig& expected,
    const std::vector<SciTruncationMaterial*>& materials, Peer* player,
    SciTruncationMetrics* batch_metrics) {
    std::unordered_set<SciTruncationMaterial*> seen;
    std::size_t count = 0;
    for (auto* material : materials) {
        if (!material || !seen.insert(material).second)
            throw std::invalid_argument("SCI truncation: null or duplicate material in batch");
        validateMaterial(role, expected, *material, player);
        count += material->count;
    }
    if (count != inputs.size() || count == 0 || count > std::numeric_limits<int>::max() / 256)
        throw std::invalid_argument("SCI truncation: grouped batch shape mismatch");
    for (const auto& input : inputs)
        if (input.bitsize != expected.contract.input_bits)
            throw std::invalid_argument("SCI truncation: grouped input ring mismatch");
    for (auto* material : materials) material->consumed = true;
    SciTruncationMaterial combined;
    combined.config = expected;
    combined.role = role;
    combined.count = static_cast<int>(count);
    for (auto* material : materials) {
        append(combined.input_mask_shares, material->input_mask_shares);
        append(combined.mask_high_shares, material->mask_high_shares);
        append(combined.mask_sign_bit_shares, material->mask_sign_bit_shares);
        append(combined.conversion_mask_bits, material->conversion_mask_bits);
        append(combined.conversion_mask_arithmetic, material->conversion_mask_arithmetic);
        append(combined.rounding.a, material->rounding.a);
        append(combined.rounding.b, material->rounding.b);
        append(combined.rounding.c, material->rounding.c);
        appendComparison(combined.low_shift, material->low_shift);
        appendComparison(combined.sign, material->sign);
        appendComparison(combined.wrap, material->wrap);
        combined.metrics.mask_preprocessing_ms += material->metrics.mask_preprocessing_ms;
        combined.metrics.comparisons_preprocessing_ms += material->metrics.comparisons_preprocessing_ms;
        combined.metrics.rounding_preprocessing_ms += material->metrics.rounding_preprocessing_ms;
        releaseSciTruncationMaterial(*material);
    }
    combined.consumed = false;
    auto outputs = sciTruncateOnlineBatch(role, inputs, expected, combined, player);
    if (batch_metrics) *batch_metrics = combined.metrics;
    return outputs;
}

void releaseSciTruncationMaterial(SciTruncationMaterial& material) {
    material.consumed = true;
    material.released = true;
    std::vector<uint64_t>().swap(material.input_mask_shares);
    std::vector<uint64_t>().swap(material.mask_high_shares);
    std::vector<u8>().swap(material.mask_sign_bit_shares);
    std::vector<u8>().swap(material.conversion_mask_bits);
    std::vector<uint64_t>().swap(material.conversion_mask_arithmetic);
    material.rounding = {};
    releaseSegmentedDcfMaterial(material.low_shift);
    releaseSegmentedDcfMaterial(material.sign);
    releaseSegmentedDcfMaterial(material.wrap);
}

}  // namespace segfss
