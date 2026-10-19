#include "dpf_lut.h"

#include <algorithm>
#include <cerrno>
#include <chrono>
#include <cstring>
#include <cstdlib>
#include <iostream>
#include <limits>
#include <random>
#include <string>
#include <stdexcept>
#include <thread>
#include <utility>

#if (defined(__i386__) || defined(__x86_64__)) && (defined(__GNUC__) || defined(__clang__))
#include <immintrin.h>
#define SEGFSS_LUT_HAS_AVX2_TARGET 1
#endif

#include <cryptoTools/Crypto/PRNG.h>

#include "mpc/secure_ops.h"
#include "cggm.h"
#include "gen.h"
#include "activation_metrics.h"
#include "fss/internal/correlated_ggm.h"
#include "fss/internal/ggm.h"
#include "fss/internal/payload_conversion.h"

namespace segfss {
namespace {

using Clock = std::chrono::steady_clock;
uint64_t elapsedUs(Clock::time_point start) {
    return static_cast<uint64_t>(std::chrono::duration_cast<std::chrono::microseconds>(
        Clock::now() - start).count());
}

struct StageSnapshot {
    Clock::time_point start;
    uint64_t sent, received, rounds;
    mutable uint64_t trace_scope = 0;
    explicit StageSnapshot(Peer* channel, int stage = -1) : start(Clock::now()),
        sent(channel->bytesSent), received(channel->bytesReceived), rounds(channel->rounds) {
        if (stage >= 0 && dfss_round::callbacks.scope_begin)
            trace_scope = dfss_round::callbacks.scope_begin("activation_stage", -1, -1, -1, -1, -1, stage);
    }
    void finish() const {
        if (trace_scope && dfss_round::callbacks.scope_end) dfss_round::callbacks.scope_end(trace_scope);
        trace_scope = 0;
    }
    ~StageSnapshot() { finish(); }
    void record(DpfLutStageMetric& stage, Peer* channel) const {
        const auto end = Clock::now();
        const auto begin_us = activationStageTimestampUs(start);
        stage.elapsed_us += std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
        if (!stage.begin_us || begin_us < stage.begin_us) stage.begin_us = begin_us;
        stage.end_us = std::max(stage.end_us, activationStageTimestampUs(end));
        stage.sent_bytes += channel->bytesSent - sent;
        stage.received_bytes += channel->bytesReceived - received;
        stage.channel_rounds += channel->rounds - rounds;
        finish();
    }
};

std::size_t maskB2aWorkingEstimate(std::size_t values) {
    // Input/output, COT correlations/messages/pointers and padded OT scratch.
    return values * (sizeof(u8) + 2 * sizeof(GroupElement) +
                     3 * sizeof(uint64_t) + 2 * sizeof(osuCrypto::block)) +
           (std::size_t(1) << 20);
}

bool lutEnvironmentFlag(const char* name) {
    const char* value = std::getenv(name);
    if (!value || std::strcmp(value, "0") == 0) return false;
    if (std::strcmp(value, "1") == 0) return true;
    throw std::invalid_argument(std::string(name) + " must be 0 or 1");
}

uint32_t lutDotProductScalar(const uint8_t* selection, const uint32_t* coefficients,
                             std::size_t count) {
    uint32_t sum = 0;
    for (std::size_t j = 0; j < count; ++j) {
        const auto* bytes = selection + 4 * j;
        const uint32_t value = uint32_t(bytes[0]) | (uint32_t(bytes[1]) << 8) |
            (uint32_t(bytes[2]) << 16) | (uint32_t(bytes[3]) << 24);
        sum += value * coefficients[j];
    }
    return sum;
}

#if defined(SEGFSS_LUT_HAS_AVX2_TARGET)
__attribute__((target("avx2")))
uint32_t lutDotProductAvx2(const uint8_t* selection, const uint32_t* coefficients,
                          std::size_t count) {
    __m256i sum = _mm256_setzero_si256();
    std::size_t j = 0;
    for (; count - j >= 8; j += 8) {
        const auto values = _mm256_loadu_si256(
            reinterpret_cast<const __m256i*>(selection + 4 * j));
        const auto table = _mm256_loadu_si256(
            reinterpret_cast<const __m256i*>(coefficients + j));
        // Integer vector operations retain exactly the low 32 bits, as does
        // the unsigned scalar arithmetic in the ring2^32 baseline.
        sum = _mm256_add_epi32(sum, _mm256_mullo_epi32(values, table));
    }
    uint32_t lanes[8];
    _mm256_storeu_si256(reinterpret_cast<__m256i*>(lanes), sum);
    uint32_t result = lutDotProductScalar(selection + 4 * j, coefficients + j, count - j);
    for (auto lane : lanes) result += lane;
    return result;
}
#endif

bool cacheLutLeaves() {
    const char* value = std::getenv("SEGFSS_CACHE_LUT_LEAVES");
    if (!value || std::strcmp(value, "0") == 0) return false;
    if (std::strcmp(value, "1") == 0) return true;
    throw std::invalid_argument("SEGFSS_CACHE_LUT_LEAVES must be 0 or 1");
}

uint64_t generationResource(const char* name, uint64_t fallback) {
    const char* value = std::getenv(name);
    if (!value) return fallback;
    char* end = nullptr;
    errno = 0;
    const auto parsed = std::strtoull(value, &end, 10);
    if (errno || !*value || *value == '-' || *end || parsed > (uint64_t(1) << 40))
        throw std::invalid_argument(std::string("invalid LUT generation resource: ") + name);
    return parsed;
}

std::size_t fullQWorkingEstimate(std::size_t count, std::size_t n, int q,
                                std::size_t participant_limit, bool cache) {
    const auto domain = std::size_t(1) << q;
    const auto participants = std::min({n, std::size_t(4), participant_limit});
    return count * q * (sizeof(u8) + sizeof(GroupElement)) +
        n * (domain * 2 * sizeof(osuCrypto::block) + std::size_t(q) * 320 + 512) +
        participants * domain * 2 * sizeof(osuCrypto::block) +
        (cache ? n * domain * sizeof(osuCrypto::block) : 0);
}

struct LutLeafCachePlan {
    std::size_t tile_size = 0, resident_bytes = 0;
    uint64_t budget_bytes = 0, reserve_bytes = 0;
};

LutLeafCachePlan planLutLeafCache(std::size_t count, int q, int output_bits,
                                std::size_t tile_size, std::size_t participant_limit) {
    LutLeafCachePlan plan;
    plan.tile_size = std::min(count, tile_size);
    // Reserve all final selectors, including requests not yet generated. The
    // existing generation reserve covers other contexts and allocator slack.
    const auto domain = std::size_t(1) << q;
    plan.resident_bytes = count * (sizeof(DpfPublicLutMaterial) +
                                  domain * std::size_t((output_bits + 7) / 8));
    plan.budget_bytes = generationResource("SEGFSS_GENERATION_BUDGET_BYTES", 0);
    plan.reserve_bytes = generationResource("SEGFSS_GENERATION_RESERVE_BYTES", uint64_t(512) << 20);
    if (!plan.budget_bytes) return plan;
    if (plan.resident_bytes >= plan.budget_bytes ||
        plan.reserve_bytes >= plan.budget_bytes - plan.resident_bytes)
        throw std::invalid_argument("LUT leaf cache budget cannot reserve resident selectors");
    const auto available = plan.budget_bytes - plan.resident_bytes - plan.reserve_bytes;
    if (maskB2aWorkingEstimate(count * q) > available ||
        fullQWorkingEstimate(count, 1, q, participant_limit, true) > available)
        throw std::invalid_argument("LUT leaf cache budget cannot fit preprocessing scratch");
    std::size_t low = 1, high = plan.tile_size;
    while (low < high) {
        const auto middle = low + (high - low + 1) / 2;
        if (fullQWorkingEstimate(count, middle, q, participant_limit, true) <= available)
            low = middle;
        else high = middle - 1;
    }
    plan.tile_size = low;
    return plan;
}

uint8_t blockLsb(const osuCrypto::block& value) {
    return static_cast<uint8_t>(_mm_cvtsi128_si64x(value) & 1);
}

// Explicit-channel opening. Sending and receiving share the same causal
// parent, so a two-way exchange is one round. The caller receives while one
// helper sends when its caller has two participants. Cap-one callers keep both
// directions on their owner, with opposite public role order to avoid deadlock.
void openAdditive(std::vector<GroupElement>& values, int bits, Peer* player) {
    const std::size_t width = static_cast<std::size_t>((bits + 7) / 8);
    std::vector<u8> outgoing(values.size() * width), incoming(outgoing.size());
    for (std::size_t i = 0; i < values.size(); ++i)
        for (std::size_t b = 0; b < width; ++b)
            outgoing[i * width + b] = static_cast<u8>(values[i].value >> (8 * b));
    if (player->useFile) {
        player->file.write(reinterpret_cast<const char*>(outgoing.data()), outgoing.size());
        player->file.read(reinterpret_cast<char*>(incoming.data()), incoming.size());
        if (!player->file) throw std::runtime_error("DPF LUT file opening failed");
    } else {
        bool send_ok = true, receive_ok = true;
        const auto send_values = [&] {
            player->trace_raw_send(outgoing.size());
            std::size_t done = 0;
            while (done < outgoing.size()) {
                const auto n = ::send(player->sendsocket, outgoing.data() + done,
                                      outgoing.size() - done, MSG_NOSIGNAL);
                if (n < 0 && errno == EINTR) continue;
                if (n <= 0) { send_ok = false; break; }
                done += static_cast<std::size_t>(n);
            }
        };
        const auto receive_values = [&] {
            std::size_t done = 0;
            while (done < incoming.size()) {
                const auto n = ::recv(player->recvsocket, incoming.data() + done,
                                      incoming.size() - done, 0);
                if (n < 0 && errno == EINTR) continue;
                if (n <= 0) { receive_ok = false; break; }
                done += static_cast<std::size_t>(n);
            }
            if (receive_ok) player->trace_raw_received(incoming.size());
        };
        const bool owned = cggmCallerThreadLimit() == 1;
        observeCallerTransportParticipants(owned ? 1 : 2);
        if (owned) {
            if (player->role() == SERVER) { send_values(); receive_values(); }
            else { receive_values(); send_values(); }
        } else {
            const auto parent = dfss_round::fork();
            uint64_t send_tail = 0, receive_tail = 0;
            std::thread sender([&] {
                dfss_round::ActorScope actor(parent);
                send_values(); send_tail = actor.finish();
            });
            {
                dfss_round::ActorScope actor(parent);
                receive_values(); receive_tail = actor.finish();
            }
            sender.join();
            dfss_round::join(send_tail);
            dfss_round::join(receive_tail);
        }
        if (!send_ok || !receive_ok)
            throw std::runtime_error("DPF LUT socket opening failed");
    }
    player->bytesSent += outgoing.size();
    player->bytesReceived += incoming.size();
    player->rounds += 2;
    for (std::size_t i = 0; i < values.size(); ++i) {
        uint64_t remote = 0;
        for (std::size_t b = 0; b < width; ++b)
            remote |= uint64_t(incoming[i * width + b]) << (8 * b);
        values[i] = values[i] + GroupElement(remote, bits);
    }
}

// Local full-domain tree expansion; there is no socket use here. It uses the
// same roots and CW format as the arithmetic correlated-DPF reference.
void expandCggmLeaves(const CggmKey& key,
                     std::vector<osuCrypto::block>& nodes,
                     std::vector<osuCrypto::block>& next) {
    nodes.resize(1);
    nodes[0] = key.root;
    for (int level = 0; level < key.depth - 1; ++level) {
        next.resize(nodes.size() * 2);
        for (std::size_t j = 0; j < nodes.size(); ++j) {
            const auto parent = nodes[j];
            const auto correction = blockLsb(parent)
                ? key.internal_cw[level] : osuCrypto::ZeroBlock;
            const auto left = dfss::internal::ccrHash(parent) ^ correction;
            next[2 * j] = left;
            next[2 * j + 1] = left ^ parent;
        }
        nodes.swap(next);
    }
    next.resize(nodes.size() * 2);
    for (std::size_t j = 0; j < nodes.size(); ++j) {
        for (int direction = 0; direction != 2; ++direction) {
            const auto correction = dfss::internal::setBlockLsb(
                key.hcw, key.lcw[direction]);
            const auto leaf = dfss::internal::ccrHash(nodes[j] ^
                (direction ? osuCrypto::OneBlock : osuCrypto::ZeroBlock));
            next[2 * j + direction] = leaf ^
                (blockLsb(nodes[j]) ? correction : osuCrypto::ZeroBlock);
        }
    }
    nodes.swap(next);
}

template <typename Function>
std::size_t parallelForRanges(std::size_t count, std::size_t participant_limit,
                              std::size_t work_per_item,
                              Function&& function) {
    constexpr std::size_t kMaxMaterializeParticipants = 4;
    const std::size_t participants =
        std::min({count, kMaxMaterializeParticipants,
                  std::max<std::size_t>(1, participant_limit)});
    const std::size_t work_items =
        count > std::numeric_limits<std::size_t>::max() / work_per_item
            ? std::numeric_limits<std::size_t>::max()
            : count * work_per_item;
    return cggmParallelForRanges(count, work_items, participants, function);
}

uint64_t maskForBits(int bits) {
    return bits == 64 ? UINT64_MAX : ((uint64_t(1) << bits) - 1);
}

void validateTable(const PublicLUTData& table) {
    if (table.Bin <= 0 || table.Bin > 24 || table.Bout <= 0 ||
        table.Bout > 62 || table.values.size() != (size_t(1) << table.Bin)) {
        throw std::invalid_argument("DPF public LUT has an invalid shape");
    }
    for (const auto& value : table.values) {
        if (value.bitsize != table.Bout) {
            throw std::invalid_argument("DPF public LUT output width mismatch");
        }
    }
}

void validateMaterial(int party_id, const GroupElement& input,
                      const DpfPublicLutMaterial& material, Peer* player,
                      uint64_t expected_id) {
    if ((party_id != SERVER && party_id != CLIENT) || player == nullptr ||
        input.bitsize != material.index_bits ||
        material.index_bits <= 0 || material.index_bits > 24 ||
        material.output_bits <= 0 ||
        material.request_id != expected_id || material.consumed ||
        material.released || material.key_group_size != material.index_bits ||
        material.selection_share.size() != (size_t(1) << material.index_bits) ||
        material.mask_share.bitsize != material.index_bits ||
        material.output_bits > 62) {
        throw std::invalid_argument("DPF public LUT material is invalid or consumed");
    }
    if (material.selection_share.bitsize() != material.output_bits ||
        material.generation_epoch == 0) {
        throw std::invalid_argument("DPF public LUT resident shape or epoch mismatch");
    }
}

}  // namespace

DpfLutOptions::DpfLutOptions()
    : skip_zero_coefficients(lutEnvironmentFlag("SEGFSS_LUT_SKIP_ZERO")),
      use_simd(lutEnvironmentFlag("SEGFSS_LUT_SIMD")) {}

bool dpfLutSimdAvailable() noexcept {
#if defined(SEGFSS_LUT_HAS_AVX2_TARGET)
    static const bool available = [] {
        __builtin_cpu_init();
        return bool(__builtin_cpu_supports("avx2"));
    }();
    return available;
#else
    return false;
#endif
}

bool DpfLutBinding::operator==(const DpfLutBinding& other) const noexcept {
    return context_id == other.context_id && query_id == other.query_id &&
        operator_id == other.operator_id && worker_id == other.worker_id &&
        slot_id == other.slot_id && element_begin == other.element_begin &&
        table_version == other.table_version && family == other.family;
}

PreparedDpfLutTable::PreparedDpfLutTable(const PublicLUTData& table, uint64_t version)
    : table_(table), version_(version) {
    validateTable(table);
    if (table.Bin != 10 || table.Bout != 32 || version == 0)
        throw std::invalid_argument("paper LUT requires q10, ring32 and a table version");
    coefficients_.reserve(table.values.size());
    fingerprint_ = 1469598103934665603ULL;
    auto mix = [&](uint64_t value) {
        for (int byte = 0; byte < 8; ++byte) {
            fingerprint_ ^= static_cast<uint8_t>(value >> (8 * byte));
            fingerprint_ *= 1099511628211ULL;
        }
    };
    mix(table.Bin); mix(table.Bout); mix(version);
    for (const auto& value : table.values) {
        coefficients_.push_back(static_cast<uint32_t>(value.value));
        mix(value.value);
    }
    for (std::size_t begin = 0; begin < coefficients_.size();) {
        if (coefficients_[begin] == 0) { ++begin; continue; }
        std::size_t end = begin + 1;
        while (end < coefficients_.size() && coefficients_[end] != 0) ++end;
        nonzero_ranges_.emplace_back(begin, end);
        nonzero_count_ += end - begin;
        begin = end;
    }
}

std::shared_ptr<const PreparedDpfLutTable> prepareDpfLutTable(
    const PublicLUTData& table, uint64_t version) {
    return std::make_shared<const PreparedDpfLutTable>(table, version);
}

PackedDpfSelectionVector::PackedDpfSelectionVector(PackedDpfSelectionVector&& other) noexcept {
    *this = std::move(other);
}

PackedDpfSelectionVector& PackedDpfSelectionVector::operator=(
    PackedDpfSelectionVector&& other) noexcept {
    if (this == &other) return *this;
    bytes_ = std::move(other.bytes_);
    count_ = std::exchange(other.count_, 0);
    output_bits_ = std::exchange(other.output_bits_, 0);
    other.bytes_.clear();
    return *this;
}

void PackedDpfSelectionVector::reset(std::size_t count, int output_bits) {
    if (output_bits <= 0 || output_bits > 62 ||
        count > SIZE_MAX / static_cast<std::size_t>((output_bits + 7) / 8))
        throw std::invalid_argument("invalid packed DPF selection shape");
    output_bits_ = output_bits;
    count_ = count;
    bytes_.assign(count * static_cast<std::size_t>((output_bits + 7) / 8), 0);
}

void PackedDpfSelectionVector::set(std::size_t index, uint64_t value) {
    if (index >= count_) throw std::out_of_range("packed DPF selection index");
    const std::size_t width = static_cast<std::size_t>((output_bits_ + 7) / 8);
    value &= maskForBits(output_bits_);
    for (std::size_t byte = 0; byte < width; ++byte)
        bytes_[index * width + byte] = static_cast<uint8_t>(value >> (8 * byte));
}

GroupElement PackedDpfSelectionVector::operator[](std::size_t index) const {
    if (index >= count_) throw std::out_of_range("packed DPF selection index");
    const std::size_t width = static_cast<std::size_t>((output_bits_ + 7) / 8);
    uint64_t value = 0;
    for (std::size_t byte = 0; byte < width; ++byte)
        value |= uint64_t(bytes_[index * width + byte]) << (8 * byte);
    return GroupElement(value, output_bits_);
}

uint32_t PackedDpfSelectionVector::shiftedDotProduct32(
    const std::vector<uint32_t>& coefficients, std::size_t shift) const {
    if (output_bits_ != 32 || coefficients.size() != count_ || shift >= count_)
        throw std::invalid_argument("packed32 LUT inner-product shape mismatch");
    uint32_t sum = 0;
    auto accumulate = [&](std::size_t begin, std::size_t length,
                          const uint32_t* table) {
        for (std::size_t j = 0; j < length; ++j) {
            const auto* bytes = bytes_.data() + 4 * (begin + j);
            const uint32_t value = uint32_t(bytes[0]) | (uint32_t(bytes[1]) << 8) |
                (uint32_t(bytes[2]) << 16) | (uint32_t(bytes[3]) << 24);
            sum += value * table[j];
        }
    };
    // Two contiguous public views implement the rotation without copying a
    // table, accessing secret-dependent memory, or recomputing each modulus.
    accumulate(0, count_ - shift, coefficients.data() + shift);
    accumulate(count_ - shift, shift, coefficients.data());
    return sum;
}

uint32_t PackedDpfSelectionVector::shiftedDotProduct32(
    const PreparedDpfLutTable& table, std::size_t shift, bool skip_zero, bool use_simd) const {
    const auto& coefficients = table.coefficients();
    if (output_bits_ != 32 || coefficients.size() != count_ || shift >= count_)
        throw std::invalid_argument("packed32 LUT inner-product shape mismatch");
    if (!skip_zero && !use_simd) return shiftedDotProduct32(coefficients, shift);
    using DotProduct = uint32_t (*)(const uint8_t*, const uint32_t*, std::size_t);
    DotProduct dot_product = lutDotProductScalar;
#if defined(SEGFSS_LUT_HAS_AVX2_TARGET)
    if (use_simd && dpfLutSimdAvailable()) dot_product = lutDotProductAvx2;
#endif
    uint32_t sum = 0;
    auto accumulate_range = [&](std::size_t begin, std::size_t end) {
        // The table and the already-opened masked shift are public. Keep the
        // complete domain when aligning a support interval with the selector.
        const std::size_t selector_begin = (begin + count_ - shift) % count_;
        const std::size_t first = std::min(end - begin, count_ - selector_begin);
        sum += dot_product(bytes_.data() + 4 * selector_begin,
                           coefficients.data() + begin, first);
        if (first < end - begin)
            sum += dot_product(bytes_.data(), coefficients.data() + begin + first,
                               end - begin - first);
    };
    if (skip_zero) {
        for (const auto& range : table.nonzeroRanges())
            accumulate_range(range.first, range.second);
    } else accumulate_range(0, count_);
    return sum;
}

void PackedDpfSelectionVector::assign(
    const std::vector<GroupElement>& values, int output_bits) {
    reset(values.size(), output_bits);
    for (std::size_t i = 0; i < values.size(); ++i) {
        if (values[i].bitsize != output_bits)
            throw std::invalid_argument("packed DPF selection ring mismatch");
        set(i, values[i].value);
    }
}

std::size_t DpfPublicLutMaterial::residentBytes() const noexcept {
    return selection_share.capacityBytes() +
           sizeof(*this);
}

bool DpfPublicLutMaterial::hasValidPaperLayout() const noexcept {
    return (family == DpfLutKeyFamily::full_q_arithmetic && lut_segments == 1 &&
            segment_bits == std::array<int, 2>{{10, 0}}) ||
           (family == DpfLutKeyFamily::segmented_boolean_tensor_b2a && lut_segments == 2 &&
            segment_bits == std::array<int, 2>{{5, 5}});
}

DpfPublicLutMaterial::DpfPublicLutMaterial(DpfPublicLutMaterial&& other) noexcept {
    *this = std::move(other);
}

DpfPublicLutMaterial& DpfPublicLutMaterial::operator=(DpfPublicLutMaterial&& other) noexcept {
    if (this == &other) return *this;
    index_bits = other.index_bits;
    output_bits = other.output_bits;
    key_group_size = other.key_group_size;
    request_id = other.request_id;
    generation_epoch = other.generation_epoch;
    owner_peer = other.owner_peer;
    binding = other.binding;
    family = other.family;
    lut_segments = other.lut_segments;
    segment_bits = other.segment_bits;
    table_fingerprint = other.table_fingerprint;
    paper_lookup = other.paper_lookup;
    lut_skip_zero = other.lut_skip_zero;
    lut_simd = other.lut_simd;
    mask_share = other.mask_share;
    selection_share = std::move(other.selection_share);
    key = std::move(other.key);
    consumed = other.consumed;
    released = other.released;
    // Moving transfers the one-use inventory, including its mask and key.
    other.selection_share.clear();
    other.key = DPFKeyPack();
    other.owner_peer = nullptr;
    other.consumed = true;
    other.released = true;
    return *this;
}

DpfPublicLutMaterial dpfPublicLutOffline(
    int party_id, const PublicLUTData& table, uint64_t request_id,
    Peer* player, uint64_t generation_epoch) {
    validateTable(table);
    if ((party_id != SERVER && party_id != CLIENT) || player == nullptr ||
        player != peer || generation_epoch == 0) {
        throw std::invalid_argument("invalid DPF public LUT party or epoch");
    }

    const int q = table.Bin;
    auto& rng = player->randomSource();
    // alpha_bits are XOR shares.  Each party samples its own share of the
    // random target; key generation reconstructs their XOR without either
    // party learning the target.  Sampling both shares is important here:
    // assigning the whole mask to SERVER would make the material lifecycle
    // asymmetric and would not satisfy the dealerless preprocessing contract.
    std::vector<u8> mask_bits(static_cast<size_t>(q), 0);
    for (auto& bit : mask_bits)
        bit = static_cast<u8>(rng.get<uint8_t>() & 1U);

    // Convert independent XOR shares of r's bits to an arithmetic share in
    // Z_(2^q).  This is deliberately part of preparation, before input load.
    std::vector<GroupElement> bit_shares(mask_bits.size(), GroupElement(0, q));
    B2A(party_id, mask_bits.data(), bit_shares.data(), q, q, player);
    GroupElement mask_share(0, q);
    for (int bit = 0; bit < q; ++bit)
        mask_share = mask_share + bit_shares[static_cast<size_t>(bit)] *
            (uint64_t(1) << bit);

    // A fixed public split (1, 0) is a valid additive share of the one-hot
    // payload.  The DPF tree and random mask still hide the target; the
    // resulting rotated table shares are independently pseudorandom.
    const GroupElement beta(party_id == SERVER ? 1 : 0, table.Bout);
    std::vector<u8> target_bits(static_cast<std::size_t>(q));
    for (int bit = 0; bit < q; ++bit)
        target_bits[static_cast<std::size_t>(bit)] =
            mask_bits[static_cast<std::size_t>(q - 1 - bit)];
    auto key = dfss::keyGenCorrelatedDPF(
        party_id, q, table.Bout, target_bits.data(), beta);
    std::vector<GroupElement> one_hot(size_t(1) << q,
                                      GroupElement(0, table.Bout));
    dfss::evalAllCorrelatedDPF(party_id, one_hot.data(), key, q);

    DpfPublicLutMaterial result;
    result.index_bits = q;
    result.output_bits = table.Bout;
    result.key_group_size = q;
    result.request_id = request_id;
    result.generation_epoch = generation_epoch;
    result.mask_share = mask_share;
    result.key = std::move(key);
    result.selection_share.assign(one_hot, table.Bout);
    // The one-hot vector is the resident material.  The DPF key has served its
    // purpose and is released before input arrival.
    freeDPFKeyPack(result.key);
    return result;
}

static std::vector<DpfPublicLutMaterial> fullQBatchOffline(
    int party_id, const PublicLUTData& table, std::size_t count,
    uint64_t request_id_base, Peer* player, uint64_t generation_epoch,
    DpfLutBatchMetrics* metrics, std::size_t tile_size,
    std::size_t participant_limit) {
    validateTable(table);
    const int q = table.Bin;
    if ((party_id != SERVER && party_id != CLIENT) || player == nullptr ||
        generation_epoch == 0 || count == 0 || tile_size == 0 ||
        count > static_cast<std::size_t>(std::numeric_limits<int>::max() / q) ||
        tile_size > static_cast<std::size_t>(std::numeric_limits<int>::max() / (2 * q)) ||
        count - 1 > UINT64_MAX - request_id_base) {
        throw std::invalid_argument("invalid DPF public LUT batch shape or identity");
    }
    const bool cache_leaves = cacheLutLeaves();
    LutLeafCachePlan cache_plan;
    if (cache_leaves) {
        cache_plan = planLutLeafCache(count, q, table.Bout, tile_size, participant_limit);
        tile_size = cache_plan.tile_size;
    }
    DpfLutBatchMetrics accounting;
    accounting.generation_epoch = generation_epoch;
    accounting.key_trees = count;
    accounting.tree_max_depth = q;
    accounting.mask_b2a_values = count * q;
    const auto start = Clock::now();
    const auto sent_before = player->bytesSent, received_before = player->bytesReceived;
    const std::size_t domain = std::size_t(1) << q;
    auto& rng = player->randomSource();
    std::vector<u8> mask_bits(count * static_cast<std::size_t>(q));
    rng.get(mask_bits.data(), mask_bits.size());
    for (u8& bit : mask_bits) bit &= 1U;
    std::vector<GroupElement> bit_shares(mask_bits.size(), GroupElement(0, q));
    const auto conversion_start = Clock::now();
    const StageSnapshot conversion_stage(player, int(ActivationStage::lut_mask));
    B2A(party_id, mask_bits.data(), bit_shares.data(),
        static_cast<int>(bit_shares.size()), q, player);
    accounting.mask_conversion_us = elapsedUs(conversion_start);
    conversion_stage.record(accounting.mask_stage, player);
    accounting.peak_working_bytes = maskB2aWorkingEstimate(mask_bits.capacity());

    std::vector<DpfPublicLutMaterial> result(count);
    for (std::size_t i = 0; i < count; ++i) {
        auto& material = result[i];
        material.index_bits = q;
        material.output_bits = table.Bout;
        material.key_group_size = q;
        material.request_id = request_id_base + i;
        material.generation_epoch = generation_epoch;
        material.owner_peer = player;
        material.mask_share = GroupElement(0, q);
        for (int bit = 0; bit < q; ++bit)
            material.mask_share = material.mask_share +
                bit_shares[i * static_cast<std::size_t>(q) + bit] * (uint64_t(1) << bit);
    }
    // The arithmetic DPF differs from a Boolean cGGM only in its payload
    // correction. Generate all independent trees in a tile at matching depth,
    // adapt their payloads in one batched COT, and open all payload CWs once.
    // There is no interactive single-key call in this loop.
    for (std::size_t begin = 0; begin < count; begin += tile_size) {
        const std::size_t n = std::min(tile_size, count - begin);
        std::vector<std::vector<u8>> targets(n, std::vector<u8>(q));
        for (std::size_t j = 0; j < n; ++j)
            for (int bit = 0; bit < q; ++bit)
                targets[j][bit] = mask_bits[(begin + j) * q + q - 1 - bit];
        const auto key_start = Clock::now();
        const StageSnapshot tree_stage(player, int(ActivationStage::lut_keygen));
        auto keys = cggmGen2pcBatch(party_id, targets, nullptr, nullptr,
                                  player, participant_limit);
        accounting.keygen_us += elapsedUs(key_start);
        tree_stage.record(accounting.tree_keygen_stage, player);
        // The cache belongs to this tile and is destroyed before the next
        // tile's key generation. It never becomes resident query material.
        std::vector<osuCrypto::block> leaf_cache(cache_leaves ? n * domain : 0);
        accounting.leaf_cache_peak_bytes = std::max(accounting.leaf_cache_peak_bytes,
            leaf_cache.capacity() * sizeof(osuCrypto::block));
        std::vector<u8> choices(n);
        std::vector<GroupElement> data0(n), data1(n), corrections(n, GroupElement(0, table.Bout));
        const auto sum_start = Clock::now();
        const StageSnapshot leaf_stage(player, int(ActivationStage::lut_materialize));
        const std::size_t participants = parallelForRanges(
            n, participant_limit, domain,
            [&](std::size_t range_begin, std::size_t range_end) {
            std::vector<osuCrypto::block> nodes, next;
            for (std::size_t j = range_begin; j < range_end; ++j) {
                expandCggmLeaves(keys[j], nodes, next);
                if (cache_leaves)
                    std::copy(nodes.begin(), nodes.end(), leaf_cache.begin() + j * domain);
                GroupElement converted_sum(0, table.Bout);
                uint64_t control_sum = 0;
                for (const auto& node : nodes) {
                    converted_sum = converted_sum +
                        dfss::internal::convertBlockToGroup(table.Bout, node);
                    control_sum += blockLsb(node);
                }
                choices[j] = dfss::internal::dpfPayloadChoiceBit(
                    party_id, control_sum, 1);
                const GroupElement beta(party_id == SERVER ? 1 : 0, table.Bout);
                const GroupElement sign(party_id == CLIENT ? 1 : -1, table.Bout);
                data0[j] = beta + converted_sum * sign;
                data1[j] = -beta + converted_sum * (-sign);
            }
        });
        accounting.eval_all_us += elapsedUs(sum_start);
        leaf_stage.record(accounting.materialize_stage, player);
        const auto payload_start = Clock::now();
        const StageSnapshot payload_stage(player, int(ActivationStage::lut_payload));
        // The legacy pointer-selection overload serializes one OT per item.
        // The vector-selection overload preserves independent choices and
        // sends each direction's ring correlations as one whole array.
        std::vector<u8> real_choices(n);
        for (std::size_t j = 0; j < n; ++j) {
            real_choices[j] = choices[j] ^ static_cast<u8>(party_id - SERVER);
            data0[j] = data0[j] - data1[j];
        }
        multiplexer(party_id, real_choices, data0.data(), corrections.data(),
                    static_cast<int32_t>(n), player);
        for (std::size_t j = 0; j < n; ++j)
            corrections[j] = corrections[j] + data1[j];
        openAdditive(corrections, table.Bout, player);
        accounting.keygen_us += elapsedUs(payload_start);
        payload_stage.record(accounting.payload_stage, player);
        const auto eval_start = Clock::now();
        const StageSnapshot selection_stage(player, int(ActivationStage::lut_materialize));
        parallelForRanges(n, participant_limit, domain,
            [&](std::size_t range_begin, std::size_t range_end) {
            std::vector<osuCrypto::block> nodes, next;
            for (std::size_t j = range_begin; j < range_end; ++j) {
                if (!cache_leaves) expandCggmLeaves(keys[j], nodes, next);
                const auto* leaves = cache_leaves ? leaf_cache.data() + j * domain : nodes.data();
                auto& material = result[begin + j];
                material.selection_share.reset(domain, table.Bout);
                const GroupElement sign(party_id == CLIENT ? -1 : 1, table.Bout);
                for (std::size_t r = 0; r < domain; ++r) {
                    const auto share = (corrections[j] *
                        static_cast<uint64_t>(blockLsb(leaves[r])) +
                        dfss::internal::convertBlockToGroup(table.Bout, leaves[r])) * sign;
                    material.selection_share.set(r, share.value);
                }
            }
        });
        for (std::size_t j = 0; j < n; ++j)
            accounting.resident_bytes += result[begin + j].residentBytes();
        accounting.eval_all_us += elapsedUs(eval_start);
        selection_stage.record(accounting.materialize_stage, player);
        ++accounting.tiles;
        const auto expansions = n * (cache_leaves ? 1 : 2);
        accounting.local_tree_expansions += expansions;
        accounting.local_internal_hashes += expansions * (domain / 2 - 1);
        accounting.local_leaf_hashes += expansions * domain;
        // Conservative allocation estimate for batched cGGM nodes/hash
        // scratch, payload/OT vectors, and reusable local tree expansion.
        // It is explicitly separate from externally measured RSS/AS.
        const std::size_t working = mask_bits.capacity() * sizeof(u8) +
            bit_shares.capacity() * sizeof(GroupElement) +
            n * (domain * 2 * sizeof(osuCrypto::block) +
                 static_cast<std::size_t>(q) * 320 + 512) +
            participants * domain * 2 * sizeof(osuCrypto::block) +
            leaf_cache.capacity() * sizeof(osuCrypto::block);
        accounting.peak_working_bytes = std::max(accounting.peak_working_bytes, working);
        if (cache_leaves && cache_plan.budget_bytes &&
            accounting.peak_working_bytes > cache_plan.budget_bytes -
                cache_plan.resident_bytes - cache_plan.reserve_bytes)
            throw std::runtime_error("LUT leaf cache exceeded its generation working estimate");
    }
    accounting.generated = accounting.remaining = count;
    accounting.eval_points = count * domain;
    accounting.offline_sent_bytes = player->bytesSent - sent_before;
    accounting.offline_received_bytes = player->bytesReceived - received_before;
    accounting.offline_us = elapsedUs(start);
    if (cache_leaves)
        std::cout << "LUT_LEAF_CACHE {\"count\":" << count << ",\"domain\":" << domain
            << ",\"tile_size\":" << tile_size << ",\"tiles\":" << accounting.tiles
            << ",\"cache_peak_bytes\":" << accounting.leaf_cache_peak_bytes
            << ",\"local_tree_expansions\":" << accounting.local_tree_expansions
            << ",\"local_internal_hashes\":" << accounting.local_internal_hashes
            << ",\"local_leaf_hashes\":" << accounting.local_leaf_hashes
            << ",\"budget_bytes\":" << cache_plan.budget_bytes
            << ",\"reserve_bytes\":" << cache_plan.reserve_bytes
            << ",\"resident_estimate_bytes\":" << cache_plan.resident_bytes
            << ",\"peak_working_bytes\":" << accounting.peak_working_bytes << "}\n";
    if (metrics) *metrics = accounting;
    return result;
}

std::vector<DpfPublicLutMaterial> dpfPublicLutBatchOffline(
    int party_id, const PublicLUTData& table, std::size_t count,
    uint64_t request_id_base, Peer* player, uint64_t generation_epoch,
    DpfLutBatchMetrics* metrics, std::size_t tile_size) {
    return fullQBatchOffline(party_id, table, count, request_id_base, player,
                            generation_epoch, metrics, tile_size, cggmThreadLimit());
}

static std::vector<DpfPublicLutMaterial> segmentedBatchOffline(
    int party_id, std::size_t count, const DpfLutBinding& binding,
    const DpfLutOptions& options, Peer* player, uint64_t generation_epoch,
    DpfLutBatchMetrics* metrics) {
    constexpr int q = 10, bits = 32, width = 5;
    constexpr std::size_t domain = 1024, digit_domain = 32;
    if (generation_epoch == 0 || count == 0 ||
        count > static_cast<std::size_t>(std::numeric_limits<int>::max() / q) ||
        std::min(count, options.tile_size) >
            static_cast<std::size_t>(std::numeric_limits<int>::max() / (2 * domain)) ||
        count - 1 > UINT64_MAX - binding.element_begin)
        throw std::invalid_argument("segmented paper LUT: invalid batch or tile");

    DpfLutBatchMetrics accounting;
    accounting.generation_epoch = generation_epoch;
    accounting.key_trees = 2 * count;
    accounting.tree_max_depth = width;
    accounting.mask_b2a_values = count * q;
    accounting.tensor_and_gates = accounting.leaf_b2a_values = count * domain;
    const auto start = Clock::now();
    const auto sent_before = player->bytesSent, received_before = player->bytesReceived;
    std::vector<u8> mask_bits(count * q);
    player->randomSource().get(mask_bits.data(), mask_bits.size());
    for (auto& bit : mask_bits) bit &= 1U;
    std::vector<GroupElement> arithmetic_bits(mask_bits.size(), GroupElement(0, q));
    const StageSnapshot mask_stage(player, int(ActivationStage::lut_mask));
    B2A(party_id, mask_bits.data(), arithmetic_bits.data(),
        static_cast<int>(mask_bits.size()), q, player);
    mask_stage.record(accounting.mask_stage, player);
    accounting.peak_working_bytes = maskB2aWorkingEstimate(mask_bits.capacity());
    accounting.mask_conversion_us = accounting.mask_stage.elapsed_us;
    std::vector<DpfPublicLutMaterial> result(count);
    for (std::size_t i = 0; i < count; ++i) {
        auto& material = result[i];
        material.index_bits = q;
        material.output_bits = bits;
        material.key_group_size = q;
        material.request_id = binding.element_begin + i;
        material.generation_epoch = generation_epoch;
        material.owner_peer = player;
        material.lut_segments = 2;
        material.segment_bits = {{width, width}};
        material.mask_share = GroupElement(0, q);
        for (int bit = 0; bit < q; ++bit)
            material.mask_share = material.mask_share +
                arithmetic_bits[i * q + bit] * (uint64_t(1) << bit);
    }

    for (std::size_t begin = 0; begin < count; begin += options.tile_size) {
        const auto n = std::min(options.tile_size, count - begin);
        const auto entries = n * domain;
        std::vector<std::vector<u8>> targets(2 * n, std::vector<u8>(width));
        for (std::size_t i = 0; i < n; ++i)
            for (int segment = 0; segment < 2; ++segment)
                for (int bit = 0; bit < width; ++bit)
                    targets[2 * i + segment][bit] = mask_bits[
                        (begin + i) * q + (1 - segment) * width + width - 1 - bit];
        const StageSnapshot tree_stage(player, int(ActivationStage::lut_keygen));
        CggmGenerationStats generation_stats;
        auto keys = cggmGen2pcBatch(party_id, targets, nullptr, &generation_stats,
                                  player, options.participant_limit);
        tree_stage.record(accounting.tree_keygen_stage, player);
        accounting.keygen_us = accounting.tree_keygen_stage.elapsed_us;

        std::vector<u8> lhs(entries), rhs(entries), equality(entries);
        std::vector<osuCrypto::block> nodes, next;
        const StageSnapshot expand_stage(player, int(ActivationStage::lut_materialize));
        for (std::size_t i = 0; i < n; ++i) {
            std::array<u8, digit_domain> high, low;
            expandCggmLeaves(keys[2 * i], nodes, next);
            for (std::size_t digit = 0; digit < digit_domain; ++digit)
                high[digit] = blockLsb(nodes[digit]);
            expandCggmLeaves(keys[2 * i + 1], nodes, next);
            for (std::size_t digit = 0; digit < digit_domain; ++digit)
                low[digit] = blockLsb(nodes[digit]);
            for (std::size_t u = 0; u < domain; ++u) {
                lhs[i * domain + u] = high[u >> width];
                rhs[i * domain + u] = low[u & (digit_domain - 1)];
            }
        }
        expand_stage.record(accounting.materialize_stage, player);
        keys.clear(); keys.shrink_to_fit();
        targets.clear(); targets.shrink_to_fit();
        // Two leaves form a one-level balanced tensor tree. Every point uses
        // a fresh Boolean triple; this is not local share multiplication.
        const StageSnapshot tensor_stage(player, int(ActivationStage::lut_tensor_and));
        {
            auto tensor_material = booleanAndOffline(party_id, static_cast<int>(entries), player);
            BooleanAndWorkspace tensor_workspace;
            booleanAndOnlineBatch(party_id,
                {{lhs.data(), rhs.data(), equality.data(), static_cast<int>(entries), &tensor_material}},
                player, &tensor_workspace);
        }
        tensor_stage.record(accounting.tensor_and_stage, player);
        lhs.clear(); lhs.shrink_to_fit();
        rhs.clear(); rhs.shrink_to_fit();

        std::vector<GroupElement> arithmetic(entries, GroupElement(0, bits));
        const StageSnapshot b2a_stage(player, int(ActivationStage::lut_leaf_b2a));
        B2A(party_id, equality.data(), arithmetic.data(), static_cast<int>(entries), bits, player);
        b2a_stage.record(accounting.leaf_b2a_stage, player);
        const StageSnapshot pack_stage(player, int(ActivationStage::lut_materialize));
        for (std::size_t i = 0; i < n; ++i) {
            auto& material = result[begin + i];
            material.selection_share.reset(domain, bits);
            for (std::size_t u = 0; u < domain; ++u)
                material.selection_share.set(u, arithmetic[i * domain + u].value);
            accounting.resident_bytes += material.residentBytes();
        }
        pack_stage.record(accounting.materialize_stage, player);
        ++accounting.tiles;
        // Conservative bounded-tile allocation estimate, including secure
        // AND/B2A OT buffers. RSS/AS snapshots and lifetime peak RSS are separate.
        const std::size_t working = mask_bits.capacity() * sizeof(u8) +
            arithmetic_bits.capacity() * sizeof(GroupElement) +
            n * (2 * digit_domain * 2 * sizeof(osuCrypto::block) + 2 * width * 320 + 512) +
            generation_stats.scratch_peak_bytes + generation_stats.workspace_capacity_bytes +
            entries * (8 * sizeof(u8) + 2 * sizeof(GroupElement) +
                       6 * sizeof(uint64_t) + 2 * sizeof(osuCrypto::block)) +
            (nodes.capacity() + next.capacity()) * sizeof(osuCrypto::block);
        accounting.peak_working_bytes = std::max(accounting.peak_working_bytes, working);
    }
    accounting.eval_all_us = accounting.materialize_stage.elapsed_us;
    accounting.generated = accounting.remaining = count;
    accounting.eval_points = count * 2 * digit_domain;
    accounting.offline_sent_bytes = player->bytesSent - sent_before;
    accounting.offline_received_bytes = player->bytesReceived - received_before;
    accounting.offline_us = elapsedUs(start);
    if (metrics) *metrics = accounting;
    return result;
}

std::vector<DpfPublicLutMaterial> dpfPublicLutBatchOffline(
    int party_id, const PreparedDpfLutTable& table, std::size_t count,
    const DpfLutBinding& binding, const DpfLutOptions& options, Peer* player,
    uint64_t generation_epoch, DpfLutBatchMetrics* metrics) {
    if (binding.context_id == 0 || binding.table_version != table.version() ||
        binding.family != options.family ||
        (options.family != DpfLutKeyFamily::full_q_arithmetic &&
         options.family != DpfLutKeyFamily::segmented_boolean_tensor_b2a) ||
        options.participant_limit == 0 || options.participant_limit > 4 ||
        options.tile_size == 0 || !player || player->role() != party_id)
        throw std::invalid_argument("paper LUT: unsupported family, binding or context");
    const std::vector<int> canonical = options.family == DpfLutKeyFamily::full_q_arithmetic
        ? std::vector<int>{10} : std::vector<int>{5, 5};
    if (!options.segment_widths.empty() && options.segment_widths != canonical)
        throw std::invalid_argument("paper LUT: segment layout does not match family");
    auto materials = options.family == DpfLutKeyFamily::full_q_arithmetic
        ? fullQBatchOffline(party_id, table.table(), count, binding.element_begin,
            player, generation_epoch, metrics, options.tile_size, options.participant_limit)
        : segmentedBatchOffline(party_id, count, binding, options, player, generation_epoch, metrics);
    for (auto& material : materials) {
        material.binding = binding;
        material.family = options.family;
        material.table_fingerprint = table.fingerprint();
        material.paper_lookup = true;
        material.lut_skip_zero = options.skip_zero_coefficients;
        material.lut_simd = options.use_simd;
    }
    return materials;
}

namespace {
std::vector<std::vector<GroupElement>> onlineSelectedTables(
    int party_id, const std::vector<GroupElement>& inputs,
    std::vector<DpfPublicLutMaterial>& materials, Peer* player,
    uint64_t request_id_base, const std::vector<const PublicLUTData*>& tables,
    DpfLutBatchMetrics* metrics,
    const std::vector<std::size_t>* table_for_input,
    const DpfLutBinding* binding = nullptr,
    const PreparedDpfLutTable* prepared_table = nullptr) {
    auto invalidateBatch = [&] {
        for (auto& material : materials) material.consumed = true;
        if (metrics) {
            metrics->consumed = materials.size();
            metrics->remaining = 0;
        }
    };
    try {
        if (tables.empty() || inputs.empty() || inputs.size() != materials.size() ||
            inputs.size() > static_cast<std::size_t>(std::numeric_limits<int32_t>::max()) ||
            inputs.size() - 1 > UINT64_MAX - request_id_base)
            throw std::invalid_argument("DPF public LUT batch size or identity mismatch");
        if (table_for_input) {
            if (table_for_input->size() != inputs.size())
                throw std::invalid_argument("DPF public LUT per-input table count mismatch");
            for (const auto t : *table_for_input)
                if (t >= tables.size())
                    throw std::invalid_argument("DPF public LUT per-input table index invalid");
        }
        int output_bits = 0;
        for (const auto* table : tables) {
            if (!table) throw std::invalid_argument("DPF public LUT null output table");
            validateTable(*table);
            if (table->Bin != materials.front().index_bits)
                throw std::invalid_argument("DPF public LUT output domains differ");
            output_bits = std::max(output_bits, table->Bout);
        }
        const auto& first = materials.front();
        for (std::size_t i = 0; i < inputs.size(); ++i) {
            validateMaterial(party_id, inputs[i], materials[i], player, request_id_base + i);
            if (materials[i].output_bits != output_bits ||
                materials[i].index_bits != first.index_bits ||
                materials[i].generation_epoch != first.generation_epoch)
                throw std::invalid_argument("DPF public LUT batch shape or epoch mismatch");
            if (binding) {
                if (!prepared_table || player->role() != party_id || binding->context_id == 0 ||
                    binding->table_version != prepared_table->version() ||
                    !materials[i].paper_lookup || materials[i].binding != *binding ||
                    materials[i].owner_peer != player ||
                    materials[i].family != binding->family ||
                    !materials[i].hasValidPaperLayout() ||
                    materials[i].family != first.family ||
                    materials[i].lut_skip_zero != first.lut_skip_zero ||
                    materials[i].lut_simd != first.lut_simd ||
                    (first.family != DpfLutKeyFamily::full_q_arithmetic &&
                     first.family != DpfLutKeyFamily::segmented_boolean_tensor_b2a) ||
                    materials[i].table_fingerprint != prepared_table->fingerprint())
                    throw std::invalid_argument("paper LUT material binding mismatch");
            } else if (materials[i].paper_lookup) {
                throw std::invalid_argument("paper LUT cannot use the legacy additive-mask API");
            }
        }
        // All output tables and every material have been validated locally.
        // Poison the whole batch before the first online socket operation.
        invalidateBatch();
        std::vector<GroupElement> masked(inputs.size());
        for (std::size_t i = 0; i < inputs.size(); ++i)
            masked[i] = binding ? inputs[i] - materials[i].mask_share
                                : inputs[i] + materials[i].mask_share;
        const auto sent_before = player->bytesSent, received_before = player->bytesReceived;
        const auto open_start = Clock::now();
        const StageSnapshot opening_stage(player, metrics ? int(ActivationStage::lut_open) : -1);
        if (metrics) ++metrics->online_open_batches;
        openAdditive(masked, first.index_bits, player);
        if (metrics) {
            metrics->online_open_us += elapsedUs(open_start);
            metrics->online_sent_bytes += player->bytesSent - sent_before;
            metrics->online_received_bytes += player->bytesReceived - received_before;
            opening_stage.record(metrics->open_stage, player);
        }
        const auto product_start = Clock::now();
        const StageSnapshot product_stage(player, metrics ? int(ActivationStage::lut_product) : -1);
        std::vector<std::vector<GroupElement>> result(tables.size());
        const uint64_t index_mask = maskForBits(first.index_bits);
        for (std::size_t t = 0; t < tables.size(); ++t) {
            const auto& table = *tables[t];
            result[t].resize(inputs.size(), GroupElement(0, table.Bout));
            for (std::size_t i = 0; i < inputs.size(); ++i) {
                if (table_for_input && (*table_for_input)[i] != t) continue;
                if (prepared_table) {
                    result[t][i] = GroupElement(materials[i].selection_share.shiftedDotProduct32(
                        *prepared_table, masked[i].value, first.lut_skip_zero, first.lut_simd), table.Bout);
                    continue;
                }
                uint64_t sum = 0;
                for (uint64_t r = 0; r <= index_mask; ++r) {
                    const uint64_t index = (masked[i].value - r) & index_mask;
                    // Reducing a wide arithmetic selection share locally is
                    // exact in every smaller 2-power output ring.
                    sum += materials[i].selection_share[static_cast<std::size_t>(r)].value *
                           table.values[static_cast<std::size_t>(index)].value;
                }
                result[t][i] = GroupElement(sum, table.Bout);
            }
        }
        if (metrics) {
            metrics->inner_product_us += elapsedUs(product_start);
            product_stage.record(metrics->product_stage, player);
        }
        return result;
    } catch (...) {
        invalidateBatch();
        throw;
    }
}
}  // namespace

std::vector<GroupElement> dpfPublicLutBatchOnline(
    int party_id, const std::vector<GroupElement>& inputs,
    std::vector<DpfPublicLutMaterial>& materials, const DpfLutBinding& expected,
    Peer* player, const PreparedDpfLutTable& table, DpfLutBatchMetrics* metrics) {
    auto result = onlineSelectedTables(party_id, inputs, materials, player,
        expected.element_begin, {&table.table()}, metrics, nullptr, &expected, &table);
    return std::move(result.front());
}

std::vector<std::vector<GroupElement>> dpfPublicLutBatchOnlineMany(
    int party_id, const std::vector<GroupElement>& inputs,
    std::vector<DpfPublicLutMaterial>& materials, Peer* player,
    uint64_t request_id_base, const std::vector<const PublicLUTData*>& tables,
    DpfLutBatchMetrics* metrics) {
    return onlineSelectedTables(party_id, inputs, materials, player,
                                request_id_base, tables, metrics, nullptr);
}

std::vector<GroupElement> dpfPublicLutBatchOnlinePerInput(
    int party_id, const std::vector<GroupElement>& inputs,
    std::vector<DpfPublicLutMaterial>& materials, Peer* player,
    uint64_t request_id_base,
    const std::vector<const PublicLUTData*>& table_for_input,
    DpfLutBatchMetrics* metrics) {
    try {
    if (table_for_input.size() != inputs.size()) {
        for (auto& material : materials) material.consumed = true;
        if (metrics) { metrics->consumed = materials.size(); metrics->remaining = 0; }
        throw std::invalid_argument("DPF public LUT per-input table count mismatch");
    }
    std::vector<const PublicLUTData*> unique_tables;
    std::vector<std::size_t> selections;
    selections.reserve(table_for_input.size());
    for (const auto* table : table_for_input) {
        const auto found = std::find(unique_tables.begin(), unique_tables.end(), table);
        selections.push_back(static_cast<std::size_t>(found - unique_tables.begin()));
        if (found == unique_tables.end()) unique_tables.push_back(table);
    }
    auto outputs = onlineSelectedTables(party_id, inputs, materials, player,
        request_id_base, unique_tables, metrics, &selections);
    std::vector<GroupElement> result(inputs.size());
    for (std::size_t i = 0; i < inputs.size(); ++i)
        result[i] = outputs[selections[i]][i];
    return result;
    } catch (...) {
        for (auto& material : materials) material.consumed = true;
        if (metrics) { metrics->consumed = materials.size(); metrics->remaining = 0; }
        throw;
    }
}

std::vector<GroupElement> dpfPublicLutBatchOnline(
    int party_id, const std::vector<GroupElement>& inputs,
    std::vector<DpfPublicLutMaterial>& materials, Peer* player,
    uint64_t request_id_base, const PublicLUTData& table) {
    auto result = dpfPublicLutBatchOnlineMany(party_id, inputs, materials, player,
                                           request_id_base, {&table});
    return std::move(result.front());
}

GroupElement dpfPublicLutOnline(
    int party_id, GroupElement input, DpfPublicLutMaterial& material,
    Peer* player, const PublicLUTData& table) {
    const uint64_t request_id = material.request_id;
    std::vector<GroupElement> inputs{input};
    std::vector<DpfPublicLutMaterial> batch;
    batch.push_back(std::move(material));
    try {
        auto result = dpfPublicLutBatchOnline(
            party_id, inputs, batch, player, request_id, table);
        material = std::move(batch.front());
        return result.front();
    } catch (...) {
        material = std::move(batch.front());
        throw;
    }
}

void releaseDpfPublicLutMaterial(DpfPublicLutMaterial& material) noexcept {
    material.consumed = true;
    material.released = true;
    material.selection_share.clear();
    material.selection_share.shrink_to_fit();
    freeDPFKeyPack(material.key);
}

}  // namespace segfss
