#include "sci_folded_bridge.h"
#include "sci_msb_bridge.h"
#include "activation.h"
#include "folded_native.h"
#include <algorithm>
#include <chrono>
#include <cstdlib>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

namespace {
using Clock = std::chrono::steady_clock;
uint64_t elapsed(Clock::time_point start) {
    return std::chrono::duration_cast<std::chrono::microseconds>(Clock::now() - start).count();
}
struct Entry {
    segfss_sci_folded_call shape{};
    segfss::SciTruncationMaterial input, output;
    std::unique_ptr<segfss::ActivationContext> core;
    segfss::ActivationMetrics metrics;
};
struct Context {
    void* network = nullptr;
    Peer* player = nullptr;
    int role = 0;
    uint64_t identity = 0, epoch = 0, setup_sent = 0, setup_received = 0;
    uint64_t public_table_prepare_us = 0, public_table_resident_bytes = 0;
    uint64_t public_table_prepare_sent_bytes = 0, public_table_prepare_received_bytes = 0;
    uint64_t generated = 0, consumed = 0, generated_elements = 0, consumed_elements = 0;
    size_t cursor = 0;
    bool poisoned = false;
    segfss::SegmentPolicy policy = segfss::SegmentPolicy::fixed_4;
    segfss::DcfGenerationOptions generation;
    segfss::SciTruncationWorkspace workspace;
    std::shared_ptr<const segfss::PreparedDpfLutTable> table;
    std::vector<Entry> entries;
    ~Context() {
        entries.clear();
        if (network) segfss_sci_destroy(network);
    }
};
thread_local std::string error;
bool reuseInputSignOption() {
    const char* value = std::getenv("SEGFSS_FOLDED_REUSE_SIGN");
    if (!value || std::string(value) == "0") return false;
    if (std::string(value) == "1") return true;
    throw std::invalid_argument("SEGFSS_FOLDED_REUSE_SIGN must be 0 or 1");
}
int clipBitsOption() {
    const char* value = std::getenv("SEGFSS_FOLDED_CLIP_BITS");
    if (!value || std::string(value) == "32") return 32;
    if (std::string(value) == "22") return 22;
    throw std::invalid_argument("SEGFSS_FOLDED_CLIP_BITS must be 32 or 22");
}
void validate(const segfss_sci_folded_call& c) {
    if (c.query < 0 || c.layer < -1 || c.slot < 0 || c.call < 0 || c.count < 1 ||
        c.input_bits != 16 || c.output_bits != 16 || c.input_scale != 7 ||
        c.kind < 0 || c.kind > 1 || c.output_scale != (c.kind ? 7 : 14) ||
        c.lut_family < 0 || c.lut_family > 1 ||
        (c.sign_backend != SEGFSS_SIGN_BITDEC && c.sign_backend != SEGFSS_SIGN_SEGMENTED_FSS) ||
        (c.sign_backend == SEGFSS_SIGN_BITDEC ? c.sign_segments != 0 : (c.sign_segments < 1 || c.sign_segments > 31)) ||
        c.table_version != 1)
        throw std::invalid_argument("invalid complete folded activation contract");
    if (c.kind && !segfss::folded::tanhNativeAvailable())
        throw std::logic_error("complete folded TanH adapter is unavailable; no SCI fallback");
}
bool same(const segfss_sci_folded_call& a, const segfss_sci_folded_call& b) {
    return a.query == b.query && a.layer == b.layer && a.slot == b.slot && a.call == b.call &&
        a.kind == b.kind && a.count == b.count && a.input_bits == b.input_bits &&
        a.output_bits == b.output_bits && a.input_scale == b.input_scale &&
        a.output_scale == b.output_scale && a.lut_family == b.lut_family &&
        a.sign_backend == b.sign_backend && a.sign_segments == b.sign_segments &&
        a.table_version == b.table_version;
}
uint64_t materialBytes(const segfss::SciTruncationMaterial& m) {
    return sizeof(m) + (m.input_mask_shares.capacity() + m.mask_high_shares.capacity() +
        m.conversion_mask_arithmetic.capacity()) * sizeof(uint64_t) + m.mask_sign_bit_shares.capacity() +
        m.conversion_mask_bits.capacity() + m.rounding.a.capacity() + m.rounding.b.capacity() +
        m.rounding.c.capacity() + m.low_shift.capacityBytes() + m.sign.capacityBytes() + m.wrap.capacityBytes();
}
uint64_t residentBytes(const Entry& e) {
    uint64_t result = materialBytes(e.input) + materialBytes(e.output);
    if (!e.core) return result;
    result += sizeof(*e.core) + e.core->sign_material.material.capacityBytes() + e.core->clip_material.material.capacityBytes();
    result += e.core->materials.capacity() * sizeof(segfss::SigmaActivationKey);
    for (const auto& key : e.core->materials) {
        result += key.dpf_lut.selection_share.capacityBytes();
        for (const auto* p : {&key.sign_and, &key.clip_and})
            result += p->a.capacity() + p->b.capacity() + p->c.capacity();
        result += key.trunc_key.context_comparison_key.internal_cw.capacity() * sizeof(osuCrypto::block);
    }
    return result;
}
void lutStages(segfss::ActivationMetrics& metrics, const segfss::DpfLutBatchMetrics& lut) {
    using S = segfss::ActivationStage;
    auto copy = [&](bool offline, S stage, const segfss::DpfLutStageMetric& source) {
        auto& target = (offline ? metrics.offline : metrics.online)[unsigned(stage)];
        target = {source.elapsed_us, source.sent_bytes, source.received_bytes, source.begin_us, source.end_us};
    };
    copy(true, S::lut_mask, lut.mask_stage); copy(true, S::lut_keygen, lut.tree_keygen_stage);
    copy(true, S::lut_materialize, lut.materialize_stage); copy(true, S::lut_tensor_and, lut.tensor_and_stage);
    copy(true, S::lut_leaf_b2a, lut.leaf_b2a_stage); copy(true, S::lut_payload, lut.payload_stage);
    copy(false, S::lut_open, lut.open_stage); copy(false, S::lut_product, lut.product_stage);
    metrics.peak_working_bytes = std::max<uint64_t>(metrics.peak_working_bytes,lut.peak_working_bytes);
    metrics.online_generated = lut.online_generated;
    metrics.online_loaded_bytes = lut.online_loaded_bytes;
}
void exportMetrics(const segfss::ActivationMetrics& source, segfss_sci_folded_metrics* target) {
    if (!target) return;
    static_assert(SEGFSS_FOLDED_STAGE_COUNT == segfss::kActivationStageCount);
    for (unsigned i = 0; i < segfss::kActivationStageCount; ++i) {
        const auto& off = source.offline[i]; const auto& on = source.online[i];
        target->offline[i] = {off.elapsed_us, off.sent_bytes, off.received_bytes, off.begin_us, off.end_us};
        target->online[i] = {on.elapsed_us, on.sent_bytes, on.received_bytes, on.begin_us, on.end_us};
    }
    target->offline_wall_us = source.offline_wall_us; target->online_wall_us = source.online_wall_us;
    target->resident_bytes = source.resident_bytes; target->peak_working_bytes = source.peak_working_bytes;
    target->online_generated = source.online_generated; target->online_loaded_bytes = source.online_loaded_bytes;
}
}
extern "C" {
void* segfss_sci_folded_create(int sci_role, const char* host, int port, uint64_t identity) {
    try {
        if ((sci_role != 1 && sci_role != 2) || !host || !identity)
            throw std::invalid_argument("invalid folded activation context");
        auto ctx = std::make_unique<Context>();
        ctx->network = segfss_sci_create(sci_role, host, port);
        if (!ctx->network) throw std::runtime_error(segfss_sci_error());
        ctx->player = static_cast<Peer*>(segfss_sci_get_peer(ctx->network));
        segfss_sci_stats stats{};
        if (segfss_sci_get_stats(ctx->network, &stats)) throw std::runtime_error(segfss_sci_error());
        ctx->setup_sent = stats.setup_sent_bytes - ctx->player->bytesSent;
        ctx->setup_received = stats.setup_received_bytes - ctx->player->bytesReceived;
        ctx->role = sci_role + 1; ctx->identity = identity;
        if (const char* policy = std::getenv("SEGFSS_SEGMENT_POLICY"))
            ctx->policy = segfss::parseSegmentPolicy(policy);
        if (ctx->policy == segfss::SegmentPolicy::explicit_counts)
            ctx->policy = segfss::SegmentPolicy::fixed_4;
        ctx->generation = segfss::dcfGenerationOptions();
        const auto table_start = Clock::now();
        const auto table_sent = ctx->player->bytesSent;
        const auto table_received = ctx->player->bytesReceived;
        ctx->table = segfss::prepareDpfLutTable(segfss::makeSigmaDeltaTable(32, 12, 10), 1);
        ctx->public_table_prepare_us = elapsed(table_start);
        ctx->public_table_resident_bytes = sizeof(*ctx->table) +
            ctx->table->table().values.capacity() * sizeof(GroupElement) +
            ctx->table->coefficients().capacity() * sizeof(uint32_t) +
            ctx->table->nonzeroRanges().capacity() * sizeof(std::pair<std::size_t, std::size_t>);
        ctx->public_table_prepare_sent_bytes = ctx->player->bytesSent - table_sent;
        ctx->public_table_prepare_received_bytes = ctx->player->bytesReceived - table_received;
        return ctx.release();
    } catch (const std::exception& ex) { error = ex.what(); return nullptr; }
}
int segfss_sci_folded_prepare(void* pointer, const segfss_sci_folded_call* calls, int count) {
    auto* ctx = static_cast<Context*>(pointer);
    try {
        if (!ctx || count < 0 || (count && !calls) || ctx->poisoned || ctx->cursor != ctx->entries.size())
            throw std::logic_error("folded previous query inventory unfinished");
        // Options are frozen with this query's material. Online never reads
        // these variables and cannot change its inventory shape after READY.
        const bool reuse_input_sign = reuseInputSignOption();
        const int clip_bits = clipBitsOption();
        for (int index = 0; index < count; ++index) {
            validate(calls[index]);
            if(calls[index].sign_backend==SEGFSS_SIGN_SEGMENTED_FSS &&
               calls[index].sign_segments!=segfss::segmentCount(31,ctx->policy))
                throw std::invalid_argument("folded sign segmentation differs from the outer public policy");
            if (calls[index].call != index) throw std::invalid_argument("folded plan call IDs are not contiguous");
        }
        ctx->entries.clear(); ctx->cursor = 0;
        ctx->generated = ctx->consumed = ctx->generated_elements = ctx->consumed_elements = 0;
        ++ctx->epoch;
        for (int index = 0; index < count; ++index) {
            Entry item; item.shape = calls[index];
            dfss_round::SemanticScope call_scope("folded_call", item.shape.query,
                item.shape.layer, item.shape.slot, item.shape.call, item.shape.kind);
            const auto start = Clock::now();
            item.input = segfss::folded::prepareInputExtension(ctx->role, item.shape.count,
                ctx->player, ctx->policy, &ctx->generation, &item.metrics);
            item.core = std::make_unique<segfss::ActivationContext>(ctx->role, ctx->player, ctx->identity);
            item.core->public_table = ctx->table;
            item.core->generation_epoch = ctx->epoch - 1;
            item.core->sign_backend = static_cast<segfss::ActivationSignBackend>(item.shape.sign_backend);
            item.core->sign_segments = item.shape.sign_segments;
            item.core->native_signed16_input = true;
            item.core->reuse_input_sign = reuse_input_sign;
            item.core->clip_bits = clip_bits;
            item.core->clip_segments = item.shape.sign_backend == SEGFSS_SIGN_SEGMENTED_FSS
                ? segfss::segmentCount(clip_bits - 1, ctx->policy) : 0;
            segfss::DpfLutOptions options;
            options.family = static_cast<segfss::DpfLutKeyFamily>(item.shape.lut_family);
            options.participant_limit = ctx->generation.participant_limit;
            segfss::DpfLutBinding binding;
            binding.context_id = ctx->identity; binding.query_id = item.shape.query;
            binding.operator_id = uint64_t(item.shape.layer + 1);
            binding.worker_id = item.shape.slot; binding.slot_id = item.shape.call;
            binding.table_version = item.shape.table_version; binding.family = options.family;
            segfss::prepareSigmaActivation(*item.core, item.shape.count, 32, 12, 10,
                options, binding, nullptr, &item.metrics);
            if (item.shape.kind) item.output = segfss::folded::prepareTanhOutput(ctx->role,
                item.shape.count, ctx->player, ctx->policy, &ctx->generation, &item.metrics);
            lutStages(item.metrics, item.core->lut_metrics);
            for(const auto* stock:{&item.core->sign_material.material,&item.core->clip_material.material})
                item.metrics.peak_working_bytes=std::max<uint64_t>(item.metrics.peak_working_bytes,
                    std::max(stock->generation_peak_working_bytes,stock->generation_workspace_capacity_bytes));
            item.metrics.offline_wall_us = elapsed(start);
            item.metrics.resident_bytes = residentBytes(item);
            ++ctx->generated; ctx->generated_elements += item.shape.count;
            ctx->entries.push_back(std::move(item));
        }
        return 0;
    } catch (const std::exception& ex) { if (ctx) ctx->poisoned = true; error = ex.what(); return -1; }
}
int segfss_sci_folded_prepared_metrics(void* pointer, int call, segfss_sci_folded_metrics* metrics) {
    auto* ctx = static_cast<Context*>(pointer);
    if (!ctx || !metrics || call < 0 || size_t(call) >= ctx->entries.size()) return -1;
    exportMetrics(ctx->entries[call].metrics, metrics); return 0;
}
int segfss_sci_folded_online(void* pointer, const segfss_sci_folded_call* call,
    const uint64_t* input, uint64_t* output, segfss_sci_folded_metrics* metrics) {
    auto* ctx = static_cast<Context*>(pointer);
    try {
        if (!ctx || !call || !input || !output || ctx->poisoned || ctx->cursor >= ctx->entries.size())
            throw std::logic_error("folded online inventory unavailable");
        validate(*call);
        auto& item = ctx->entries[ctx->cursor];
        if (!same(*call, item.shape)) throw std::invalid_argument("folded online immutable call mismatch");
        segfss::validateActivationSignInventory(*item.core, call->count, 32);
        ++ctx->cursor; ++ctx->consumed; ctx->consumed_elements += call->count;
        dfss_round::SemanticScope call_scope("folded_call", call->query,
            call->layer, call->slot, call->call, call->kind);
        const auto start = Clock::now();
        segfss::ActivationInputSignShares input_signs;
        input_signs.owner_peer = ctx->player;
        input_signs.binding = item.core->lut_binding;
        input_signs.generation_epoch = item.core->generation_epoch;
        auto q12 = segfss::folded::nativeInputToQ12(ctx->role, input, call->count,
            item.input, ctx->player, &ctx->workspace, &item.metrics,
            item.core->reuse_input_sign ? &input_signs.shares : nullptr);
        if (call->kind) q12 = segfss::folded::tanhCoreInput(q12);
        auto result = segfss::sigmaBatchOnline(*item.core, q12, &item.metrics,
            item.core->reuse_input_sign ? &input_signs : nullptr);
        if (call->kind) result = segfss::folded::tanhNativeOutput(ctx->role, result,
            item.output, ctx->player, &ctx->workspace, &item.metrics);
        else {
            segfss::ActivationStageScope stage(&item.metrics, false, segfss::ActivationStage::output_adapter, ctx->player);
            for (auto& value : result) value = GroupElement(value.value * uint64_t(4), 16);
        }
        for (int index = 0; index < call->count; ++index) output[index] = result.at(index).value & 65535;
        lutStages(item.metrics, item.core->lut_metrics);
        segfss::releaseSciTruncationMaterial(item.input); segfss::releaseSciTruncationMaterial(item.output);
        item.core.reset();
        item.metrics.online_wall_us = elapsed(start);
        exportMetrics(item.metrics, metrics);
        return 0;
    } catch (const std::exception& ex) { if (ctx) ctx->poisoned = true; error = ex.what(); return -1; }
}
int segfss_sci_folded_finish_query(void* pointer) {
    auto* ctx = static_cast<Context*>(pointer);
    if (!ctx || ctx->poisoned || ctx->cursor != ctx->entries.size() ||
        ctx->generated != ctx->consumed || ctx->generated_elements != ctx->consumed_elements) {
        if (ctx) ctx->poisoned = true; error = "folded query inventory unbalanced"; return -1;
    }
    return 0;
}
int segfss_sci_folded_counters_get(void* pointer, segfss_sci_folded_counters* counters) {
    auto* ctx = static_cast<Context*>(pointer);
    if (!ctx || !counters) return -1;
    uint64_t resident = 0;
    for (const auto& entry : ctx->entries) resident += residentBytes(entry);
    *counters = {ctx->setup_sent + ctx->player->bytesSent, ctx->setup_received + ctx->player->bytesReceived,
        ctx->epoch, resident, ctx->generated, ctx->consumed, ctx->generated_elements, ctx->consumed_elements, 0, 0,
        ctx->public_table_prepare_us, ctx->public_table_resident_bytes,
        ctx->public_table_prepare_sent_bytes, ctx->public_table_prepare_received_bytes};
    return 0;
}
void segfss_sci_folded_destroy(void* pointer) { delete static_cast<Context*>(pointer); }
const char* segfss_sci_folded_error(void) { return error.c_str(); }
}
