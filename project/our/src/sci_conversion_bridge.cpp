#include "sci_conversion_bridge.h"
#include "sci_msb_bridge.h"
#include "sci_truncation.h"
#include <cstdio>
#include <algorithm>
#include <chrono>
#include <cstdlib>
#include <deque>
#include <fstream>
#include <memory>
#include <mutex>
#include <unordered_set>
#include <iostream>
#include <stdexcept>
#include <type_traits>
#include <sys/resource.h>
#include <unistd.h>

namespace {
using Clock = std::chrono::steady_clock;
double elapsedMs(Clock::time_point start) {
    return std::chrono::duration<double, std::milli>(Clock::now() - start).count();
}
struct Context {
    void* network = nullptr;
    Peer* connection = nullptr;
    FILE* spool = nullptr;
    int role = 0, max_segments = 0;
    segfss::DcfBackend backend = segfss::DcfBackend::segmented;
    segfss::SegmentPolicy segment_policy = segfss::SegmentPolicy::explicit_counts;
    std::vector<segfss_sci_conversion_call> calls;
    std::size_t cursor = 0;
    uint64_t epoch = 0, setup_sent = 0, setup_received = 0;
    bool poisoned = false;
    bool contiguous = false;
    struct Loaded {
        segfss::SciTruncationMaterial material;
        double milliseconds = 0;
        uint64_t serialized_bytes = 0, resident_bytes = 0;
    };
    std::deque<Loaded> prefetched;
    uint64_t prefetch_budget = 0, resident_bytes = 0;
    segfss::SciTruncationWorkspace workspace;
    segfss::DcfGenerationOptions generation_options;
    std::vector<GroupElement> input_workspace;
};
thread_local Context* active = nullptr;
thread_local std::string error;
std::mutex contexts_mutex;
std::unordered_set<Context*> contexts;
struct ContextScope {
    Context* previous;
    explicit ContextScope(void* ptr) : previous(active) {
        std::lock_guard<std::mutex> lock(contexts_mutex);
        auto* requested = static_cast<Context*>(ptr);
        active = contexts.count(requested) ? requested : nullptr;
    }
    ~ContextScope() { active = previous; }
};

template<class T> void write(FILE* file, const T& value) {
    static_assert(std::is_trivially_copyable<T>::value, "spool scalar required");
    if (fwrite(&value, sizeof(T), 1, file) != 1) throw std::runtime_error("conversion spool write failed");
}
template<class T> T read(FILE* file) {
    T value{};
    if (fread(&value, sizeof(T), 1, file) != 1) throw std::runtime_error("conversion spool read failed");
    return value;
}
template<class T> void writeVector(FILE* file, const std::vector<T>& values) {
    write(file, uint64_t(values.size()));
    if (!values.empty() && fwrite(values.data(), sizeof(T), values.size(), file) != values.size())
        throw std::runtime_error("conversion spool vector write failed");
}
template<class T> std::vector<T> readVector(FILE* file, uint64_t expected) {
    const auto size = read<uint64_t>(file);
    if (size != expected) throw std::runtime_error("conversion spool vector shape differs from plan");
    std::vector<T> values(size);
    if (size && fread(values.data(), sizeof(T), size, file) != size)
        throw std::runtime_error("conversion spool vector read failed");
    return values;
}
void writeAnd(FILE* file, const segfss::BooleanAndPreprocessing& material) {
    writeVector(file, material.a); writeVector(file, material.b); writeVector(file, material.c);
}
segfss::BooleanAndPreprocessing readAnd(FILE* file, uint64_t count) {
    segfss::BooleanAndPreprocessing material;
    material.a = readVector<u8>(file, count); material.b = readVector<u8>(file, count);
    material.c = readVector<u8>(file, count);
    return material;
}
void writeComparison(FILE* file, const segfss::SegmentedDcfMaterial& material) {
    if (!material.count) return;
    write(file, int(material.segment_policy));
    write(file, int(material.need_e));
    write(file, int(material.prune_equality));
    write(file, int(material.prune_leaf_hashes));
    writeVector(file, material.layout.width); writeVector(file, material.layout.shift);
    write(file, material.metrics);
    if (active->contiguous) {
        writeVector(file, material.compact_keys.records);
        writeVector(file, material.compact_keys.correction_words);
        writeAnd(file, material.scan);
        return;
    }
    for (const auto& key : material.keys) {
        write(file, key.root); writeVector(file, key.internal_cw); write(file, key.hcw);
        write(file, key.lcw[0]); write(file, key.lcw[1]);
    }
    writeAnd(file, material.scan);
}
segfss::SegmentedDcfMaterial readComparison(FILE* file, int role, int count, int bits, int nseg, bool need_e) {
    segfss::SegmentedDcfMaterial material;
    if (!bits) return material;
    material.role = role; material.count = count; material.bits = bits; material.nseg = nseg;
    material.layout = segfss::SegmentLayout(bits, nseg); material.consumed = false;
    material.segment_policy = active->segment_policy;
    if (read<int>(file) != int(material.segment_policy))
        throw std::runtime_error("conversion spool comparison policy differs from plan");
    const int stored_need_e = read<int>(file), prune_equality = read<int>(file), prune_leaf = read<int>(file);
    if (stored_need_e != int(need_e) || (prune_equality != 0 && prune_equality != 1) ||
        (prune_leaf != 0 && prune_leaf != 1))
        throw std::runtime_error("conversion spool comparison demand differs from plan");
    material.need_e = need_e; material.prune_equality = prune_equality; material.prune_leaf_hashes = prune_leaf;
    if (
        readVector<int>(file, nseg) != material.layout.width ||
        readVector<int>(file, nseg) != material.layout.shift)
        throw std::runtime_error("conversion spool comparison policy/layout differs from plan");
    material.metrics = read<segfss::SegmentedDcfMetrics>(file);
    if (active->contiguous) {
        material.compact_keys.records = readVector<segfss::CggmKeyRecord>(file, uint64_t(count) * nseg);
        material.compact_keys.correction_words = readVector<osuCrypto::block>(file, uint64_t(count) * (bits - nseg));
        uint64_t offset = 0;
        for (std::size_t index = 0; index < material.keyCount(); ++index) {
            const auto key = material.keyView(index);
            if (key.depth != material.layout.width[index % nseg] ||
                material.compact_keys.records[index].cw_offset != offset)
                throw std::runtime_error("conversion spool compact key contract differs");
            offset += key.depth - 1;
        }
        material.scan = readAnd(file, uint64_t(count) *
            segfss::planDcfScan(nseg, need_e, material.prune_equality).and_gates_per_row);
        return material;
    }
    material.keys.reserve(static_cast<std::size_t>(count) * nseg);
    for (int row = 0; row < count; ++row) for (int segment = 0; segment < nseg; ++segment) {
        segfss::CggmKey key; key.depth = material.layout.width[segment];
        key.root = read<osuCrypto::block>(file);
        key.internal_cw = readVector<osuCrypto::block>(file, key.depth - 1);
        key.hcw = read<osuCrypto::block>(file);
        key.lcw[0] = read<u8>(file); key.lcw[1] = read<u8>(file);
        material.keys.push_back(std::move(key));
    }
    material.scan = readAnd(file, uint64_t(count) *
        segfss::planDcfScan(nseg, need_e, material.prune_equality).and_gates_per_row);
    return material;
}
segfss::SciTruncationConfig config(const segfss_sci_conversion_call& call) {
    using namespace segfss::reference;
    const Contract contract{call.input_bits, call.output_bits, call.shift,
        static_cast<SciCall>(call.call), bool(call.signed_arithmetic), NumericMode::sci_compatible};
    return active->segment_policy == segfss::SegmentPolicy::explicit_counts ?
        segfss::makeSciTruncationConfig(contract, active->backend, active->max_segments) :
        segfss::makeSciTruncationConfig(contract, active->backend, active->segment_policy);
}
void writeMaterial(FILE* file, const segfss::SciTruncationMaterial& material) {
    const auto& c = material.config.contract;
    write(file, uint64_t(0x534443464d415432));
    for (int field : {5, material.role, material.count, c.input_bits, c.output_bits, c.shift,
                     int(c.call), int(c.signed_arithmetic), int(c.numeric_mode), int(material.config.backend),
                     material.config.low_shift_nseg, material.config.sign_nseg, material.config.wrap_nseg,
                     int(active->contiguous), int(material.config.segment_policy),
                     int(material.config.fuse_sign_wrap)}) write(file, field);
    write(file, material.metrics);
    writeVector(file, material.input_mask_shares); writeVector(file, material.mask_high_shares);
    writeVector(file, material.mask_sign_bit_shares);
    writeComparison(file, material.low_shift); writeComparison(file, material.sign); writeComparison(file, material.wrap);
    writeAnd(file, material.rounding); writeVector(file, material.conversion_mask_bits);
    writeVector(file, material.conversion_mask_arithmetic);
}
segfss::SciTruncationMaterial readMaterial(FILE* file, const segfss_sci_conversion_call& call) {
    segfss::SciTruncationMaterial material;
    material.config = config(call); material.role = active->role; material.count = call.count; material.consumed = false;
    const auto& c = material.config.contract;
    if (read<uint64_t>(file) != uint64_t(0x534443464d415432))
        throw std::runtime_error("conversion spool format mismatch");
    for (int expected : {5, material.role, material.count, c.input_bits, c.output_bits, c.shift,
                         int(c.call), int(c.signed_arithmetic), int(c.numeric_mode), int(material.config.backend),
                         material.config.low_shift_nseg, material.config.sign_nseg, material.config.wrap_nseg,
                         int(active->contiguous), int(material.config.segment_policy),
                         int(material.config.fuse_sign_wrap)})
        if (read<int>(file) != expected) throw std::runtime_error("conversion spool contract mismatch");
    const auto domains = segfss::sciTruncationDomains(material.config);
    material.metrics = read<segfss::SciTruncationMetrics>(file);
    const uint64_t count = call.count;
    material.input_mask_shares = readVector<uint64_t>(file, domains.local_only ? 0 : count);
    material.mask_high_shares = readVector<uint64_t>(file, domains.local_only ? 0 : count);
    material.mask_sign_bit_shares = readVector<u8>(file, domains.need_sign ? count : 0);
    material.low_shift = readComparison(file, active->role, call.count, domains.low_shift_bits,
                                       material.config.low_shift_nseg, domains.toward_zero);
    material.sign = readComparison(file, active->role, call.count, domains.sign_bits, material.config.sign_nseg, false);
    material.wrap = readComparison(file, active->role, call.count, domains.wrap_bits, material.config.wrap_nseg, false);
    material.rounding = readAnd(file, count * (int(domains.toward_zero) + int(domains.fused_sign_wrap)));
    material.conversion_mask_bits = readVector<u8>(file, count * domains.correction_count);
    material.conversion_mask_arithmetic = readVector<uint64_t>(file, count * domains.correction_count);
    return material;
}
void check(void* ptr) {
    if (!ptr || ptr != active) throw std::invalid_argument("invalid conversion bridge context");
}

uint64_t materialCapacity(const segfss::SciTruncationMaterial& m) {
    return sizeof(m) + (m.input_mask_shares.capacity() + m.mask_high_shares.capacity() +
        m.conversion_mask_arithmetic.capacity()) * sizeof(uint64_t) + m.mask_sign_bit_shares.capacity() +
        m.conversion_mask_bits.capacity() + m.rounding.a.capacity() + m.rounding.b.capacity() +
        m.rounding.c.capacity() + m.low_shift.capacityBytes() + m.sign.capacityBytes() + m.wrap.capacityBytes();
}

uint64_t addressSpaceBytes() {
    uint64_t pages = 0;
    std::ifstream stream("/proc/self/statm");
    if (!(stream >> pages)) throw std::runtime_error("cannot account prefetch address space");
    return pages * static_cast<uint64_t>(sysconf(_SC_PAGESIZE));
}

void prefetchPrefix(const std::vector<uint64_t>& sizes) {
    if (!active->prefetch_budget) return;
    rlimit limit{};
    if (getrlimit(RLIMIT_AS, &limit)) throw std::runtime_error("cannot read prefetch resource budget");
    uint64_t workspace_reserve = 64ULL << 20;
    for (const auto& call : active->calls)
        workspace_reserve = std::max(workspace_reserve, uint64_t(call.count) * 512);
    for (std::size_t index = 0; index < sizes.size(); ++index) {
        // Reserve allocator slack and the online scratch in addition to the
        // already resident process state; only this query's prefix is loaded.
        const uint64_t allocation = sizes[index] * 2 + (1ULL << 20);
        const uint64_t used = addressSpaceBytes();
        if (allocation > active->prefetch_budget - active->resident_bytes ||
            (limit.rlim_cur != RLIM_INFINITY &&
             (used >= limit.rlim_cur || workspace_reserve >= limit.rlim_cur - used ||
              allocation >= limit.rlim_cur - used - workspace_reserve))) break;
        const auto start = Clock::now();
        Context::Loaded loaded;
        loaded.material = readMaterial(active->spool, active->calls[index]);
        loaded.milliseconds = elapsedMs(start);
        loaded.serialized_bytes = sizes[index];
        loaded.resident_bytes = materialCapacity(loaded.material);
        active->resident_bytes += loaded.resident_bytes;
        if (active->resident_bytes > active->prefetch_budget)
            throw std::runtime_error("conversion prefetch exceeded resident budget");
        active->prefetched.push_back(std::move(loaded));
    }
    std::cout << "CONVERSION_PREFETCH {\"mode\":\"prepared_prefix\",\"calls\":"
        << active->prefetched.size() << ",\"resident_bytes\":" << active->resident_bytes
        << ",\"budget_bytes\":" << active->prefetch_budget
        << ",\"workspace_reserve_bytes\":" << workspace_reserve << ",\"address_space_bytes\":"
        << addressSpaceBytes() << ",\"charged_phase\":\"offline\"}" << std::endl;
}
bool same(const segfss_sci_conversion_call& a, const segfss_sci_conversion_call& b) {
    return a.count == b.count && a.input_bits == b.input_bits && a.output_bits == b.output_bits &&
        a.shift == b.shift && a.call == b.call && a.signed_arithmetic == b.signed_arithmetic;
}

void confirmPublicSchedule() {
    Peer* peer = active->connection;
    const auto& options = active->generation_options;
    std::vector<uint64_t> own{3, uint64_t(active->backend), uint64_t(active->segment_policy),
        uint64_t(active->max_segments), uint64_t(active->contiguous), options.tile_keys,
        options.budget_bytes, options.reserve_bytes, options.participant_limit, options.resident_bytes,
        active->prefetch_budget};
    for (const auto& call : active->calls) {
        const auto c = config(call);
        const auto d = segfss::sciTruncationDomains(c);
        for (int value : {call.count, call.input_bits, call.output_bits, call.shift, call.call,
                          call.signed_arithmetic, c.low_shift_nseg, c.sign_nseg, c.wrap_nseg,
                          int(c.fuse_sign_wrap)})
            own.push_back(uint64_t(value));
        for (const auto shape : {std::pair<int,int>{d.low_shift_bits, c.low_shift_nseg},
                                {d.sign_bits, c.sign_nseg}, {d.wrap_bits, c.wrap_nseg}}) {
            own.push_back(shape.first);
            if (shape.first) for (int width : segfss::SegmentLayout(shape.first, shape.second).width)
                own.push_back(width);
        }
    }
    uint64_t remote_size = 0;
    if (active->role == SERVER) { peer->send_u64(own.size()); remote_size = peer->recv_u64(); }
    else { remote_size = peer->recv_u64(); peer->send_u64(own.size()); }
    if (remote_size != own.size()) throw std::invalid_argument("conversion public schedule size mismatch");
    std::vector<uint64_t> remote(own.size());
    if (active->role == SERVER) {
        peer->send_u64(own.data(), own.size()); peer->recv_u64(remote.data(), remote.size());
    } else {
        peer->recv_u64(remote.data(), remote.size()); peer->send_u64(own.data(), own.size());
    }
    if (remote != own) throw std::invalid_argument("conversion public policy/schedule mismatch");
}
}

extern "C" {
void* segfss_sci_conversion_create(int sci_party, const char* host, int port, int full_tree, int max_segments) {
    try {
        if (max_segments < 1 || max_segments > 64) throw std::invalid_argument("invalid conversion setup");
        auto ctx = std::make_unique<Context>();
        ctx->contiguous = segfss::contiguousDcfMaterials();
        if (const char* policy = std::getenv("SEGFSS_SEGMENT_POLICY"))
            ctx->segment_policy = segfss::parseSegmentPolicy(policy);
        if (const char* bytes = std::getenv("SEGFSS_PREFETCH_BYTES")) {
            const std::string value(bytes);
            if (value.empty() || value.find_first_not_of("0123456789") != std::string::npos)
                throw std::invalid_argument("SEGFSS_PREFETCH_BYTES must be nonnegative integer");
            ctx->prefetch_budget = std::stoull(value);
            if (ctx->prefetch_budget > (1ULL << 40) || (ctx->prefetch_budget && !ctx->contiguous))
                throw std::invalid_argument("prefetch requires contiguous layout and bounded budget");
        }
        ctx->network = segfss_sci_create(sci_party, host, port);
        if (!ctx->network) throw std::runtime_error(segfss_sci_error());
        ctx->generation_options = segfss::dcfGenerationOptions();
        ctx->connection = static_cast<Peer*>(segfss_sci_get_peer(ctx->network));
        ctx->role = sci_party + 1; ctx->max_segments = max_segments;
        ctx->backend = full_tree ? segfss::DcfBackend::full_tree : segfss::DcfBackend::segmented;
        segfss_sci_stats stats{}; segfss_sci_get_stats(ctx->network, &stats);
        ctx->setup_sent = stats.setup_sent_bytes; ctx->setup_received = stats.setup_received_bytes;
        std::lock_guard<std::mutex> lock(contexts_mutex);
        contexts.insert(ctx.get());
        return ctx.release();
    } catch (const std::exception& e) { error = e.what(); return nullptr; }
}
int segfss_sci_conversion_set_generation_resident(void* ptr, uint64_t bytes) {
    ContextScope scope(ptr);
    try {
        check(ptr);
        if (active->spool || active->poisoned)
            throw std::logic_error("cannot alter prepared conversion memory reservation");
        active->generation_options.resident_bytes = bytes;
        return 0;
    } catch (const std::exception& e) { error = e.what(); return -1; }
}
int segfss_sci_conversion_set_prefetch_budget(void* ptr, uint64_t bytes) {
    ContextScope scope(ptr);
    try {
        check(ptr);
        if (active->spool || active->poisoned)
            throw std::logic_error("cannot alter prepared conversion prefetch budget");
        if (bytes > (1ULL << 40) || (bytes && !active->contiguous))
            throw std::invalid_argument("prefetch requires contiguous layout and bounded budget");
        active->prefetch_budget = bytes;
        return 0;
    } catch (const std::exception& e) { error = e.what(); return -1; }
}
int segfss_sci_conversion_prepare(void* ptr, const segfss_sci_conversion_call* calls, int count) {
    ContextScope scope(ptr);
    try {
        check(ptr);
        if (active->spool || active->poisoned || count < 0 || (count && !calls))
            throw std::logic_error("unconsumed or poisoned conversion query");
        if (count) active->calls.assign(calls, calls + count);
        else active->calls.clear();
        active->cursor = 0;
        confirmPublicSchedule();
        active->spool = std::tmpfile();
        if (!active->spool) throw std::runtime_error("cannot create private conversion spool");
        std::size_t ordinal = 0;
        std::vector<uint64_t> serialized_sizes;
        for (const auto& call : active->calls) {
            ++active->epoch;
            auto material = segfss::sciTruncateOfflineBatch(active->role, call.count, config(call), active->connection,
                                                          &active->generation_options);
            const auto begin = ftell(active->spool);
            writeMaterial(active->spool, material);
            const auto end = ftell(active->spool);
            if (begin < 0 || end < begin) throw std::runtime_error("cannot account serialized material");
            serialized_sizes.push_back(end - begin);
            uint64_t internal_hashes = 0, leaf_hashes = 0, participants = 0, aes_scratch = 0;
            bool batched_aes = false;
            for (const auto* comparison : {&material.low_shift, &material.sign, &material.wrap}) {
                internal_hashes += comparison->metrics.internal_hashes;
                leaf_hashes += comparison->metrics.leaf_hashes;
                participants = std::max<uint64_t>(participants, comparison->metrics.generation_participants);
                aes_scratch = std::max<uint64_t>(aes_scratch, comparison->metrics.keygen_aes_scratch_bytes);
                batched_aes = batched_aes || comparison->metrics.aes_offline;
            }
            std::cout << "CONVERSION_PREPROCESS_DETAIL {\"ordinal\":" << ordinal++
                << ",\"mask_ms\":" << material.metrics.mask_preprocessing_ms
                << ",\"comparison_ms\":" << material.metrics.comparisons_preprocessing_ms
                << ",\"rounding_ms\":" << material.metrics.rounding_preprocessing_ms
                << ",\"internal_hashes\":" << internal_hashes << ",\"leaf_hashes\":" << leaf_hashes
                << ",\"generation_participants\":" << participants << ",\"aes_scratch_peak_bytes\":" << aes_scratch
                << ",\"aes_offline\":" << (batched_aes ? "true" : "false")
                << ",\"fused_sign_wrap\":" << (segfss::sciTruncationDomains(material.config).fused_sign_wrap ? "true" : "false")
                << ",\"fresh\":true,\"dealerless\":true}" << std::endl;
        }
        if (fflush(active->spool) != 0 || fseek(active->spool, 0, SEEK_SET) != 0)
            throw std::runtime_error("cannot rewind conversion spool");
        prefetchPrefix(serialized_sizes);
        return 0;
    } catch (const std::exception& e) { if (active) active->poisoned = true; error = e.what(); return -1; }
}
int segfss_sci_conversion_online(void* ptr, const segfss_sci_conversion_call* call,
                                const uint64_t* input, uint64_t* output, segfss_sci_conversion_metrics* metrics) {
    ContextScope scope(ptr);
    try {
        const auto online_start = Clock::now();
        check(ptr);
        if (!call || !input || !output || !metrics || active->poisoned || !active->spool ||
            active->cursor >= active->calls.size() || !same(*call, active->calls[active->cursor]))
            throw std::logic_error("conversion call differs from fresh offline schedule");
        ++active->cursor;
        auto start = Clock::now();
        segfss::SciTruncationMaterial material;
        double material_load_ms = 0, material_preload_ms = 0;
        uint64_t serialized_bytes = 0;
        const bool prefetched = !active->prefetched.empty();
        if (prefetched) {
            auto& loaded = active->prefetched.front();
            material = std::move(loaded.material);
            material_preload_ms = loaded.milliseconds;
            serialized_bytes = loaded.serialized_bytes;
            active->resident_bytes -= loaded.resident_bytes;
            active->prefetched.pop_front();
        } else {
            const auto material_begin = ftell(active->spool);
            material = readMaterial(active->spool, *call);
            const auto material_end = ftell(active->spool);
            material_load_ms = elapsedMs(start);
            if (material_begin < 0 || material_end < material_begin)
                throw std::runtime_error("cannot measure conversion spool position");
            serialized_bytes = material_end - material_begin;
        }
        start = Clock::now();
        std::vector<GroupElement> local_shares;
        auto& shares = active->contiguous ? active->input_workspace : local_shares;
        shares.clear(); shares.reserve(call->count);
        for (int row = 0; row < call->count; ++row) shares.emplace_back(input[row], call->input_bits);
        const auto query_prepare_ms = elapsedMs(start);
        auto results = segfss::sciTruncateOnlineBatch(active->role, shares, material.config, material, active->connection,
            active->contiguous ? &active->workspace : nullptr);
        for (int row = 0; row < call->count; ++row) output[row] = results[row].value;
        const auto& m = material.metrics;
        *metrics = {m.mask_preprocessing_ms, m.comparisons_preprocessing_ms, m.rounding_preprocessing_ms,
            m.masked_input_open_ms, m.comparisons_online_ms, m.rounding_online_ms, m.masked_bit_conversion_ms,
            m.masked_input_values, m.masked_conversion_bits, m.online_logical_rounds};
        metrics->material_load_ms = material_load_ms;
        metrics->material_bytes = serialized_bytes;
        metrics->material_preload_ms = material_preload_ms;
        metrics->prefetched = prefetched;
        metrics->contiguous_material = active->contiguous;
        metrics->resident_bytes = active->resident_bytes + materialCapacity(material);
        metrics->prefetch_budget_bytes = active->prefetch_budget;
        metrics->query_prepare_ms = query_prepare_ms;
        metrics->scan_ms = m.comparison_scan_ms;
        for (const auto* comparison : {&material.low_shift, &material.sign, &material.wrap}) {
            metrics->query_prepare_ms += comparison->metrics.query_prepare_ms;
            metrics->local_evaluation_ms += comparison->metrics.local_evaluation_ms;
            metrics->comparison_keys += comparison->keyCount();
            metrics->scan_and_gates += comparison->scan.cursor;
            metrics->eval_participants = std::max<uint64_t>(metrics->eval_participants, comparison->metrics.eval_participants);
            metrics->keygen_internal_hashes += comparison->metrics.internal_hashes;
            metrics->keygen_leaf_hashes += comparison->metrics.leaf_hashes;
            metrics->keygen_participants = std::max<uint64_t>(metrics->keygen_participants,
                                                           comparison->metrics.generation_participants);
            metrics->keygen_aes_scratch_bytes = std::max<uint64_t>(metrics->keygen_aes_scratch_bytes,
                                                               comparison->metrics.keygen_aes_scratch_bytes);
            metrics->eval_internal_hashes += comparison->metrics.eval_internal_hashes;
            metrics->eval_leaf_hashes += comparison->metrics.eval_leaf_hashes;
            metrics->pruned_e_outputs += comparison->metrics.pruned_e_outputs;
            metrics->skipped_leaf_hashes += comparison->metrics.skipped_leaf_hashes;
            if (comparison->count) {
                metrics->prune_equality |= comparison->prune_equality;
                metrics->prune_leaf_hashes |= comparison->prune_leaf_hashes;
            }
            metrics->eval_aes_scratch_bytes = std::max<uint64_t>(metrics->eval_aes_scratch_bytes,
                                                             comparison->metrics.eval_aes_scratch_bytes);
            metrics->aes_online |= comparison->metrics.aes_online;
            metrics->aes_offline |= comparison->metrics.aes_offline;
        }
        start = Clock::now();
        segfss::releaseSciTruncationMaterial(material);
        metrics->release_ms = elapsedMs(start);
        metrics->total_online_ms = elapsedMs(online_start);
        return 0;
    } catch (const std::exception& e) { if (active) active->poisoned = true; error = e.what(); return -1; }
}
int segfss_sci_conversion_finish_query(void* ptr) {
    ContextScope scope(ptr);
    try {
        check(ptr);
        if (active->poisoned || !active->spool || active->cursor != active->calls.size())
            throw std::logic_error("query left conversion materials unconsumed");
        fclose(active->spool); active->spool = nullptr; active->calls.clear(); return 0;
    } catch (const std::exception& e) { error = e.what(); return -1; }
}
int segfss_sci_conversion_get_counters(void* ptr, segfss_sci_conversion_counters* output) {
    ContextScope scope(ptr);
    Peer* peer = active ? active->connection : nullptr;
    try {
        check(ptr); if (!output) throw std::invalid_argument("missing conversion counter output");
        *output = {active->setup_sent + peer->bytesSent, active->setup_received + peer->bytesReceived, active->epoch};
        return 0;
    } catch (const std::exception& e) { error = e.what(); return -1; }
}
void segfss_sci_conversion_destroy(void* ptr) {
    ContextScope scope(ptr);
    if (ptr != active || !active) return;
    if (active->spool) fclose(active->spool);
    segfss_sci_destroy(active->network);
    { std::lock_guard<std::mutex> lock(contexts_mutex); contexts.erase(active); }
    delete active; active = nullptr;
}
const char* segfss_sci_conversion_error() { return error.c_str(); }
}
