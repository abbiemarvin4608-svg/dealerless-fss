// Reuse the existing trusted local MSB spool format and isolated runtime.
#include "../../src/sci_msb_bridge.cpp"
#include "../../src/relu.h"
#include "../../src/segmented_dcf.h"
#include "../../src/ring_packing.h"
#include "../../src/runtime_options.h"
#include "mpc/secure_ops.h"
#include "cnn_bridge.h"

namespace {
struct CnnContext {
    void* network = nullptr;
    Peer* connection = nullptr;
    int role = 0;
    FILE* file = nullptr;
    std::vector<CnnCall> calls;
    size_t cursor = 0;
    uint64_t bytes = 0, setup_sent = 0, setup_received = 0;
    bool poisoned = false;
    segfss::SegmentedDcfGroupWorkspace msb_workspace;
};
thread_local std::string cnn_error;
void cnnCheck(bool condition, const char* message) {
    if (!condition) throw std::runtime_error(message);
}
void writeGe(FILE* file, const GroupElement& value) { writePod(file, value.value); }
GroupElement readGe(FILE* file, int bits) { return GroupElement(readPod<uint64_t>(file), bits); }
template<class T> void writeCnnVector(FILE* file, const std::vector<T>& v) {
    writePod(file, uint64_t(v.size()));
    if (!v.empty() && std::fwrite(v.data(), sizeof(T), v.size(), file) != v.size())
        throw std::runtime_error("CNN material vector write failed");
}
template<class T> std::vector<T> readCnnVector(FILE* file, uint64_t expected) {
    cnnCheck(readPod<uint64_t>(file) == expected, "CNN material vector shape mismatch");
    std::vector<T> v(expected);
    if (expected && std::fread(v.data(), sizeof(T), expected, file) != expected)
        throw std::runtime_error("CNN material vector read failed");
    return v;
}
int cnnSegments(int width) {
    const char* policy = std::getenv("SEGFSS_SEGMENT_POLICY");
    return policy ? segfss::segmentCount(width - 1, segfss::parseSegmentPolicy(policy)) : std::min(4, width - 1);
}
void prepareMsb(CnnContext& ctx, const CnnCall& call) {
    const int width = call.bits, count = call.count;
    std::vector<u8> mask_bits(size_t(count) * width);
    osuCrypto::PRNG rng(osuCrypto::sysRandomSeed()); rng.get(mask_bits.data(), mask_bits.size());
    for (auto& bit : mask_bits) bit &= 1;
    std::vector<GroupElement> arithmetic(mask_bits.size(), GroupElement(0, width));
    B2A(ctx.role, mask_bits.data(), arithmetic.data(), arithmetic.size(), width, ctx.connection);
    std::vector<uint64_t> masks(count);
    std::vector<u8> signs(count);
    std::vector<std::vector<u8>> thresholds(count, std::vector<u8>(width - 1));
    const auto mask = segfss::reference::mask(width);
    for (int row = 0; row < count; ++row) {
        for (int bit = 0; bit < width; ++bit) masks[row] += arithmetic[row * width + bit].value << bit;
        masks[row] &= mask; signs[row] = mask_bits[row * width + width - 1];
        for (int bit = 0; bit < width - 1; ++bit) thresholds[row][bit] = mask_bits[row * width + width - 2 - bit];
    }
    auto material = segfss::segmentedDcfOfflineBatch(ctx.role, thresholds, width - 1,
        cnnSegments(width), ctx.connection, segfss::dcfGenerationOptions(), false);
    material.compactKeys();
    writeCnnVector(ctx.file, masks); writeCnnVector(ctx.file, signs);
    writePod(ctx.file, int(material.prune_equality)); writePod(ctx.file, int(material.prune_leaf_hashes));
    writeCnnVector(ctx.file, material.compact_keys.records);
    writeCnnVector(ctx.file, material.compact_keys.correction_words);
    writeCnnVector(ctx.file, material.scan.a); writeCnnVector(ctx.file, material.scan.b); writeCnnVector(ctx.file, material.scan.c);
}
void onlineMsb(CnnContext& ctx, const CnnCall& call, const uint64_t* input, uint8_t* output) {
    const auto masks = readCnnVector<uint64_t>(ctx.file, call.count);
    const auto signs = readCnnVector<u8>(ctx.file, call.count);
    segfss::SegmentedDcfMaterial material;
    material.role = ctx.role; material.bits = call.bits - 1; material.nseg = cnnSegments(call.bits);
    material.count = call.count; material.need_e = false; material.consumed = false;
    material.layout = segfss::SegmentLayout(material.bits, material.nseg);
    material.prune_equality = readPod<int>(ctx.file); material.prune_leaf_hashes = readPod<int>(ctx.file);
    material.compact_keys.records = readCnnVector<segfss::CggmKeyRecord>(ctx.file, size_t(call.count) * material.nseg);
    material.compact_keys.correction_words = readCnnVector<osuCrypto::block>(ctx.file, size_t(call.count) * (material.bits - material.nseg));
    const auto gates = size_t(call.count) * segfss::planDcfScan(material.nseg, false, material.prune_equality).and_gates_per_row;
    material.scan.a = readCnnVector<u8>(ctx.file, gates); material.scan.b = readCnnVector<u8>(ctx.file, gates);
    material.scan.c = readCnnVector<u8>(ctx.file, gates);
    std::vector<uint64_t> opened(call.count);
    const auto mask = segfss::reference::mask(call.bits);
    for (int row = 0; row < call.count; ++row) opened[row] = (input[row] + masks[row]) & mask;
    const auto remote = segfss::exchangeRingValues(ctx.role == SERVER, opened, call.bits,
        [&](const u8* data, int size) { ctx.connection->send_u8(data, size); },
        [&](u8* data, int size) { ctx.connection->recv_u8(data, size); });
    std::vector<uint64_t> queries(call.count);
    for (int row = 0; row < call.count; ++row) {
        opened[row] = (opened[row] + remote[row]) & mask;
        queries[row] = opened[row] & segfss::reference::mask(call.bits - 1);
    }
    auto values = segfss::segmentedDcfOnlineBatchGroup(ctx.role, {{&queries, &material, false}},
        ctx.connection, &ctx.msb_workspace).front();
    for (int row = 0; row < call.count; ++row)
        output[row] = values[row].L ^ signs[row] ^ (ctx.role == SERVER ? opened[row] >> (call.bits - 1) : 0);
}
}
extern "C" {
void* segfss_sci_cnn_create(int role, const char* host, int port) {
    try {
        auto ctx = std::make_unique<CnnContext>();
        ctx->network = segfss_sci_create(role, host, port);
        cnnCheck(ctx->network, segfss_sci_error());
        ctx->connection = static_cast<Peer*>(segfss_sci_get_peer(ctx->network));
        ctx->role = role + 1;
        segfss_sci_stats stats{};
        segfss_sci_get_stats(ctx->network, &stats);
        ctx->setup_sent = stats.setup_sent_bytes;
        ctx->setup_received = stats.setup_received_bytes;
        return ctx.release();
    } catch (const std::exception& e) { cnn_error = e.what(); return nullptr; }
}
int segfss_sci_cnn_prepare(void* pointer, const CnnCall* calls, int count, uint64_t budget) {
    auto* ctx = static_cast<CnnContext*>(pointer);
    try {
        cnnCheck(ctx && !ctx->file && !ctx->poisoned && count >= 0, "invalid CNN preparation state");
        ctx->calls.assign(calls, calls + count); ctx->cursor = 0;
        ctx->file = std::tmpfile(); cnnCheck(ctx->file, "cannot create CNN private material spool");
        size_t prepared = 0; const auto started = Clock::now();
        for (const auto& call : ctx->calls) {
            cnnCheck(call.count > 0 && call.bits >= 2 && call.bits <= 64, "invalid CNN primitive shape");
            writePod(ctx->file, call);
            if (call.kind == 1) {
                prepareMsb(*ctx, call);
            } else if (call.kind == 2) {
                auto material = segfss::bitxaOffline(ctx->role, call.count, call.bits, ctx->connection);
                for (const auto& value : material) {
                    writeGe(ctx->file, value.delta_x); writePod(ctx->file, value.delta_y_bool);
                    writeGe(ctx->file, value.delta_y_arith); writeGe(ctx->file, value.delta_z);
                }
            } else if (call.kind == 3) {
                auto material = segfss::booleanAndOffline(ctx->role, call.count, ctx->connection);
                writeBytes(ctx->file, material.a); writeBytes(ctx->file, material.b); writeBytes(ctx->file, material.c);
            } else throw std::runtime_error("unknown CNN primitive kind");
            const auto size = std::ftell(ctx->file);
            cnnCheck(size >= 0, "CNN spool accounting failed"); ctx->bytes = size;
            cnnCheck(!budget || ctx->bytes <= budget, "CNN preprocessing spool exceeds --spool-budget");
            if (++prepared % 100 == 0 || prepared == ctx->calls.size()) std::fprintf(stderr,
                "CNN_NONLINEAR_PREPARED calls=%zu/%zu seconds=%.6f spool_bytes=%llu\n", prepared, ctx->calls.size(), ms(started) / 1000.0, static_cast<unsigned long long>(ctx->bytes));
        }
        cnnCheck(std::fflush(ctx->file) == 0 && std::fseek(ctx->file, 0, SEEK_SET) == 0, "CNN spool rewind failed");
        return 0;
    } catch (const std::exception& e) { if (ctx) ctx->poisoned = true; cnn_error = e.what(); return -1; }
}
int segfss_sci_cnn_online(void* pointer, const CnnCall* call, const uint64_t* input,
    const uint8_t* a, const uint8_t* b, uint64_t* output, uint8_t* bits) {
    auto* ctx = static_cast<CnnContext*>(pointer);
    try {
        cnnCheck(ctx && ctx->file && !ctx->poisoned && call && ctx->cursor < ctx->calls.size(), "CNN material exhausted");
        const auto stored = readPod<CnnCall>(ctx->file);
        cnnCheck(stored.kind == call->kind && stored.count == call->count && stored.bits == call->bits,
            "CNN primitive differs from prepared schedule");
        ++ctx->cursor;
        if (call->kind == 1) {
            onlineMsb(*ctx, *call, input, bits);
        } else if (call->kind == 2) {
            std::vector<segfss::BitXAPreprocessing> material(call->count);
            std::vector<GroupElement> values; values.reserve(call->count);
            std::vector<u8> selection(a, a + call->count);
            for (int i = 0; i < call->count; ++i) {
                auto& m = material[i]; m.bitwidth = call->bits;
                m.delta_x = readGe(ctx->file, call->bits); m.delta_y_bool = readPod<u8>(ctx->file);
                m.delta_y_arith = readGe(ctx->file, call->bits); m.delta_z = readGe(ctx->file, call->bits);
                values.emplace_back(input[i], call->bits);
            }
            auto result = segfss::bitxaOnline(ctx->role, values, selection, material, ctx->connection);
            for (int i = 0; i < call->count; ++i) output[i] = result[i].value;
        } else if (call->kind == 3) {
            segfss::BooleanAndPreprocessing material;
            material.a = readBytes(ctx->file, call->count, "and-a");
            material.b = readBytes(ctx->file, call->count, "and-b");
            material.c = readBytes(ctx->file, call->count, "and-c");
            segfss::booleanAndOnline(ctx->role, a, b, bits, call->count, material, ctx->connection);
        }
        return 0;
    } catch (const std::exception& e) { if (ctx) ctx->poisoned = true; cnn_error = e.what(); return -1; }
}
int segfss_sci_cnn_counters(void* pointer, uint64_t* sent, uint64_t* received, uint64_t* bytes) {
    auto* ctx = static_cast<CnnContext*>(pointer); if (!ctx) return -1;
    *sent = ctx->setup_sent + ctx->connection->bytesSent;
    *received = ctx->setup_received + ctx->connection->bytesReceived; *bytes = ctx->bytes; return 0;
}
int segfss_sci_cnn_finish(void* pointer) {
    auto* ctx = static_cast<CnnContext*>(pointer);
    if (!ctx || ctx->poisoned || ctx->cursor != ctx->calls.size()) { cnn_error = "CNN prepared inventory not consumed"; return -1; }
    if (ctx->file) std::fclose(ctx->file); ctx->file = nullptr; ctx->calls.clear(); return 0;
}
void segfss_sci_cnn_destroy(void* pointer) {
    auto* ctx = static_cast<CnnContext*>(pointer); if (!ctx) return;
    if (ctx->file) std::fclose(ctx->file); segfss_sci_destroy(ctx->network); delete ctx;
}
const char* segfss_sci_cnn_error() { return cnn_error.c_str(); }
}
