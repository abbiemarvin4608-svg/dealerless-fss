#ifndef SDCF_NATIVE_PREPROCESSING_H
#define SDCF_NATIVE_PREPROCESSING_H
#include "runtime_log.h"
#include <algorithm>
#include <atomic>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <map>
#include <mutex>
#include <stdexcept>
#include <string>
#include <vector>

namespace sdcf_preprocess {
struct MaterialCall {
    int query = -1, call = -1;
    std::string protocol;
    int count = 0, message_bits = 0, messages_per_ot = 1;
    std::vector<int> widths;
};
enum class Phase { Idle, Preparing, Sealed, Ready, Input, Online, Output };
struct Inventory {
    int context = 0;
    std::string direction;
    std::vector<MaterialCall> calls;
    size_t cursor = 0;
    uint64_t requested = 0, generated = 0, initialized = 0, consumed = 0;
};
inline std::mutex mutex;
inline std::map<const void*, Inventory> inventories;
inline std::atomic<uint64_t> generation_epoch{0};
inline std::atomic<bool> planning_active{false};
inline std::atomic<int> offline_purpose_depth{0};
inline bool nested_offline() { return offline_purpose_depth.load() != 0; }
class GenerationPurposeScope {
    bool enabled_;
public:
    explicit GenerationPurposeScope(bool enabled = true) : enabled_(enabled) {
        if (enabled_) ++offline_purpose_depth;
    }
    ~GenerationPurposeScope() { if (enabled_) --offline_purpose_depth; }
    GenerationPurposeScope(const GenerationPurposeScope&) = delete;
    GenerationPurposeScope& operator=(const GenerationPurposeScope&) = delete;
};
inline Phase phase = Phase::Idle;
inline int current_query = -1;
inline bool strict = false;

inline bool enabled() {
    const char* value = std::getenv("SDCF_NATIVE_PREPROCESSING");
    return value && std::string(value) != "0";
}
inline void require_query(int query) {
    if (query != current_query) throw std::runtime_error("native material query mismatch");
}
inline void register_context(const void* ot, int context, const std::string& direction) {
    std::lock_guard<std::mutex> lock(mutex);
    auto& stock = inventories[ot];
    stock.context = context; stock.direction = direction;
}
inline void begin_prepare(int query, bool strict_mode) {
    std::lock_guard<std::mutex> lock(mutex);
    if (phase != Phase::Idle || query <= current_query)
        throw std::runtime_error("native material query reused or unfinished");
    current_query = query; strict = strict_mode; phase = Phase::Preparing;
    for (auto& item : inventories) {
        auto& stock = item.second;
        stock.calls.clear(); stock.cursor = 0;
        stock.requested = stock.generated = stock.initialized = stock.consumed = 0;
    }
}
inline void plan(const void* ot, const std::vector<MaterialCall>& calls) {
    std::lock_guard<std::mutex> lock(mutex);
    if (phase != Phase::Preparing || !strict)
        throw std::runtime_error("native material plan outside prepare");
    auto& stock = inventories.at(ot);
    if (!stock.calls.empty()) throw std::runtime_error("native material plan repeated");
    for (const auto& call : calls) {
        if (call.query != current_query || call.call < 0 || call.count < 1 ||
            call.message_bits < 1 || call.message_bits > 64 || call.messages_per_ot < 1)
            throw std::runtime_error("invalid native material contract");
        if (!call.widths.empty() && (call.count % call.widths.size() ||
                *std::max_element(call.widths.begin(), call.widths.end()) != call.message_bits ||
                std::any_of(call.widths.begin(), call.widths.end(), [](int width) { return width < 1 || width > 64; })))
            throw std::runtime_error("invalid native material full-width contract");
        stock.requested += uint64_t(call.count);
    }
    stock.calls = calls;
}
inline void generation_begin(const void*) {
    if (planning_active.load()) throw std::runtime_error("unplanned material generation during public planning");
    std::lock_guard<std::mutex> lock(mutex);
    if (strict && phase != Phase::Preparing)
        throw std::runtime_error("material generation after prepare");
}
inline void generation_complete(const void* ot, int count) {
    std::lock_guard<std::mutex> lock(mutex);
    ++generation_epoch;
    const auto found = inventories.find(ot);
    if (strict && !nested_offline() && found != inventories.end() && found->second.requested)
        found->second.generated += uint64_t(count);
}
inline void random_ot_complete(const void* ot, int count) {
    std::lock_guard<std::mutex> lock(mutex);
    if (strict && phase != Phase::Preparing)
        throw std::runtime_error("random OT initialization after prepare");
    const auto found = inventories.find(ot);
    if (strict && !nested_offline() && found != inventories.end()) found->second.initialized += uint64_t(count);
}
inline void seal_prepare(int query) {
    std::lock_guard<std::mutex> lock(mutex);
    require_query(query);
    if (phase != Phase::Preparing) throw std::runtime_error("native prepare phase mismatch");
    for (const auto& item : inventories) {
        const auto& stock = item.second;
        if (strict && stock.generated < stock.requested)
            throw std::runtime_error("native material not generated before input");
        if (strict && stock.initialized < stock.requested)
            throw std::runtime_error("native random OT outputs incomplete");
        if (strict && stock.requested)
            sdcf_log::Line() << "NATIVE_MATERIAL_PREPARED {\"query\":" << query
                << ",\"context\":" << stock.context << ",\"direction\":\"" << stock.direction
                << "\",\"requested\":" << stock.requested << ",\"generated\":" << stock.generated << "}\n";
    }
    phase = Phase::Sealed;
}
inline void material_ready(int query) {
    std::lock_guard<std::mutex> lock(mutex);
    require_query(query);
    if (phase != Phase::Sealed) throw std::runtime_error("native material-ready phase mismatch");
    phase = Phase::Ready;
    sdcf_log::Line() << "MATERIAL_READY {\"query\":" << query
              << ",\"native_generation_epoch\":" << generation_epoch.load()
              << ",\"strict\":" << (strict ? "true" : "false") << "}\n";
}
inline void begin_input(int query) {
    std::lock_guard<std::mutex> lock(mutex);
    require_query(query);
    if (phase != Phase::Ready) throw std::runtime_error("input before native material-ready");
    phase = Phase::Input;
}
inline void enter_online(int query) {
    std::lock_guard<std::mutex> lock(mutex);
    require_query(query);
    if (phase != Phase::Input) throw std::runtime_error("native online phase mismatch");
    phase = Phase::Online;
}
inline void consume(const void* ot, const std::string& protocol, int count,
                    int message_bits, int messages_per_ot = 1, uint64_t correction_bits = 0,
                    const std::vector<int>& widths = {}) {
    std::lock_guard<std::mutex> lock(mutex);
    if (!strict) return;
    if (phase != Phase::Online) throw std::runtime_error("native material consumption outside online");
    const auto found = inventories.find(ot);
    if (found == inventories.end() || found->second.cursor >= found->second.calls.size())
        throw std::runtime_error("material inventory exhausted");
    auto& stock = found->second;
    const auto& call = stock.calls[stock.cursor];
    if (call.query != current_query || call.protocol != protocol || call.count != count ||
        call.message_bits != message_bits || call.messages_per_ot != messages_per_ot ||
        (!call.widths.empty() && call.widths != widths))
        throw std::runtime_error("material contract mismatch");
    ++stock.cursor; stock.consumed += uint64_t(count);
    sdcf_log::Line log;
    log << "NATIVE_MATERIAL_CONSUMED {\"query\":" << current_query
        << ",\"call\":" << call.call << ",\"context\":" << stock.context
        << ",\"direction\":\"" << stock.direction << "\",\"protocol\":\"" << protocol
        << "\",\"message_bits\":" << message_bits << ",\"messages_per_ot\":" << messages_per_ot << ",\"count\":" << count
        << ",\"choice_correction_bits\":" << (correction_bits ? correction_bits : uint64_t(count)) << ",\"widths\":[";
    for (size_t index = 0; index < widths.size(); ++index) { if (index) log << ','; log << widths[index]; }
    log << "]}\n";
}
inline void leave_online(int query) {
    std::lock_guard<std::mutex> lock(mutex);
    require_query(query);
    if (phase != Phase::Online) throw std::runtime_error("native output phase mismatch");
    for (const auto& item : inventories) {
        const auto& stock = item.second;
        if (strict && (stock.cursor != stock.calls.size() || stock.consumed != stock.requested))
            throw std::runtime_error("unconsumed native material");
        if (strict && stock.requested)
            sdcf_log::Line() << "NATIVE_MATERIAL_BALANCE {\"query\":" << query
                << ",\"context\":" << stock.context << ",\"direction\":\"" << stock.direction
                << "\",\"requested\":" << stock.requested << ",\"generated\":" << stock.generated
                << ",\"consumed\":" << stock.consumed << ",\"padding_discarded\":"
                << stock.generated - stock.consumed << ",\"remaining\":0}\n";
    }
    phase = Phase::Output;
}
inline void finish_query(int query) {
    std::lock_guard<std::mutex> lock(mutex);
    require_query(query);
    if (phase != Phase::Output) throw std::runtime_error("native query completion phase mismatch");
    phase = Phase::Idle; strict = false;
}
}
#endif
