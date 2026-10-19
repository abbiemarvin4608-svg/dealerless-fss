#ifndef SDCF_PREPARED_TRIPLES_H
#define SDCF_PREPARED_TRIPLES_H
#include "native_preprocessing.h"
#include "native_planning.h"
#include <algorithm>
#include <cstring>
#include <deque>

namespace sdcf_triples {
struct Call {
    int count = 0, method = 0;
    bool packed = false;
    int offset = 1;
};
struct Material {
    Call call;
    std::vector<uint8_t> a, b, c;
};
struct Inventory {
    int query = -1, context = -1;
    std::vector<Call> calls;
    std::deque<Material> stock;
    size_t consumed_calls = 0;
    uint64_t requested = 0, generated = 0, consumed = 0;
};
inline std::mutex mutex;
inline std::map<const void*, Inventory> inventories;
inline std::atomic<uint64_t> generation_epoch{0};

inline size_t bytes(const Call& call) {
    return call.packed ? size_t(call.count / 8) : size_t(call.count);
}
inline void validate(const Call& call) {
    if (call.method != 2 && call.method != 3)
        throw std::runtime_error("unsafe bit-triple generation method");
    if (call.count < 1 || call.count % 2 || call.offset < 1 ||
        call.count % call.offset || (call.packed && call.count % 8) ||
        (call.method == 3 && call.count % (2 * call.offset)))
        throw std::runtime_error("invalid bit-triple material contract");
}
inline void plan(const void* pack, int query, int context, const std::vector<Call>& calls) {
    sdcf_preprocess::require_query(query);
    if (!sdcf_preprocess::strict || sdcf_preprocess::phase != sdcf_preprocess::Phase::Preparing)
        throw std::runtime_error("bit-triple plan outside prepare");
    std::lock_guard<std::mutex> lock(mutex);
    auto& inventory = inventories[pack];
    if (!inventory.stock.empty() || inventory.consumed != inventory.requested)
        throw std::runtime_error("unconsumed bit-triple material from previous query");
    if (inventory.query == query) throw std::runtime_error("bit-triple plan repeated");
    inventory = Inventory{};
    inventory.query = query; inventory.context = context; inventory.calls = calls;
    for (size_t index = 0; index < calls.size(); ++index) {
        const auto& call = calls[index];
        validate(call); inventory.requested += call.count;
        sdcf_log::Line() << "NATIVE_PUBLIC_TRIPLE_PLAN {\"query\":" << query << ",\"context\":" << context
            << ",\"call\":" << index << ",\"count\":" << call.count << ",\"method\":" << call.method
            << ",\"packed\":" << (call.packed ? "true" : "false") << ",\"offset\":" << call.offset << "}\n";
    }
}
template <class Generator> void prepare(Generator* generator) {
    std::vector<Call> calls;
    {
        std::lock_guard<std::mutex> lock(mutex);
        const auto& inventory = inventories.at(generator->otpack);
        sdcf_preprocess::require_query(inventory.query);
        if (inventory.generated || !inventory.stock.empty())
            throw std::runtime_error("bit-triple preparation repeated");
        calls = inventory.calls;
    }
    for (const auto& call : calls) {
        Material material;
        material.call = call;
        material.a.resize(bytes(call)); material.b.resize(bytes(call)); material.c.resize(bytes(call));
        generator->generate(generator->otpack->party, material.a.data(), material.b.data(), material.c.data(),
                            call.count, static_cast<typename Generator::sdcf_triple_method>(call.method), call.packed, call.offset);
        std::lock_guard<std::mutex> lock(mutex);
        auto& inventory = inventories.at(generator->otpack);
        inventory.generated += call.count;
        inventory.stock.push_back(std::move(material));
    }
}
inline bool intercept(const void* pack, uint8_t* a, uint8_t* b, uint8_t* c,
                      int count, int method, bool packed, int offset) {
    const Call call{count, method, packed, offset};
    if (sdcf_planning::active()) {
        validate(call);
        sdcf_planning::record_triple(pack, count, method, packed, offset);
        std::memset(a, 0, bytes(call)); std::memset(b, 0, bytes(call)); std::memset(c, 0, bytes(call));
        return true;
    }
    if (!sdcf_preprocess::strict) return false;
    if (sdcf_preprocess::phase == sdcf_preprocess::Phase::Preparing) { validate(call); return false; }
    if (sdcf_preprocess::phase != sdcf_preprocess::Phase::Online)
        throw std::runtime_error("bit-triple consumption outside online");
    std::lock_guard<std::mutex> lock(mutex);
    auto found = inventories.find(pack);
    if (found == inventories.end() || found->second.stock.empty())
        throw std::runtime_error("bit-triple inventory exhausted");
    auto& inventory = found->second;
    sdcf_preprocess::require_query(inventory.query);
    const auto& wanted = inventory.stock.front().call;
    if (wanted.count != count || wanted.method != method || wanted.packed != packed || wanted.offset != offset)
        throw std::runtime_error("bit-triple material contract mismatch");
    auto material = std::move(inventory.stock.front());
    inventory.stock.pop_front(); ++inventory.consumed_calls; inventory.consumed += count;
    std::memcpy(a, material.a.data(), bytes(call)); std::memcpy(b, material.b.data(), bytes(call));
    std::memcpy(c, material.c.data(), bytes(call));
    sdcf_log::Line() << "NATIVE_TRIPLE_CONSUMED {\"query\":" << inventory.query
        << ",\"context\":" << inventory.context << ",\"call\":" << inventory.consumed_calls - 1
        << ",\"count\":" << count << ",\"method\":" << method
        << ",\"packed\":" << (packed ? "true" : "false") << ",\"offset\":" << offset << "}\n";
    return true;
}
inline void completed() {
    ++generation_epoch;
    ++sdcf_preprocess::generation_epoch;
}
inline void seal(int query) {
    if (!sdcf_preprocess::strict) return;
    std::lock_guard<std::mutex> lock(mutex);
    for (const auto& item : sdcf_planning::triple_streams) {
        if (item.second.calls.empty()) continue;
        const auto found = inventories.find(item.first);
        if (found == inventories.end() || found->second.query != query)
            throw std::runtime_error("missing native triple query inventory");
        const auto& planned = item.second.calls;
        const auto& stocked = found->second.calls;
        if (planned.size() != stocked.size()) throw std::runtime_error("native triple query plan mismatch");
        for (size_t index = 0; index < planned.size(); ++index)
            if (planned[index].count != stocked[index].count || planned[index].method != stocked[index].method ||
                planned[index].packed != stocked[index].packed || planned[index].offset != stocked[index].offset)
                throw std::runtime_error("native triple query plan mismatch");
    }
    for (const auto& item : inventories) {
        const auto& inventory = item.second;
        if (inventory.query != query || !inventory.requested) continue;
        if (inventory.generated != inventory.requested || inventory.stock.size() != inventory.calls.size())
            throw std::runtime_error("bit-triple material not generated before input");
        size_t standard = 0, correlated = 0;
        for (const auto& call : inventory.calls) { standard += call.method == 2; correlated += call.method == 3; }
        sdcf_log::Line() << "NATIVE_TRIPLE_PREPARED {\"query\":" << query << ",\"context\":" << inventory.context
            << ",\"requested\":" << inventory.requested << ",\"generated\":" << inventory.generated
            << ",\"standard_calls\":" << standard << ",\"correlated_calls\":" << correlated << "}\n";
    }
}
inline void balance(int query) {
    if (!sdcf_preprocess::strict) return;
    std::lock_guard<std::mutex> lock(mutex);
    for (const auto& item : inventories) {
        const auto& inventory = item.second;
        if (inventory.query != query || !inventory.requested) continue;
        if (!inventory.stock.empty() || inventory.consumed != inventory.requested ||
            inventory.consumed_calls != inventory.calls.size())
            throw std::runtime_error("unconsumed bit-triple material");
        sdcf_log::Line() << "NATIVE_TRIPLE_BALANCE {\"query\":" << query << ",\"context\":" << inventory.context
            << ",\"requested\":" << inventory.requested << ",\"generated\":" << inventory.generated
            << ",\"consumed\":" << inventory.consumed << ",\"remaining\":0,\"padding_discarded\":0}\n";
    }
}
}
#endif
