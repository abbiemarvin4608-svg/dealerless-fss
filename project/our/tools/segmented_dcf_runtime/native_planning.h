#ifndef SDCF_NATIVE_PLANNING_H
#define SDCF_NATIVE_PLANNING_H
#include "native_preprocessing.h"
#include "native_dpf_lut.h"
#include "native_folded.h"
#include <algorithm>
#include <functional>
#include <sstream>

namespace sdcf_planning {
struct OTCall {
    std::string protocol;
    int count, bits, group;
    std::vector<int> widths;
};
struct TripleCall { int count, method; bool packed; int offset; };
struct Stream {
    int context;
    std::string direction;
    std::vector<OTCall> calls;
    std::function<void(int, const std::vector<OTCall>&)> prepare;
};
struct TripleStream { int context; std::vector<TripleCall> calls; };
inline std::mutex mutex;
inline std::map<const void*, Stream> streams;
inline std::map<const void*, TripleStream> triple_streams;
inline std::atomic<uint64_t> prg_blocks{0};
inline bool active() { return sdcf_preprocess::planning_active.load(); }
inline void register_ot(const void* ot, int context, const std::string& direction,
                        std::function<void(int, const std::vector<OTCall>&)> prepare) {
    std::lock_guard<std::mutex> lock(mutex);
    auto found = streams.find(ot);
    if (found != streams.end()) {
        if (found->second.context != context || found->second.direction != direction)
            throw std::runtime_error("public plan context changed");
        return;
    }
    for (const auto& item : streams)
        if (item.second.context == context && item.second.direction == direction)
            throw std::runtime_error("duplicate public plan context");
    streams.emplace(ot, Stream{context, direction, {}, std::move(prepare)});
    sdcf_preprocess::register_context(ot, context, direction);
}
inline void register_triples(const void* pack, int context) {
    std::lock_guard<std::mutex> lock(mutex);
    auto found = triple_streams.find(pack);
    if (found != triple_streams.end() && found->second.context != context)
        throw std::runtime_error("public triple context changed");
    triple_streams[pack].context = context;
}
inline void begin() {
    std::lock_guard<std::mutex> lock(mutex);
    if (active() || sdcf_preprocess::phase != sdcf_preprocess::Phase::Preparing)
        throw std::runtime_error("public planning outside prepare");
    for (auto& item : streams) item.second.calls.clear();
    for (auto& item : triple_streams) item.second.calls.clear();
    if (sdcf_dpf::plan_begin) sdcf_dpf::plan_begin();
    if (sdcf_folded::plan_begin) sdcf_folded::plan_begin();
    sdcf_preprocess::planning_active = true;
}
inline bool record_ot(const void* ot, const std::string& protocol, int count, int bits,
                      int group = 1, const std::vector<int>& widths = {}) {
    if (!active()) return false;
    if (count < 1 || bits < 1 || bits > 64 || group < 1)
        throw std::runtime_error("invalid public OT plan");
    std::lock_guard<std::mutex> lock(mutex);
    streams.at(ot).calls.push_back({protocol, count, bits, group, widths});
    return true;
}
inline void record_triple(const void* pack, int count, int method, bool packed, int offset) {
    if (!active()) throw std::runtime_error("triple request outside public planning");
    std::lock_guard<std::mutex> lock(mutex);
    triple_streams.at(pack).calls.push_back({count, method, packed, offset});
}
inline std::vector<TripleCall> triple_calls(const void* pack) {
    std::lock_guard<std::mutex> lock(mutex);
    return triple_streams.at(pack).calls;
}
inline int context(const void* pack) {
    std::lock_guard<std::mutex> lock(mutex);
    return triple_streams.at(pack).context;
}
inline std::vector<const void*> ordered_streams() {
    std::vector<const void*> result;
    for (const auto& item : streams) result.push_back(item.first);
    std::sort(result.begin(), result.end(), [](const void* a, const void* b) {
        const auto& x = streams.at(a); const auto& y = streams.at(b);
        return std::make_pair(x.context, x.direction) < std::make_pair(y.context, y.direction);
    });
    return result;
}
inline std::string end() {
    std::lock_guard<std::mutex> lock(mutex);
    if (!active()) throw std::runtime_error("public planning not active");
    sdcf_preprocess::planning_active = false;
    std::ostringstream result;
    for (const void* ot : ordered_streams()) {
        const auto& stream = streams.at(ot);
        for (size_t index = 0; index < stream.calls.size(); ++index) {
            const auto& call = stream.calls[index];
            result << "O " << stream.context << ' ' << stream.direction << ' ' << index << ' '
                << call.protocol << ' ' << call.count << ' ' << call.bits << ' ' << call.group;
            for (int width : call.widths) result << ' ' << width;
            result << '\n';
        }
    }
    std::vector<const void*> packs;
    for (const auto& item : triple_streams) packs.push_back(item.first);
    std::sort(packs.begin(), packs.end(), [](const void* a, const void* b) {
        return triple_streams.at(a).context < triple_streams.at(b).context;
    });
    for (const void* pack : packs) {
        const auto& stream = triple_streams.at(pack);
        for (size_t index = 0; index < stream.calls.size(); ++index) {
            const auto& call = stream.calls[index];
            result << "T " << stream.context << ' ' << index << ' ' << call.count << ' '
                << call.method << ' ' << int(call.packed) << ' ' << call.offset << '\n';
        }
    }
    if (sdcf_dpf::plan_end) result << sdcf_dpf::plan_end();
    if (sdcf_folded::plan_end) result << sdcf_folded::plan_end();
    size_t ot_calls = 0, triple_calls = 0;
    for (const auto& item : streams) ot_calls += item.second.calls.size();
    for (const auto& item : triple_streams) triple_calls += item.second.calls.size();
    sdcf_log::Line() << "NATIVE_PUBLIC_PLAN_SUMMARY {\"query\":" << sdcf_preprocess::current_query
        << ",\"registered_ot_streams\":" << streams.size()
        << ",\"registered_triple_contexts\":" << triple_streams.size()
        << ",\"ot_calls\":" << ot_calls << ",\"triple_calls\":" << triple_calls << "}\n";
    return result.str();
}
inline void prepare_ots(int query) {
    if (active()) throw std::runtime_error("material replay during planning");
    sdcf_preprocess::require_query(query);
    for (const void* ot : ordered_streams()) {
        const auto& stream = streams.at(ot);
        if (stream.calls.empty()) continue;
        std::vector<sdcf_preprocess::MaterialCall> calls;
        for (size_t index = 0; index < stream.calls.size(); ++index) {
            const auto& call = stream.calls[index];
            calls.push_back({query, int(index), call.protocol, call.count, call.bits, call.group, call.widths});
            sdcf_log::Line log;
            log << "NATIVE_PUBLIC_PLAN {\"query\":" << query << ",\"context\":" << stream.context
                << ",\"direction\":\"" << stream.direction << "\",\"call\":" << index
                << ",\"protocol\":\"" << call.protocol << "\",\"count\":" << call.count
                << ",\"message_bits\":" << call.bits << ",\"messages_per_ot\":" << call.group << ",\"widths\":[";
            for (size_t i = 0; i < call.widths.size(); ++i) { if (i) log << ','; log << call.widths[i]; }
            log << "]}\n";
        }
        sdcf_preprocess::plan(ot, calls);
        stream.prepare(query, stream.calls);
    }
}
template<class IO> void compare_plan(IO* io, int role, const std::string& plan) {
    if (active()) throw std::runtime_error("plan agreement during planning");
    const uint64_t length = plan.size(); uint64_t remote_length = 0;
    auto send_length = [&] { io->send_data(&length, sizeof(length)); io->flush(); };
    auto recv_length = [&] { io->recv_data(&remote_length, sizeof(remote_length)); };
    if (role == 1) { send_length(); recv_length(); } else { recv_length(); send_length(); }
    if (length != remote_length) throw std::runtime_error("public material plan differs between parties");
    std::string remote(length, '\0');
    auto send = [&] { io->send_data(plan.data(), int(length)); io->flush(); };
    auto receive = [&] { io->recv_data(remote.data(), int(length)); };
    if (role == 1) { send(); receive(); } else { receive(); send(); }
    if (plan != remote) throw std::runtime_error("public material plan differs between parties");
}
}
#endif
