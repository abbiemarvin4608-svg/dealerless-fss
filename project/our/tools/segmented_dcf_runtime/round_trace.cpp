#include "round_trace.h"
#include <cstdio>
#include <cstdlib>
#include <mutex>
#include <map>
#include <stdexcept>
#include <string>
#include <vector>

namespace {
std::mutex mutex;
FILE* output = nullptr;
bool enabled = false;
uint64_t next_event = 0, next_actor = 1;
uint64_t next_scope = 0;
std::vector<uint64_t> predecessors(2, 0);
struct Semantic {
    std::string label;
    int query, layer, slot, call, kind, stage;
};
struct ScopeRecord { Semantic semantic; uint64_t owner, parent; bool open = true; };
std::map<uint64_t, ScopeRecord> scopes;
std::vector<std::vector<uint64_t>> actor_scopes(2);
std::map<uint64_t, std::vector<uint64_t>> fork_scopes;
thread_local uint64_t actor = 0;
std::string phase = "unassigned";
int query = -1;
char buffer[512 * 1024];

uint64_t event(const char* kind, uint64_t dependency = 0, int channel = 0,
               uint64_t offset = 0, uint64_t length = 0, const ScopeRecord* boundary = nullptr) {
    if (!actor) throw std::runtime_error("round trace encountered an unregistered communication actor");
    const uint64_t identifier = ++next_event;
    const int status = std::fprintf(output,
        "{\"type\":\"%s\",\"id\":%llu,\"actor\":%llu,\"prev\":%llu,"
        "\"phase\":\"%s\",\"query\":%d,\"dependency\":%llu,"
        "\"channel\":%d,\"offset\":%llu,\"length\":%llu,\"scope\":%llu",
        kind, static_cast<unsigned long long>(identifier), static_cast<unsigned long long>(actor),
        static_cast<unsigned long long>(predecessors.at(actor)), phase.c_str(), query,
        static_cast<unsigned long long>(dependency), channel,
        static_cast<unsigned long long>(offset), static_cast<unsigned long long>(length),
        static_cast<unsigned long long>(actor_scopes.at(actor).empty() ? 0 : actor_scopes.at(actor).back()));
    if (status < 0) throw std::runtime_error("round trace write failed");
    if (boundary) {
        const auto& s = boundary->semantic;
        if (std::fprintf(output, ",\"parent_scope\":%llu,\"semantic\":{\"label\":\"%s\","
            "\"query\":%d,\"layer\":%d,\"slot\":%d,\"call\":%d,\"kind\":%d,\"stage\":%d}",
            static_cast<unsigned long long>(boundary->parent), s.label.c_str(),
            s.query, s.layer, s.slot, s.call, s.kind, s.stage) < 0)
            throw std::runtime_error("round trace semantic write failed");
    }
    if (std::fputs("}\n", output) < 0) throw std::runtime_error("round trace write failed");
    predecessors[actor] = identifier;
    return identifier;
}
}

namespace sdcf_round {
void initialize(int party) {
    const char* path = std::getenv("SDCF_ROUND_TRACE");
    if (!path || !*path) return;
    if (party != 1 && party != 2) throw std::invalid_argument("round trace party must be 1 or 2");
    std::lock_guard<std::mutex> lock(mutex);
    if (output) throw std::logic_error("round trace initialized twice");
    output = std::fopen(path, "w");
    if (!output) throw std::runtime_error("cannot open round trace output");
    std::setvbuf(output, buffer, _IOFBF, sizeof(buffer));
    enabled = true; actor = 1;
    std::fprintf(output, "{\"type\":\"meta\",\"version\":1,\"party\":%d}\n", party);
}
bool active() { return enabled; }
void phase_begin(const char* name, int value) {
    if (!enabled) return;
    std::lock_guard<std::mutex> lock(mutex);
    phase = name; query = value;
    event("phase_begin");
}
void phase_end(const char* name, int value) {
    if (!enabled) return;
    std::lock_guard<std::mutex> lock(mutex);
    if (phase != name || query != value) throw std::runtime_error("round trace phase boundary mismatch");
    if (!actor_scopes.at(actor).empty()) throw std::runtime_error("phase ended with an open semantic scope");
    event("phase_end");
}
void finish() {
    if (!enabled) return;
    std::lock_guard<std::mutex> lock(mutex);
    std::fprintf(output, "{\"type\":\"complete\",\"events\":%llu}\n",
                 static_cast<unsigned long long>(next_event));
    if (std::fclose(output)) throw std::runtime_error("round trace close failed");
    output = nullptr; enabled = false;
}
}

extern "C" void sdcf_round_trace_io(int direction, int channel, uint64_t offset, uint64_t length) {
    if (!enabled || !length) return;
    if (direction != 0 && direction != 1) throw std::invalid_argument("invalid trace I/O direction");
    std::lock_guard<std::mutex> lock(mutex);
    event(direction ? "recv" : "send", 0, channel, offset, length);
}
extern "C" uint64_t sdcf_round_actor_fork() {
    if (!enabled) return 0;
    std::lock_guard<std::mutex> lock(mutex);
    const auto fork = event("fork");
    fork_scopes.emplace(fork, actor_scopes.at(actor));
    return fork;
}
extern "C" uint64_t sdcf_round_actor_enter(uint64_t fork_event) {
    if (!enabled) return 0;
    std::lock_guard<std::mutex> lock(mutex);
    const auto previous = actor;
    actor = ++next_actor;
    predecessors.push_back(0);
    actor_scopes.push_back(fork_scopes.at(fork_event));
    event("start", fork_event);
    return previous;
}
extern "C" uint64_t sdcf_round_actor_leave(uint64_t previous_actor) {
    if (!enabled) return 0;
    std::lock_guard<std::mutex> lock(mutex);
    for (auto scope : actor_scopes.at(actor))
        if (scopes.at(scope).owner == actor)
            throw std::runtime_error("actor ended with an open semantic scope");
    const auto end = event("end");
    actor = previous_actor;
    return end;
}
extern "C" void sdcf_round_actor_join(uint64_t child_end) {
    if (!enabled) return;
    std::lock_guard<std::mutex> lock(mutex);
    if (!child_end) throw std::runtime_error("round trace child completion missing");
    event("join", child_end);
}

extern "C" uint64_t sdcf_round_scope_begin(const char* label, int value, int layer,
    int slot, int call, int kind, int stage) {
    if (!enabled) return 0;
    std::lock_guard<std::mutex> lock(mutex);
    const std::string name = label ? label : "";
    if (name != "activation" && name != "folded_call" && name != "activation_stage" && name != "folded_prepare")
        throw std::invalid_argument("unknown semantic trace scope");
    auto& stack = actor_scopes.at(actor);
    const auto parent = stack.empty() ? 0 : stack.back();
    const Semantic previous = parent ? scopes.at(parent).semantic : Semantic{"", query, -1, -1, -1, -1, -1};
    Semantic semantic{name, value < 0 ? query : value,
        layer < 0 ? previous.layer : layer, slot < 0 ? previous.slot : slot,
        call < 0 ? previous.call : call, kind < 0 ? previous.kind : kind, stage};
    if (semantic.query != query) throw std::invalid_argument("semantic trace query differs from its phase");
    const auto identifier = ++next_scope;
    auto inserted = scopes.emplace(identifier, ScopeRecord{semantic, actor, parent});
    stack.push_back(identifier);
    event("scope_begin", 0, 0, 0, 0, &inserted.first->second);
    return identifier;
}
extern "C" void sdcf_round_scope_end(uint64_t identifier) {
    if (!enabled) return;
    std::lock_guard<std::mutex> lock(mutex);
    auto& stack = actor_scopes.at(actor);
    if (!identifier || stack.empty() || stack.back() != identifier ||
        scopes.at(identifier).owner != actor || !scopes.at(identifier).open)
        throw std::runtime_error("semantic trace scope end is not owned by its actor");
    event("scope_end");
    scopes.at(identifier).open = false;
    stack.pop_back();
}
