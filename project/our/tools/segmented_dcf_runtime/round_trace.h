#ifndef SDCF_ROUND_TRACE_H
#define SDCF_ROUND_TRACE_H
#include <cstdint>
#include <memory>
#include <thread>
#include <tuple>
#include <utility>

extern "C" {
void sdcf_round_trace_io(int direction, int channel, uint64_t offset, uint64_t length) __attribute__((weak));
uint64_t sdcf_round_actor_fork();
uint64_t sdcf_round_actor_enter(uint64_t fork_event);
uint64_t sdcf_round_actor_leave(uint64_t previous_actor);
void sdcf_round_actor_join(uint64_t child_end);
uint64_t sdcf_round_scope_begin(const char* label, int query, int layer,
                              int slot, int call, int kind, int stage);
void sdcf_round_scope_end(uint64_t scope);
}

namespace sdcf_round {
void initialize(int party);
bool active();
void phase_begin(const char* phase, int query);
void phase_end(const char* phase, int query);
void finish();

class SemanticScope {
    uint64_t scope_ = 0;
public:
    SemanticScope(const char* label, int query = -1, int layer = -1,
                  int slot = -1, int call = -1, int kind = -1, int stage = -1)
        : scope_(sdcf_round_scope_begin(label, query, layer, slot, call, kind, stage)) {}
    ~SemanticScope() { if (scope_) sdcf_round_scope_end(scope_); }
    SemanticScope(const SemanticScope&) = delete;
    SemanticScope& operator=(const SemanticScope&) = delete;
};

class ActorScope {
    uint64_t previous_;
    bool finished_ = false;
public:
    explicit ActorScope(uint64_t fork) : previous_(sdcf_round_actor_enter(fork)) {}
    ~ActorScope() { if (!finished_) sdcf_round_actor_leave(previous_); }
    uint64_t finish() { finished_ = true; return sdcf_round_actor_leave(previous_); }
};

class Thread {
    std::thread thread_;
    std::shared_ptr<uint64_t> end_;
public:
    Thread() = default;
    template <class Function, class... Arguments>
    explicit Thread(Function&& function, Arguments&&... arguments) {
        const auto fork = sdcf_round_actor_fork();
        end_ = std::make_shared<uint64_t>(0);
        thread_ = std::thread([fork, end = end_, fn = std::forward<Function>(function),
                               args = std::make_tuple(std::forward<Arguments>(arguments)...)]() mutable {
            ActorScope actor(fork);
            std::apply(fn, std::move(args));
            *end = actor.finish();
        });
    }
    Thread(Thread&&) = default;
    Thread& operator=(Thread&&) = default;
    bool joinable() const { return thread_.joinable(); }
    void join() { thread_.join(); sdcf_round_actor_join(*end_); }
};

class DirectionGroup {
    bool parallel_;
    uint64_t fork_, ends_[2] = {};
public:
    explicit DirectionGroup(int participants)
        : parallel_(participants > 1), fork_(parallel_ ? sdcf_round_actor_fork() : 0) {}
    bool parallel() const { return parallel_; }
    uint64_t fork() const { return fork_; }
    void complete(int direction, uint64_t end) { ends_[direction] = end; }
    void join() { if (parallel_) for (auto end : ends_) sdcf_round_actor_join(end); }
};

class DirectionScope {
    DirectionGroup& group_;
    int direction_;
    uint64_t previous_ = 0;
public:
    DirectionScope(DirectionGroup& group, int direction) : group_(group), direction_(direction) {
        if (group_.parallel()) previous_ = sdcf_round_actor_enter(group_.fork());
    }
    ~DirectionScope() {
        if (group_.parallel()) group_.complete(direction_, sdcf_round_actor_leave(previous_));
    }
};
}

#endif
