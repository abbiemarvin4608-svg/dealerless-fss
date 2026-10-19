#pragma once
#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <arpa/inet.h>
#include <sys/socket.h>

namespace dfss_round {
using IoCallback = void (*)(int, int, uint64_t, uint64_t);
using ForkCallback = uint64_t (*)();
using EnterCallback = uint64_t (*)(uint64_t);
using LeaveCallback = uint64_t (*)(uint64_t);
using JoinCallback = void (*)(uint64_t);
using ScopeBeginCallback = uint64_t (*)(const char*, int, int, int, int, int, int);
using ScopeEndCallback = void (*)(uint64_t);
using CallerLimitCallback = std::size_t (*)();
using ParticipantsCallback = void (*)(std::size_t);
struct Callbacks {
    IoCallback io = nullptr;
    ForkCallback fork = nullptr;
    EnterCallback enter = nullptr;
    LeaveCallback leave = nullptr;
    JoinCallback join = nullptr;
    ScopeBeginCallback scope_begin = nullptr;
    ScopeEndCallback scope_end = nullptr;
    CallerLimitCallback caller_limit = nullptr;
    ParticipantsCallback transport_participants = nullptr;
};
extern Callbacks callbacks;

inline int socketChannel(int socket, bool server) {
    sockaddr_in endpoint{};
    socklen_t length = sizeof(endpoint);
    const int result = server ? getsockname(socket, reinterpret_cast<sockaddr*>(&endpoint), &length) :
        getpeername(socket, reinterpret_cast<sockaddr*>(&endpoint), &length);
    if (result || endpoint.sin_family != AF_INET)
        throw std::runtime_error("round trace cannot identify socket endpoint");
    return ntohs(endpoint.sin_port);
}
inline void io(int direction, int socket, bool server, uint64_t offset, uint64_t length,
               int channel_family = 100000) {
    // Peer and NetIO can accept separate connections on the same public port.
    if (length && callbacks.io)
        callbacks.io(direction, channel_family + socketChannel(socket, server), offset, length);
}
inline uint64_t fork() { return callbacks.fork ? callbacks.fork() : 0; }
inline void join(uint64_t child) { if (callbacks.join) callbacks.join(child); }
class SemanticScope {
    uint64_t identifier_ = 0;
public:
    SemanticScope(const char* label, int query = -1, int layer = -1, int slot = -1,
                  int call = -1, int kind = -1, int stage = -1, bool enabled = true) {
        if (enabled && callbacks.scope_begin)
            identifier_ = callbacks.scope_begin(label, query, layer, slot, call, kind, stage);
    }
    ~SemanticScope() { if (identifier_ && callbacks.scope_end) callbacks.scope_end(identifier_); }
    SemanticScope(const SemanticScope&) = delete;
    SemanticScope& operator=(const SemanticScope&) = delete;
};
class ActorScope {
public:
    explicit ActorScope(uint64_t parent) {
        if (callbacks.enter) previous_ = callbacks.enter(parent);
    }
    ~ActorScope() { finish(); }
    uint64_t finish() {
        if (finished_) return tail_;
        finished_ = true;
        if (callbacks.leave) tail_ = callbacks.leave(previous_);
        return tail_;
    }
private:
    uint64_t previous_ = 0, tail_ = 0;
    bool finished_ = false;
};
}
