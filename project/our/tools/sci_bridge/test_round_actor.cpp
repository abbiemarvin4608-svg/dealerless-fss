#include "dfss/utils/round_trace.h"
#ifdef NDEBUG
#undef NDEBUG
#endif
#include <cassert>
#include <unistd.h>

dfss_round::Callbacks dfss_round::callbacks;
namespace {
uint64_t actor = 7, leaves = 0, joins = 0, events = 0;
int last_channel = 0;
uint64_t fork_actor() { return 41; }
uint64_t enter_actor(uint64_t token) { assert(token == 41); const auto old = actor; actor = 9; return old; }
uint64_t leave_actor(uint64_t previous) { assert(actor == 9); actor = previous; ++leaves; return 53; }
void join_actor(uint64_t child) { assert(child == 53); ++joins; }
void trace(int, int channel, uint64_t offset, uint64_t length) {
    assert(offset == 0 && length == 3);
    last_channel = channel; ++events;
}
}
int main() {
    dfss_round::callbacks = {trace, fork_actor, enter_actor, leave_actor, join_actor};
    {
        dfss_round::ActorScope scope(dfss_round::fork());
        assert(actor == 9 && scope.finish() == 53 && scope.finish() == 53);
        dfss_round::join(scope.finish());
    }
    assert(actor == 7 && leaves == 1 && joins == 1);
    const int listener = socket(AF_INET, SOCK_STREAM, 0);
    sockaddr_in address{};
    address.sin_family = AF_INET; address.sin_addr.s_addr = htonl(INADDR_LOOPBACK);
    assert(bind(listener, reinterpret_cast<sockaddr*>(&address), sizeof(address)) == 0);
    assert(listen(listener, 1) == 0);
    socklen_t length = sizeof(address);
    assert(getsockname(listener, reinterpret_cast<sockaddr*>(&address), &length) == 0);
    const int client = socket(AF_INET, SOCK_STREAM, 0);
    assert(connect(client, reinterpret_cast<sockaddr*>(&address), sizeof(address)) == 0);
    const int server = accept(listener, nullptr, nullptr);
    const int port = ntohs(address.sin_port);
    dfss_round::io(0, server, true, 0, 3);
    assert(last_channel == 100000 + port);
    dfss_round::io(1, client, false, 0, 3);
    assert(last_channel == 100000 + port);
    dfss_round::io(0, server, true, 0, 3, 200000);
    assert(last_channel == 200000 + port);
    dfss_round::io(1, client, false, 0, 3, 200000);
    assert(last_channel == 200000 + port && events == 4);
    dfss_round::io(0, -1, true, 0, 0);
    assert(events == 4);
    dfss_round::callbacks = {};
    { dfss_round::ActorScope scope(dfss_round::fork()); assert(scope.finish() == 0); }
    dfss_round::io(0, -1, true, 0, 3);
    close(server); close(client); close(listener);
}
