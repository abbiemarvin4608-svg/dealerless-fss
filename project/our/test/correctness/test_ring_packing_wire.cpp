#include "ring_packing.h"
#include <cstring>
#include <iostream>
#include <stdexcept>
#include <sys/socket.h>
#include <thread>
#include <unistd.h>

namespace {
void require(bool condition) {
    if (!condition) throw std::runtime_error("packed wire boundary regression");
}
template<class Action> void rejects(Action action) {
    bool caught = false;
    try { action(); } catch (const std::invalid_argument&) { caught = true; }
    require(caught);
}
void exactReceive(int socket, uint8_t* data, int count) {
    for (int offset = 0; offset < count;) {
        const auto got = ::recv(socket, data + offset, count - offset, 0);
        if (got <= 0) throw std::runtime_error("packed wire EOF");
        offset += int(got);
    }
}
void shapeBoundary() {
    auto message = segfss::packRingMessage({1, 2, 3}, 9);
    const auto next = segfss::packRingMessage({7}, 7);
    message.insert(message.end(), next.begin(), next.end());
    for (const auto shape : {std::pair<std::size_t,int>{2,9}, {3,8}}) {
        std::size_t cursor = 0;
        rejects([&] { segfss::receiveRingMessage(shape.first, shape.second,
            [&](uint8_t* bytes, int count) {
                require(cursor + count <= message.size());
                std::memcpy(bytes, message.data() + cursor, count); cursor += count;
            }); });
        // Reject after exactly the shape header, before adjacent payloads.
        require(cursor == segfss::kRingPackingHeaderBytes);
    }
    for (const auto offset : {0, 2, 3, 12}) {
        auto malformed = message;
        malformed[offset] = offset == 3 ? 0 : 255;
        rejects([&] { segfss::receiveRingMessage(3, 9,
            [&](uint8_t* bytes, int count) {
                require(count == int(segfss::kRingPackingHeaderBytes));
                std::memcpy(bytes, malformed.data(), count);
            }); });
    }
    auto tail = segfss::packRingMessage({1, 0, 1}, 1);
    tail.back() |= 0x80;
    std::size_t cursor = 0;
    rejects([&] { segfss::receiveRingMessage(3, 1,
        [&](uint8_t* bytes, int count) {
            std::memcpy(bytes, tail.data() + cursor, count); cursor += count;
        }); });
}
void fragmentedSocketFrames() {
    int sockets[2]; require(::socketpair(AF_UNIX, SOCK_STREAM, 0, sockets) == 0);
    std::exception_ptr sender_error;
    std::thread sender([&] {
        try {
            for (int width : {1,7,9,13,64}) {
                const auto message = segfss::packRingMessage({0, 1, UINT64_MAX}, width);
                // Deliberately fragment both header and payload at transport.
                for (auto byte : message) require(::send(sockets[0], &byte, 1, 0) == 1);
            }
            auto truncated = segfss::packRingMessage({1,2,3}, 13);
            require(::send(sockets[0], truncated.data(), truncated.size() - 1, 0) ==
                    ssize_t(truncated.size() - 1));
        } catch (...) { sender_error = std::current_exception(); }
        ::shutdown(sockets[0], SHUT_WR);
    });
    for (int width : {1,7,9,13,64}) {
        const auto values = segfss::receiveRingMessage(3, width,
            [&](uint8_t* bytes, int count) { exactReceive(sockets[1], bytes, count); });
        require(values == std::vector<uint64_t>({0, 1, segfss::reference::mask(width)}));
    }
    bool eof = false;
    try { segfss::receiveRingMessage(3, 13,
        [&](uint8_t* bytes, int count) { exactReceive(sockets[1], bytes, count); }); }
    catch (const std::runtime_error&) { eof = true; }
    sender.join(); ::close(sockets[0]); ::close(sockets[1]);
    if (sender_error) std::rethrow_exception(sender_error);
    require(eof);
}
}
int main() {
    shapeBoundary(); fragmentedSocketFrames();
    std::cout << "PACKED_WIRE widths=1,7,9,13,64 header_shape,adjacent_frames,fragmentation,tail,EOF status=passed\n";
}
