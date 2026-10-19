#include "sci_msb_bridge.h"
#include "sci_conversion_bridge.h"
#include <array>
#include <cstdlib>
#include <iostream>
#include <map>
#include <stdexcept>
#include <utility>

namespace {
std::map<std::pair<int, int>, uint64_t> offsets;
uint64_t sent = 0, received = 0, events = 0;
void trace(int direction, int channel, uint64_t offset, uint64_t length) {
    if ((direction != 0 && direction != 1) || !length || channel < 100000 || channel >= 300000)
        throw std::runtime_error("invalid bridge trace event");
    auto& cursor = offsets[{direction, channel}];
    if (cursor != offset) throw std::runtime_error("bridge trace stream has gap or overlap");
    cursor += length;
    (direction ? received : sent) += length;
    ++events;
}
void check_bytes(void* bridge) {
    segfss_sci_conversion_counters counters{};
    if (segfss_sci_conversion_get_counters(bridge, &counters))
        throw std::runtime_error(segfss_sci_conversion_error());
    if (sent != counters.sent_bytes || received != counters.received_bytes)
        throw std::runtime_error("callback bytes omitted or counted twice");
}
}

int main(int argc, char** argv) {
    if (argc != 3) return 2;
    const int role = std::atoi(argv[1]), port = std::atoi(argv[2]);
    try {
        segfss_sci_set_round_trace_callbacks(trace, nullptr, nullptr, nullptr, nullptr);
        void* bridge = segfss_sci_conversion_create(role, "127.0.0.1", port, 0, 2);
        if (!bridge) throw std::runtime_error(segfss_sci_conversion_error());
        check_bytes(bridge);
        const segfss_sci_conversion_call call{6, 4, 8, 0, 3, 1};
        for (int query = 0; query < 2; ++query) {
            if (segfss_sci_conversion_prepare(bridge, &call, 1))
                throw std::runtime_error(segfss_sci_conversion_error());
            check_bytes(bridge);
            std::array<uint64_t, 6> input{}, output{};
            segfss_sci_conversion_metrics metrics{};
            if (segfss_sci_conversion_online(bridge, &call, input.data(), output.data(), &metrics) ||
                segfss_sci_conversion_finish_query(bridge))
                throw std::runtime_error(segfss_sci_conversion_error());
            check_bytes(bridge);
        }
        bool raw = false, ot = false;
        for (const auto& entry : offsets) {
            raw |= entry.first.second >= 200000;
            ot |= entry.first.second < 200000;
        }
        if (!raw || !ot) throw std::runtime_error("missing transport family");
        const auto before = events;
        segfss_sci_set_round_trace_callbacks(nullptr, nullptr, nullptr, nullptr, nullptr);
        if (segfss_sci_conversion_prepare(bridge, &call, 1))
            throw std::runtime_error(segfss_sci_conversion_error());
        std::array<uint64_t, 6> input{}, output{};
        segfss_sci_conversion_metrics metrics{};
        if (segfss_sci_conversion_online(bridge, &call, input.data(), output.data(), &metrics) ||
            segfss_sci_conversion_finish_query(bridge))
            throw std::runtime_error(segfss_sci_conversion_error());
        if (events != before) throw std::runtime_error("disabled callback still active");
        segfss_sci_conversion_destroy(bridge);
        std::cout << "ROUND_TRACE_CALLBACK_TEST {\"role\":" << role << ",\"events\":" << events
            << ",\"sent_bytes\":" << sent << ",\"received_bytes\":" << received << ",\"passed\":true}\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
