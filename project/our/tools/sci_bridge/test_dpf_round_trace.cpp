#include "sci_dpf_bridge.h"
#include "sci_msb_bridge.h"
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
        throw std::runtime_error("invalid DPF bridge trace event");
    auto& cursor = offsets[{direction, channel}];
    if (cursor != offset) throw std::runtime_error("DPF trace stream has gap or overlap");
    cursor += length;
    (direction ? received : sent) += length;
    ++events;
}
segfss_sci_dpf_counters check_bytes(void* bridge, int role, const char* phase, int query) {
    segfss_sci_dpf_counters counters{};
    if (segfss_sci_dpf_counters_get(bridge, &counters))
        throw std::runtime_error(segfss_sci_dpf_error());
    std::cout << "DPF_COUNTER_TRACE {\"role\":" << role << ",\"phase\":\"" << phase
        << "\",\"query\":" << query << ",\"sent_bytes\":" << counters.sent_bytes
        << ",\"received_bytes\":" << counters.received_bytes
        << ",\"trace_sent_bytes\":" << sent << ",\"trace_received_bytes\":" << received
        << ",\"events\":" << events << "}\n";
    if (sent != counters.sent_bytes || received != counters.received_bytes)
        throw std::runtime_error("DPF callback bytes omitted or counted twice");
    return counters;
}
}

int main(int argc, char** argv) {
    if (argc != 3) return 2;
    const int role = std::atoi(argv[1]), port = std::atoi(argv[2]);
    void* bridge = nullptr;
    try {
        segfss_sci_set_round_trace_callbacks(trace, nullptr, nullptr, nullptr, nullptr);
        bridge = segfss_sci_dpf_create(role, "127.0.0.1", port);
        if (!bridge) throw std::runtime_error(segfss_sci_dpf_error());
        const auto setup = check_bytes(bridge, role, "setup", -1);
        if (!sent || !received || setup.generation_epoch || setup.generated_requests || setup.consumed_requests)
            throw std::runtime_error("DPF setup must include actual base OT but no lookup inventory");
        std::array<uint64_t, 8> table{{13, 7, 41, 11, 29, 3, 61, 19}};
        segfss_sci_dpf_call call{};
        call.count = 7; call.index_bits = 3; call.tables = 1;
        call.output_bits[0] = 6; call.values[0] = table.data();
        for (int query = 0; query < 2; ++query) {
            call.query = query;
            if (segfss_sci_dpf_prepare(bridge, &call, 1))
                throw std::runtime_error(segfss_sci_dpf_error());
            const auto offline = check_bytes(bridge, role, "offline", query);
            if (offline.generation_epoch != uint64_t(query + 1) || offline.generated_requests != 7 ||
                offline.consumed_requests != 0 || offline.sent_bytes <= setup.sent_bytes ||
                offline.received_bytes <= setup.received_bytes)
                throw std::runtime_error("DPF offline inventory or traffic missing");
            // Inputs appear only after the complete lookup inventory is ready.
            std::array<uint64_t, 7> input{{0, 1, 2, 3, 4, 5, 6}}, output{};
            if (segfss_sci_dpf_online(bridge, &call, input.data(), output.data(), nullptr) ||
                segfss_sci_dpf_finish_query(bridge))
                throw std::runtime_error(segfss_sci_dpf_error());
            const auto online = check_bytes(bridge, role, "online", query);
            if (online.generation_epoch != offline.generation_epoch || online.generated_requests != 7 ||
                online.consumed_requests != 7 || online.sent_bytes <= offline.sent_bytes ||
                online.received_bytes <= offline.received_bytes)
                throw std::runtime_error("DPF lookup traffic or one-use inventory missing");
            // Repeated snapshots must not re-add the fixed creation-time offset.
            const auto repeated = check_bytes(bridge, role, "snapshot", query);
            if (online.sent_bytes != repeated.sent_bytes || online.received_bytes != repeated.received_bytes)
                throw std::runtime_error("DPF counter reads change totals");
        }
        bool raw = false, ot = false;
        for (const auto& entry : offsets) {
            raw |= entry.first.second >= 200000;
            ot |= entry.first.second < 200000;
        }
        if (!raw || !ot) throw std::runtime_error("DPF trace is missing a transport family");
        segfss_sci_dpf_destroy(bridge); bridge = nullptr;
        segfss_sci_set_round_trace_callbacks(nullptr, nullptr, nullptr, nullptr, nullptr);
        std::cout << "DPF_ROUND_TRACE_TEST {\"role\":" << role << ",\"passed\":true}\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        if (bridge) segfss_sci_dpf_destroy(bridge);
        return 1;
    }
}
