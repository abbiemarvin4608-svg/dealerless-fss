#include "sci_conversion_bridge.h"
#include "utils/net_io_channel.h"
#include <cstdlib>
#include <array>
#include <future>
#include <iostream>
#include <stdexcept>
#include <vector>

int main(int argc, char** argv) {
    if (argc != 4) return 2;
    const int role = std::atoi(argv[1]), port = std::atoi(argv[2]), full = std::atoi(argv[3]);
    sci::NetIO reveal(role == 1 ? nullptr : "127.0.0.1", port + 100, true);
    void* bridge = segfss_sci_conversion_create(role, "127.0.0.1", port, full, 2);
    if (!bridge) { std::cerr << segfss_sci_conversion_error() << '\n'; return 3; }
    const uint64_t input[] = {0, 1, 7, 8, 15, 13};
    const uint64_t expected[4][6] = {{0, 1, 7, 248, 255, 253}, {0, 0, 3, 12, 0, 15},
        {0, 0, 3, 12, 15, 14}, {0, 0, 3, 4, 7, 6}};
    const segfss_sci_conversion_call calls[] = {{6,4,8,0,3,1}, {6,4,4,1,0,1},
        {6,4,4,1,1,1}, {6,4,3,1,2,1}};
    int failures = 0;
    for (int query = 0; query < 2; ++query) {
        if (segfss_sci_conversion_prepare(bridge, calls, 4)) {
            std::cerr << segfss_sci_conversion_error() << '\n'; return 4;
        }
        for (int call = 0; call < 4; ++call) {
            uint64_t shares[6], output[6], remote[6];
            for (int row = 0; row < 6; ++row) {
                const uint64_t mask = (row * 7 + query * 3) & 15;
                shares[row] = role == 1 ? mask : (input[row] - mask) & 15;
            }
            segfss_sci_conversion_metrics metrics{};
            if (segfss_sci_conversion_online(bridge, &calls[call], shares, output, &metrics)) {
                std::cerr << segfss_sci_conversion_error() << '\n'; return 5;
            }
            failures += metrics.material_load_ms < 0 || metrics.query_prepare_ms < 0 ||
                metrics.local_evaluation_ms < 0 || metrics.scan_ms < 0 || metrics.release_ms < 0;
            failures += metrics.material_bytes == 0 || metrics.comparison_keys == 0;
            failures += metrics.total_online_ms < metrics.material_load_ms + metrics.query_prepare_ms +
                metrics.local_evaluation_ms + metrics.scan_ms + metrics.release_ms;
            failures += metrics.eval_participants < 1;
            if (const char* layout = std::getenv("SEGFSS_MATERIAL_LAYOUT"))
                failures += metrics.contiguous_material != (std::string(layout) == "contiguous");
            if (call == 0 && std::getenv("SEGFSS_PREFETCH_BYTES")) {
                const auto budget = std::stoull(std::getenv("SEGFSS_PREFETCH_BYTES"));
                if (budget >= 8388608) failures += !metrics.prefetched || metrics.material_load_ms != 0;
                if (budget == 1) failures += metrics.prefetched != 0;
            }
            if (role == 1) { reveal.send_data(output, sizeof(output)); reveal.recv_data(remote, sizeof(remote)); }
            else { reveal.recv_data(remote, sizeof(remote)); reveal.send_data(output, sizeof(output)); reveal.flush(); }
            const uint64_t mask = (uint64_t(1) << calls[call].output_bits) - 1;
            for (int row = 0; row < 6; ++row) failures += ((output[row] + remote[row]) & mask) != expected[call][row];
        }
        failures += segfss_sci_conversion_finish_query(bridge) != 0;
    }
    segfss_sci_conversion_counters counters{};
    failures += segfss_sci_conversion_get_counters(bridge, &counters) != 0 || counters.generation_epoch != 8;
    uint64_t ignored[6]{}; segfss_sci_conversion_metrics metrics{};
    failures += segfss_sci_conversion_prepare(bridge, calls, 1) != 0;
    auto mismatch = calls[0]; ++mismatch.output_bits;
    failures += segfss_sci_conversion_online(bridge, &mismatch, ignored, ignored, &metrics) == 0;
    failures += segfss_sci_conversion_prepare(bridge, calls, 1) == 0;
    failures += segfss_sci_conversion_online(bridge, &calls[0], ignored, ignored, &metrics) == 0;
    segfss_sci_conversion_destroy(bridge);
    if (std::getenv("SEGFSS_TEST_MULTI_CONTEXT")) {
        std::array<void*, 2> contexts{};
        for (int task = 0; task < 2; ++task) {
            contexts[task] = segfss_sci_conversion_create(role, "127.0.0.1", port + 200 + task * 900, full, 2);
            if (!contexts[task]) { std::cerr << segfss_sci_conversion_error() << '\n'; return 7; }
        }
        for (int query = 0; query < 2; ++query) {
            for (auto context : contexts)
                if (segfss_sci_conversion_prepare(context, calls, 4)) return 8;
            std::array<std::array<std::array<uint64_t, 6>, 4>, 2> outputs{};
            std::array<std::future<int>, 2> tasks;
            for (int task = 0; task < 2; ++task) tasks[task] = std::async(std::launch::async, [&, task]() -> int {
                for (int call = 0; call < 4; ++call) {
                    uint64_t shares[6];
                    for (int row = 0; row < 6; ++row) {
                        const uint64_t mask = (row * 7 + query * 3 + task) & 15;
                        shares[row] = role == 1 ? mask : (input[row] - mask) & 15;
                    }
                    segfss_sci_conversion_metrics metrics{};
                    if (segfss_sci_conversion_online(contexts[task], &calls[call], shares,
                                                    outputs[task][call].data(), &metrics)) return 1;
                }
                return segfss_sci_conversion_finish_query(contexts[task]) != 0;
            });
            for (auto& task : tasks) failures += task.get();
            decltype(outputs) remote{};
            if (role == 1) { reveal.send_data(&outputs, sizeof(outputs)); reveal.recv_data(&remote, sizeof(remote)); }
            else { reveal.recv_data(&remote, sizeof(remote)); reveal.send_data(&outputs, sizeof(outputs)); reveal.flush(); }
            for (int task = 0; task < 2; ++task) for (int call = 0; call < 4; ++call)
                for (int row = 0; row < 6; ++row) {
                    const uint64_t mask = (uint64_t(1) << calls[call].output_bits) - 1;
                    failures += ((outputs[task][call][row] + remote[task][call][row]) & mask) != expected[call][row];
                }
        }
        for (auto context : contexts) {
            failures += segfss_sci_conversion_get_counters(context, &counters) != 0 || counters.generation_epoch != 8;
            segfss_sci_conversion_destroy(context);
        }
    }
    if (std::getenv("SEGFSS_TEST_POLICY_MISMATCH")) {
        setenv("SEGFSS_SEGMENT_POLICY", role == 1 ? "fixed-2" : "depth-cap-8", 1);
        bridge = segfss_sci_conversion_create(role, "127.0.0.1", port + 200, full, 2);
        if (!bridge) { std::cerr << segfss_sci_conversion_error() << '\n'; return 6; }
        failures += segfss_sci_conversion_prepare(bridge, calls, 4) == 0;
        failures += segfss_sci_conversion_get_counters(bridge, &counters) != 0 || counters.generation_epoch != 0;
        failures += segfss_sci_conversion_prepare(bridge, calls, 4) == 0;
        segfss_sci_conversion_destroy(bridge);
    }
    std::cout << "CONVERSION_BRIDGE_TEST role=" << role << " full_tree=" << full
        << " values=" << (std::getenv("SEGFSS_TEST_MULTI_CONTEXT") ? 144 : 48)
        << " failures=" << failures << '\n';
    return failures ? 1 : 0;
}
