#pragma once
#include "backend_adapter.h"
#include "io_counters.h"
#include "native_preprocessing.h"
#include "native_dpf_lut.h"
#include "native_folded.h"
#include "round_trace.h"
#include <chrono>
#include <iostream>
#include <stdexcept>
#include <time.h>

namespace sdcf_measure {
using Clock = std::chrono::steady_clock;
struct Point {
    Clock::time_point time;
    double process_cpu_seconds;
    uint64_t sent, received, sends, recvs, offline_epoch, native_epoch, dpf_epoch, folded_epoch;
    bool io_calls_known, strict_preprocessing;
};
inline double processCpuSeconds() {
    timespec value{};
    if (clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &value) != 0)
        throw std::runtime_error("cannot read process CPU clock");
    return static_cast<double>(value.tv_sec) + static_cast<double>(value.tv_nsec) * 1e-9;
}
inline Point point() {
    const auto bridge = sdcf_backend::counters();
    const auto dpf = sdcf_dpf::counters ? sdcf_dpf::counters() : sdcf_dpf::Counters{};
    const auto folded = sdcf_folded::counters ? sdcf_folded::counters() : sdcf_folded::Counters{};
    // The DPF bridge reports bytes but has no total I/O-call counter. Keep
    // the instrumented SCI call counts separate when DPF traffic is present.
    const bool io_calls_known = bridge.io_calls_known && dpf.sent == 0 && dpf.received == 0 &&
        folded.sent == 0 && folded.received == 0;
    return {Clock::now(), processCpuSeconds(), sent + bridge.sent_bytes + dpf.sent + folded.sent,
            received + bridge.received_bytes + dpf.received + folded.received,
            sends + bridge.send_calls, recvs + bridge.recv_calls,
            bridge.offline_generation_epoch, sdcf_preprocess::generation_epoch.load(), dpf.epoch, folded.epoch,
            io_calls_known, sdcf_preprocess::strict};
}
inline void report(const char* phase, int query, const Point& before) {
    const auto after = point();
    const std::string phase_name(phase);
    const bool after_prepare = phase_name == "input" || phase_name == "ready" || phase_name == "online" || phase_name == "output";
    const bool strict_post_prepare = after_prepare && (before.strict_preprocessing || after.strict_preprocessing);
    if ((phase_name == "online" || strict_post_prepare) && after.offline_epoch != before.offline_epoch)
        throw std::runtime_error("offline material generated during online phase");
    if (strict_post_prepare && after.native_epoch != before.native_epoch)
        throw std::runtime_error("native offline material generated after prepare");
    if (strict_post_prepare && after.dpf_epoch != before.dpf_epoch)
        throw std::runtime_error("DPF material generated after prepare");
    if (strict_post_prepare && after.folded_epoch != before.folded_epoch)
        throw std::runtime_error("complete folded material generated after prepare");
    sdcf_log::Line log;
    log.precision(std::cout.precision());
    log << "METRIC {\"phase\":\"" << phase << "\",\"query\":" << query
        << ",\"seconds\":" << std::chrono::duration<double>(after.time - before.time).count()
        << ",\"process_cpu_seconds\":" << after.process_cpu_seconds - before.process_cpu_seconds
        << ",\"sent_bytes\":" << after.sent - before.sent
        << ",\"received_bytes\":" << after.received - before.received
        << ",\"send_calls\":";
    if (before.io_calls_known && after.io_calls_known) log << after.sends - before.sends;
    else log << "null";
    log << ",\"recv_calls\":";
    if (before.io_calls_known && after.io_calls_known) log << after.recvs - before.recvs;
    else log << "null";
    log << ",\"SCI_send_calls\":" << after.sends - before.sends
        << ",\"SCI_recv_calls\":" << after.recvs - before.recvs
        << ",\"logical_rounds\":null,\"native_generation_epoch_delta\":"
        << after.native_epoch - before.native_epoch << ",\"bridge_generation_epoch_delta\":"
        << after.offline_epoch - before.offline_epoch
        << ",\"dpf_generation_epoch_delta\":" << after.dpf_epoch - before.dpf_epoch
        << ",\"folded_generation_epoch_delta\":" << after.folded_epoch - before.folded_epoch
        << ",\"round_trace_active\":" << (sdcf_round::active() ? "true" : "false") << "}\n";
    if (phase_name != "query_total" && phase_name != "first_query")
        sdcf_round::phase_end(phase, query);
}
}
