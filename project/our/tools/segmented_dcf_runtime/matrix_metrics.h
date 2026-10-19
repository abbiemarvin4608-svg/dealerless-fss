#pragma once
#include <chrono>
#include <cstdint>

namespace sdcf_local_matrix {
using Clock=std::chrono::steady_clock;
struct Metrics {
    double layout_copy_seconds=0;
    double compute_seconds=0;
    uint64_t calls=0;
};
inline thread_local Metrics metrics;
inline double seconds(Clock::time_point begin,Clock::time_point end) {
    return std::chrono::duration<double>(end-begin).count();
}
}
