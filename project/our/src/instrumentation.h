#pragma once

#include <chrono>
#include <string>
#include <unordered_map>

// PhaseTimer: accumulates wall-clock time for named phases.
// Usage:
//   PhaseTimer timer;
//   timer.start("offline");
//   // ... do work ...
//   timer.stop("offline");
//   timer.start("online");
//   // ... do work ...
//   timer.stop("online");
//   std::cout << "Offline: " << timer.get_duration_ms("offline") << " ms\n";
//
// Multiple start/stop pairs for the same phase name accumulate.
class PhaseTimer {
public:
    using clock = std::chrono::high_resolution_clock;
    using time_point = std::chrono::time_point<clock>;

    PhaseTimer() = default;

    void start(const std::string& phase_name);
    void stop(const std::string& phase_name);

    bool has_phase(const std::string& phase_name) const;
    double get_duration_ms(const std::string& phase_name) const;

private:
    struct PhaseData {
        double accumulated_ms = 0.0;
        time_point start_time;
        bool running = false;
    };

    std::unordered_map<std::string, PhaseData> phases_;
};
