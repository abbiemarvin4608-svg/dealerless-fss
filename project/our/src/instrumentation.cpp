#include "instrumentation.h"

void PhaseTimer::start(const std::string& phase_name) {
    PhaseData& phase = phases_[phase_name];
    phase.start_time = clock::now();
    phase.running = true;
}

void PhaseTimer::stop(const std::string& phase_name) {
    auto stop_time = clock::now();
    auto it = phases_.find(phase_name);
    if (it == phases_.end() || !it->second.running) {
        return;
    }

    PhaseData& phase = it->second;
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(
        stop_time - phase.start_time);
    phase.accumulated_ms += duration.count() / 1000.0;
    phase.running = false;
}

bool PhaseTimer::has_phase(const std::string& phase_name) const {
    return phases_.find(phase_name) != phases_.end();
}

double PhaseTimer::get_duration_ms(const std::string& phase_name) const {
    auto it = phases_.find(phase_name);
    if (it == phases_.end()) {
        return 0.0;
    }
    return it->second.accumulated_ms;
}
