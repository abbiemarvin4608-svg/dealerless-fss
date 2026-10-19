#include "instrumentation.h"

#include <iostream>
#include <string>
#include <thread>
#include <chrono>

class ResultLog {
public:
    void check_phase_exists(const std::string& test_name, PhaseTimer& timer,
                           const std::string& phase_name) {
        const bool ok = timer.has_phase(phase_name);
        std::cout << (ok ? "[PASS] " : "[FAIL] ") << test_name
                  << " phase=" << phase_name
                  << (ok ? " exists" : " missing") << '\n';
        failures_ += ok ? 0 : 1;
    }

    void check_phase_duration_positive(const std::string& test_name,
                                      PhaseTimer& timer,
                                      const std::string& phase_name) {
        const double duration = timer.get_duration_ms(phase_name);
        const bool ok = duration > 0.0;
        std::cout << (ok ? "[PASS] " : "[FAIL] ") << test_name
                  << " phase=" << phase_name << " duration=" << duration
                  << "ms" << '\n';
        failures_ += ok ? 0 : 1;
    }

    void check_duration_in_range(const std::string& test_name, PhaseTimer& timer,
                                const std::string& phase_name,
                                double min_ms, double max_ms) {
        const double duration = timer.get_duration_ms(phase_name);
        const bool ok = duration >= min_ms && duration <= max_ms;
        std::cout << (ok ? "[PASS] " : "[FAIL] ") << test_name
                  << " phase=" << phase_name << " duration=" << duration
                  << "ms (expected " << min_ms << "-" << max_ms << "ms)"
                  << '\n';
        failures_ += ok ? 0 : 1;
    }

    int failures() const { return failures_; }

private:
    int failures_ = 0;
};

void test_start_stop_single_phase(ResultLog& log) {
    PhaseTimer timer;

    timer.start("phase1");
    std::this_thread::sleep_for(std::chrono::milliseconds(10));
    timer.stop("phase1");

    log.check_phase_exists("start_stop_single", timer, "phase1");
    log.check_phase_duration_positive("start_stop_single", timer, "phase1");
    log.check_duration_in_range("start_stop_single", timer, "phase1", 9.0, 50.0);
}

void test_multiple_phases_independent(ResultLog& log) {
    PhaseTimer timer;

    timer.start("A");
    std::this_thread::sleep_for(std::chrono::milliseconds(10));
    timer.stop("A");

    timer.start("B");
    std::this_thread::sleep_for(std::chrono::milliseconds(15));
    timer.stop("B");

    log.check_phase_exists("multiple_phases_A", timer, "A");
    log.check_phase_exists("multiple_phases_B", timer, "B");
    log.check_duration_in_range("multiple_phases_A", timer, "A", 9.0, 50.0);
    log.check_duration_in_range("multiple_phases_B", timer, "B", 14.0, 50.0);
}

void test_phase_accumulation(ResultLog& log) {
    PhaseTimer timer;

    timer.start("loop");
    std::this_thread::sleep_for(std::chrono::milliseconds(5));
    timer.stop("loop");

    timer.start("loop");
    std::this_thread::sleep_for(std::chrono::milliseconds(5));
    timer.stop("loop");

    log.check_phase_exists("accumulation", timer, "loop");
    log.check_duration_in_range("accumulation", timer, "loop", 9.0, 60.0);
}

void test_nested_phase_summation(ResultLog& log) {
    PhaseTimer timer;

    timer.start("total");

    timer.start("sub1");
    std::this_thread::sleep_for(std::chrono::milliseconds(5));
    timer.stop("sub1");

    timer.start("sub2");
    std::this_thread::sleep_for(std::chrono::milliseconds(8));
    timer.stop("sub2");

    timer.stop("total");

    const double total = timer.get_duration_ms("total");
    const double sub1 = timer.get_duration_ms("sub1");
    const double sub2 = timer.get_duration_ms("sub2");
    const double sub_sum = sub1 + sub2;

    // Total should be >= sum of sub-phases (timing overhead is positive)
    // and within 20% tolerance for sleep-based test
    const bool ok = (total >= sub_sum) && (total <= sub_sum * 1.2);
    std::cout << (ok ? "[PASS] " : "[FAIL] ") << "nested_summation"
              << " total=" << total << "ms sub_sum=" << sub_sum << "ms"
              << " (sub1=" << sub1 << "ms, sub2=" << sub2 << "ms)\n";
    log.check_phase_exists("nested_total", timer, "total");
    log.check_phase_exists("nested_sub1", timer, "sub1");
    log.check_phase_exists("nested_sub2", timer, "sub2");
}

struct Case {
    int id;
    const char* name;
    void (*run)(ResultLog&);
};

Case cases[] = {
    {1, "start_stop_single_phase", test_start_stop_single_phase},
    {2, "multiple_phases_independent", test_multiple_phases_independent},
    {3, "phase_accumulation", test_phase_accumulation},
    {4, "nested_phase_summation", test_nested_phase_summation},
};

int main(int argc, char** argv) {
    ResultLog log;
    const int n_cases = sizeof(cases) / sizeof(cases[0]);

    std::cout << "=== Instrumentation Tests ===\n";

    for (int i = 0; i < n_cases; ++i) {
        std::cout << "\nTest " << cases[i].id << ": " << cases[i].name << '\n';
        cases[i].run(log);
    }

    std::cout << "\n=== Summary ===\n";
    std::cout << "Total failures: " << log.failures() << '\n';

    return log.failures();
}
