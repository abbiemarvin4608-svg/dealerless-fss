#ifndef SDCF_NATIVE_WORKERS_H
#define SDCF_NATIVE_WORKERS_H
#include <cstdint>
#include <atomic>
#include <functional>
#include <string>
#include <sstream>
#include "round_trace.h"

class MathFunctions;
class LinearOT;
class Truncation;
class AuxProtocols;
class XTProtocol;

namespace sdcf_native {
struct TaskIdentity {
    int query = -1, layer = -1, first_slot = 0, slots = 0;
};

// A scope selects owned protocol contexts; it never changes SCI globals.
class TaskScope {
public:
    TaskScope(int query, int layer, int first_slot = 0, int slots = 0);
    ~TaskScope();
    TaskScope(const TaskScope&) = delete;
    TaskScope& operator=(const TaskScope&) = delete;
    int observed_ot_participants() const;
    int observed_compute_participants() const;
private:
    TaskIdentity previous_;
    std::atomic<int> observed_{1};
    std::atomic<int> compute_{1};
    std::atomic<int>* previous_observer_;
    std::atomic<int>* previous_compute_;
    void* cggm_scope_ = nullptr;
};

void setup(int role, int port, const std::string& host, MathFunctions& primary,
           int required_contexts = 1);
void finish();
int activation_workers();
int context_count();
MathFunctions& context(int slot);
MathFunctions* selected_math(int relative_slot = 0);
LinearOT* selected_mult(int relative_slot = 0);
Truncation* selected_truncation(int relative_slot = 0);
AuxProtocols* selected_aux(int relative_slot = 0);
XTProtocol* selected_xt(int relative_slot = 0);
TaskIdentity task_identity();
std::string graph_mode();
bool batched_graph_openings();
bool tail_parallel();
bool isolated_graph();
int graph_branch(int layer);
int selected_branch();
void run_worker_group(int first_slot, int slots, const std::function<void(int)>& work);
void run_task_pair(const std::function<void()>& first, const std::function<void()>& second);
int ot_participants();
std::atomic<int>* team_observer();
void observe_ot_team(std::atomic<int>* observer, int participants);
void observe_compute_participants(int participants);
void log(const std::string& lines);
class LogLine : public std::ostringstream {
public:
    ~LogLine() { log(str()); }
};
bool record_native_conversion();
void activation(bool tanh, uint64_t* input, uint64_t* output, int count,
                int input_bits, int output_bits, int input_scale, int output_scale);
}
#endif
