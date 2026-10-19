#include "native_workers.h"
#include "Math/math-functions.h"
#include "native_planning.h"
#include "native_folded.h"
#include "runtime_log.h"
#include "round_trace.h"
#include "sci_msb_bridge.h"
#include <algorithm>
#include <chrono>
#include <condition_variable>
#include <cstdlib>
#include <exception>
#include <iostream>
#include <memory>
#include <mutex>
#include <sstream>
#include <stdexcept>
#include <thread>
#include <vector>
#pragma weak segfss_sci_cggm_scope_begin
#pragma weak segfss_sci_cggm_scope_stats
#pragma weak segfss_sci_cggm_scope_end

namespace sdcf_native {
namespace {
using Clock = std::chrono::steady_clock;
thread_local TaskIdentity identity;
thread_local int ot_limit = 2;
thread_local uint64_t* native_calls = nullptr;
thread_local std::atomic<int>* ot_observer = nullptr;
thread_local std::atomic<int>* compute_observer = nullptr;
int requested_workers = 1;
std::string execution_mode = "sequential";
std::string opening_mode = "independent";
bool parallel_tail = false;

int bounded_environment(const char* name, int fallback) {
    const char* value = std::getenv(name);
    if (!value) return fallback;
    std::string text(value);
    if (text.empty() || text.find_first_not_of("0123456789") != std::string::npos)
        throw std::invalid_argument(std::string(name) + " must be an integer in 1..4");
    const int result = std::stoi(text);
    if (result < 1 || result > 4)
        throw std::invalid_argument(std::string(name) + " must be an integer in 1..4");
    return result;
}

struct Protocol {
    std::unique_ptr<sci::IOPack> io;
    std::unique_ptr<sci::OTPack> ot;
    std::unique_ptr<MathFunctions> owned_math;
    MathFunctions* math = nullptr;
};
std::vector<std::unique_ptr<Protocol>> protocols;

class Worker {
public:
    Worker() : thread_([this] { loop(); }) {
        std::unique_lock<std::mutex> lock(mutex_);
        completed_.wait(lock, [&] { return ready_; });
    }
    ~Worker() {
        { std::lock_guard<std::mutex> lock(mutex_); stopping_ = true; }
        available_.notify_one();
        thread_.join();
    }
    void submit(std::function<void()> task) {
        std::lock_guard<std::mutex> lock(mutex_);
        if (busy_) throw std::logic_error("protocol worker already has an owner");
        const auto fork = sdcf_round_actor_fork();
        task_ = [this, task = std::move(task), fork] {
            sdcf_round::ActorScope actor(fork);
            try { task(); } catch (...) { end_ = actor.finish(); throw; }
            end_ = actor.finish();
        };
        error_ = nullptr; busy_ = true;
        available_.notify_one();
    }
    std::exception_ptr wait() {
        std::unique_lock<std::mutex> lock(mutex_);
        completed_.wait(lock, [&] { return !busy_; });
        const auto error = error_;
        const auto end = end_;
        lock.unlock();
        sdcf_round_actor_join(end);
        return error;
    }
private:
    void loop() {
        std::unique_lock<std::mutex> lock(mutex_);
        ready_ = true; completed_.notify_one();
        for (;;) {
            available_.wait(lock, [&] { return busy_ || stopping_; });
            if (stopping_) return;
            auto task = std::move(task_);
            lock.unlock();
            std::exception_ptr error;
            try { task(); } catch (...) { error = std::current_exception(); }
            lock.lock(); error_ = error; busy_ = false;
            completed_.notify_one();
        }
    }
    std::mutex mutex_;
    std::condition_variable available_, completed_;
    bool busy_ = false, stopping_ = false, ready_ = false;
    std::function<void()> task_;
    std::exception_ptr error_;
    uint64_t end_ = 0;
    std::thread thread_;
};
std::vector<std::unique_ptr<Worker>> workers;

struct WorkerMetrics {
    int slot = 0, begin = 0, count = 0, ot = 1, observed_ot = 1;
    int cggm_limit = 1, observed_cggm = 0, observed_transport = 0, observed_compute = 1;
    uint64_t calls = 0, sent = 0;
    double seconds = 0;
};
struct NativeScope {
    int previous_ot = ot_limit;
    uint64_t* previous_calls = native_calls;
    void* cggm_scope = nullptr;
    NativeScope(int ot, uint64_t& calls) {
        if (segfss_sci_cggm_scope_begin) {
            cggm_scope = segfss_sci_cggm_scope_begin(ot);
            if (!cggm_scope) throw std::runtime_error("cannot restrict activation cGGM participants");
        }
        ot_limit = ot; native_calls = &calls;
    }
    void stats(int& limit, int& peak, int& transport) const {
        if (cggm_scope && segfss_sci_cggm_scope_stats(cggm_scope, &limit, &peak, &transport))
            throw std::runtime_error("cannot observe activation cGGM participants");
    }
    ~NativeScope() {
        if (cggm_scope) segfss_sci_cggm_scope_end(cggm_scope);
        ot_limit = previous_ot; native_calls = previous_calls;
    }
};
}

TaskScope::TaskScope(int query, int layer, int first_slot, int slots)
    : previous_(identity), previous_observer_(ot_observer), previous_compute_(compute_observer) {
    if (first_slot < 0 || slots < 0 || (slots && first_slot + slots > context_count()))
        throw std::invalid_argument("native task protocol range is unavailable");
    if (segfss_sci_cggm_scope_begin) {
        cggm_scope_ = segfss_sci_cggm_scope_begin(isolated_graph() ? 2 : 4);
        if (!cggm_scope_) throw std::runtime_error("cannot restrict graph cGGM participants");
    }
    identity = {query, layer, first_slot, slots};
    ot_observer = &observed_;
    compute_observer = &compute_;
}
TaskScope::~TaskScope() {
    if (cggm_scope_) observe_compute_participants(segfss_sci_cggm_scope_end(cggm_scope_));
    observe_ot_team(previous_observer_, observed_.load());
    observe_ot_team(previous_compute_, compute_.load());
    identity = previous_; ot_observer = previous_observer_; compute_observer = previous_compute_;
}
int TaskScope::observed_ot_participants() const { return observed_.load(); }
int TaskScope::observed_compute_participants() const {
    int cggm_limit = 0, cggm_peak = 0, transport_peak = 0;
    if (cggm_scope_ && segfss_sci_cggm_scope_stats(cggm_scope_, &cggm_limit, &cggm_peak, &transport_peak))
        throw std::runtime_error("cannot observe graph cGGM participants");
    return std::max({compute_.load(), observed_.load(), cggm_peak, transport_peak});
}
TaskIdentity task_identity() { return identity; }
int activation_workers() { return requested_workers; }
int context_count() { return int(protocols.size()); }
MathFunctions& context(int slot) { return *protocols.at(slot)->math; }
MathFunctions* selected_math(int relative_slot) { return &context(identity.first_slot + relative_slot); }
LinearOT* selected_mult(int relative_slot) { return selected_math(relative_slot)->mult; }
Truncation* selected_truncation(int relative_slot) { return selected_math(relative_slot)->trunc; }
AuxProtocols* selected_aux(int relative_slot) { return selected_math(relative_slot)->aux; }
XTProtocol* selected_xt(int relative_slot) { return selected_math(relative_slot)->xt; }
std::string graph_mode() { return execution_mode; }
bool batched_graph_openings() { return opening_mode == "batched"; }
bool tail_parallel() { return parallel_tail; }
bool isolated_graph() { return execution_mode != "sequential"; }
int graph_branch(int layer) {
    const int row = layer % 15;
    return isolated_graph() && (row == 2 || row == 3 || row == 7 || row == 8 ||
        (parallel_tail && row >= 10 && row <= 13)) ? 1 : 0;
}
int selected_branch() { return isolated_graph() ? identity.first_slot / 2 : 0; }
int ot_participants() { return ot_limit; }
std::atomic<int>* team_observer() { return ot_observer; }
void observe_ot_team(std::atomic<int>* observer, int participants) {
    if (!observer) return;
    int previous = observer->load();
    while (previous < participants && !observer->compare_exchange_weak(previous, participants)) {}
}
void observe_compute_participants(int participants) { observe_ot_team(compute_observer, participants); }
void log(const std::string& lines) {
    if (sdcf_planning::active()) return;
    sdcf_log::emit(lines);
}
bool record_native_conversion() {
    if (!native_calls) return false;
    ++*native_calls;
    return true;
}

void setup(int role, int port, const std::string& host, MathFunctions& primary, int required_contexts) {
    if (!protocols.empty()) throw std::logic_error("native contexts are already initialized");
    requested_workers = bounded_environment("SDCF_ACTIVATION_WORKERS", 1);
    const int cggm = bounded_environment("SEGFSS_CGGM_THREADS", 1);
    const char* overlap_value = std::getenv("SDCF_OFFLINE_OVERLAP");
    const bool offline_overlap = overlap_value && std::string(overlap_value) == "1";
    if (const char* mode = std::getenv("SDCF_GRAPH_MODE")) execution_mode = mode;
    if (const char* mode = std::getenv("SDCF_GRAPH_OPENINGS")) opening_mode = mode;
    if (const char* mode = std::getenv("SDCF_TAIL_PARALLEL")) {
        if (std::string(mode) != "0" && std::string(mode) != "1")
            throw std::invalid_argument("tail parallel must be 0 or 1");
        parallel_tail = std::string(mode) == "1";
    }
    if (parallel_tail && execution_mode != "parallel")
        throw std::invalid_argument("tail parallel requires parallel graph contexts");
    if (execution_mode != "sequential" && execution_mode != "isolated" && execution_mode != "parallel")
        throw std::invalid_argument("unsupported graph mode");
    if (opening_mode != "independent" && opening_mode != "batched")
        throw std::invalid_argument("unsupported graph opening mode");
    if (isolated_graph() && (requested_workers > 2 || cggm > 2))
        throw std::invalid_argument("unsupported graph branch allocation: activation and cGGM limits must be at most 2");
    const int count = isolated_graph() ? 4 : std::max(required_contexts, requested_workers);
    if (count < 1 || count > 4 || role != primary.party)
        throw std::invalid_argument("invalid native protocol allocation");
    const int config[6] = {count, requested_workers, execution_mode == "parallel" ? 2 : isolated_graph() ? 1 : 0, int(offline_overlap), int(batched_graph_openings()), int(parallel_tail)};
    int remote[6] = {};
    if (role == sci::ALICE) {
        primary.iopack->io->send_data(config, sizeof(config));
        primary.iopack->io->recv_data(remote, sizeof(remote));
    } else {
        primary.iopack->io->recv_data(remote, sizeof(remote));
        primary.iopack->io->send_data(config, sizeof(config));
    }
    if (!std::equal(std::begin(config), std::end(config), std::begin(remote)))
        throw std::invalid_argument("native worker counts differ between parties");
    for (int slot = 0; slot < count; ++slot) {
        auto protocol = std::make_unique<Protocol>();
        if (!slot) protocol->math = &primary;
        else {
            protocol->io = std::make_unique<sci::IOPack>(role, port + 300 + 150 * (slot - 1), host);
            protocol->ot = std::make_unique<sci::OTPack>(protocol->io.get(), role);
            protocol->owned_math = std::make_unique<MathFunctions>(role, protocol->io.get(), protocol->ot.get());
            protocol->math = protocol->owned_math.get();
        }
        if (sdcf_dpf::register_math) sdcf_dpf::register_math(protocol->math, role, slot, port, host);
        if (sdcf_folded::register_math) sdcf_folded::register_math(protocol->math, role, slot, port, host);
        protocols.push_back(std::move(protocol));
        workers.emplace_back(slot ? std::make_unique<Worker>() : nullptr);
    }
    sdcf_log::Line config_log;
    config_log << "NATIVE_WORKER_CONFIG {\"requested_activation_workers\":" << requested_workers
        << ",\"protocol_contexts\":" << count << ",\"resident_workers\":" << count - 1
        << ",\"cggm_limit\":" << cggm << ",\"eigen_threads\":1,\"default_ot_participants\":2"
        << ",\"graph_mode\":\"" << execution_mode << "\",\"graph_branches\":" << (isolated_graph() ? 2 : 1)
        << ",\"graph_openings\":\"" << opening_mode << "\""
        << ",\"tail_parallel\":" << (parallel_tail ? "true" : "false")
        << ",\"branch_compute_limit\":" << (isolated_graph() ? 2 : 4)
        << ",\"activation_ot_participants\":" << (requested_workers == 1 ? 2 : 1)
        << ",\"activation_cggm_cap\":" << (requested_workers == 1 ? 2 : 1)
        << ",\"worker_local_cggm_cap\":true"
        << ",\"offline_overlap\":" << (offline_overlap ? "true" : "false")
        << ",\"max_compute_participants\":" << (execution_mode == "parallel" || offline_overlap ? 4 : std::max({2, cggm, requested_workers}))
        << ",\"ot_direction_schedule\":\"both_directions_static_loop\",\"channel_offsets\":[0";
    for (int slot = 1; slot < count; ++slot) config_log << ',' << 300 + 150 * (slot - 1);
    config_log << "]}\n";
}

void finish() {
    if (sdcf_folded::finish) sdcf_folded::finish();
    if (sdcf_dpf::finish) sdcf_dpf::finish();
    workers.clear();
    protocols.clear();
}

void run_worker_group(int first_slot, int slots, const std::function<void(int)>& work) {
    if (slots < 1 || first_slot < 0 || first_slot + slots > context_count())
        throw std::invalid_argument("invalid native worker group");
    int submitted = 0;
    std::exception_ptr error;
    try {
        for (int index = 1; index < slots; ++index) {
            const int slot = first_slot + index;
            workers.at(slot)->submit([&, slot] { work(slot); });
            ++submitted;
        }
        work(first_slot);
    } catch (...) { error = std::current_exception(); }
    for (int index = 1; index <= submitted; ++index) {
        auto worker_error = workers.at(first_slot + index)->wait();
        if (!error) error = worker_error;
    }
    if (error) std::rethrow_exception(error);
}

void run_task_pair(const std::function<void()>& first, const std::function<void()>& second) {
    if (execution_mode != "parallel") { first(); second(); return; }
    const auto task = identity;
    const auto begin = Clock::now();
    double first_seconds = 0, second_seconds = 0;
    int second_compute = 1;
    workers.at(2)->submit([&] {
        TaskScope branch(task.query, task.layer, 2, 2);
        const auto start = Clock::now();
        second();
        second_seconds = std::chrono::duration<double>(Clock::now() - start).count();
        second_compute = branch.observed_compute_participants();
    });
    std::exception_ptr error;
    try {
        const auto start = Clock::now(); first();
        first_seconds = std::chrono::duration<double>(Clock::now() - start).count();
    } catch (...) { error = std::current_exception(); }
    const auto other = workers.at(2)->wait();
    if (!error) error = other;
    if (error) std::rethrow_exception(error);
    const int first_compute = std::max(ot_observer ? ot_observer->load() : 1,
                                       compute_observer ? compute_observer->load() : 1);
    if (first_compute > 2 || second_compute > 2)
        throw std::runtime_error("observed graph branch exceeded its compute allocation");
    LogLine() << "GRAPH_GROUP {\"query\":" << task.query << ",\"index\":" << task.layer
        << ",\"branches\":2,\"branch_limit\":2,\"requested_compute_participants\":4"
        << ",\"observed_first_branch_peak\":" << first_compute
        << ",\"observed_second_branch_peak\":" << second_compute
        << ",\"observed_compute_upper_bound\":" << first_compute + second_compute
        << ",\"first_branch_seconds\":" << first_seconds << ",\"second_branch_seconds\":" << second_seconds
        << ",\"wall_seconds\":" << std::chrono::duration<double>(Clock::now() - begin).count() << "}\n";
}

void activation(bool tanh, uint64_t* input, uint64_t* output, int count,
                int input_bits, int output_bits, int input_scale, int output_scale) {
    if (count < 1) throw std::invalid_argument("activation count must be positive");
    const auto task = identity;
    const int limit = task.slots ? std::min(task.slots, requested_workers) : requested_workers;
    const int participants = std::min(limit, (count + 127) / 128);
    const int ot = participants == 1 ? 2 : 1;
    sdcf_round::SemanticScope trace("activation", task.query, task.layer, -1, -1, int(tanh));
    const auto start = Clock::now();
    std::vector<WorkerMetrics> metrics(participants);
    run_worker_group(task.first_slot, participants, [&](int slot) {
        const int index = slot - task.first_slot;
        auto& result = metrics[index];
        result.slot = slot; result.ot = ot;
        const int chunk = count / participants, remainder = count % participants;
        result.begin = index * chunk + std::min(index, remainder);
        result.count = chunk + (index < remainder);
        auto& math = context(slot);
        const auto sent_before = math.iopack->get_comm();
        TaskScope task_scope(task.query, task.layer, slot, 1);
        NativeScope native_scope(ot, result.calls);
        const auto worker_start = Clock::now();
        const bool folded = sdcf_folded::activation && sdcf_folded::activation(&math, tanh,
            result.count, input + result.begin, output + result.begin,
            input_bits, output_bits, input_scale, output_scale);
        if (!folded) {
            if (tanh) math.tanh(result.count, input + result.begin, output + result.begin,
                                input_bits, output_bits, input_scale, output_scale);
            else math.sigmoid(result.count, input + result.begin, output + result.begin,
                              input_bits, output_bits, input_scale, output_scale);
        }
        result.seconds = std::chrono::duration<double>(Clock::now() - worker_start).count();
        result.sent = math.iopack->get_comm() - sent_before;
        result.observed_ot = folded ? 1 : task_scope.observed_ot_participants();
        native_scope.stats(result.cggm_limit, result.observed_cggm, result.observed_transport);
        result.observed_compute = task_scope.observed_compute_participants();
        if (result.observed_compute > ot)
            throw std::runtime_error("activation worker exceeded its caller compute allocation");
    });
    const double wall = std::chrono::duration<double>(Clock::now() - start).count();
    std::ostringstream log;
    log.precision(12);
    double work_seconds = 0;
    int observed_compute = 0;
    for (const auto& result : metrics) {
        work_seconds += result.seconds;
        observed_compute += result.observed_compute;
        log << "ACTIVATION_WORKER {\"query\":" << task.query << ",\"index\":" << task.layer
            << ",\"function\":\"" << (tanh ? "TanH" : "Sigmoid") << "\",\"slot\":" << result.slot
            << ",\"begin\":" << result.begin << ",\"count\":" << result.count
            << ",\"seconds\":" << result.seconds << ",\"ot_participants\":" << result.observed_ot
            << ",\"requested_ot_participants\":" << result.ot
            << ",\"cggm_participant_limit\":" << result.cggm_limit
            << ",\"observed_cggm_participants\":" << result.observed_cggm
            << ",\"observed_transport_participants\":" << result.observed_transport
            << ",\"compute_participants\":" << result.observed_compute
            << ",\"native_conversion_calls\":" << result.calls << ",\"sent_bytes\":" << result.sent << "}\n";
    }
    log << "ACTIVATION_DETAIL {\"query\":" << task.query << ",\"index\":" << task.layer
        << ",\"function\":\"" << (tanh ? "TanH" : "Sigmoid") << "\",\"count\":" << count
        << ",\"requested_workers\":" << requested_workers << ",\"actual_workers\":" << participants
        << ",\"compute_participants\":" << observed_compute << ",\"requested_compute_participants\":" << participants * ot << ",\"wall_seconds\":" << wall
        << ",\"worker_seconds\":" << work_seconds << ",\"conversion\":\""
        << ((sdcf_folded::selected && sdcf_folded::selected(tanh)) ? "folded_sigma_f12_q10" : "native_SCI") << "\"}\n";
    sdcf_native::log(log.str());
    observe_compute_participants(observed_compute);
}
}
