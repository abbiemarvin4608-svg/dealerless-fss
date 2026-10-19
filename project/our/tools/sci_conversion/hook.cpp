#include "hook.h"
#include "../segmented_dcf_runtime/native_workers.h"
#include "../segmented_dcf_runtime/native_preprocessing.h"
#include "../../src/sci_numeric_oracle.h"
#include "../../src/runtime_options.h"
#include "../../src/sci_msb_bridge.h"
#include <algorithm>
#include <cstdlib>
#include <iostream>
#include <map>
#include <stdexcept>

extern "C" void sdcf_round_trace_io(int, int, uint64_t, uint64_t) __attribute__((weak));
extern "C" uint64_t sdcf_round_actor_fork() __attribute__((weak));
extern "C" uint64_t sdcf_round_actor_enter(uint64_t) __attribute__((weak));
extern "C" uint64_t sdcf_round_actor_leave(uint64_t) __attribute__((weak));
extern "C" void sdcf_round_actor_join(uint64_t) __attribute__((weak));

namespace {
using Call = segfss_sci_conversion_call;
struct Context {
    std::string backend;
    std::string segment_policy = "legacy";
    int role = 0, query = -1, max_segments = 4;
    void* bridge = nullptr;
    bool prepared = false, online = false;
    const sdcf_runtime::Layer* layer = nullptr;
    std::map<int, std::vector<Call>> plan;
    std::size_t cursor = 0;
    sdcf_runtime::Counters final_counters;
};
Context contexts[2];
Context& current_context() { return contexts[sdcf_native::selected_branch()]; }
#define context current_context()
int context_count() { return sdcf_native::isolated_graph() ? 2 : 1; }
std::vector<sdcf_runtime::Layer> branch_schedule(const std::vector<sdcf_runtime::Layer>& schedule, int branch) {
    std::vector<sdcf_runtime::Layer> result;
    for (const auto& layer : schedule) if (sdcf_native::graph_branch(layer.index) == branch) result.push_back(layer);
    return result;
}
bool enabled() {
    return context.backend == "SCI_ours" || context.backend == "SCI_Beaver_full" ||
        context.backend == "SCI_Beaver_segmented" || context.backend == "SCI_Beaver_segmented_DPF_LUT" ||
        context.backend == "SCI_Beaver_segmented_PaperLUT";
}
bool beaver() { return context.backend.compare(0, 10, "SCI_Beaver") == 0; }
int log2Exact(int64_t value) {
    if (value < 1 || (value & (value - 1))) throw std::invalid_argument("non-power-of-two graph scale");
    int shift = 0; while (value > 1) { value >>= 1; ++shift; } return shift;
}
int depth(int count) { int result = 0; for (int value = 1; value < count; value <<= 1) ++result; return result; }
void append(std::vector<Call>& calls, int count, int in, int out, int shift, int call, bool sign = true) {
    segfss::reference::validate({in, out, shift, static_cast<segfss::reference::SciCall>(call), sign});
    if (shift == 0 && out <= in) return;
    calls.push_back({count, in, out, shift, call, int(sign)});
}
std::vector<Call> plan(const sdcf_runtime::Layer& layer) {
    std::vector<Call> result;
    auto a = layer.numeric_arguments;
    if (sdcf_native::batched_graph_openings()) {
        if (layer.index % 15 == 7) return result;
        if (layer.index % 15 == 5) a[0] *= 2;
    }
    if (layer.function == "MatMul") {
        if (a.size() != 12) throw std::invalid_argument("MatMul conversion graph changed");
        const int d = depth(a[1]), product = a[8] + a[9], count = a[0] * a[2];
        append(result, a[0] * a[1], a[8], beaver() ? product + d : a[8] + d, 0, 3);
        append(result, count, product + d, product, d, 2);
        const int shift = log2Exact(a[3]) + log2Exact(a[4]) + log2Exact(a[7]) + a[5] - d;
        if (shift > 0) append(result, count, product, product, shift, 0);
    } else if (layer.function == "MulCir") {
        if (a.size() != 9) throw std::invalid_argument("MulCir conversion graph changed");
        const int count = a[0] * a[1], product = a[5] + a[6];
        if (beaver()) {
            if (segfss::runtimeFlag("SDCF_PAIR_EXTENSIONS") && a[5] == a[6]) {
                append(result, 2 * count, a[5], product, 0, 3);
            } else {
                append(result, count, a[5], product, 0, 3);
                append(result, count, a[6], product, 0, 3);
            }
        }
        append(result, count, product, product, log2Exact(a[2]) + log2Exact(a[3]) + log2Exact(a[4]), 0);
    } else if (layer.function == "ScalarMul") {
        if (a.size() != 9) throw std::invalid_argument("ScalarMul conversion graph changed");
        const int count = a[0] * a[1];
        append(result, count, a[6], a[7], 0, 3);
        append(result, count, a[7], a[7], log2Exact(a[2]) + log2Exact(a[3]) + log2Exact(a[4]), 0);
    } else if (layer.function == "MatAdd" || layer.function == "MatSubBroadCastA" ||
               layer.function == "MatAddBroadCastB") {
        if (a.size() != 10) throw std::invalid_argument("MatAdd conversion graph changed");
        const int count = a[0] * a[1], temp = a[8];
        const bool scalar_a = layer.function == "MatSubBroadCastA";
        const bool scalar_b = layer.function == "MatAddBroadCastB";
        append(result, count, scalar_a ? temp : a[6], temp, 0, 3);
        append(result, count, scalar_b ? temp : a[7], temp, 0, 3);
        append(result, count, temp, temp, scalar_a ? 0 : log2Exact(a[2]) + log2Exact(a[4]), 0);
        append(result, count, temp, temp, scalar_b ? 0 : log2Exact(a[3]) + log2Exact(a[4]), 0);
        append(result, count, temp, temp, log2Exact(a[5]), 0);
    } else if (layer.function != "Sigmoid" && layer.function != "TanH") {
        throw std::invalid_argument("unplanned conversion graph function: " + layer.function);
    }
    return result;
}
void bridgeCheck(int status) {
    if (status) throw std::runtime_error(segfss_sci_conversion_error());
}
void printWidths(std::ostream& output, int bits, int segments) {
    output << '[';
    for (int index = 0; index < segments; ++index) {
        if (index) output << ',';
        output << bits / segments + (index < bits % segments ? 1 : 0);
    }
    output << ']';
}
void printCall(const char* prefix, const sdcf_runtime::Layer& layer, std::size_t slot, const Call& c) {
    sdcf_native::LogLine output;
    const bool toward_zero = c.call == 0 && c.signed_arithmetic && c.shift > 0;
    const bool extend = c.output_bits > c.input_bits - c.shift;
    const int low = c.shift, sign = c.signed_arithmetic && (extend || toward_zero) ? c.input_bits - 1 : 0;
    const bool fused_sign_wrap = extend && c.signed_arithmetic &&
        segfss::runtimeFlag("SEGFSS_FUSE_SIGN_WRAP");
    const int wrap = extend && !fused_sign_wrap ? c.input_bits : 0;
    const int limit = context.backend == "SCI_Beaver_full" ? 1 : context.max_segments;
    const auto segments = [limit](int bits) {
        if (context.backend == "SCI_Beaver_full") return std::min(1, bits);
        if (context.segment_policy == "fixed-2") return std::min(2, bits);
        if (context.segment_policy == "fixed-4") return std::min(4, bits);
        if (context.segment_policy == "depth-cap-8") return (bits + 7) / 8;
        return std::min(limit, bits);
    };
    output << prefix << " {\"query\":" << context.query << ",\"index\":" << layer.index
        << ",\"context\":" << sdcf_native::selected_branch()
        << ",\"step\":" << layer.step << ",\"operator\":\"" << layer.operator_name
        << "\",\"slot\":" << slot << ",\"count\":" << c.count << ",\"input_bits\":" << c.input_bits
        << ",\"output_bits\":" << c.output_bits << ",\"shift\":" << c.shift << ",\"call\":" << c.call
        << ",\"numeric_mode\":\"sci_compatible\",\"comparison_backend\":\""
        << (context.backend == "SCI_Beaver_full" ? "full_tree" : "segmented_dcf") << "\",\"low_comparison_bits\":" << low
        << ",\"sign_comparison_bits\":" << sign << ",\"wrap_comparison_bits\":" << wrap
        << ",\"segment_policy\":\"" << context.segment_policy << "\""
        << ",\"fused_sign_wrap\":" << (fused_sign_wrap ? "true" : "false")
        << ",\"low_nseg\":" << segments(low) << ",\"sign_nseg\":" << segments(sign)
        << ",\"wrap_nseg\":" << segments(wrap) << ",\"low_widths\":";
    printWidths(output, low, segments(low));
    output << ",\"sign_widths\":"; printWidths(output, sign, segments(sign));
    output << ",\"wrap_widths\":"; printWidths(output, wrap, segments(wrap));
    const char* workers = std::getenv("SEGFSS_CGGM_THREADS");
    output << ",\"cggm_requested_workers\":" << (workers ? std::stoi(workers) : 0)
        << ",\"sample_groups\":1}" << std::endl;
}
}

namespace sdcf_conversion {
void setup_one(const std::string& backend, int role, const std::vector<sdcf_runtime::Layer>& schedule) {
    if (context.prepared || context.online || context.bridge) throw std::logic_error("conversion setup state");
    context.backend = backend; context.role = role;
    if (!enabled()) return;
    if (std::getenv("SDCF_ROUND_TRACE"))
        segfss_sci_set_round_trace_callbacks(sdcf_round_trace_io, sdcf_round_actor_fork,
            sdcf_round_actor_enter, sdcf_round_actor_leave, sdcf_round_actor_join);
    const char* limit = std::getenv("SDCF_MAX_SEGMENTS");
    context.max_segments = limit ? std::stoi(limit) : 4;
    if (const char* policy = std::getenv("SEGFSS_SEGMENT_POLICY")) context.segment_policy = policy;
    const char* host = std::getenv("SDCF_HOST"); const char* port = std::getenv("SDCF_PORT");
    if (!port) throw std::invalid_argument("SDCF_PORT is required for conversion channel");
    context.bridge = segfss_sci_conversion_create(role, host ? host : "127.0.0.1", std::stoi(port) + 800 * sdcf_native::selected_branch(),
        backend == "SCI_Beaver_full", context.max_segments);
    if (!context.bridge) throw std::runtime_error(segfss_sci_conversion_error());
    const char* prefetch = std::getenv("SEGFSS_PREFETCH_BYTES");
    const uint64_t configured_budget = prefetch ? std::stoull(prefetch) : 0;
    const char* dpf_reserve = std::getenv("SEGFSS_DPF_RESERVE_BYTES");
    const uint64_t reserved_budget = dpf_reserve ? std::stoull(dpf_reserve) : 0;
    if(reserved_budget > configured_budget)
        throw std::runtime_error("joint conversion and DPF resident budget is insufficient");
    const uint64_t party_budget = configured_budget - reserved_budget;
    const int branches = context_count(), branch = sdcf_native::selected_branch();
    const uint64_t context_budget = party_budget / branches + (uint64_t(branch) < party_budget % branches);
    bridgeCheck(segfss_sci_conversion_set_prefetch_budget(context.bridge, context_budget));
    sdcf_native::LogLine() << "CONVERSION_PREFETCH_BUDGET {\"context\":" << branch
        << ",\"contexts\":" << branches << ",\"context_budget_bytes\":" << context_budget
        << ",\"party_budget_bytes\":" << party_budget
        << ",\"configured_party_budget_bytes\":" << configured_budget
        << ",\"dpf_reserved_bytes\":" << reserved_budget << "}\n";
    context.plan.clear();
    for (const auto& layer : schedule) if (!context.plan.emplace(layer.index, plan(layer)).second)
        throw std::invalid_argument("duplicate conversion layer");
}
void prepare_one(int query, const std::vector<sdcf_runtime::Layer>& schedule) {
    if (!enabled()) return;
    if (context.prepared || context.online) throw std::logic_error("unconsumed conversion query");
    context.query = query;
    std::vector<Call> calls;
    for (const auto& layer : schedule) {
        const auto& planned = context.plan.at(layer.index);
        for (std::size_t index = 0; index < planned.size(); ++index) {
            printCall("CONVERSION_OFFLINE", layer, index, planned[index]); calls.push_back(planned[index]);
        }
    }
    bridgeCheck(segfss_sci_conversion_prepare(context.bridge, calls.data(), calls.size()));
    context.prepared = true;
}
void enter_online_one(int query) {
    if (!enabled()) return;
    if (!context.prepared || context.online || context.query != query) throw std::logic_error("conversion online entry state");
    context.online = true;
}
void layer_begin(const sdcf_runtime::Layer& layer) {
    if (!enabled()) return;
    if ((!context.online && !sdcf_preprocess::planning_active.load()) || context.layer) throw std::logic_error("conversion layer entry state");
    context.layer = &layer; context.cursor = 0;
}
void layer_end(const sdcf_runtime::Layer& layer) {
    if (!enabled()) return;
    if (context.layer != &layer || context.cursor != context.plan.at(layer.index).size())
        throw std::logic_error("conversion layer did not consume its complete call plan");
    const bool folded_activation = context.backend == "SCI_Beaver_segmented_PaperLUT" &&
        ((layer.function == "Sigmoid" && segfss::runtimeFlag("SDCF_FOLDED_SIGMOID")) ||
         (layer.function == "TanH" && segfss::runtimeFlag("SDCF_FOLDED_TANH")));
    if ((layer.function == "Sigmoid" || layer.function == "TanH") && !folded_activation)
        sdcf_native::LogLine() << "CONVERSION_REMAINING_SCI {\"query\":" << context.query << ",\"index\":" << layer.index
            << ",\"function\":\"" << layer.function << "\",\"reason\":\"native_activation_internals\"}\n";
    context.layer = nullptr;
}
void leave_online_one(int query) {
    if (!enabled()) return;
    if (!context.online || context.layer || context.query != query) throw std::logic_error("conversion online exit state");
    bridgeCheck(segfss_sci_conversion_finish_query(context.bridge));
    context.online = false; context.prepared = false;
}
sdcf_runtime::Counters counters_one() {
    if (!context.bridge) return context.final_counters;
    segfss_sci_conversion_counters stats{};
    bridgeCheck(segfss_sci_conversion_get_counters(context.bridge, &stats));
    sdcf_runtime::Counters result;
    result.sent_bytes = stats.sent_bytes; result.received_bytes = stats.received_bytes;
    result.offline_generation_epoch = stats.generation_epoch;
    result.io_calls_known = false;
    return result;
}
void finish_one() {
    if (context.prepared || context.online) throw std::logic_error("unfinished conversion query");
    context.final_counters = counters_one();
    if (context.bridge) segfss_sci_conversion_destroy(context.bridge);
    context.bridge = nullptr; context.plan.clear(); context.backend.clear();
}
void setup(const std::string& backend, int role, const std::vector<sdcf_runtime::Layer>& schedule) {
    for (int branch = 0; branch < context_count(); ++branch) {
        sdcf_native::TaskScope scope(-1, -1, branch * 2);
        setup_one(backend, role, branch_schedule(schedule, branch));
    }
}
void prepare(int query, const std::vector<sdcf_runtime::Layer>& schedule) {
    for (int branch = 0; branch < context_count(); ++branch) {
        sdcf_native::TaskScope scope(query, -1, branch * 2);
        prepare_one(query, branch_schedule(schedule, branch));
    }
}
void offline_reservation(uint64_t bytes) {
    if (const char* prefetch = std::getenv("SEGFSS_PREFETCH_BYTES"))
        bytes += std::stoull(prefetch);
    for (int branch = 0; branch < context_count(); ++branch) {
        sdcf_native::TaskScope scope(-1, -1, branch * 2);
        if (context.bridge) bridgeCheck(segfss_sci_conversion_set_generation_resident(context.bridge, bytes));
    }
}
void abort_preparation() {
    for (int branch = 0; branch < context_count(); ++branch) {
        sdcf_native::TaskScope scope(-1, -1, branch * 2);
        if (context.bridge) segfss_sci_conversion_destroy(context.bridge);
        context.bridge = nullptr; context.prepared = false; context.online = false;
        context.plan.clear();
    }
}
void enter_online(int query) {
    for (int branch = 0; branch < context_count(); ++branch) {
        sdcf_native::TaskScope scope(query, -1, branch * 2); enter_online_one(query);
    }
}
void leave_online(int query) {
    for (int branch = 0; branch < context_count(); ++branch) {
        sdcf_native::TaskScope scope(query, -1, branch * 2); leave_online_one(query);
    }
}
sdcf_runtime::Counters counters() {
    sdcf_runtime::Counters result;
    for (int branch = 0; branch < context_count(); ++branch) {
        sdcf_native::TaskScope scope(-1, -1, branch * 2);
        const auto item = counters_one();
        result.sent_bytes += item.sent_bytes; result.received_bytes += item.received_bytes;
        result.offline_generation_epoch += item.offline_generation_epoch;
        result.io_calls_known = result.io_calls_known && item.io_calls_known;
    }
    return result;
}
void finish() {
    for (int branch = 0; branch < context_count(); ++branch) {
        sdcf_native::TaskScope scope(-1, -1, branch * 2); finish_one();
    }
}
bool convert(int role, uint64_t* input, uint64_t* output, int count,
             int input_bits, int output_bits, int shift, int call, bool signed_arithmetic) {
    if (sdcf_native::record_native_conversion()) return false;
    if (!enabled()) return false;
    const bool planning = sdcf_preprocess::planning_active.load();
    if ((!context.online && !planning) || role != context.role) return false;
    if (!context.layer || context.layer->function == "Sigmoid" || context.layer->function == "TanH") {
        sdcf_native::LogLine() << "CONVERSION_NATIVE_CALL {\"query\":" << context.query
            << ",\"index\":" << (context.layer ? context.layer->index : -1)
            << ",\"count\":" << count << ",\"input_bits\":" << input_bits << ",\"output_bits\":" << output_bits
            << ",\"shift\":" << shift << ",\"call\":" << call << ",\"reason\":\""
            << (context.layer ? "native_activation_internals" : "classifier_output_or_argmax") << "\"}\n";
        return false;
    }
    if (shift == 0 && output_bits <= input_bits) return false;
    Call actual{count, input_bits, output_bits, shift, call, int(signed_arithmetic)};
    const auto& planned = context.plan.at(context.layer->index);
    if (context.cursor >= planned.size()) throw std::logic_error("unplanned conversion subcall");
    const auto& expected = planned[context.cursor];
    if (actual.count != expected.count || actual.input_bits != expected.input_bits ||
        actual.output_bits != expected.output_bits || actual.shift != expected.shift || actual.call != expected.call ||
        actual.signed_arithmetic != expected.signed_arithmetic) {
        std::cerr << "CONVERSION_MISMATCH layer=" << context.layer->operator_name << " slot=" << context.cursor
            << " actual=" << count << ',' << input_bits << ',' << output_bits << ',' << shift << ',' << call
            << " expected=" << expected.count << ',' << expected.input_bits << ',' << expected.output_bits
            << ',' << expected.shift << ',' << expected.call << '\n';
        throw std::logic_error("conversion contract does not match frozen graph plan");
    }
    if (planning) {
        std::fill_n(output, count, uint64_t(0));
        ++context.cursor;
        return true;
    }
    segfss_sci_conversion_metrics metrics{};
    bridgeCheck(segfss_sci_conversion_online(context.bridge, &actual, input, output, &metrics));
    sdcf_native::observe_compute_participants(std::max<uint64_t>(1, metrics.eval_participants));
    printCall("CONVERSION_ONLINE", *context.layer, context.cursor, actual);
    sdcf_native::LogLine() << "CONVERSION_DETAIL {\"query\":" << context.query << ",\"index\":" << context.layer->index
        << ",\"slot\":" << context.cursor++
        << ",\"masked_open_ms\":" << metrics.masked_open_ms << ",\"comparison_online_ms\":" << metrics.comparison_online_ms
        << ",\"rounding_online_ms\":" << metrics.rounding_online_ms << ",\"b2a_online_ms\":" << metrics.b2a_online_ms
        << ",\"masked_values\":" << metrics.masked_values << ",\"masked_bits\":" << metrics.masked_bits
        << ",\"logical_rounds\":" << metrics.logical_rounds
        << ",\"material_preload_ms\":" << metrics.material_preload_ms
        << ",\"prefetch_wait_ms\":" << metrics.prefetch_wait_ms
        << ",\"contiguous_material\":" << metrics.contiguous_material
        << ",\"prefetched\":" << metrics.prefetched
        << ",\"resident_bytes\":" << metrics.resident_bytes
        << ",\"prefetch_budget_bytes\":" << metrics.prefetch_budget_bytes
        << ",\"material_load_ms\":" << metrics.material_load_ms << ",\"query_prepare_ms\":" << metrics.query_prepare_ms
        << ",\"local_evaluation_ms\":" << metrics.local_evaluation_ms << ",\"scan_ms\":" << metrics.scan_ms
        << ",\"release_ms\":" << metrics.release_ms << ",\"total_online_ms\":" << metrics.total_online_ms
        << ",\"material_bytes\":" << metrics.material_bytes << ",\"comparison_keys\":" << metrics.comparison_keys
        << ",\"scan_and_gates\":" << metrics.scan_and_gates << ",\"eval_participants\":" << metrics.eval_participants
        << ",\"keygen_internal_hashes\":" << metrics.keygen_internal_hashes
        << ",\"keygen_leaf_hashes\":" << metrics.keygen_leaf_hashes
        << ",\"keygen_participants\":" << metrics.keygen_participants
        << ",\"keygen_aes_scratch_bytes\":" << metrics.keygen_aes_scratch_bytes
        << ",\"eval_internal_hashes\":" << metrics.eval_internal_hashes
        << ",\"eval_leaf_hashes\":" << metrics.eval_leaf_hashes
        << ",\"pruned_e_outputs\":" << metrics.pruned_e_outputs
        << ",\"skipped_leaf_hashes\":" << metrics.skipped_leaf_hashes
        << ",\"prune_equality\":" << metrics.prune_equality
        << ",\"prune_leaf_hashes\":" << metrics.prune_leaf_hashes
        << ",\"eval_aes_scratch_bytes\":" << metrics.eval_aes_scratch_bytes
        << ",\"aes_online\":" << metrics.aes_online << ",\"aes_offline\":" << metrics.aes_offline
        << ",\"timing_kind\":\"serial_phase_wall\",\"raw_secret_openings\":0}\n";
    return true;
}
bool extend_pair(int role, uint64_t* lhs, uint64_t* rhs, uint64_t* wide_lhs,
                 uint64_t* wide_rhs, int count, int lhs_bits, int rhs_bits, int output_bits) {
    if (!enabled() || !segfss::runtimeFlag("SDCF_PAIR_EXTENSIONS") || lhs_bits != rhs_bits)
        return false;
    if (count < 1 || count > INT32_MAX / 2)
        throw std::invalid_argument("paired extension count exceeds conversion range");
    std::vector<uint64_t> input(size_t(count) * 2), output(input.size());
    std::copy_n(lhs, count, input.begin());
    std::copy_n(rhs, count, input.begin() + count);
    if (!convert(role, input.data(), output.data(), 2 * count, lhs_bits, output_bits, 0, 3, true))
        return false;
    std::copy_n(output.begin(), count, wide_lhs);
    std::copy_n(output.begin() + count, count, wide_rhs);
    sdcf_native::LogLine() << "CONVERSION_GROUP {\"query\":" << context.query << ",\"index\":" << context.layer->index
        << ",\"slot\":" << context.cursor - 1 << ",\"input_groups\":2,\"group_count\":" << count
        << ",\"mapping\":\"lhs_then_rhs\",\"contract\":\"homogeneous_signed_extension\"}\n";
    return true;
}
}
