#include "dpf_lut.h"
#include "native_dpf_lut.h"
#include "native_preprocessing.h"
#include "native_workers.h"
#include "sci_dpf_bridge.h"
#include <algorithm>
#include <cstdlib>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

int32_t bitlength = 64;

namespace {
struct Bridge {
    uint64_t resident = 0, generated = 0;
};
uint64_t material_bytes(int index_bits, int output_bits) {
    segfss::DpfPublicLutMaterial material;
    material.selection_share.reset(uint64_t(1) << index_bits, output_bits);
    return material.residentBytes();
}
uint64_t lookup_exp(uint64_t value, int32_t, int32_t) { return value; }
uint64_t lookup_reciprocal(uint64_t value, int) { return value; }
}

namespace sdcf_native {
TaskIdentity task_identity() { return {0, 8, 0, 1}; }
}

extern "C" {
uint64_t sdcf_round_actor_fork() { return 0; }
uint64_t sdcf_round_actor_enter(uint64_t) { return 0; }
uint64_t sdcf_round_actor_leave(uint64_t) { return 0; }
void sdcf_round_actor_join(uint64_t) {}
void* segfss_sci_dpf_create(int, const char*, int) { return new Bridge; }
uint64_t segfss_sci_dpf_material_metadata_bytes() {
    return segfss::DpfPublicLutMaterial{}.residentBytes();
}
int segfss_sci_dpf_prepare(void* pointer, const segfss_sci_dpf_call* calls, int count) {
    auto& bridge = *static_cast<Bridge*>(pointer);
    bridge = {};
    for (int i = 0; i < count; ++i) {
        const auto& call = calls[i];
        bridge.resident += uint64_t(call.count) *
            material_bytes(call.index_bits, std::max(call.output_bits[0], call.output_bits[1]));
        bridge.generated += call.count;
        for (int t = 0; t < call.tables; ++t) {
            std::vector<GroupElement> table;
            for (uint64_t leaf = 0; leaf < (uint64_t(1) << call.index_bits); ++leaf)
                table.emplace_back(call.values[t][leaf], call.output_bits[t]);
            bridge.resident += table.capacity() * sizeof(GroupElement);
        }
    }
    return 0;
}
int segfss_sci_dpf_counters_get(void* pointer, segfss_sci_dpf_counters* counters) {
    const auto& bridge = *static_cast<Bridge*>(pointer);
    *counters = {};
    counters->resident_bytes = bridge.resident;
    counters->generated_requests = bridge.generated;
    return 0;
}
void segfss_sci_dpf_destroy(void* pointer) { delete static_cast<Bridge*>(pointer); }
const char* segfss_sci_dpf_error() { return "budget regression bridge"; }
}

int main(int argc, char** argv) {
    try {
        const std::string mode = argv[1], check = argv[2];
        const bool sigmoid = mode != "tanh", tanh = mode != "sigmoid";
        constexpr int count = 12800;
        setenv("SDCF_NATIVE_PREPROCESSING", "1", 1);
        setenv("SDCF_DPF_SIGMOID", sigmoid ? "1" : "0", 1);
        setenv("SDCF_DPF_TANH", tanh ? "1" : "0", 1);
        int identity = 0;
        auto* math = reinterpret_cast<MathFunctions*>(&identity);
        sdcf_dpf::register_math(math, 1, 0, 10000, "127.0.0.1");
        sdcf_dpf::reserve_schedule(count, count);
        const uint64_t reserve = std::stoull(std::getenv("SEGFSS_DPF_RESERVE_BYTES"));
        sdcf_preprocess::begin_prepare(0, true);
        sdcf_preprocess::planning_active = true;
        sdcf_dpf::plan_begin();
        std::vector<uint64_t> input(2 * count), output(2 * count), other(count);
        for (bool is_tanh : {false, true}) {
            if ((is_tanh && !tanh) || (!is_tanh && !sigmoid)) continue;
            sdcf_dpf::ActivationScope activation(is_tanh);
            sdcf_dpf::lookup_exp(math, count, input.data(), output.data(), 2, 8, 8,
                is_tanh ? 6 : 7, is_tanh ? 7 : 14, lookup_exp);
            sdcf_dpf::lookup_reciprocal(math, count, input.data(), output.data(), other.data(),
                is_tanh ? 3 : 6, lookup_reciprocal, lookup_reciprocal);
        }
        sdcf_dpf::plan_end();
        sdcf_preprocess::planning_active = false;
        uint64_t actual = 0;
        if (sigmoid) actual += count * (2 * material_bytes(8, 16) + material_bytes(6, 15)) +
            (2 * 256 + 2 * 64) * sizeof(GroupElement);
        if (tanh) actual += count * (2 * material_bytes(8, 9) + material_bytes(3, 9)) +
            (2 * 256 + 2 * 8) * sizeof(GroupElement);
        const uint64_t planned = sdcf_dpf::planned_resident_bytes();
        std::cout << "DPF_BUDGET_CHECK mode=" << mode << " check=" << check
            << " metadata=" << segfss::DpfPublicLutMaterial{}.residentBytes()
            << " planned=" << planned << " actual=" << actual << " reserve=" << reserve << '\n';
        if (check == "planned" && planned != actual)
            throw std::runtime_error("native plan does not match actual resident material bytes");
        if (check == "reserve" && reserve < actual)
            throw std::runtime_error("native reserve underestimates actual resident material bytes");
        if (check == "prepare") {
            sdcf_dpf::prepare_query(0);
            if (sdcf_dpf::counters().resident != actual)
                throw std::runtime_error("prepared resident count differs from actual materials");
        }
        sdcf_dpf::finish();
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "DPF_BUDGET_ERROR " << error.what() << '\n';
        return 2;
    }
}
