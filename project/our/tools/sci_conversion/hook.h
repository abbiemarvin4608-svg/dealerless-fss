#pragma once
#include "backend_types.h"
#include "../../src/sci_conversion_bridge.h"

namespace sdcf_conversion {
void setup(const std::string&, int, const std::vector<sdcf_runtime::Layer>&);
void prepare(int, const std::vector<sdcf_runtime::Layer>&);
void offline_reservation(uint64_t bytes);
void abort_preparation();
void enter_online(int);
void layer_begin(const sdcf_runtime::Layer&);
void layer_end(const sdcf_runtime::Layer&);
void leave_online(int);
void finish();
sdcf_runtime::Counters counters();
bool convert(int role, uint64_t* input, uint64_t* output, int count,
             int input_bits, int output_bits, int shift, int call, bool signed_arithmetic);
bool extend_pair(int role, uint64_t* lhs, uint64_t* rhs, uint64_t* wide_lhs,
                 uint64_t* wide_rhs, int count, int lhs_bits, int rhs_bits, int output_bits);
}
