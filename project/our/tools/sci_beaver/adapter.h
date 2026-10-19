#pragma once
#include "backend_types.h"

namespace sdcf_backend {
std::vector<std::string> supported_backends();
void validate(const std::string& name);
void setup(const std::string& name,int role,const std::vector<sdcf_runtime::Layer>& schedule);
void prepare(int query,const std::vector<sdcf_runtime::Layer>& schedule);
void enter_online(int query);
void layer_begin(const sdcf_runtime::Layer& layer);
void layer_end(const sdcf_runtime::Layer& layer);
void leave_online(int query);
void finish();
sdcf_runtime::Counters counters();
}

namespace sdcf_beaver {
bool matmul(int tid,uint64_t* lhs,uint64_t* rhs,uint64_t* out,
            int rows,int inner,int cols,int lhs_bits,int rhs_bits);
bool mulcir(int tid,uint64_t* lhs,uint64_t* rhs,uint64_t* out,
            int elements,int lhs_bits,int rhs_bits,int temp_bits,int out_bits,
            int64_t shrA,int64_t shrB,int64_t demote);
}
