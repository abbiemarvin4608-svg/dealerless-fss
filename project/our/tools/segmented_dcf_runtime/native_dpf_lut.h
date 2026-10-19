#pragma once
#include <cstdint>
#include <string>
class MathFunctions;
namespace sdcf_dpf {
struct Counters {uint64_t sent=0,received=0,epoch=0,resident=0,generated=0,consumed=0;};
int activation_enter(bool tanh) __attribute__((weak));
void activation_leave(int previous) __attribute__((weak));
class ActivationScope {
    int previous_;
public:
    explicit ActivationScope(bool tanh):previous_(activation_enter?activation_enter(tanh):-1){}
    ~ActivationScope(){if(activation_leave)activation_leave(previous_);}
};
bool lookup_exp(MathFunctions*,int,uint64_t*,uint64_t*,int,int,int,int,int,
                uint64_t (*)(uint64_t,int32_t,int32_t)) __attribute__((weak));
bool lookup_reciprocal(MathFunctions*,int,uint64_t*,uint64_t*,uint64_t*,int,
                       uint64_t (*)(uint64_t,int),uint64_t (*)(uint64_t,int)) __attribute__((weak));
void register_math(MathFunctions*,int role,int slot,int port,const std::string& host) __attribute__((weak));
void plan_begin() __attribute__((weak));
std::string plan_end() __attribute__((weak));
void reserve_schedule(uint64_t sigmoid_count,uint64_t tanh_count) __attribute__((weak));
void prepare_query(int query) __attribute__((weak));
void balance(int query) __attribute__((weak));
void finish() __attribute__((weak));
Counters counters() __attribute__((weak));
uint64_t planned_resident_bytes() __attribute__((weak));
}
