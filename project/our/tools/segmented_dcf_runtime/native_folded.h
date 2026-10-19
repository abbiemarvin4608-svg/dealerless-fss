#ifndef SDCF_NATIVE_FOLDED_H
#define SDCF_NATIVE_FOLDED_H
#include <cstdint>
#include <string>
class MathFunctions;
namespace sdcf_folded {
struct Counters {
    uint64_t sent=0,received=0,epoch=0,resident=0,generated=0,consumed=0;
    uint64_t generated_elements=0,consumed_elements=0,online_generated=0,online_loaded_bytes=0;
};
bool activation(MathFunctions*,bool,int,const uint64_t*,uint64_t*,int,int,int,int) __attribute__((weak));
bool selected(bool tanh) __attribute__((weak));
void register_math(MathFunctions*,int,int,int,const std::string&) __attribute__((weak));
void plan_begin() __attribute__((weak));
std::string plan_end() __attribute__((weak));
void reserve_schedule(uint64_t,uint64_t) __attribute__((weak));
void prepare_query(int) __attribute__((weak));
void balance(int) __attribute__((weak));
void finish() __attribute__((weak));
Counters counters() __attribute__((weak));
uint64_t planned_resident_bytes() __attribute__((weak));
}
#endif
