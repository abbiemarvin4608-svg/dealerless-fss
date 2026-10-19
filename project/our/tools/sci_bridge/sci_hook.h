#pragma once
#include "sci_msb_bridge.h"
#include <algorithm>
#include <cstdlib>
#include <cstdio>
#include <stdexcept>
#include <string>

// Include only in the isolated SCI aux-protocols.cpp. Use when SCI owns exactly
// one protocol thread; explicitly call finish() before SCI final metrics.
namespace sci_segmented {
inline void*& context(){static void* value=nullptr;return value;}
inline void ensure(int role){
    if(context())return;
    const char* host=std::getenv("SEGFSS_SCI_HOST");
    const char* port=std::getenv("SEGFSS_SCI_PORT");
    if(!port)throw std::runtime_error("SEGFSS_SCI_PORT is required");
    context()=segfss_sci_create(role,host?host:"127.0.0.1",std::atoi(port));
    if(!context())throw std::runtime_error(segfss_sci_error());
}
inline bool msb(int role,uint64_t* input,uint8_t* output,int count,int width) {
    const char* enabled=std::getenv("SEGFSS_SCI_ENABLE");
    if(!enabled||std::atoi(enabled)!=1)return false;
    const char* selected=std::getenv("SEGFSS_SCI_WIDTH");
    if(width!=(selected?std::atoi(selected):16))return false;
    ensure(role);
    const char* segments=std::getenv("SEGFSS_SCI_NSEG");
    const int nseg=std::min(width-1,segments?std::atoi(segments):4);
    if(segfss_sci_msb(context(),input,output,count,width,nseg))throw std::runtime_error(segfss_sci_error());
    return true;
}
inline segfss_sci_stats stats(){segfss_sci_stats out{};if(context())segfss_sci_get_stats(context(),&out);return out;}
inline void precompute_recorded(){
    if(!context())throw std::runtime_error("cannot precompute FSS schedule before bridge creation");
    if(segfss_sci_precompute_recorded(context()))throw std::runtime_error(segfss_sci_error());
}
inline void precompute_schedule(int count,int width,int nseg,int calls){
    if(!context())throw std::runtime_error("cannot precompute FSS schedule before bridge creation");
    if(segfss_sci_precompute_schedule(context(),count,width,nseg,calls))
        throw std::runtime_error(segfss_sci_error());
}
inline void precompute_activations(int batch,int steps,int elements_per_call,int bits){
    if(!context())throw std::runtime_error("cannot precompute activations before bridge creation");
    if(segfss_sci_precompute_activations(context(),batch,steps,elements_per_call,bits))
        throw std::runtime_error(segfss_sci_error());
}
inline bool activation_enabled(){
    const char* enabled=std::getenv("SEGFSS_FSS_ACTIVATION");
    return enabled&&std::atoi(enabled)==1;
}
inline void activation(int kind,uint64_t* input,uint64_t* output,int count,int bits){
    if(!activation_enabled())throw std::runtime_error("FSS activation is disabled");
    if(!context())throw std::runtime_error("activation bridge is not initialized");
    if(segfss_sci_activation(context(),kind,input,output,count,bits))
        throw std::runtime_error(segfss_sci_error());
}
inline void finish(){
    if(context()){
        const int status=segfss_sci_finish(context());
        const std::string error=segfss_sci_error();
        void* bridge=context();
        segfss_sci_destroy(bridge);context()=nullptr;
        if(status)throw std::runtime_error(error);
    }
}
}
