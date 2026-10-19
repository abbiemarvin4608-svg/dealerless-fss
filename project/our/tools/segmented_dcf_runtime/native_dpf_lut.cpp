#include "native_dpf_lut.h"
#include "sci_dpf_bridge.h"
#include "sci_msb_bridge.h"
#include "native_preprocessing.h"
#include "native_workers.h"
#include "round_trace.h"
#include <algorithm>
#include <array>
#include <cstdlib>
#include <map>
#include <mutex>
#include <sstream>
#include <stdexcept>
#include <vector>
#pragma weak segfss_sci_dpf_create
#pragma weak segfss_sci_dpf_material_metadata_bytes
#pragma weak segfss_sci_dpf_prepare
#pragma weak segfss_sci_dpf_online
#pragma weak segfss_sci_dpf_finish_query
#pragma weak segfss_sci_dpf_counters_get
#pragma weak segfss_sci_dpf_destroy
#pragma weak segfss_sci_dpf_error
#pragma weak segfss_sci_set_round_trace_callbacks
namespace sdcf_dpf {
namespace {
thread_local int activation=-1;
struct Call {
    segfss_sci_dpf_call shape{};
    std::array<std::vector<uint64_t>,2> tables;
    void bind() {for(int t=0;t<shape.tables;++t) shape.values[t]=tables[t].data();}
};
struct Slot {
    int slot=0;
    void* bridge=nullptr;
    std::vector<Call> calls;
    size_t cursor=0;
};
std::map<const MathFunctions*,Slot> slots;
std::mutex mutex;
Counters retired;
uint64_t reserved_bytes=0;
bool flag(const char* name) {const char* value=std::getenv(name);return value && std::string(value)!="0";}
bool selected() {return activation>=0 && flag(activation?"SDCF_DPF_TANH":"SDCF_DPF_SIGMOID");}
bool enabled() {return flag("SDCF_DPF_SIGMOID") || flag("SDCF_DPF_TANH");}
uint64_t material_bytes(int index_bits,int output_bits) {
    if(!segfss_sci_dpf_material_metadata_bytes)
        throw std::runtime_error("DPF native material layout bridge is not linked");
    return (uint64_t(1)<<index_bits)*((output_bits+7)/8)+
        segfss_sci_dpf_material_metadata_bytes();
}
void check(int result) {if(result)throw std::runtime_error(std::string("SCI DPF: ")+segfss_sci_dpf_error());}
bool same(const Call& a,const Call& b) {
    const auto& x=a.shape;const auto& y=b.shape;
    return x.query==y.query && x.layer==y.layer && x.slot==y.slot && x.call==y.call &&
        x.family==y.family && x.count==y.count && x.index_bits==y.index_bits && x.tables==y.tables &&
        x.grouped_tables==y.grouped_tables && x.output_bits[0]==y.output_bits[0] &&
        x.output_bits[1]==y.output_bits[1];
}
Call descriptor(Slot& slot,int family,int count,int bits,int tables,bool grouped) {
    Call call;
    call.shape.query=sdcf_preprocess::current_query;
    call.shape.layer=sdcf_native::task_identity().layer;
    call.shape.slot=slot.slot;
    call.shape.call=int(sdcf_preprocess::planning_active?slot.calls.size():slot.cursor);
    call.shape.family=family+2*activation;
    call.shape.count=count;call.shape.index_bits=bits;
    call.shape.tables=tables;call.shape.grouped_tables=grouped;
    return call;
}
void run(Slot& slot,Call call,uint64_t* input,uint64_t* output0,uint64_t* output1=nullptr) {
    call.bind();
    if(sdcf_preprocess::planning_active) {
        slot.calls.push_back(std::move(call));
        std::fill(output0,output0+slot.calls.back().shape.count,0);
        if(output1)std::fill(output1,output1+slot.calls.back().shape.count,0);
        return;
    }
    if(!sdcf_preprocess::strict || sdcf_preprocess::phase!=sdcf_preprocess::Phase::Online ||
       slot.cursor>=slot.calls.size() || !same(slot.calls[slot.cursor],call))
        throw std::runtime_error("SCI DPF online schedule mismatch or unavailable material");
    ++slot.cursor;
    auto shape=call.shape;
    for(int t=0;t<shape.tables;++t)shape.values[t]=slot.calls[slot.cursor-1].tables[t].data();
    segfss_sci_dpf_counters before{},after{};
    check(segfss_sci_dpf_counters_get(slot.bridge,&before));
    check(segfss_sci_dpf_online(slot.bridge,&shape,input,output0,output1));
    check(segfss_sci_dpf_counters_get(slot.bridge,&after));
    sdcf_log::Line() << "DPF_LUT_CONSUMED {\"query\":" << call.shape.query << ",\"layer\":"
        << call.shape.layer << ",\"slot\":" << slot.slot << ",\"call\":" << call.shape.call
        << ",\"family\":" << call.shape.family << ",\"count\":" << call.shape.count
        << ",\"index_bits\":" << call.shape.index_bits
        << ",\"tables\":" << call.shape.tables << ",\"output_bits\":[" << call.shape.output_bits[0]
        << ',' << call.shape.output_bits[1] << "],\"grouped_tables\":" << call.shape.grouped_tables
        << ",\"online_open_us\":" << after.online_open_us-before.online_open_us
        << ",\"inner_product_us\":" << after.inner_product_us-before.inner_product_us
        << ",\"sent_bytes\":" << after.sent_bytes-before.sent_bytes
        << ",\"received_bytes\":" << after.received_bytes-before.received_bytes
        << ",\"online_full_domain_eval\":false,\"online_loaded_bytes\":0,\"online_generated\":0}\n";
}
}
int activation_enter(bool tanh){const int previous=activation;activation=tanh?1:0;return previous;}
void activation_leave(int previous){activation=previous;}
void register_math(MathFunctions* math,int role,int slot,int port,const std::string& host) {
    if(!enabled())return;
    if(!sdcf_preprocess::enabled())throw std::runtime_error("DPF native activations require strict pre-input preparation");
    if(!segfss_sci_dpf_create)throw std::runtime_error("DPF native activation bridge is not linked");
    if(segfss_sci_set_round_trace_callbacks)
        segfss_sci_set_round_trace_callbacks(sdcf_round_trace_io,sdcf_round_actor_fork,
            sdcf_round_actor_enter,sdcf_round_actor_leave,sdcf_round_actor_join);
    Slot item;item.slot=slot;item.bridge=segfss_sci_dpf_create(role,host.c_str(),port+2000+150*slot);
    if(!item.bridge)throw std::runtime_error(segfss_sci_dpf_error());
    if(!slots.emplace(math,std::move(item)).second)throw std::runtime_error("DPF native context repeated");
    sdcf_log::Line() << "DPF_ACTIVATION_CONFIG {\"slot\":" << slot
        << ",\"sigmoid\":\"" << (flag("SDCF_DPF_SIGMOID")?"public_dpf_lut":"native_sci")
        << "\",\"tanh\":\"" << (flag("SDCF_DPF_TANH")?"public_dpf_lut":"native_sci")
        << "\",\"key_family\":\"dealerless_arithmetic_cGGM_DPF\",\"port_offset\":"
        << 2000+150*slot << ",\"selection_storage\":\"packed_ring_shares\"}\n";
}
void plan_begin() {
    if(!enabled())return;
    for(auto& item:slots){item.second.calls.clear();item.second.cursor=0;}
}
std::string plan_end() {
    std::ostringstream plan;
    for(int index=0;index<4;++index)for(auto& item:slots) {
        auto& slot=item.second;if(slot.slot!=index)continue;
        for(auto& call:slot.calls) {
            call.bind();const auto& c=call.shape;
            plan << "D " << c.query << ' ' << c.layer << ' ' << c.slot << ' ' << c.call << ' '
                << c.family << ' ' << c.count << ' ' << c.index_bits << ' ' << c.tables << ' '
                << c.output_bits[0] << ' ' << c.output_bits[1] << ' ' << c.grouped_tables;
            for(int table=0;table<c.tables;++table)for(auto value:call.tables[table])plan << ' ' << value;
            plan << '\n';
        }
    }
    return plan.str();
}
void reserve_schedule(uint64_t sigmoid_count,uint64_t tanh_count) {
    if(!enabled())return;
    // Packed DPF selection rings are at most two bytes for every requested
    // leaf; reserve metadata, allocator headroom and public tables as well.
    reserved_bytes=(flag("SDCF_DPF_SIGMOID")?sigmoid_count*(2*material_bytes(8,16)+material_bytes(6,15)):0)+
        (flag("SDCF_DPF_TANH")?tanh_count*(2*material_bytes(8,9)+material_bytes(3,9)):0)+(1ULL<<20);
    const std::string reserve=std::to_string(reserved_bytes);
    setenv("SEGFSS_DPF_RESERVE_BYTES",reserve.c_str(),1);
    sdcf_log::Line() << "DPF_LUT_RESERVE {\"resident_budget_bytes\":" << reserved_bytes
        << ",\"sigmoid_count\":" << sigmoid_count << ",\"tanh_count\":" << tanh_count
        << ",\"per_party_address_space_bytes\":1610612736,\"includes_allocator_headroom\":true}\n";
}
void prepare_query(int query) {
    if(!enabled())return;
    sdcf_preprocess::require_query(query);
    if(sdcf_preprocess::phase!=sdcf_preprocess::Phase::Preparing || !sdcf_preprocess::strict ||
       sdcf_preprocess::planning_active)throw std::runtime_error("DPF native prepare outside offline");
    if(reserved_bytes && planned_resident_bytes()>reserved_bytes)
        throw std::runtime_error("DPF planned resident materials exceed reserved budget");
    for(int index=0;index<4;++index)for(auto& item:slots) {
        auto& slot=item.second;if(slot.slot!=index)continue;
        std::vector<segfss_sci_dpf_call> calls;
        for(auto& call:slot.calls) {
            call.bind();calls.push_back(call.shape);const auto& c=call.shape;
            sdcf_log::Line() << "DPF_LUT_PLAN {\"query\":" << c.query << ",\"layer\":" << c.layer
                << ",\"slot\":" << c.slot << ",\"call\":" << c.call << ",\"family\":" << c.family
                << ",\"count\":" << c.count << ",\"index_bits\":" << c.index_bits
                << ",\"tables\":" << c.tables << ",\"output_bits\":[" << c.output_bits[0]
                << ',' << c.output_bits[1] << "],\"grouped_tables\":" << c.grouped_tables << "}\n";
        }
        check(segfss_sci_dpf_prepare(slot.bridge,calls.data(),int(calls.size())));
        segfss_sci_dpf_counters stats{};check(segfss_sci_dpf_counters_get(slot.bridge,&stats));
        sdcf_log::Line() << "DPF_LUT_PREPARED {\"query\":" << query << ",\"slot\":" << slot.slot
            << ",\"calls\":" << calls.size() << ",\"requests\":" << stats.generated_requests
            << ",\"generation_epoch\":" << stats.generation_epoch << ",\"resident_bytes\":"
            << stats.resident_bytes << ",\"mask_conversion_us\":" << stats.mask_conversion_us
            << ",\"keygen_us\":" << stats.keygen_us << ",\"eval_all_us\":" << stats.eval_all_us
            << ",\"offline_us\":" << stats.offline_us << ",\"eval_points\":" << stats.eval_points
            << ",\"pre_input_eval_all\":true,\"no_online_reload\":true}\n";
    }
    if(reserved_bytes && counters().resident>reserved_bytes)
        throw std::runtime_error("DPF resident materials exceed reserved budget");
}
uint64_t planned_resident_bytes() {
    uint64_t bytes=0;
    for(const auto& item:slots)for(const auto& call:item.second.calls) {
        const auto& c=call.shape;
        // Packed selection rings plus material identity; the process-wide
        // address-space cap also includes scratch, thread stacks and tables.
        bytes+=uint64_t(c.count)*material_bytes(c.index_bits,std::max(c.output_bits[0],c.output_bits[1]));
        for(int t=0;t<c.tables;++t)bytes+=call.tables[t].capacity()*16;
    }
    return bytes;
}
Counters counters() {
    Counters result=retired;
    for(const auto& item:slots) {
        segfss_sci_dpf_counters stats{};check(segfss_sci_dpf_counters_get(item.second.bridge,&stats));
        result.sent+=stats.sent_bytes;result.received+=stats.received_bytes;result.epoch+=stats.generation_epoch;
        result.resident+=stats.resident_bytes;result.generated+=stats.generated_requests;result.consumed+=stats.consumed_requests;
    }
    return result;
}
void balance(int query) {
    if(!enabled())return;
    sdcf_preprocess::require_query(query);
    for(auto& item:slots) {
        if(item.second.cursor!=item.second.calls.size())throw std::runtime_error("DPF native schedule unfinished");
        check(segfss_sci_dpf_finish_query(item.second.bridge));
    }
    const auto stats=counters();
    sdcf_log::Line() << "DPF_LUT_BALANCE {\"query\":" << query << ",\"generated\":" << stats.generated
        << ",\"consumed\":" << stats.consumed << ",\"remaining\":" << stats.generated-stats.consumed
        << ",\"online_generation_epoch_delta\":0}\n";
}
void finish(){retired=counters();for(auto& item:slots)segfss_sci_dpf_destroy(item.second.bridge);slots.clear();}
bool lookup_exp(MathFunctions* math,int dim,uint64_t* inputs,uint64_t* output,
        int digits,int digit_bits,int last_bits,int input_scale,int output_scale,
        uint64_t (*lookup)(uint64_t,int32_t,int32_t)) {
    if(!selected())return false;
    auto& slot=slots.at(math);
    if(digits!=2 || digit_bits!=last_bits || digit_bits!=8 ||
       input_scale!=(activation?6:7) || output_scale!=(activation?7:14))
        throw std::runtime_error("unsupported native DPF exponent public shape");
    auto call=descriptor(slot,0,dim*digits,digit_bits,digits,true);
    for(int digit=0;digit<digits;++digit) {
        call.shape.output_bits[digit]=output_scale+2;
        if(sdcf_preprocess::planning_active) for(int i=0;i<(1<<digit_bits);++i)
            call.tables[digit].push_back(lookup(i,input_scale-digit_bits*digit,output_scale));
    }
    run(slot,std::move(call),inputs,output);return true;
}
bool lookup_reciprocal(MathFunctions* math,int dim,uint64_t* input,uint64_t* output0,
        uint64_t* output1,int bits,uint64_t (*lookup0)(uint64_t,int),uint64_t (*lookup1)(uint64_t,int)) {
    if(!selected())return false;
    auto& slot=slots.at(math);
    if(bits!=(activation?3:6))throw std::runtime_error("unsupported native DPF reciprocal shape");
    auto call=descriptor(slot,1,dim,bits,2,false);
    call.shape.output_bits[0]=bits+4;call.shape.output_bits[1]=2*bits+3;
    if(sdcf_preprocess::planning_active) for(int i=0;i<(1<<bits);++i){call.tables[0].push_back(lookup0(i,bits));call.tables[1].push_back(lookup1(i,bits));}
    run(slot,std::move(call),input,output0,output1);return true;
}
}
