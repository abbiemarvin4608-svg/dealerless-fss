#include "native_folded.h"
#include "sci_folded_bridge.h"
#include "sci_msb_bridge.h"
#include "native_preprocessing.h"
#include "prepared_triples.h"
#include "native_workers.h"
#include "round_trace.h"
#include <algorithm>
#include <chrono>
#include <cstdlib>
#include <map>
#include <sstream>
#include <stdexcept>
#include <vector>
#pragma weak segfss_sci_folded_create
#pragma weak segfss_sci_folded_prepare
#pragma weak segfss_sci_folded_prepared_metrics
#pragma weak segfss_sci_folded_online
#pragma weak segfss_sci_folded_finish_query
#pragma weak segfss_sci_folded_counters_get
#pragma weak segfss_sci_folded_destroy
#pragma weak segfss_sci_folded_error
#pragma weak segfss_sci_set_round_trace_callbacks
#pragma weak segfss_sci_set_round_scope_callbacks
namespace sdcf_folded {
namespace {
struct Slot {int slot=0;void* bridge=nullptr;std::vector<segfss_sci_folded_call> calls;size_t cursor=0;};
std::map<const MathFunctions*,Slot> slots;
Counters retired;
uint64_t reserved_bytes=0;
bool flag(const char* name) {const char* value=std::getenv(name);return value && std::string(value)!="0";}
bool enabled() {return selected(false)||selected(true);}
int family() {
    const char* value=std::getenv("SDCF_FOLDED_LUT_FAMILY");
    const int result=value?std::stoi(value):0;
    if(result<0||result>1)throw std::invalid_argument("unsupported folded LUT key family");
    return result;
}
int signBackend() {
    const char* value=std::getenv("SDCF_FOLDED_SIGN_BACKEND");
    const int result=value?std::stoi(value):0;
    if(result<0||result>1)throw std::invalid_argument("unsupported folded sign backend");
    return result;
}
int signSegments() {
    if(!signBackend())return 0;
    const char* value=std::getenv("SDCF_FOLDED_SIGN_SEGMENTS");
    const int result=value?std::stoi(value):4;
    if(result<1||result>31)throw std::invalid_argument("unsupported folded sign segment count");
    return result;
}
uint64_t perElementReserve() {return 8192+(signBackend()?1024+uint64_t(256)*signSegments():0);}
void check(int result) {
    if(result)throw std::runtime_error(std::string("complete folded activation: ")+segfss_sci_folded_error());
}
struct NativeConsumption {uint64_t ot=0,triples=0;};
NativeConsumption nativeConsumption(int slot) {
    NativeConsumption result;
    {
        std::lock_guard<std::mutex> lock(sdcf_preprocess::mutex);
        for(const auto& item:sdcf_preprocess::inventories)
            if(item.second.context==slot)result.ot+=item.second.consumed;
    }
    {
        std::lock_guard<std::mutex> lock(sdcf_triples::mutex);
        for(const auto& item:sdcf_triples::inventories)
            if(item.second.context==slot)result.triples+=item.second.consumed;
    }
    return result;
}
bool same(const segfss_sci_folded_call& a,const segfss_sci_folded_call& b) {
    return a.query==b.query&&a.layer==b.layer&&a.slot==b.slot&&a.call==b.call&&a.kind==b.kind&&
        a.count==b.count&&a.input_bits==b.input_bits&&a.output_bits==b.output_bits&&
        a.input_scale==b.input_scale&&a.output_scale==b.output_scale&&a.lut_family==b.lut_family&&
        a.sign_backend==b.sign_backend&&a.sign_segments==b.sign_segments&&a.table_version==b.table_version;
}
void shape(sdcf_log::Line& log,const segfss_sci_folded_call& c) {
    log << "\"query\":" << c.query << ",\"layer\":" << c.layer << ",\"slot\":" << c.slot
        << ",\"call\":" << c.call << ",\"kind\":" << c.kind << ",\"count\":" << c.count
        << ",\"input_bits\":" << c.input_bits << ",\"output_bits\":" << c.output_bits
        << ",\"input_scale\":" << c.input_scale << ",\"output_scale\":" << c.output_scale
        << ",\"lut_family\":" << c.lut_family << ",\"sign_backend\":" << c.sign_backend
        << ",\"sign_segments\":" << c.sign_segments << ",\"table_version\":" << c.table_version;
}
void stageArray(sdcf_log::Line& log,const char* name,const segfss_sci_folded_stage* stages) {
    log << ",\"" << name << "\":[";
    for(int i=0;i<SEGFSS_FOLDED_STAGE_COUNT;++i) {
        if(i)log << ',';
        log << "{\"elapsed_us\":" << stages[i].elapsed_us << ",\"sent_bytes\":" << stages[i].sent_bytes
            << ",\"received_bytes\":" << stages[i].received_bytes
            << ",\"begin_us\":" << stages[i].begin_us << ",\"end_us\":" << stages[i].end_us << '}';
    }
    log << ']';
}
void metrics(sdcf_log::Line& log,const segfss_sci_folded_metrics& m) {
    log << ",\"offline_wall_us\":" << m.offline_wall_us << ",\"online_wall_us\":" << m.online_wall_us
        << ",\"resident_bytes\":" << m.resident_bytes << ",\"peak_working_bytes\":" << m.peak_working_bytes
        << ",\"online_generated\":" << m.online_generated << ",\"online_loaded_bytes\":" << m.online_loaded_bytes;
    stageArray(log,"offline",m.offline);stageArray(log,"online",m.online);
}
}
bool selected(bool tanh) {return flag(tanh?"SDCF_FOLDED_TANH":"SDCF_FOLDED_SIGMOID");}
void register_math(MathFunctions* math,int role,int slot,int port,const std::string& host) {
    if(!enabled())return;
    if(flag("SDCF_DPF_SIGMOID")||flag("SDCF_DPF_TANH"))
        throw std::invalid_argument("whole folded activation and legacy SCI DPF hooks share owned ports");
    if(!sdcf_preprocess::enabled())throw std::runtime_error("folded activations require pre-input preparation");
    if(!segfss_sci_folded_create)throw std::runtime_error("complete folded activation bridge is not linked");
    if(segfss_sci_set_round_trace_callbacks)
        segfss_sci_set_round_trace_callbacks(sdcf_round_trace_io,sdcf_round_actor_fork,
            sdcf_round_actor_enter,sdcf_round_actor_leave,sdcf_round_actor_join);
    if(segfss_sci_set_round_scope_callbacks)
        segfss_sci_set_round_scope_callbacks(sdcf_round_scope_begin,sdcf_round_scope_end);
    Slot item;item.slot=slot;
    item.bridge=segfss_sci_folded_create(role,host.c_str(),port+2000+150*slot,uint64_t(slot)+1);
    if(!item.bridge)throw std::runtime_error(segfss_sci_folded_error());
    segfss_sci_folded_counters setup{};
    check(segfss_sci_folded_counters_get(item.bridge,&setup));
    if(!slots.emplace(math,std::move(item)).second)throw std::runtime_error("folded native context repeated");
    sdcf_log::Line() << "FOLDED_PUBLIC_TABLE_PREPARATION {\"slot\":" << slot
        << ",\"table_entries\":1024,\"table_version\":1,\"elapsed_us\":" << setup.public_table_prepare_us
        << ",\"resident_bytes\":" << setup.public_table_resident_bytes
        << ",\"sent_bytes\":" << setup.public_table_prepare_sent_bytes
        << ",\"received_bytes\":" << setup.public_table_prepare_received_bytes << "}\n";
    const int lut_family=family();
    sdcf_log::Line() << "FOLDED_ACTIVATION_CONFIG {\"slot\":" << slot
        << ",\"sigmoid\":\"" << (selected(false)?"folded_sigma_f12_q10":"native_sci")
        << "\",\"tanh\":\"" << (selected(true)?"folded_sigma_f12_q10":"native_sci")
        << "\",\"lut_family\":" << lut_family
        << ",\"lut_segments\":" << (lut_family?2:1)
        << ",\"lut_segment_widths\":" << (lut_family?"[5,5]":"[10]")
        << ",\"sign_backend\":" << signBackend() << ",\"sign_segments\":" << signSegments()
        << ",\"index_bits\":10"
        << ",\"table_entries\":1024,\"table_version\":1,\"port_offset\":" << 2000+150*slot
        << ",\"selection_storage\":\"u32_ring_shares\"}\n";
}
bool activation(MathFunctions* math,bool tanh,int count,const uint64_t* input,uint64_t* output,
    int input_bits,int output_bits,int input_scale,int output_scale) {
    if(!selected(tanh))return false;
    auto& slot=slots.at(math);
    const auto task=sdcf_native::task_identity();
    segfss_sci_folded_call call{task.query,task.layer,slot.slot,
        int(sdcf_preprocess::planning_active?slot.calls.size():slot.cursor),int(tanh),count,
        input_bits,output_bits,input_scale,output_scale,family(),signBackend(),signSegments(),1};
    if(input_bits!=16||output_bits!=16||input_scale!=7||output_scale!=(tanh?7:14)||count<1)
        throw std::invalid_argument("unsupported native folded activation shape");
    if(sdcf_preprocess::planning_active) {
        slot.calls.push_back(call);std::fill(output,output+count,0);return true;
    }
    if(!sdcf_preprocess::strict||sdcf_preprocess::phase!=sdcf_preprocess::Phase::Online||
        slot.cursor>=slot.calls.size()||!same(slot.calls[slot.cursor],call))
        throw std::runtime_error("folded native online schedule mismatch or unavailable material");
    ++slot.cursor;
    segfss_sci_folded_counters before{},after{};
    segfss_sci_folded_metrics stats{};
    const auto native_before=nativeConsumption(slot.slot);
    check(segfss_sci_folded_counters_get(slot.bridge,&before));
    check(segfss_sci_folded_online(slot.bridge,&call,input,output,&stats));
    check(segfss_sci_folded_counters_get(slot.bridge,&after));
    const auto native_after=nativeConsumption(slot.slot);
    const auto native_ot_consumed=native_after.ot-native_before.ot;
    const auto bit_triples_consumed=native_after.triples-native_before.triples;
    if(after.generation_epoch!=before.generation_epoch||stats.online_generated||stats.online_loaded_bytes)
        throw std::runtime_error("folded online generated or loaded material");
    if(native_ot_consumed||bit_triples_consumed)
        throw std::runtime_error("complete folded activation consumed native SCI inventory");
    sdcf_log::Line log;log << "FOLDED_ACTIVATION_CONSUMED {";shape(log,call);metrics(log,stats);
    log << ",\"sent_bytes\":" << after.sent_bytes-before.sent_bytes
        << ",\"received_bytes\":" << after.received_bytes-before.received_bytes
        << ",\"native_ot_consumed\":" << native_ot_consumed
        << ",\"bit_triples_consumed\":" << bit_triples_consumed << "}\n";
    return true;
}
void plan_begin() {if(enabled())for(auto& item:slots){item.second.calls.clear();item.second.cursor=0;}}
std::string plan_end() {
    std::ostringstream plan;
    for(int index=0;index<4;++index)for(const auto& item:slots)if(item.second.slot==index)
        for(const auto& c:item.second.calls)
            plan << "F " << c.query << ' ' << c.layer << ' ' << c.slot << ' ' << c.call << ' '
                << c.kind << ' ' << c.count << ' ' << c.input_bits << ' ' << c.output_bits << ' '
                << c.input_scale << ' ' << c.output_scale << ' ' << c.lut_family << ' ' << c.sign_backend
                << ' ' << c.sign_segments << ' ' << c.table_version << '\n';
    return plan.str();
}
void reserve_schedule(uint64_t sigmoid_count,uint64_t tanh_count) {
    if(!enabled())return;
    const auto elements=(selected(false)?sigmoid_count:0)+(selected(true)?tanh_count:0);
    reserved_bytes=elements*perElementReserve()+(8ULL<<20);
    setenv("SEGFSS_DPF_RESERVE_BYTES",std::to_string(reserved_bytes).c_str(),1);
    sdcf_log::Line() << "FOLDED_ACTIVATION_RESERVE {\"resident_budget_bytes\":" << reserved_bytes
        << ",\"sigmoid_count\":" << sigmoid_count << ",\"tanh_count\":" << tanh_count
        << ",\"per_party_address_space_bytes\":1610612736,\"includes_allocator_headroom\":true}\n";
}
uint64_t planned_resident_bytes() {
    uint64_t bytes=0;for(const auto& item:slots)for(const auto& call:item.second.calls)
        bytes+=uint64_t(call.count)*perElementReserve();
    return bytes;
}
void prepare_query(int query) {
    if(!enabled())return;
    sdcf_preprocess::require_query(query);
    if(sdcf_preprocess::phase!=sdcf_preprocess::Phase::Preparing||!sdcf_preprocess::strict||
        sdcf_preprocess::planning_active)throw std::runtime_error("folded native prepare outside offline");
    if(reserved_bytes&&planned_resident_bytes()>reserved_bytes)
        throw std::runtime_error("folded planned materials exceed reserved budget");
    sdcf_round::SemanticScope trace("folded_prepare", query);
    const auto start=std::chrono::steady_clock::now();
    const auto before_group=counters();
    for(int index=0;index<4;++index)for(auto& item:slots) {
        auto& slot=item.second;if(slot.slot!=index)continue;
        for(const auto& call:slot.calls) {
            sdcf_log::Line log;log << "FOLDED_ACTIVATION_PLAN {";shape(log,call);log << "}\n";
        }
        segfss_sci_folded_counters before{};check(segfss_sci_folded_counters_get(slot.bridge,&before));
        check(segfss_sci_folded_prepare(slot.bridge,slot.calls.data(),int(slot.calls.size())));
        segfss_sci_folded_counters counters{};check(segfss_sci_folded_counters_get(slot.bridge,&counters));
        uint64_t attributed_sent=0,attributed_received=0;
        for(const auto& call:slot.calls) {
            segfss_sci_folded_metrics stats{};check(segfss_sci_folded_prepared_metrics(slot.bridge,call.call,&stats));
            uint64_t sent=0,received=0;
            for(const auto& stage:stats.offline){sent+=stage.sent_bytes;received+=stage.received_bytes;}
            attributed_sent+=sent;attributed_received+=received;
            sdcf_log::Line log;log << "FOLDED_ACTIVATION_PREPARED {";shape(log,call);metrics(log,stats);
            log << ",\"calls\":" << slot.calls.size() << ",\"requests\":" << counters.generated_requests
                << ",\"generation_epoch\":" << counters.generation_epoch << ",\"pre_input\":true"
                << ",\"sent_bytes\":" << sent << ",\"received_bytes\":" << received << "}\n";
        }
        if(attributed_sent!=counters.sent_bytes-before.sent_bytes ||
            attributed_received!=counters.received_bytes-before.received_bytes)
            throw std::runtime_error("folded offline stage traffic does not cover complete preparation");
    }
    if(reserved_bytes&&counters().resident>reserved_bytes)
        throw std::runtime_error("folded resident materials exceed reserved budget");
    const auto after_group=counters();
    sdcf_log::Line() << "FOLDED_PREPARE_GROUP {\"query\":" << query
        << ",\"wall_seconds\":" << std::chrono::duration<double>(std::chrono::steady_clock::now()-start).count()
        << ",\"sent_bytes\":" << after_group.sent-before_group.sent
        << ",\"received_bytes\":" << after_group.received-before_group.received
        << ",\"schedule\":\"serial_slot_major\",\"kind_timers\":\"work_attribution\"}\n";
}
Counters counters() {
    Counters result=retired;
    for(const auto& item:slots) {
        segfss_sci_folded_counters stats{};check(segfss_sci_folded_counters_get(item.second.bridge,&stats));
        result.sent+=stats.sent_bytes;result.received+=stats.received_bytes;result.epoch+=stats.generation_epoch;
        result.resident+=stats.resident_bytes;result.generated+=stats.generated_requests;result.consumed+=stats.consumed_requests;
        result.generated_elements+=stats.generated_elements;result.consumed_elements+=stats.consumed_elements;
        result.online_generated+=stats.online_generated;result.online_loaded_bytes+=stats.online_loaded_bytes;
    }
    return result;
}
void balance(int query) {
    if(!enabled())return;
    sdcf_preprocess::require_query(query);
    for(auto& item:slots) {
        if(item.second.cursor!=item.second.calls.size())throw std::runtime_error("folded native schedule unfinished");
        check(segfss_sci_folded_finish_query(item.second.bridge));
    }
    const auto stats=counters();
    sdcf_log::Line() << "FOLDED_ACTIVATION_BALANCE {\"query\":" << query << ",\"generated\":" << stats.generated
        << ",\"consumed\":" << stats.consumed << ",\"remaining\":" << stats.generated-stats.consumed
        << ",\"generated_elements\":" << stats.generated_elements << ",\"consumed_elements\":" << stats.consumed_elements
        << ",\"online_generation_epoch_delta\":0,\"online_generated\":" << stats.online_generated
        << ",\"online_loaded_bytes\":" << stats.online_loaded_bytes << "}\n";
}
void finish() {
    retired=counters();for(auto& item:slots)segfss_sci_folded_destroy(item.second.bridge);slots.clear();
}
}
