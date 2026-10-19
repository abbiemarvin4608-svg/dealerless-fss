#include "adapter.h"
#include "beaver.h"
#include "globals.h"
#include "../segmented_dcf_runtime/native_workers.h"
#include "../segmented_dcf_runtime/native_preprocessing.h"
#include "Math/math-functions.h"
#include "../../src/sci_numeric_oracle.h"
#ifdef SDCF_CONVERSION_RUNTIME
#include "../sci_conversion/hook.h"
#endif
#include <algorithm>
#include <cstdlib>
#include <iostream>
#include <iomanip>
#include <map>
#include <stdexcept>
#include <sstream>
#include <chrono>
#include <exception>
#include <thread>

extern "C" uint64_t sdcf_round_actor_fork() __attribute__((weak));
extern "C" uint64_t sdcf_round_actor_enter(uint64_t) __attribute__((weak));
extern "C" uint64_t sdcf_round_actor_leave(uint64_t) __attribute__((weak));
extern "C" void sdcf_round_actor_join(uint64_t) __attribute__((weak));

namespace {
struct Context {
    std::string backend;
    int role=0,query=-1;
    bool prepared=false,online=false,poisoned=false;
    int tile_rows=32;
    bool overlap=false;
    std::ostringstream offline_log;
    uint64_t epoch=0;
    const sdcf_runtime::Layer* layer=nullptr;
    sdcf_beaver::MatmulOptions matmul_options;
    sdcf_beaver::MatmulWorkspace matmul_workspace;
    std::vector<uint64_t> matmul_output;
    std::map<int,sdcf_beaver::MatmulMaterial> matrices;
    std::map<int,sdcf_beaver::MulCirMaterial> mulcirs;
};
Context contexts[2];
Context& current_context() { return contexts[sdcf_native::selected_branch()]; }
#define context current_context()
int context_count() { return sdcf_native::isolated_graph() ? 2 : 1; }
std::vector<sdcf_runtime::Layer> branch_schedule(const std::vector<sdcf_runtime::Layer>& schedule, int branch) {
    std::vector<sdcf_runtime::Layer> result;
    for(const auto& layer:schedule) if(sdcf_native::graph_branch(layer.index)==branch) result.push_back(layer);
    return result;
}
bool configured_flag(const char* name) {
    const char* value=std::getenv(name);
    if(!value || std::string(value)=="0") return false;
    if(std::string(value)=="1") return true;
    throw std::invalid_argument(std::string(name)+" must be 0 or 1");
}
int configured_tile() {
    const char* text=std::getenv("SDCF_BEAVER_TILE_ROWS");
    const int tile=text?std::stoi(text):32;
    if(tile!=1 && tile!=8 && tile!=32 && tile!=128)
        throw std::invalid_argument("Beaver tile rows must be 1, 8, 32 or 128");
    return tile;
}
bool use_beaver();
uint64_t offline_reservation(const std::vector<sdcf_runtime::Layer>& schedule) {
    if(!use_beaver()) return 0;
    uint64_t resident=0,working=0;
    for(const auto& layer:schedule) {
        const auto& a=layer.numeric_arguments;
        if(layer.function=="MatMul") {
            if(a.size()!=12) throw std::invalid_argument("MatMul schedule contract changed");
            const auto estimate=sdcf_beaver::estimate_matmul_generation(a[0],a[1],a[2],
                a[8]+a[9]+sdcf_beaver::accumulation_bits(a[1]),context.tile_rows);
            resident+=estimate.resident_bytes;
            working=std::max(working,estimate.working_bytes);
        } else if(layer.function=="MulCir") {
            if(a.size()!=9) throw std::invalid_argument("MulCir schedule contract changed");
            const uint64_t values=uint64_t(a[0])*a[1];
            resident+=3*values*sizeof(uint64_t);
            working=std::max(working,values*uint64_t(a[5]+a[6])*96+(uint64_t(2)<<20));
        }
    }
    return resident+working;
}
std::string seconds_or_null(bool instrumented,double seconds) {
    if(!instrumented) return "null";
    std::ostringstream value;value<<std::setprecision(17)<<seconds;return value.str();
}
bool use_beaver() { return context.backend.compare(0,10,"SCI_Beaver")==0; }
const char* conversion_backend() {
    if(context.backend=="SCI_Beaver_full") return "full_tree";
    if(context.backend=="SCI_ours" || context.backend=="SCI_Beaver_segmented" || context.backend=="SCI_Beaver_segmented_DPF_LUT" || context.backend=="SCI_Beaver_segmented_PaperLUT") return "segmented_dcf";
    return "native_SCI";
}
}

namespace sdcf_backend {
std::vector<std::string> supported_backends(){
#ifdef SDCF_CONVERSION_RUNTIME
    return {"SCI","SCI_preprocessed","SCI_ours","SCI_Beaver","SCI_Beaver_full","SCI_Beaver_segmented","SCI_Beaver_segmented_DPF_LUT","SCI_Beaver_segmented_PaperLUT"};
#else
    return {"SCI","SCI_preprocessed","SCI_Beaver"};
#endif
}
void validate(const std::string& name) {
    const auto names=supported_backends();
    if(std::find(names.begin(),names.end(),name)==names.end())
        throw std::invalid_argument("unavailable backend: "+name);
}
void setup_one(const std::string& name,int role,const std::vector<sdcf_runtime::Layer>& schedule) {
    validate(name);
    if((role!=1 && role!=2) || context.online || context.prepared)
        throw std::logic_error("invalid Beaver setup state");
    context.backend=name;context.role=role;
    context.tile_rows=configured_tile();context.overlap=configured_flag("SDCF_OFFLINE_OVERLAP");
    if(context.overlap && name!="SCI_Beaver_full" && name!="SCI_Beaver_segmented" && name!="SCI_Beaver_segmented_DPF_LUT" && name!="SCI_Beaver_segmented_PaperLUT")
        throw std::invalid_argument("offline overlap requires Beaver and FSS conversion");
    if(context.overlap) {
        const char* workers=std::getenv("SEGFSS_CGGM_THREADS");
        const char* budget=std::getenv("SEGFSS_GENERATION_BUDGET_BYTES");
        if(!workers || std::stoi(workers)<1 || std::stoi(workers)>2 || !budget || !std::stoull(budget))
            throw std::invalid_argument("offline overlap requires cGGM limit 1 or 2 and explicit generation budget");
    }
    context.matmul_options={configured_flag("SDCF_BEAVER_ALGEBRA"),
                            configured_flag("SDCF_BEAVER_MAPPED_LAYOUT")};
}
void prepare_one(int query,const std::vector<sdcf_runtime::Layer>& schedule) {
    if(context.online || context.prepared || context.poisoned) throw std::logic_error("unconsumed or poisoned query preparation");
    context.offline_log.str("");context.offline_log.clear();
    context.matrices.clear();context.mulcirs.clear();context.query=query;
    if(use_beaver()) {
        for(const auto& layer:schedule) if(layer.function=="MatMul") {
            const auto& a=layer.numeric_arguments;
            if(a.size()!=12) throw std::invalid_argument("MatMul schedule contract changed");
            const int bits=a[8]+a[9]+sdcf_beaver::accumulation_bits(a[1]);
            ++context.epoch;
            auto m=sdcf_beaver::prepare_matmul(context.role,*sdcf_native::selected_mult(),a[0],a[1],a[2],bits,context.tile_rows,context.matmul_options);
            context.offline_log<<"BEAVER_OFFLINE {\"query\":"<<query<<",\"index\":"<<layer.index
                <<",\"operator\":\""<<layer.operator_name<<"\",\"rows\":"<<m.rows
                <<",\"inner\":"<<m.inner<<",\"cols\":"<<m.cols<<",\"ring_bits\":"<<bits
                <<",\"tiles\":"<<m.offline_tiles<<",\"material_elements\":"<<m.u.size()+m.v.size()+m.w.size()
                <<",\"tile_rows\":"<<m.offline_tile_rows<<",\"tail_rows\":"<<m.offline_tail_rows
                <<",\"working_bytes\":"<<m.offline_working_bytes<<",\"resident_bytes\":"<<m.offline_resident_bytes
                <<",\"sent_bytes\":"<<m.offline_sent_bytes<<",\"received_bytes\":"
                <<(m.offline_receive_instrumented?std::to_string(m.offline_received_bytes):"null")
                <<",\"receive_instrumented\":"<<(m.offline_receive_instrumented?"true":"false")
                <<",\"io_rounds\":"<<m.offline_io_rounds
                <<",\"cross_term_batches\":"<<2*m.offline_tiles
                <<",\"algebra\":"<<(m.options.algebra?"true":"false")
                <<",\"mapped_layout\":"<<(m.options.mapped_layout?"true":"false")
                <<",\"local_kernel_seconds\":"<<m.offline_local.local_kernel_seconds
                <<",\"buffer_seconds\":"<<m.offline_local.buffer_seconds
                <<",\"local_products\":"<<m.offline_local.local_products
                <<",\"kernel_layout_copy_seconds\":"<<seconds_or_null(m.offline_local.copy_instrumented,m.offline_local.layout_copy_seconds)
                <<",\"kernel_compute_seconds\":"<<seconds_or_null(m.offline_local.copy_instrumented,m.offline_local.compute_seconds)
                <<",\"local_kernel_includes_layout_copy\":"<<(m.options.mapped_layout?"false":"true")
                <<",\"material_kind\":\"multiplication_matrix_triple\",\"triples\":1,\"activation_triples\":0"
                <<",\"fresh\":true,\"dealerless\":true}\n";
            if(!context.matrices.emplace(layer.index,std::move(m)).second)
                throw std::invalid_argument("duplicate scheduled matrix");
        }
        for(const auto& layer:schedule) if(layer.function=="MulCir") {
            const auto& a=layer.numeric_arguments;
            if(a.size()!=9) throw std::invalid_argument("MulCir schedule contract changed");
            const int elements = a[0] * a[1];
            const int bits = a[5] + a[6];
            ++context.epoch;
            auto m=sdcf_beaver::prepare_mulcir(context.role,*sdcf_native::selected_mult(),elements,bits);
            context.offline_log<<"BEAVER_OFFLINE {\"query\":"<<query<<",\"index\":"<<layer.index
                <<",\"operator\":\""<<layer.operator_name<<"\",\"function\":\"MulCir\",\"elements\":"<<m.elements
                <<",\"ring_bits\":"<<m.bits<<",\"material_elements\":"<<m.u.size()+m.v.size()+m.w.size()
                <<",\"material_kind\":\"multiplication_elementwise_triples\",\"triples\":"<<m.elements<<",\"activation_triples\":0"
                <<",\"fresh\":true,\"dealerless\":true}\n";
            if(!context.mulcirs.emplace(layer.index,std::move(m)).second)
                throw std::invalid_argument("duplicate scheduled MulCir");
        }
    }
    context.prepared=true;
}
void enter_online_one(int query) {
    if(!context.prepared || context.online || context.query!=query)
        throw std::logic_error("online query has no matching preprocessing");
    context.online=true;
}
void layer_begin(const sdcf_runtime::Layer& layer) {
    if((!context.online && !sdcf_preprocess::planning_active.load()) || context.layer) throw std::logic_error("invalid layer entry");
    context.layer=&layer;
#ifdef SDCF_CONVERSION_RUNTIME
    sdcf_conversion::layer_begin(layer);
#endif
}
void layer_end(const sdcf_runtime::Layer& layer) {
    if(context.layer!=&layer) throw std::logic_error("mismatched current layer");
    const bool planning = sdcf_preprocess::planning_active.load();
    if(!planning && use_beaver() && layer.function=="MatMul" && !context.matrices.at(layer.index).consumed)
        throw std::logic_error("scheduled matrix Beaver hook was not called");
    if(!planning && use_beaver() && layer.function=="MulCir" && !context.mulcirs.at(layer.index).consumed)
        throw std::logic_error("scheduled MulCir Beaver hook was not called");
    if(use_beaver() && layer.function=="ScalarMul")
        sdcf_native::LogLine()<<"PUBLIC_SCALARMUL {\"query\":"<<context.query<<",\"index\":"<<layer.index
            <<",\"operator\":\""<<layer.operator_name<<"\",\"elements\":"<<layer.count
            <<",\"operand_visibility\":\"public_model_scalar\",\"multiplication\":\"local\",\"multiplication_triples\":0"
            <<",\"conversion\":\""<<conversion_backend()<<"\"}\n";
#ifdef SDCF_CONVERSION_RUNTIME
    sdcf_conversion::layer_end(layer);
#endif
    context.layer=nullptr;
}
void leave_online_one(int query) {
    if(!context.online || context.layer || context.query!=query) throw std::logic_error("invalid online exit");
    for(const auto& entry:context.matrices) if(!entry.second.consumed)
        throw std::logic_error("query left matrix material unconsumed");
    for(const auto& entry:context.mulcirs) if(!entry.second.consumed)
        throw std::logic_error("query left MulCir material unconsumed");
    context.online=false;context.prepared=false;
}
void finish_one() {
    if(context.online || context.prepared) throw std::logic_error("unfinished Beaver query");
    context.matrices.clear();context.mulcirs.clear();context.backend.clear();
    context.matmul_workspace={};context.matmul_output.clear();context.matmul_output.shrink_to_fit();
}
void setup(const std::string& name,int role,const std::vector<sdcf_runtime::Layer>& schedule) {
    for(int branch=0;branch<context_count();++branch) {
        sdcf_native::TaskScope scope(-1,-1,branch*2); setup_one(name,role,branch_schedule(schedule,branch));
    }
#ifdef SDCF_CONVERSION_RUNTIME
    sdcf_conversion::setup(name,role,schedule);
#endif
}
void prepare(int query,const std::vector<sdcf_runtime::Layer>& schedule) {
    using Clock=std::chrono::steady_clock;
    const bool overlap=contexts[0].overlap;
    const uint64_t reservation=offline_reservation(schedule);
#ifdef SDCF_CONVERSION_RUNTIME
    sdcf_conversion::offline_reservation(reservation);
#endif
    const auto wall_start=Clock::now();
    double beaver_seconds=0,fss_seconds=0;
    int observed_ot=1;
    std::exception_ptr beaver_error,fss_error;
    auto beaver=[&] {
        const auto start=Clock::now();
        try {
            for(int branch=0;branch<context_count();++branch) {
                sdcf_native::TaskScope scope(query,-1,branch*2);
                prepare_one(query,branch_schedule(schedule,branch));
                observed_ot=std::max(observed_ot,scope.observed_ot_participants());
            }
        } catch(...) { beaver_error=std::current_exception(); }
        beaver_seconds=std::chrono::duration<double>(Clock::now()-start).count();
    };
    auto fss=[&] {
        const auto start=Clock::now();
        try {
#ifdef SDCF_CONVERSION_RUNTIME
            sdcf_conversion::prepare(query,schedule);
#endif
        } catch(...) { fss_error=std::current_exception(); }
        fss_seconds=std::chrono::duration<double>(Clock::now()-start).count();
    };
    if(overlap) {
        const auto fork=sdcf_round_actor_fork?sdcf_round_actor_fork():0;
        uint64_t child=0;
        std::thread task([&] {
            const auto previous=sdcf_round_actor_enter?sdcf_round_actor_enter(fork):0;
            fss();
            if(sdcf_round_actor_leave) child=sdcf_round_actor_leave(previous);
        });
        beaver();task.join();
        if(sdcf_round_actor_join) sdcf_round_actor_join(child);
    }
    else { beaver(); if(!beaver_error) fss(); }
    const double wall_seconds=std::chrono::duration<double>(Clock::now()-wall_start).count();
    const char* cggm_limit=std::getenv("SEGFSS_CGGM_THREADS");
    const int per_task_limit=overlap?2:std::max(2,cggm_limit?std::stoi(cggm_limit):1);
    for(int branch=0;branch<context_count();++branch) sdcf_native::log(contexts[branch].offline_log.str());
    sdcf_native::LogLine()<<"OFFLINE_PREPARATION {\"query\":"<<query
        <<",\"overlap\":"<<(overlap?"true":"false")<<",\"beaver_work_seconds\":"<<beaver_seconds
        <<",\"fss_work_seconds\":"<<fss_seconds<<",\"wall_seconds\":"<<wall_seconds
        <<",\"joint_beaver_reservation_bytes\":"<<reservation<<",\"beaver_ot_participants\":"<<observed_ot
        <<",\"task_limit\":"<<(overlap?2:1)<<",\"per_task_compute_limit\":"<<per_task_limit
        <<",\"beaver_requested_ot_participants\":2,\"ready\":"<<(!beaver_error && !fss_error?"true":"false")<<"}\n";
    if(beaver_error || fss_error) {
        for(auto& item:contexts) {
            item.matrices.clear();item.mulcirs.clear();item.prepared=false;item.poisoned=true;
        }
#ifdef SDCF_CONVERSION_RUNTIME
        sdcf_conversion::abort_preparation();
#endif
        std::rethrow_exception(beaver_error?beaver_error:fss_error);
    }
}
void enter_online(int query) {
    for(int branch=0;branch<context_count();++branch) {
        sdcf_native::TaskScope scope(query,-1,branch*2); enter_online_one(query);
    }
#ifdef SDCF_CONVERSION_RUNTIME
    sdcf_conversion::enter_online(query);
#endif
}
void leave_online(int query) {
    for(int branch=0;branch<context_count();++branch) {
        sdcf_native::TaskScope scope(query,-1,branch*2); leave_online_one(query);
    }
#ifdef SDCF_CONVERSION_RUNTIME
    sdcf_conversion::leave_online(query);
#endif
}
void finish() {
    for(int branch=0;branch<context_count();++branch) {
        sdcf_native::TaskScope scope(-1,-1,branch*2); finish_one();
    }
#ifdef SDCF_CONVERSION_RUNTIME
    sdcf_conversion::finish();
#endif
}
sdcf_runtime::Counters counters(){
    sdcf_runtime::Counters result;
#ifdef SDCF_CONVERSION_RUNTIME
    result=sdcf_conversion::counters();
#endif
    for(int branch=0;branch<context_count();++branch) result.offline_generation_epoch+=contexts[branch].epoch;
    return result;
}
}

namespace sdcf_beaver {
bool matmul(int tid,uint64_t* lhs,uint64_t* rhs,uint64_t* out,
            int rows,int inner,int cols,int lhs_bits,int rhs_bits) {
    if(!use_beaver()) return false;
    const bool planning=sdcf_preprocess::planning_active.load();
    if((!context.online && !planning) || !context.layer ||
       context.layer->function!="MatMul" || tid!=0)
        throw std::logic_error("matrix Beaver hook called outside its scheduled layer");
    const int depth=accumulation_bits(inner),product_bits=lhs_bits+rhs_bits;
    const int ring_bits=product_bits+depth;
    const auto& a=context.layer->numeric_arguments;
    if(a.size()!=12 || a[0]!=rows || a[1]!=inner || a[2]!=cols || a[8]!=lhs_bits || a[9]!=rhs_bits)
        throw std::invalid_argument("matrix Beaver public contract mismatch");
    auto* material=planning?nullptr:&context.matrices.at(context.layer->index);
    if(material && (material->rows!=rows || material->inner!=inner || material->cols!=cols ||
                    material->bits!=ring_bits || material->consumed))
        throw std::invalid_argument("matrix Beaver material contract mismatch");
    std::vector<uint64_t> narrow(size_t(rows)*inner),wide(narrow.size()),weights(size_t(inner)*cols);
    for(size_t i=0;i<narrow.size();++i) narrow[i]=lhs[i]&segfss::reference::mask(lhs_bits);
    for(size_t i=0;i<weights.size();++i) {
        if(context.role==2 && (rhs[i]&segfss::reference::mask(rhs_bits)))
            throw std::invalid_argument("native MatMul requires server-private weights");
        weights[i]=context.role==1?uint64_t(segfss::reference::signedValue(rhs[i],rhs_bits))&segfss::reference::mask(ring_bits):0;
    }
    try {
#ifdef SDCF_CONVERSION_RUNTIME
        if(!sdcf_conversion::convert(context.role,narrow.data(),wide.data(),narrow.size(),lhs_bits,ring_bits,0,3,true))
#endif
            sdcf_native::selected_xt(tid)->s_extend(narrow.size(),narrow.data(),wide.data(),lhs_bits,ring_bits);
        auto& raw=context.matmul_output;
        if(planning) raw.assign(size_t(rows)*cols,0);
        else multiply_matmul(context.role,*sdcf_native::selected_mult(tid)->iopack,*sdcf_native::selected_mult(tid),wide,weights,*material,raw,&context.matmul_workspace);
#ifdef SDCF_CONVERSION_RUNTIME
        if(!sdcf_conversion::convert(context.role,raw.data(),out,raw.size(),ring_bits,product_bits,depth,2,true))
#endif
            sdcf_native::selected_truncation(tid)->truncate_and_reduce(raw.size(),raw.data(),out,depth,ring_bits);
    } catch(...) {if(material) material->consumed=true;throw;}
    if(planning) return true;
    const auto& m=*material;
    sdcf_native::LogLine()<<"BEAVER_ONLINE {\"query\":"<<context.query<<",\"index\":"<<context.layer->index
        <<",\"operator\":\""<<context.layer->operator_name<<"\",\"ring_bits\":"<<m.bits
        <<",\"product_bits\":"<<product_bits<<",\"floor_shift\":"<<depth
        <<",\"E_values\":"<<m.u.size()<<",\"F_values\":"<<m.v.size()
        <<",\"algebra\":"<<(m.options.algebra?"true":"false")
        <<",\"mapped_layout\":"<<(m.options.mapped_layout?"true":"false")
        <<",\"local_kernel_seconds\":"<<m.online_local.local_kernel_seconds
        <<",\"buffer_seconds\":"<<m.online_local.buffer_seconds
        <<",\"local_products\":"<<m.online_local.local_products
        <<",\"kernel_layout_copy_seconds\":"<<seconds_or_null(m.online_local.copy_instrumented,m.online_local.layout_copy_seconds)
        <<",\"kernel_compute_seconds\":"<<seconds_or_null(m.online_local.copy_instrumented,m.online_local.compute_seconds)
        <<",\"local_kernel_includes_layout_copy\":"<<(m.options.mapped_layout?"false":"true")
        <<",\"mapped_kernel_participants\":"<<(m.options.mapped_layout?"1":"null")
        <<",\"opening_groups\":1,\"opening_logical_rounds\":1,\"consumed\":true,\"conversion\":\""<<conversion_backend()<<"\"}\n";
    return true;
}

bool mulcir(int tid,uint64_t* lhs,uint64_t* rhs,uint64_t* out,
            int elements,int lhs_bits,int rhs_bits,int temp_bits,int out_bits,
            int64_t shrA,int64_t shrB,int64_t demote) {
    if(!use_beaver()) return false;
    const bool planning=sdcf_preprocess::planning_active.load();
    if((!context.online && !planning) || !context.layer ||
       context.layer->function!="MulCir" || tid!=0)
        throw std::logic_error("MulCir Beaver hook called outside its scheduled layer");
    const int product_bits=lhs_bits+rhs_bits;
    const auto& public_args=context.layer->numeric_arguments;
    if(public_args.size()!=9 || public_args[0]*public_args[1]!=elements || public_args[2]!=shrA ||
       public_args[3]!=shrB || public_args[4]!=demote || public_args[5]!=lhs_bits ||
       public_args[6]!=rhs_bits || public_args[7]!=temp_bits || public_args[8]!=out_bits)
        throw std::invalid_argument("MulCir Beaver public contract mismatch");
    auto* material=planning?nullptr:&context.mulcirs.at(context.layer->index);
    if(material && (material->elements!=elements || material->bits!=product_bits || material->consumed))
        throw std::invalid_argument("MulCir Beaver material contract mismatch");
    const auto lhs_mask=segfss::reference::mask(lhs_bits);
    const auto rhs_mask=segfss::reference::mask(rhs_bits);
    std::vector<uint64_t> a(elements),b(elements),wide_a(elements),wide_b(elements),raw;
    for(int i=0;i<elements;++i) { a[i]=lhs[i]&lhs_mask; b[i]=rhs[i]&rhs_mask; }
    try {
#ifdef SDCF_CONVERSION_RUNTIME
        if (!sdcf_conversion::extend_pair(context.role, a.data(), b.data(), wide_a.data(), wide_b.data(),
                                          elements, lhs_bits, rhs_bits, product_bits)) {
        if(!sdcf_conversion::convert(context.role,a.data(),wide_a.data(),elements,lhs_bits,product_bits,0,3,true))
#endif
            sdcf_native::selected_xt(tid)->s_extend(elements,a.data(),wide_a.data(),lhs_bits,product_bits);
#ifdef SDCF_CONVERSION_RUNTIME
        if(!sdcf_conversion::convert(context.role,b.data(),wide_b.data(),elements,rhs_bits,product_bits,0,3,true))
#endif
            sdcf_native::selected_xt(tid)->s_extend(elements,b.data(),wide_b.data(),rhs_bits,product_bits);
#ifdef SDCF_CONVERSION_RUNTIME
        }
#endif
        if(planning) raw.assign(elements,0);
        else raw=multiply_mulcir(context.role,*sdcf_native::selected_mult(tid)->iopack,*sdcf_native::selected_mult(tid),wide_a,wide_b,*material);
        // The native MulCir_thread performs the conversion after this hook.
        std::copy(raw.begin(),raw.end(),out);
        (void)temp_bits; (void)out_bits; (void)shrA; (void)shrB; (void)demote;
    } catch(...) { if(material) material->consumed=true; throw; }
    if(planning) return true;
    const auto& m=*material;
    sdcf_native::LogLine()<<"BEAVER_ONLINE {\"query\":"<<context.query<<",\"index\":"<<context.layer->index
        <<",\"operator\":\""<<context.layer->operator_name<<"\",\"function\":\"MulCir\",\"ring_bits\":"<<m.bits
        <<",\"product_bits\":"<<product_bits<<",\"E_values\":"<<elements<<",\"F_values\":"<<elements
        <<",\"opening_groups\":1,\"opening_logical_rounds\":1,\"consumed\":true,\"conversion\":\""<<conversion_backend()<<"\"}\n";
    return true;
}
}
